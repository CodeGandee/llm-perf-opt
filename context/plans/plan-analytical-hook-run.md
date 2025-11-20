# Plan: Analytical Hook Run for DeepSeek‑OCR

**Goal**  
Run the DeepSeek‑OCR model on real inputs while simultaneously executing the
analytic layer stack as a *shadow model*, so we can collect accurate,
data‑driven FLOPs/IO/memory/KV statistics for **prefill** and **decode** without
assuming output length.

This plan focuses on:

- Where to hook (in the real model and session code),
- How to map runtime shapes/configs onto analytic layers, and
- What artifacts we want to produce from a hooked run.

---

## 1. Context and Existing Building Blocks

### 1.1 Vendor model structure (DeepSeek‑OCR)

Source: `models/deepseek-ocr/modeling_deepseekocr.py`

Key classes:

```python
class DeepseekOCRModel(DeepseekV2Model):
    config_class = DeepseekOCRConfig

    def __init__(self, config: DeepseekV2Config):
        super(DeepseekOCRModel, self).__init__(config)

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()
        n_embed = 1280
        self.projector =  MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))
        ...

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        ...
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        ...
        # Vision path: sam_model, vision_model, projector, masked_scatter_ into inputs_embeds
        ...
        return super(DeepseekOCRModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids=position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class DeepseekOCRForCausalLM(DeepseekV2ForCausalLM):
    config_class = DeepseekOCRConfig

    def __init__(self, config):
        super(DeepseekV2ForCausalLM, self).__init__(config)
        self.model = DeepseekOCRModel(config)
        ...
```

The decoder stack lives in `DeepseekV2Model` / `DeepseekV2DecoderLayer` in
`models/deepseek-ocr/modeling_deepseekv2.py`.

### 1.2 Session and runners

Source: `src/llm_perf_opt/runners/dsocr_session.py`

- `DeepSeekOCRSession.from_local(...)` loads the HF model and tokenizer and
  stores them as `m_model`, `m_tokenizer`, `m_device`, etc.
- `DeepSeekOCRSession` exposes inference helpers (prefill + decode, NVTX
  annotations) that are already used by Stage‑1 profiling.

### 1.3 Existing runtime patching pattern

Source: `src/llm_perf_opt/patches/dsocr_torchlens.py`

We already monkey‑patch vendor modules at runtime for TorchLens tracing:

- `_patch_deepseek_ocr_model_forward()` wraps `DeepseekOCRModel.forward` and
  logs vision‑stage metadata.
- `_patch_deepseek_ocr_for_causal_lm_forward()` wraps
  `DeepseekOCRForCausalLM.forward` and logs decoder stack metadata.

This is an ideal reference for how to add **analytics‑only hooks** without
modifying vendor source files on disk.

### 1.4 Analytic layer stack

Relevant analytic classes in `extern/modelmeter/models/deepseek_ocr/layers/`:

- Vision:
  - `ImageEncoderViT`, `VitModel`, `MlpProjector`, `NoTP*`, etc.
- Decoder:
  - `DeepseekV2DecoderLayer`, `DeepseekV2MLP`, `DeepseekV2MoE`, `DeepseekV2RMSNorm`, `MoEGate`.
- LLaMA primitives:
  - `LlamaFlashAttention2`, `LlamaRotaryEmbedding`.
- Aggregator:
  - `DeepseekOCRModel` (analytic root layer),
  - `_CompositeLayer` (internal helper for composite vision stack).

Each implements:

- `forward_tensor_core_flops`, `forward_cuda_core_flops`,
- `forward_cal_io`, `forward_arithmetic_intensity`,
- `forward_memory_weight`, `forward_memory_activation`,
  `forward_memory_kvcache`,
- and backward analogs (not used for inference).

---

## 2. Design: Analytic Shadow Model via Hooks

### 2.1 Objectives

For each real inference run (per document):

- Attribute analytic FLOPs/IO/memory/KV to:
  - Vision stack (SAM + CLIP + projector) – prefill only.
  - Decoder stack – split into prefill vs decode.
- Derive:
  - FLOPs per document, FLOPs per token, FLOPs per stage.
  - KV‑cache size after prefill and as decode progresses.
  - Activation and weight footprints for capacity planning.

All of this should:

- Avoid extra tensor work (shape‑only arithmetic in hooks),
- Not modify vendor source files on disk,
- Respect the ~10% runtime overhead budget from `research.md`.

### 2.2 Key idea

1. Build an **analytic shadow model** from HF config:
   - Vision: instantiate analytic `ImageEncoderViT`, `VitModel`, `MlpProjector`
     with DeepSeek‑OCR‑like dimensions.
   - Decoder: instantiate one analytic `DeepseekV2DecoderLayer` with HF config
     (`hidden_size`, `intermediate_size`, MoE parameters, num_heads, num_kv_heads).
   - Aggregator: `DeepseekOCRModel.from_layers(_CompositeLayer(vision_layers),
     decoder_layer, num_decoder_layers=config.num_hidden_layers)`.
2. Attach **forward hooks** to selected vendor modules:
   - Vision path: `sam_model`, `vision_model`, `projector` in `DeepseekOCRModel`.
   - Decoder layers: each `DeepseekV2DecoderLayer` in `DeepseekV2Model.layers`.
3. Each hook:
   - Reads runtime shapes (`B`, `S`, spatial grids),
   - Updates the analytic layer’s shape parameters (`m_seq_len`, `m_batch_size`,
     etc.),
   - Computes analytic metrics (FLOPs, IO, memory, KV) and appends them to an
     in‑process accumulator with a `stage` label (`prefill` or `decode`).
4. A small `analytic_stage(...)` context manager (or equivalent flag) around
   prefill vs decode loops tells hooks which stage they are in.

---

## 3. Hook Locations and Mapping

### 3.1 Vision modules

Vendor: `DeepseekOCRModel.__init__` and `forward` in `modeling_deepseekocr.py`:

```python
class DeepseekOCRModel(DeepseekV2Model):
    def __init__(self, config: DeepseekV2Config):
        super(DeepseekOCRModel, self).__init__(config)

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()
        n_embed = 1280
        self.projector =  MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))
        ...

    def forward(..., images, images_seq_mask, images_spatial_crop, ...):
        ...
        sam_model = getattr(self, "sam_model", None)
        vision_model = getattr(self, "vision_model", None)
        ...
        if sam_model is not None and (input_ids.shape[1] != 1 or self.training) and torch.sum(images[0][1]).item() != 0:
            ...
            local_features_1 = sam_model(patches)
            local_features_2 = vision_model(patches, local_features_1)
            ...
            global_features_1 = sam_model(image_ori)
            global_features_2 = vision_model(image_ori, global_features_1)
            ...
            global_features = self.projector(global_features)
            ...
            inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)
        ...
        return super(DeepseekOCRModel, self).forward(...)
```

Hook plan:

- **SAM encoder**: hook `self.sam_model`:
  - Inputs: `[B, 3, H, W]` global; `[P, 3, crop_h, crop_w]` patches.
  - Map to analytic `ImageEncoderViT(img_size=base_size, patch_size=16, ...)`
    and accumulate FLOPs/IO/memory for each call.
- **CLIP encoder**: hook `self.vision_model`:
  - Inputs: `(patches or image_ori, sam_features)`; shapes give token counts.
  - Map to analytic `VitModel` with matching `hidden_size`, `seq_len`, `batch_size`.
- **Projector**: hook `self.projector`:
  - Inputs: `global_features` / `local_features` `[B, N, 2048]`.
  - Map to analytic `MlpProjector(input_dim=2048, output_dim=1280, num_tokens=N, batch_size=B, ...)`.

We can either:

- Attach module hooks directly to `sam_model`, `vision_model`, `projector`, or  
- Wrap `DeepseekOCRModel.forward` (as in `dsocr_torchlens`) and call analytic
  vision layers using pre‑computed shapes (preferred if we want to avoid
  touching vendor submodules).

### 3.2 Decoder layers

Decoder stack is in `DeepseekV2Model` (in `modeling_deepseekv2.py`), roughly:

```python
class DeepseekV2Model(DeepseekV2PreTrainedModel):
    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [DeepseekV2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

Hook plan:

- Iterate over `self.model.layers` in `DeepseekOCRForCausalLM` (or via
  `DeepSeekOCRSession.m_model.model.layers`) and attach hooks:

  ```python
  for layer_idx, layer in enumerate(model.layers):
      layer.register_forward_hook(
          make_decoder_hook(layer_idx, analytic_decoder_layer)
      )
  ```

- Inside `make_decoder_hook`:
  - `inputs[0]` is `hidden_states` of shape `[B, S, h]`.
  - Update analytic `DeepseekV2DecoderLayer`:
    - `m_batch_size = B`, `m_seq_len = S`.
  - Compute FLOPs/IO/KV/memory and log them with:
    - `layer_idx`,
    - `stage = "prefill"` or `"decode"` (from a context manager), and
    - `token_count = S` (for normalization if needed).

### 3.3 Stage attribution (prefill vs decode)

Stage boundaries live in the runner/session layer, not inside the HF model.
We should:

- Add a lightweight `analytic_stage` context manager:

  ```python
  from contextlib import contextmanager
  import contextvars

  _CURRENT_STAGE = contextvars.ContextVar("dsocr_analytic_stage", default="unknown")

  @contextmanager
  def analytic_stage(name: str):
      token = _CURRENT_STAGE.set(name)
      try:
          yield
      finally:
          _CURRENT_STAGE.reset(token)

  def current_stage() -> str:
      return _CURRENT_STAGE.get()
  ```

- Use it inside `DeepSeekOCRSession` prefill/decode flows, e.g.:

  ```python
  with analytic_stage("prefill"):
      run_prefill(...)

  with analytic_stage("decode"):
      for _ in range(max_new_tokens):
          run_one_decode_step(...)
  ```

Hooks call `current_stage()` to tag events.

---

## 4. Stats Collector and Output Artifacts

### 4.1 In‑process accumulator

Define a small accumulator object (similar to `DsocrTorchlensRuntimeLog`) to
collect analytic events:

```python
@define(kw_only=True)
class AnalyticRunStats:
    decoder_steps: list[dict[str, Any]] = field(factory=list)
    vision_steps: list[dict[str, Any]] = field(factory=list)

    def add_decoder_step(self, *, layer_idx: int, stage: str,
                         flops_tflops: float, io_tb: float, kv_gb: float) -> None:
        self.decoder_steps.append(
            {
                "layer_idx": layer_idx,
                "stage": stage,
                "flops_tflops": flops_tflops,
                "io_tb": io_tb,
                "kv_gb": kv_gb,
            }
        )

    def add_vision_step(self, *, module: str, flops_tflops: float,
                         io_tb: float, act_gb: float, w_gb: float) -> None:
        self.vision_steps.append(
            {
                "module": module,
                "flops_tflops": flops_tflops,
                "io_tb": io_tb,
                "act_gb": act_gb,
                "w_gb": w_gb,
            }
        )
```

We can keep a global instance for the duration of a run:

- `reset_analytic_run_stats()`,
- `get_analytic_run_stats()`,
- `write_analytic_run_stats(path)` (JSON).

### 4.2 Aggregation into AnalyticModelReport

After the run:

- Aggregate per‑step events into:
  - per‑layer, per‑stage totals,
  - per‑token averages (decode).
- Map these aggregates into:
  - `ModuleMetricsSnapshot` entries for:
    - `vision/image_encoder_vit`,
    - `vision/vit_model`,
    - `vision/mlp_projector`,
    - `decoder/deepseek_v2_decoder_layer`.
  - Updated `AnalyticModelReport` (or a variant that includes *observed* token
    counts).

This can either:

- reuse `DeepseekOCRStaticAnalyzer.run_analytic(...)` and adjust the
  `module_metrics` based on hook data, or  
- build a separate “runtime‑aware” report under a distinct report id.

### 4.3 Filesystem layout

Write artifacts under the existing run directory structure, e.g.:

```text
tmp/profile-output/<run_id>/
  static_analysis/
    analytic_model/
      report.json
      report.yaml
      layers/...
  analytic_hooks/
    decoder_steps.json
    vision_steps.json
```

This keeps hook‑based analytics separate but discoverable, and allows later
tools to correlate them with Nsight, TorchProfiler, and static analyzer
outputs.

---

## 5. Implementation Steps

1. **Add analytic stage context**
   - Implement `analytic_stage(...)` and `current_stage()` in a small helper
     module (e.g., `src/llm_perf_opt/profiling/analytic_context.py`).
   - Wrap prefill/decode loops in `DeepSeekOCRSession` with `analytic_stage`.

2. **Build analytic shadow layers from config**
   - Implement a helper `build_analytic_layers_from_config(cfg)` that returns:
     - `vision_layers` (`ImageEncoderViT`, `VitModel`, `MlpProjector`),
     - `decoder_layer` (`DeepseekV2DecoderLayer`), and
     - `num_layers`.
   - Reuse logic from `DeepseekOCRStaticAnalyzer._infer_model_dims()` and
     `_build_vision_layers()` / `_build_decoder_layer()`.

3. **Implement `AnalyticRunStats` accumulator**
   - Add a small module (e.g., `src/llm_perf_opt/profiling/analytic_hooks.py`)
     with:
     - `AnalyticRunStats` data model,
     - global instance + reset/get/write helpers.

4. **Attach hooks to decoder layers**
   - In `DeepSeekOCRSession.from_local(...)` (or a dedicated hook registration
     function), after the HF model is constructed:
     - locate `DeepseekV2Model` via `session.m_model.model`,
     - iterate `model.layers` and register `forward_hook`s using `make_decoder_hook`.

5. **Attach hooks to vision path**
   - Either:
     - module‑level hooks on `sam_model`, `vision_model`, `projector`, or  
     - a single wrapper around `DeepseekOCRModel.forward` (similar to
       `dsocr_torchlens._patch_deepseek_ocr_model_forward`) that:
       - inspects `images`, `images_spatial_crop` to infer patch/global shapes,
       - calls analytic vision layers once per document with those shapes, and
       - records a single combined vision cost event in `AnalyticRunStats`.

6. **Add CLI or configuration switch**
   - Add a Hydra flag or CLI option (e.g., `analytic_hooks.enable=true`) to
     turn hooks on/off for a run.

7. **Aggregate and persist results**
   - At the end of a run (e.g., in `llm_profile_runner` or a dedicated
     analytic driver), call `write_analytic_run_stats(...)` into the run
     directory.
   - Optionally post‑process into a `AnalyticModelReport` variant that includes
     observed decode lengths and hook‑based metrics.

8. **Validation**
   - Compare:
     - hook‑based FLOPs vs static analyzer FLOPs,
     - hook‑based per‑token FLOPs vs the `Simulated Run` formulas in
       `HOLISTIC_ANALYSIS.md`,
     - hook‑based KV sizes vs direct KV tensor inspections for a few
       small‑scale runs.
   - Ensure runtime overhead stays within the agreed budget.

This plan reuses existing monkeypatching patterns (from `dsocr_torchlens`) and
the analytic layer stack to provide a **hooked analytical mode** that tracks
prefill/decode costs on real workloads without assuming decode length in
advance.

