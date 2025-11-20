**What to Refactor**
- Files (ModelMeter analytic layers and docs):
  - `extern/modelmeter/models/deepseek_ocr/layers/llama/llama_flash_attention2.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/decoder/deepseek_v2_decoder_layer.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - `extern/modelmeter/models/deepseek_ocr/HOLISTIC_ANALYSIS.md`
  - `extern/modelmeter/models/deepseek_ocr/LAYER_IO_ESTIMATION_GUIDE.md`
- Scope:
  - Introduce explicit **query vs context** sequence-length modeling in the analytic attention and decoder layers (prefill vs decode).
  - Add first-class **prefill / decode cost helpers** on `DeepseekV2DecoderLayer` and `DeepseekOCRModel` (FLOPs / I/O / memory).
  - Keep the existing “single `seq_len`” API surface backward compatible for current call sites (tests, scripts, `llm_perf_opt` integration).

**Why Refactor**
- Today, prefill vs decode are modeled **by convention**:
  - Prefill uses `seq_len = S_prefill`.
  - Decode uses `seq_len = 1` and we *interpret* the formulas as “per-token over cached context” (see `HOLISTIC_ANALYSIS.md` §3 and §6).
  - There is no way to express `S_q` vs `S_kv` directly in the analytic APIs.
- This makes it harder to:
  - Express per-token decode FLOPs / I/O for arbitrary context length (`S_prefill`) in a single, self-documenting call.
  - Systematically derive prefill vs decode costs in downstream tooling without re-encoding the same conventions.
  - Evolve KV-aware modeling (e.g., separate `context_len` / `generated_len`) as outlined in the TODO sections of `HOLISTIC_ANALYSIS.md`.
- `HOLISTIC_ANALYSIS.md` already sketches a better abstraction:
  - Explicit `seq_len_q` / `seq_len_kv`.
  - Decode-mode helpers that take a context length and return **per-token** metrics.
  - Root-model helpers like `DeepseekOCRModel.estimate_prefill_cost(...)` to expose a clean “stage cost” API.
  Refactoring the code to match this design will align implementation with documentation and reduce duplication in consumers like `llm_perf_opt`.

**Key Questions and Answers**
1) **How do we distinguish prefill vs decode analytically, and support both modes?**
   - Prefill is modeled as a full-sequence pass with **query length equals context length**:
     `S_q = S_kv = S_prefill`. Vision + decoder FLOPs / I/O / memory are attributed to this single stage.
   - Decode (per token) is modeled as a **single-token query over an existing context**:
     `S_q = 1`, `S_kv = S_prefill (+ t)` at decode step `t`. All metrics are reported **per new token**, with context length explicit.
   - The refactor will make both modes **first-class** by:
     - Extending attention / decoder layers to track `seq_len_q` and `seq_len_kv` instead of a single `seq_len`.
     - Adding helpers such as `estimate_prefill_cost(...)` and `estimate_decode_cost_per_token(context_len=...)` on `DeepseekV2DecoderLayer` and `DeepseekOCRModel` so callers do not have to re-encode these conventions.

2) **How do we model KV‑cache size from prefill and its growth during decode? Where does the “fake kv_cache” live?**
   - Prefill KV‑cache size is a pure function of configuration: batch size, number of KV heads, head dim, number of layers, and **context length** `S_prefill`. We will:
     - Provide a helper like `estimate_kv_cache_size_gb(context_len, decode_len=0)` on the decoder or root model, implemented using the same formulas as `forward_memory_kvcache()`.
   - For simulated runs that grow KV‑cache during decode, we will introduce a lightweight **synthetic KV‑cache meta object** that records both shape and derived sizes, for example:
     ```python
     @dataclass
     class SyntheticKVCache:
         batch: int
         num_kv_heads: int
         head_dim: int
         context_len: int
         decode_len: int = 0
         bytes_per_val: float = 2.0  # fp16/bf16

         def total_len(self) -> int:
             return self.context_len + self.decode_len

         def size_bytes(self) -> float:
             s_total = self.total_len()
             # 2 (K,V) * B * H_kv * S_total * d * bytes_per_val
             return 2.0 * self.batch * self.num_kv_heads * s_total * self.head_dim * self.bytes_per_val

         def size_gb(self) -> float:
             return self.size_bytes() / (1024.0**3)
     ```
     - This object carries only lengths (no real tensors) and will live next to the decoder analytic code (same module or a small `kv_cache_meta.py` under `models/deepseek_ocr/layers/decoder/`).
     - `HOLISTIC_ANALYSIS.md`’s simulated-run examples will be updated to use `SyntheticKVCache` plus `estimate_kv_cache_size_gb(...)` / `size_gb()` to show KV‑cache growth as decode progresses.

3) **How do we verify analytic prefill/decode FLOP counts against the real model (like `.verify_by_impl()`)?**
   - We will add a dedicated verification path, analogous to the per-layer `.verify_by_impl()` helpers, but targeted at whole‑stage prefill/decode FLOPs:
     - A new script (e.g., `models/deepseek_ocr/scripts/run_verify_prefill_decode.py`) will:
       - Instantiate the reference DeepSeek‑OCR model from its HF or local implementation (using `from_pretrained` or direct import from `models/deepseek-ocr/modeling_deepseekocr.py`), **without modifying its source files**.
       - Use `torch.utils.flop_counter.FlopCounterMode` to measure:
         - A **prefill** forward pass at context length `S_prefill` (image + prompt).
         - A **single decode** step with `S_q = 1`, `S_kv = S_prefill` by manually calling the underlying `DeepseekV2ForCausalLM` or `DeepseekV2Model` with `past_key_values`.
       - Construct the corresponding analytic objects and compare:
         - Measured FLOPs vs `DeepseekOCRModel.estimate_prefill_cost(...)`.
         - Measured per-token decode FLOPs vs `DeepseekV2DecoderLayer.estimate_decode_cost_per_token(context_len=S_prefill)`.
   - A concrete verification sketch (pseudocode):
     ```python
     import torch
     from torch.utils.flop_counter import FlopCounterMode
     from models.deepseek_ocr.modeling_deepseekocr import DeepseekV2ForCausalLM
     from modelmeter.models.deepseek_ocr.layers.core.deepseek_ocr_model import DeepseekOCRModel
     from modelmeter.models.deepseek_ocr.layers.decoder.deepseek_v2_decoder_layer import DeepseekV2DecoderLayer

     # 1) Load reference model (weights + config)
     hf_model = DeepseekV2ForCausalLM.from_pretrained("<DEEPSEEK_OCR_MODEL_PATH>").eval().to("cuda")

     # 2) Build synthetic prefill inputs (same shapes as used in dsocr_analyzer)
     input_ids, model_kwargs = build_prefill_inputs(S_prefill=1024, device="cuda")  # helper mirroring DeepseekOCRStaticAnalyzer.prepare_inputs

     # 3) Measure prefill FLOPs with FlopCounterMode
     with torch.no_grad():
         with FlopCounterMode(mods=hf_model, display=False) as flop_counter:
             out = hf_model(
                 input_ids=input_ids,
                 use_cache=True,
                 **model_kwargs,
             )
         flops_prefill_measured = flop_counter.get_total_flops()
         past_kv = out.past_key_values

     # 4) Measure one-step decode FLOPs (S_q = 1, S_kv = S_prefill)
     next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
     with torch.no_grad():
         with FlopCounterMode(mods=hf_model, display=False) as flop_counter:
             out_decode = hf_model(
                 input_ids=next_token,
                 past_key_values=past_kv,
                 use_cache=True,
             )
         flops_decode_one_step_measured = flop_counter.get_total_flops()

     # 5) Build analytic model with matching config and shapes
     decoder_layer = DeepseekV2DecoderLayer(
         hidden_size=hf_model.config.hidden_size,
         num_heads=hf_model.config.num_attention_heads,
         seq_len_q=1,
         seq_len_kv=input_ids.size(1),
         intermediate_size=hf_model.config.intermediate_size,
         num_experts=hf_model.config.n_routed_experts,
         batch_size=input_ids.size(0),
         num_key_value_heads=hf_model.config.num_key_value_heads,
         k_active=hf_model.config.k_active,
         num_shared_experts=hf_model.config.num_shared_experts,
     )
     # vision_stack construction omitted here; it follows the same pattern as dsocr_analyzer
     deepseek_model = DeepseekOCRModel.from_layers(vision_stack, decoder_layer, num_decoder_layers=hf_model.config.num_hidden_layers)

     prefill_cost = deepseek_model.estimate_prefill_cost()
     decode_cost = decoder_layer.estimate_decode_cost_per_token(context_len=input_ids.size(1))

     # 6) Compare FLOPs (convert TFLOPs to FLOPs)
     tol = 0.05  # 5% relative tolerance
     def _rel_diff(a, b):
         denom = max(abs(b), 1.0)
         return abs(a - b) / denom

     prefill_rel_diff = _rel_diff(prefill_cost.flops_tflops * 1.0e12, flops_prefill_measured)
     decode_rel_diff = _rel_diff(decode_cost.flops_tflops * 1.0e12, flops_decode_one_step_measured)
     assert prefill_rel_diff <= tol
     assert decode_rel_diff <= tol
     ```
   - If we need to control the path inside the HF model for measurement (e.g., bypass `.generate()` and call the low-level model directly), we can do so by:
     - Calling `hf_model.model` or `hf_model.base_model` explicitly with `input_ids` and `past_key_values`.
     - Monkeypatching *in-memory attributes* (e.g., swapping out attention modules with wrappers) for measurement purposes, **without modifying the DeepSeek-OCR source files** on disk.
   - Optionally, smaller helpers like `verify_prefill_flops_by_impl(...)` / `verify_decode_flops_by_impl(...)` can be added inside the analytic modules (accepting `impl_file`, `impl_class`, `device`) to reuse this logic from both scripts and tests, mirroring the existing per-layer `.verify_by_impl()` convention.

**How to Refactor**
1) Baseline audit and call-site mapping
   - Enumerate where `DeepseekV2DecoderLayer` and `LlamaFlashAttention2` are instantiated:
     - Analytic layers/tests: `extern/modelmeter/models/deepseek_ocr/layers/**.py`
     - Verify scripts: `extern/modelmeter/models/deepseek_ocr/scripts/run_verify_*.py`
     - External consumers: `src/llm_perf_opt/runners/dsocr_analyzer.py`, `tests/unit/deepseek_ocr/test_analytic_layers_scaling.py`.
   - Document the current expectations around `seq_len` in these call sites (e.g., always “full context length”, or “1 for per-token decode”).

2) Extend `LlamaFlashAttention2` with query/context lengths (backward compatible)
   - Internals:
     - Introduce fields `m_seq_len_q` and `m_seq_len_kv`.
     - Keep the existing `seq_len: int` parameter in `__init__`, but implement it as a **compatibility alias**:
       - If only `seq_len` is provided, set `m_seq_len_q = m_seq_len_kv = seq_len`.
       - Provide an optional new initializer path that accepts `seq_len_q` and `seq_len_kv` explicitly.
   - FLOPs / I/O methods:
     - Update internal formulas to use `S_q` and `S_kv` separately:
       ```python
       # Before (conceptual)
       s = float(self.m_seq_len)
       flops_attn = 4.0 * b * h * s * d * s  # s × s

       # After (conceptual)
       s_q = float(self.m_seq_len_q)
       s_kv = float(self.m_seq_len_kv)
       flops_attn = 4.0 * b * h * s_q * d * s_kv  # q × kv
       ```
     - Keep the public method names (`forward_tensor_core_flops`, `forward_cal_io`, etc.) unchanged, but make their behavior depend on `m_seq_len_q` / `m_seq_len_kv`.
   - Optional convenience helpers:
     - Add thin wrappers (non-breaking) that make intent explicit:
       ```python
       def forward_flops_prefill(self) -> float: ...
       def forward_flops_decode_per_token(self, *, context_len: int) -> float: ...
       ```
       These can be implemented in terms of the existing per-layer methods but provide a clearer semantic entry point for prefill vs decode.

3) Extend `DeepseekV2DecoderLayer` with prefill/decode-aware APIs
   - Constructor:
     - Mirror the `LlamaFlashAttention2` change by holding onto both `seq_len_q` and `seq_len_kv`, while keeping the existing `seq_len` argument for compatibility.
     - Ensure that all internal sublayers (attention, MLP/MoE, RMSNorm) use the new lengths where appropriate.
   - Stage-specific helpers:
     - Introduce small, self-contained helpers that return structured metrics:
       ```python
       @dataclass
       class StageCost:
           flops_tflops: float
           io_tb: float
           weights_gb: float
           activations_gb: float
           kv_gb: float

       def estimate_prefill_cost(self) -> StageCost:
           # uses seq_len_q == seq_len_kv == S_prefill
           ...

       def estimate_decode_cost_per_token(self, *, context_len: int) -> StageCost:
           # internally treat S_q = 1, S_kv = context_len
           ...
       ```
     - Implement these in terms of the existing analytic methods:
       - FLOPs: `forward_tensor_core_flops` + `forward_cuda_core_flops`.
       - I/O: `forward_cal_io`.
       - Memory: `forward_memory_weight` / `forward_memory_activation` / `forward_memory_kvcache`.
   - Tests:
     - Add unit tests under `extern/modelmeter/models/deepseek_ocr/layers/tests/` (或现有测试模块) 以验证：
       - `estimate_prefill_cost(seq_len_q=S, seq_len_kv=S)` 在数值上与当前 `seq_len=S` 行为一致。
       - `estimate_decode_cost_per_token(context_len=S_prefill)` 与 `HOLISTIC_ANALYSIS.md` 中使用 `seq_len=1` 的近似保持一致（在容忍误差范围内）。

4) Add root-level helpers on `DeepseekOCRModel`
   - Implement methods to compute full-model stage costs:
     ```python
     class DeepseekOCRModel(BaseLayer):
         def estimate_prefill_cost(self) -> StageCost:
             # aggregates vision stack + decoder stack in prefill mode
             ...

         def estimate_decode_cost_per_token(self, *, context_len: int) -> StageCost:
             # aggregates decoder stack in decode mode
             ...
     ```
   - Ensure these helpers:
     - Use the new `StageCost` dataclass (or similar) to keep return types simple and serializable.
     - Respect the existing `_CompositeLayer` grouping for the vision stack.
     - Are pure analytic calculations (no runtime tensors).

5) Thread the new APIs into `DeepseekOCRStaticAnalyzer` and usage sites (optional but recommended)
   - Update `src/llm_perf_opt/runners/dsocr_analyzer.py` to:
     - Prefer `DeepseekOCRModel.estimate_prefill_cost(...)` and `estimate_decode_cost_per_token(...)` where today it manually constructs decoder layers and recomputes the same formulas.
     - Use the structured `StageCost` results when populating `AnalyticModelReport` fields for prefill vs decode.
   - Keep the analyzer logic read-only with respect to model weights; only static configuration should change.

6) Documentation alignment
   - Update `extern/modelmeter/models/deepseek_ocr/HOLISTIC_ANALYSIS.md`:
     - Replace the pseudo-code in sections 2–3 and 6–8 with snippets that call the new helpers directly (e.g., `decoder.estimate_prefill_cost()`, `decoder.estimate_decode_cost_per_token(context_len=...)`).
     - Clarify that “Option A” (fixed-context per-token model) is now directly supported by the analytic API.
   - Update `LAYER_IO_ESTIMATION_GUIDE.md` to:
     - Mention that attention/decoder layers now distinguish `seq_len_q` vs `seq_len_kv`, and how this affects I/O formulas and KV-cache calculations.

7) Migration and deprecation strategy
   - Keep existing signatures usable:
     - `__init__(..., seq_len: int, ...)` continues to work; internally maps to `seq_len_q = seq_len_kv = seq_len`.
     - Existing `forward_*` methods keep their names and semantics when called on such objects.
   - Optionally:
     - Add warnings (via logging) if a caller uses a no-longer-recommended path (e.g., `seq_len=1` with large implicit KV), pointing them to the new helpers instead.
     - Mark any future-breaking parameters in docstrings rather than raising at runtime.


**Before/After Snippets**
- Before: decode is modeled via `seq_len=1` and interpreted by the caller.
```python
# Decoder analytic layer instantiation (decode approximation today)
decoder_layer_decode = DeepseekV2DecoderLayer(
    hidden_size=hidden_size,
    num_heads=num_heads,
    seq_len=1,  # per-token decode, context implied
    intermediate_size=intermediate_size,
    num_experts=num_experts,
    batch_size=batch_size,
)

flops_tflops_decode_per_token = (
    decoder_layer_decode.forward_tensor_core_flops()
    + decoder_layer_decode.forward_cuda_core_flops()
) * num_layers
io_tb_decode_per_token = decoder_layer_decode.forward_cal_io() * num_layers
```

- After: explicit query/context lengths and a dedicated helper.
```python
# Decoder analytic layer with explicit query/context lengths
decoder_layer = DeepseekV2DecoderLayer(
    hidden_size=hidden_size,
    num_heads=num_heads,
    seq_len_q=1,                  # one new token
    seq_len_kv=S_prefill,         # cached context length
    intermediate_size=intermediate_size,
    num_experts=num_experts,
    batch_size=batch_size,
)

decode_cost = decoder_layer.estimate_decode_cost_per_token(context_len=S_prefill)
flops_tflops_decode_per_token = decode_cost.flops_tflops * num_layers
io_tb_decode_per_token = decode_cost.io_tb * num_layers
```

- Before: root model prefill cost assembled manually.
```python
vision_stack = _CompositeLayer(layers=[image_encoder_vit, vit_model, mlp_projector])
decoder_layer = DeepseekV2DecoderLayer(...)
deepseek_model = DeepseekOCRModel.from_layers(
    vision_stack,
    decoder_layer,
    num_decoder_layers=num_layers,
)

flops_tflops_prefill = (
    deepseek_model.forward_tensor_core_flops()
    + deepseek_model.forward_cuda_core_flops()
)
io_tb_prefill = deepseek_model.forward_cal_io()
```

- After: root model exposes explicit prefill helper.
```python
deepseek_model = DeepseekOCRModel.from_layers(
    vision_stack,
    decoder_layer,
    num_decoder_layers=num_layers,
)

prefill_cost = deepseek_model.estimate_prefill_cost()
flops_tflops_prefill = prefill_cost.flops_tflops
io_tb_prefill = prefill_cost.io_tb
weights_gb_prefill = prefill_cost.weights_gb
kv_gb_prefill = prefill_cost.kv_gb
```

**Impact Analysis**
- Functional impact:
  - Enables direct, API-level access to **prefill vs decode** FLOPs / I/O / memory without re-implementing formulas in downstream code.
  - Improves modeling fidelity for decode, since attention FLOPs can explicitly depend on `S_q × S_kv` rather than an implicit `seq_len=1` convention.
  - Makes KV-cache modeling more transparent by associating cache size and per-token cost with dedicated helpers.
- Compatibility:
  - Existing code that constructs `DeepseekV2DecoderLayer(seq_len=S)` and calls `forward_*` methods will continue to behave as before (internally, `seq_len_q = seq_len_kv = S`).
  - Existing tests and verify scripts should only need minimal adjustments (or none) if we keep the default constructor signature and add helpers instead of changing existing method contracts.
- Risks:
  - Mis-wiring `seq_len_q` / `seq_len_kv` could silently skew analytic FLOPs/IO numbers; mitigate via unit tests that compare new helpers to the current formulas in `HOLISTIC_ANALYSIS.md`.
  - If downstream tools start relying on the new helpers, any future changes to their return type must be done carefully (e.g., adding fields to `StageCost` instead of changing the shape).
  - Additional complexity in the analytic layer classes; mitigated by keeping helpers small, pure, and well-documented, and by centralizing shared math.

**Expected Outcome**
- A clear, documented analytic API for DeepSeek-OCR that:
  - Separately models prefill and per-token decode costs.
  - Exposes FLOPs, I/O, and memory metrics for each stage via small, composable helpers.
  - Matches the design sketched in `HOLISTIC_ANALYSIS.md` and simplifies the logic in consumers like `dsocr_analyzer`.
- Downstream tools can compute:
  - Total prefill FLOPs / I/O / memory.
  - Per-token decode cost at a given context length.
  - KV-cache footprint and its growth over decode tokens.
  with minimal bespoke code and reduced risk of modeling drift.

**References**
- Code and docs
  - `extern/modelmeter/models/deepseek_ocr/layers/decoder/deepseek_v2_decoder_layer.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/llama/llama_flash_attention2.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - `extern/modelmeter/models/deepseek_ocr/HOLISTIC_ANALYSIS.md`
  - `extern/modelmeter/models/deepseek_ocr/LAYER_IO_ESTIMATION_GUIDE.md`
  - `extern/modelmeter/models/deepseek_ocr/LAYER_IMPL_GUIDE.md`
  - `src/llm_perf_opt/runners/dsocr_analyzer.py`
- Third-party libraries (Context7 IDs)
  - `/pytorch/pytorch`
  - `/huggingface/transformers`
