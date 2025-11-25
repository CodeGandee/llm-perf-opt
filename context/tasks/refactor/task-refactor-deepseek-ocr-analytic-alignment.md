# Refactor Plan: Align Analytic DeepSeek-OCR Model with Vendor Implementation

## What to Refactor

- **Analytic root model**
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py` (`DeepseekOCRModel`, `start_vision`, `start_prefill`, `start_decode`, `decode_one_token`).
  - Vision composite layer wiring (SAM + CLIP + projector) and how per-stage FLOPs are aggregated via `_iter_layers()`.
- **Analytic vision stack**
  - `ImageEncoderViT`, `VitModel`, `NoTPTransformer`, `MlpProjector` analytic layers and their configs under:
    - `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`
    - `extern/modelmeter/models/deepseek_ocr/configs/deepseek_ocr.yaml`
  - Missing workload parameters (multi-view count, crop grid, projector token count) that currently assume a single 1024×1024 view and a fixed token count.
- **Decoder / attention FLOP policy**
  - Analytic decoder layer configuration (`DeepseekV2DecoderLayer` analytic) vs vendor `DeepseekV2Model` with FlashAttention2.
  - How `_set_ignore_torch_unsupported_flop_count(...)` is used in verification scripts to match `FlopCounterMode`.
- **Hydra configs and runtime orchestration**
  - Model configs under `conf/model/deepseek_ocr/**` (especially `arch` and `infer` groups) that describe preprocess, prompt, and runtime batch shapes.
  - Runners and analyzers that construct analytic workloads:
    - `src/llm_perf_opt/runners/dsocr_analyzer.py`
    - `src/llm_perf_opt/runners/dsocr_session.py`
    - `src/llm_perf_opt/runners/llm_profile_runner.py`
  - ModelMeter verification scripts:
    - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_prefill_decode.py`
    - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end.py`
    - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end_vision.py`
- **Static analysis / reports**
  - Torchinfo baseline in `reports/20211117-dsorc-op-analysis/static-20251118-130533/torchinfo-summary.txt` that captures SAM/CLIP/projector shapes and multiplicity.
  - Issue docs already describing the gap:
    - `extern/modelmeter/models/deepseek_ocr/docs/issues/issue-end2end-flop-gap.md`
    - `extern/modelmeter/models/deepseek_ocr/docs/issues/issue-vision-flop-underestimation.md`

## Why Refactor

- **Correctness of analytic FLOPs**
  - Current analytic vision FLOPs underestimate real workload by ~78% on OmniDocBench examples because they ignore multi-view processing, full CLIP depth, and dynamic projector token counts.
  - End-to-end analytic FLOPs similarly diverge from `FlopCounterMode` measurements, especially in prefill/vision stages.
- **Workload fidelity**
  - The analytic model was derived from static torchinfo shapes and does not encode runtime multiplicity (e.g., global + local crops, crop grid, valid image tokens) implemented in `modeling_deepseekocr.py`.
  - Runners and verification scripts sometimes hard-code workload assumptions instead of reusing a single config source of truth.
- **Maintainability and extension**
  - As DeepSeek-OCR evolves (different crop strategies, prompts, or model sizes), we need clear configuration points (Hydra + analytic params) to keep analytic modeling aligned.
  - Centralizing workload parameters (prompt, base/image size, crop_mode, num_views, etc.) in config reduces duplication across runners, analyzers, and verification scripts.
- **Debuggability**
  - Mismatches between analytic and measured FLOPs are currently explained in issue docs but not encoded as invariants/tests.
  - A structured refactor with explicit knobs for attention implementation, multi-view scaling, and projector tokens will make it easier to reason about future discrepancies.

## How to Refactor

### 1. Introduce explicit workload parameters to the analytic vision stack

- **Add workload fields to analytic model**
  - Extend `DeepseekOCRModel` to track vision workload beyond `batch_size`:
    - `m_num_views` (total views per image: global + crops).
    - Optional `m_global_view_tokens` and `m_local_view_tokens` (or encode via a small `VisionWorkload` dataclass).
  - Update `start_vision(...)` to accept these parameters or a structured workload object.

  **Before (simplified):**
  ```python
  def start_vision(self, *, batch_size: int = 1) -> None:
      if batch_size <= 0:
          raise ValueError("batch_size must be positive")
      if self.m_vision_layer is None:
          raise ValueError("Vision layer must be configured before start_vision")
      self.m_stage = "vision"
      self.m_batch_size = batch_size
  ```

  **After (idea):**
  ```python
  def start_vision(
      self,
      *,
      batch_size: int = 1,
      num_views: int = 1,
  ) -> None:
      if batch_size <= 0 or num_views <= 0:
          raise ValueError("batch_size and num_views must be positive")
      if self.m_vision_layer is None:
          raise ValueError("Vision layer must be configured before start_vision")
      self.m_stage = "vision"
      self.m_batch_size = batch_size * num_views
      self.m_num_views = num_views
  ```

- **Propagate workload into vision sublayers**
  - For the `_CompositeLayer` that wraps `ImageEncoderViT`, `VitModel`, and `MlpProjector`, make sure each sublayer can receive:
    - Effective batch size (`batch_size * num_views`).
    - Token counts for projector (global + patches) consistent with vendor logic.
  - Option A: keep sublayers unchanged and apply multiplicative factors at the composite level (e.g., scale FLOPs by `num_views` and token-count ratios).
  - Option B: add small `set_workload(...)` methods to vision layers so that FLOP helpers can use `num_views`, image size, and crop grid explicitly.

### 2. Align CLIP depth, projector tokens, and multi-view multiplicity

- **Match CLIP transformer depth**
  - Update `vision.notp_transformer.blocks` in the analytic config to represent all 24 CLIP layers (as in `vit_model_cfg.num_layers = 24`), or document and implement a 6× scaling factor if we intentionally keep 4 analytic blocks.
  - Prefer explicit blocks (24) to avoid implicit multipliers that are easy to forget in future changes.

- **Encode multi-view behavior**
  - Derive `num_views` and crop grid `(w_crop, h_crop)` from the same logic used in:
    - `modeling_deepseekocr.dynamic_preprocess(...)`
    - The reimplementation in `dsocr_analyzer.py` / `dsocr_session.py` and manual scripts.
  - Treat cropping as **purely shape-dependent**, matching vendor behavior:
    - Compute `aspect_ratio = orig_width / orig_height`.
    - Enumerate candidate grid shapes `(i, j)` with `min_num <= i*j <= max_num`.
    - Use `find_closest_aspect_ratio(...)` with the same tie-break rule on image area to pick `(w_crop, h_crop)`.
    - Resize to `(image_size * w_crop, image_size * h_crop)` and cut a regular grid of `w_crop * h_crop` tiles.
    - Do **not** introduce any content-dependent heuristics (no saliency/edge-based cropping); reuse the existing helper to avoid drift.
  - Encapsulate this into a shared helper (e.g., `compute_crop_grid_from_shape(...)`) that both the analytic workload builder and vendor-aligned runners call, so the grid is guaranteed to be identical.
  - For the default DeepSeek-OCR workload (`base_size=1024`, `image_size=640`, `crop_mode=True`), expose:
    - `vision.workload.num_views` (e.g., `1 + w_crop * h_crop`).
    - Optional `vision.workload.global_tokens` and `vision.workload.local_tokens_per_view` used by the projector layer.

  **Before (config, simplified):**
  ```yaml
  vision:
    projector:
      num_tokens: 577
  ```

  **After (idea):**
  ```yaml
  vision:
    workload:
      num_views: 7          # 1 global + 6 crops (example)
      global_tokens: 256
      local_tokens_per_view: 100
    projector:
      num_tokens: ${vision.workload.global_tokens} +
                  ${vision.workload.local_tokens_per_view} *
                  (${vision.workload.num_views} - 1)
  ```

- **Projector FLOPs**
  - Update analytic `MlpProjector` configuration (`input_dim`, `num_tokens`, downsample ratio) to reflect the concatenated SAM + CLIP features for all views and the actual number of tokens fed into the projector.
  - Ensure that any downsampling or token pooling used in the real projector (`downsample_ratio`, `token_pooling`) is included in the analytic `get_flops_per_sample`.

### 3. Wire Hydra configs and orchestration scripts to the analytic workload

- **Single source of truth for prompt and preprocess**
  - Confirm that `conf/model/deepseek_ocr/infer` contains:
    - `decoder_prompt`
    - `preprocess.base_size`, `preprocess.image_size`, `preprocess.crop_mode`, `preprocess.patch_size`, `preprocess.downsample_ratio`
  - Ensure all DeepSeek-OCR runners and scripts use `infer.decoder_prompt` and preprocess settings from Hydra, not hard-coded strings or shapes.

- **Connect runners to analytic model parameters**
  - In `dsocr_analyzer.py` and `dsocr_session.py`, when constructing `AnalysisConfig` or calling `DeepseekOCRModel.start_vision` / `start_prefill`:
    - Compute `(w_crop, h_crop)` using the dynamic-preprocess logic already present.
    - Set `num_views` and token counts on the analytic vision stack via new methods or via a `VisionWorkload` object.
  - Ensure `llm_profile_runner` passes the same workload parameters when it instantiates analytic models for profiling runs.

- **Verification scripts**
  - Update `run_verify_end2end.py` and `run_verify_end2end_vision.py` to:
    - Derive analytic workload parameters from the same `_build_end_to_end_inputs(...)` used for vendor runs (image path, base/image size, crop_mode, crop grid).
    - Call the new `start_vision(..., num_views=...)` and projector token configuration so analytic FLOPs reflect the exact image and crop grid used in the test.

### 4. Make FLOP-count policy for FlashAttention and SDPA explicit

- **Decoder attention policy**
  - Keep the existing behavior where analytic attention FLOPs for FlashAttention2 are optionally ignored to match `FlopCounterMode` (which cannot see them).
  - Expose a clear config switch, e.g.:
    - `decoder.ignore_torch_unsupported_flops` (bool), default `true` for verification parity.
  - Ensure `DeepseekOCRModel` and decoder analytic layers read this flag and adjust attention FLOPs accordingly.

- **Documented behavior**
  - In the analytic config or a short docstring, document that:
    - When `ignore_torch_unsupported_flops=true`, FlashAttention2 core FLOPs are not counted, matching PyTorch’s flop counter.
    - When `false`, analytic FLOPs reflect the full algorithmic cost, which will exceed `FlopCounterMode` numbers.

### 5. Strengthen tests and verification coverage

- **Extend vision-only verification**
  - Add small table-driven test cases for `run_verify_end2end_vision.py`:
    - Multiple OmniDocBench images with varying aspect ratios → different crop grids.
    - Check relative FLOP difference for vision ≤ 5–10%.
  - Consider a unit-style harness around the analytic vision stack that:
    - Accepts a synthetic `VisionWorkload` (num_views, tokens, image_size).
    - Compares analytic FLOPs to torchinfo-based baselines for standard shapes.

- **End-to-end verifier**
  - Ensure `run_verify_end2end.py`:
    - Uses `decoder_prompt` from Hydra infer config.
    - Caps decode steps (e.g., 10) but configures the analytic decode workload to match prefill context length and decode length.
  - Add assertions or logging when relative FLOP difference exceeds a threshold, with breakdown by stage (vision vs prefill vs decode).

### 6. Migration and cleanup

- **Backwards compatibility**
  - Provide sensible defaults for new workload parameters (e.g., `num_views=1`) so existing callers that only care about single-view runs continue to work.
  - Keep old APIs (e.g., `start_vision(batch_size=...)`) callable, possibly by adding defaulted parameters or thin wrappers.

- **Remove duplicated logic**
  - Once the analytic workload is fully driven by Hydra configs and shared helpers:
    - Remove any remaining hard-coded crop grids, prompt strings, or token counts from verification scripts and manual tests.
  - Update documentation under `docs/` and `extern/modelmeter/models/deepseek_ocr/docs/` to reference the new workload knobs and explain how to retarget FLOP modeling for different resolutions or crop strategies.

## Impact Analysis

- **Analytic FLOP values will change**
  - Vision FLOPs will increase (multi-view, full CLIP depth, realistic projector tokens), moving from ~1.0 TF to closer to the ~4.5 TF measured by `FlopCounterMode` for typical OCR pages.
  - End-to-end FLOP estimates (prefill + decode) will shift accordingly. Existing dashboards or reports that assume the old numbers will need to be updated.

- **Configuration surface grows**
  - Additional workload parameters (num_views, crop grid, token counts) increase config complexity but make workloads explicit and reproducible.
  - Mitigation: document defaults and provide high-level presets (e.g., `deepseek_ocr.default`) that keep common cases simple.

- **Risk of divergence between vendors and analytic model**
  - If future changes in `modeling_deepseekocr.py` adjust dynamic cropping or prompt formatting, analytic configs could drift again.
  - Mitigation:
    - Centralize vendor-aligned logic (dynamic_preprocess, prompt building) in shared helpers reused by both runners and verification scripts.
    - Add regular checks or CI jobs running `run_verify_end2end.py` / `run_verify_end2end_vision.py` on a small image set.

- **Runtime and test cost**
  - Vision-only and end-to-end verification scripts will still be bounded (e.g., 10 decode tokens) but will incur slightly more analytic computation.
  - Mitigation: keep verification image set small and reuse cached torchinfo/static analyses when possible.

## Expected Outcome

- Analytic DeepSeek-OCR FLOP estimates (vision, prefill, decode) closely match:
  - Vendor model FLOPs measured by `torch.utils.flop_counter.FlopCounterMode` for the same images, prompts, and decode lengths.
  - Static `torchinfo` summaries for canonical shapes and batch sizes.
- Workload parameters (prompt, image resolution, crop mode, crop grid, number of views, projector tokens) are configured once via Hydra and propagated consistently to:
  - Analytic model construction.
  - Profiling runners.
  - Verification scripts and tests.
- The project gains a clearer separation between:
  - Architectural parameters (depth, width, heads).
  - Workload parameters (sequence length, views, tokens).
  - FLOP accounting policy (how to treat FlashAttention2 and other fused kernels).

## Implementation Summary (Current Status)

- **CLIP depth alignment**
  - `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml` now configures `vision.notp_transformer.blocks` with 24 `NoTPTransformerBlock` entries, matching the vendor CLIP-L `vit_model_cfg.num_layers = 24`.

- **Vision workload multiplier in analytic root**
  - `DeepseekOCRModel` (`extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`) gained:
    - `self.m_vision_workload_multiplier: float = 1.0`.
    - `set_vision_workload_multiplier(multiplier: float)` to control multi-view scaling (global view + crops).
    - An extended `start_vision(..., vision_workload_multiplier: Optional[float] = None)` that can optionally override the multiplier.
  - Aggregation methods now scale **vision-layer contributions** by `m_vision_workload_multiplier` in `"vision"` and `"prefill"` modes for:
    - FLOPs: `forward_tensor_core_flops`, `forward_cuda_core_flops`, `backward_tensor_core_flops`, `backward_cuda_core_flops`.
    - I/O: `forward_cal_io`, `backward_cal_io`.
    - Activations/KV cache: `forward_memory_activation`, `backward_memory_activation`, `forward_memory_kvcache`.
  - Parameter memory (`forward_memory_weight`, `backward_memory_weight`) remains unscaled.

- **Shape-dependent multi-view wiring in verification scripts**
  - `run_verify_end2end_vision.py`:
    - `_measure_vision_flops_ref` now returns `(total_flops, context_len, w_crop, h_crop)` by decoding `images_spatial_crop` from `_build_end_to_end_inputs(...)`.
    - `main()` computes `num_crops = w_crop * h_crop` when `crop_mode=1` and the grid is non-trivial, calls `analytic_model.set_vision_workload_multiplier(float(num_crops))`, then runs `start_vision(batch_size=cfg.runtime.batch_size)` to obtain analytic FLOPs.
  - `run_verify_end2end.py`:
    - `_measure_end_to_end_flops` now returns `(total_flops, context_len, decode_steps, w_crop, h_crop)` with the crop grid decoded from `images_spatial_crop`.
    - `main()` uses the same shape-only logic to compute `num_crops` and applies `set_vision_workload_multiplier(float(num_crops))` on the analytic model before calling `_compute_analytic_total_tflops(...)`.
  - In both scripts, the analytic model’s vision workload is now tied to the **same shape-dependent crop grid** (no content heuristics) used by vendor `dynamic_preprocess`.

## References

- **Code and configs**
  - Analytic root model: `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - Analytic configs: `extern/modelmeter/models/deepseek_ocr/configs/deepseek_ocr.yaml`
  - Vision configs: `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`
  - Vendor encoder/vision: `models/deepseek-ocr/deepencoder.py`
  - Vendor inference path & dynamic crops: `models/deepseek-ocr/modeling_deepseekocr.py`
  - Hydra model configs: `conf/model/deepseek_ocr/**`
  - Verification scripts: `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end*.py`
  - Static torchinfo report: `reports/20211117-dsorc-op-analysis/static-20251118-130533/torchinfo-summary.txt`
  - Issue docs: 
    - `extern/modelmeter/models/deepseek_ocr/docs/issues/issue-end2end-flop-gap.md`
    - `extern/modelmeter/models/deepseek_ocr/docs/issues/issue-vision-flop-underestimation.md`

- **3rd-party libraries (Context7 IDs)**
  - PyTorch: `/pytorch/pytorch`
  - Hugging Face Transformers: `/huggingface/transformers`
