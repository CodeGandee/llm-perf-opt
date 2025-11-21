**What to Refactor**
- Files (DeepSeek-OCR analytic model and consumers):
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/*.py` (read-only, for behavior reference)
  - `extern/modelmeter/models/deepseek_ocr/HOLISTIC_ANALYSIS.md`
  - `src/llm_perf_opt/runners/dsocr_analyzer.py`
- Scope:
  - Extend the stateful analytic root model so it has an explicit **vision stage** in addition to the existing **prefill** and **decode** modes.
  - Make the root model’s stage semantics line up with the real DeepSeek-OCR pipeline:
    - Vision encoder + projector (SAM-B + CLIP-L + MLP projector).
    - Decoder/LLaMA stack (DeepseekV2DecoderLayer × N) operating on text+image tokens.
    - KV cache only for decoder stages.
  - Provide a single, consistent way to query **per-stage analytic costs** (vision / prefill / decode) without duplicating APIs.

**Why Refactor**
- Today, `DeepseekOCRModel` exposes stateful **prefill** / **decode** control (`start_prefill`, `start_decode`, `decode_one_token`) and an `operation_mode` property, but there is **no explicit notion of a vision stage**:
  - The vision stack (SAM encoder, CLIP vision tower, and projector) is wrapped in `_CompositeLayer` and always included in `_iter_layers()`, regardless of whether we are in `"prefill"` or `"decode"` mode.
  - In practice, we conceptually treat **vision as part of prefill** (see `HOLISTIC_ANALYSIS.md`), but this is not encoded in the API or enforced by the implementation.
- This leads to confusing or misleading semantics:
  - When `operation_mode == "decode"`, `forward_*` on `DeepseekOCRModel` still aggregates **vision + decoder + head** costs, even though the real model does not re-run SAM/CLIP/projector during per-token decode.
  - Call sites that need **vision-only** stats either:
    - poke into `vision_layer` directly, or
    - over-count vision as part of “prefill” without a way to separate it out.
  - Static analysis (`DeepseekOCRStaticAnalyzer`) already has separate stages (`"sam"`, `"clip"`, `"projector"`, `"prefill"`, `"decode"`), but the analytic model only exposes `"prefill"` / `"decode"`.
- Adding an explicit **vision stage** at the analytic root will:
  - Align the analytic model with the real DeepSeek-OCR flow (`DeepseekOCRModel.forward` in `models/deepseek-ocr/modeling_deepseekocr.py`):
    - Images go through SAM → CLIP → MLP projector once.
    - Projected image tokens are concatenated with text tokens and fed to the decoder stack for prefill.
    - Decode steps reuse the KV cache and do not re-run vision.
  - Make **per-stage FLOPs/IO/memory breakdowns** straightforward (vision vs prefill vs decode) without ad-hoc bookkeeping.
  - Simplify future analytic workflows (e.g., MFU reports, hook-based runs) that want to reason about “vision-heavy” vs “decoder-heavy” workloads.

**How to Refactor**
1) **Stage model and API design**
   - Extend the root analytic model to treat the operation mode as a three-state enum:
     - `"vision"` – only the vision stack (SAM-B, CLIP-L, projector).
     - `"prefill"` – vision + decoder stack prefill, including image tokens.
     - `"decode"` – decoder-only per-token decode on top of existing KV cache.
   - Keep the **public API surface minimal**:
     - Preserve existing methods:
       - `start_prefill(context_len, batch_size, kv_cache=None) -> SyntheticKVCache`
       - `start_decode(kv_cache=None) -> SyntheticKVCache`
       - `decode_one_token() -> SyntheticKVCache`
       - `operation_mode: str`
     - Add a single new method for vision:
       - `start_vision(batch_size: int = 1) -> None`
         - Sets `operation_mode` / `m_stage` to `"vision"`.
         - Validates that `vision_layer` is configured and records batch size if needed.
         - Does **not** touch KV cache (vision is KV-free).
     - Do not add new “estimate_*_cost” helpers; callers should continue to use `get_forward_cost()` / `get_backward_cost()` and the existing stage entry points.

2) **Root model stage semantics**
   - Rework `DeepseekOCRModel._iter_layers()` and its `forward_*` aggregators so that:
     - In `"vision"` mode:
       - Only the vision stack contributes to `forward_*` / `backward_*` / memory metrics.
       - Decoder and head layers are excluded from `_iter_layers()`.
     - In `"prefill"` mode:
       - Vision + decoder stack + optional head contribute (same as the “holistic prefill” definition in `HOLISTIC_ANALYSIS.md`).
       - KV cache is configured and owned by the decoder layer(s) only.
     - In `"decode"` mode:
       - Only the decoder stack + optional head contribute.
       - Vision layers are excluded to reflect that the real model does not re-run SAM/CLIP/projector on decode.
   - Maintain backwards compatibility for existing usage:
     - `start_prefill` and `start_decode` continue to work as today, but their semantics will now be:
       - `start_prefill` → `operation_mode == "prefill"` (vision + decoder).
       - `start_decode` / `decode_one_token` → `operation_mode == "decode"` (decoder-only).
     - `operation_mode` will additionally be `"vision"` after `start_vision` is called.

   - Example (before):
     ```python
     # Only prefill/decode modes; decode still counts vision.
     model = DeepseekOCRModel.from_layers(vision_layer, decoder_layer, num_decoder_layers=24)
     model.start_prefill(context_len=S_prefill, batch_size=B)
     flops_prefill = model.get_forward_cost().flops_tflops

     model.start_decode()
     flops_decode_per_token = model.get_forward_cost().flops_tflops  # includes vision
     ```

   - Example (after):
     ```python
     model = DeepseekOCRModel.from_layers(vision_layer, decoder_layer, num_decoder_layers=24)

     # 1) Vision-only stage (SAM + CLIP + projector)
     model.start_vision(batch_size=B)
     vision_cost = model.get_forward_cost()

     # 2) Holistic prefill (vision + decoder) including image tokens
     kv_cache = model.start_prefill(context_len=S_prefill, batch_size=B)
     prefill_cost = model.get_forward_cost()

     # 3) Decode (decoder-only per token, with KV cache)
     model.start_decode(kv_cache=None)
     decode_cost_per_token = model.get_forward_cost()
     model.decode_one_token()  # advance KV/decode state
     ```

3) **Vision state wiring and batch-size handling**
   - Audit the existing vision analytic layers (SAM ViT encoder, CLIP ViT, projector) to understand how they encode shapes:
     - `ImageEncoderViT` already stores `m_img_size`, `m_patch_size`, `m_batch_size`, and internal `m_seq_len`.
     - `VitModel` and `MlpProjector` similarly capture their own shapes and batch size at construction.
   - Decide on a simple convention for the root vision stage:
     - Treat `vision_layer` as owning its own image resolution/patch metadata.
     - Let `start_vision(batch_size=B)` optionally override the batch size on the vision stack if needed:
       - For example, `_CompositeLayer` could forward a `set_batch_size(B)` call to its component layers when present.
     - Avoid replicating the full dynamic cropping logic from `models/deepseek-ocr/modeling_deepseekocr.py`; instead:
       - Keep the analytic vision stage parameterized by a small set of knobs (e.g., base image size, crop grid, number of views) that are already baked into the analytic layers.
   - Ensure that prefill semantics still reflect “vision + decoder”:
     - `start_prefill(...)` should **not** implicitly call `start_vision`, but its `get_forward_cost()` in `"prefill"` mode should sum:
       - whatever cost `vision_layer` has been configured with (vision stage), plus
       - the decoder stack cost at `context_len` with the KV cache state configured by `start_prefill`.

4) **Expose per-stage cost helpers (without bloating core APIs)**
   - Keep `DeepseekOCRModel` itself minimal and stage-based; add small **wrapper helpers** as free functions for ergonomics:
     - Example in a new helper module (or an existing one such as `layers/core/__init__.py`):
       ```python
       from modelmeter.models.deepseek_ocr.layers.core.deepseek_ocr_model import DeepseekOCRModel


       def get_vision_cost(model: DeepseekOCRModel) -> StageCost:
           model.start_vision(batch_size=model.m_batch_size or 1)  # or a safe default
           return model.get_forward_cost()


       def get_prefill_cost(model: DeepseekOCRModel, *, context_len: int, batch_size: int) -> StageCost:
           model.start_prefill(context_len=context_len, batch_size=batch_size)
           return model.get_forward_cost()


       def get_decode_cost_per_token(model: DeepseekOCRModel) -> StageCost:
           if model.operation_mode != "decode":
               model.start_decode(kv_cache=None)
           return model.get_forward_cost()
       ```
   - Call sites that want higher-level “vision / prefill / decode” numbers can use these helpers instead of re-implementing stage switching.

5) **Align docs and static analyzer with the new vision stage**
   - Update `extern/modelmeter/models/deepseek_ocr/HOLISTIC_ANALYSIS.md`:
     - Clarify that the analytic model has an explicit `"vision"` operation mode.
     - Show how to compute:
       - Vision-only cost (`start_vision` + `get_forward_cost()`).
       - Holistic prefill cost (`start_prefill` + `get_forward_cost()`).
       - Decode-per-token cost (`start_decode` + `get_forward_cost()` / `decode_one_token`).
     - Ensure the pseudo-code in Sections 1–3 matches the new API.
   - Update `src/llm_perf_opt/runners/dsocr_analyzer.py`:
     - Where it reports per-stage numbers (`"sam"`, `"clip"`, `"projector"`, `"prefill"`, `"decode"`), add a clear mapping:
       - Vision stage (analytic) ≈ `"sam" + "clip" + "projector"` (static).
       - Prefill / decode (analytic) ≈ existing fvcore stages for transformer + LM head.
     - Optionally add analytic cross-checks:
       - Compare analytic `get_vision_cost()` vs fvcore `"sam" + "clip" + "projector"` FLOPs for sanity.

6) **Regression checks and verification**
   - Re-run all DeepSeek-OCR analytic verification scripts:
     - `extern/modelmeter/models/deepseek_ocr/scripts/run_verify_core.py`
     - `extern/modelmeter/models/deepseek_ocr/scripts/run_verify_vision.py`
     - `extern/modelmeter/models/deepseek_ocr/scripts/run_verify_decoder.py`
     - `extern/modelmeter/models/deepseek_ocr/scripts/run_verify_llama.py`
     - `extern/modelmeter/models/deepseek_ocr/scripts/run_verify_prefill_decode.py`
   - Ensure that:
     - The new `"vision"` operation mode does **not** change layer-level formulas; only root aggregation changes.
     - `run_verify_prefill_decode.py` uses the updated stage semantics where needed (for example, decode should no longer implicitly include vision cost).
   - Add small unit tests (if not already present) that:
     - Construct a toy `DeepseekOCRModel` with stubbed vision/decoder/head layers that return easily distinguishable FLOPs.
     - Assert that `start_vision` / `start_prefill` / `start_decode` select the correct subset of layers under `_iter_layers()` when computing `get_forward_cost()`.

**Impact Analysis**
- **Behavioral changes**
  - Decode-stage analytic metrics from `DeepseekOCRModel` will change:
    - Previously they included vision costs; after the refactor they will be **decoder-only**.
    - Any downstream code that assumed “decode FLOPs per token” included a fixed vision overhead will need to be updated (or to explicitly add `get_vision_cost()` once).
  - Prefill remains “vision + decoder” at the root model level, but callers now have a way to:
    - isolate vision-only costs, and
    - cross-check that `F_prefill ≈ F_vision + F_decoder_prefill` analytically.
- **Risk and mitigation**
  - Risk: Some scripts or docs might still refer to the old implicit semantics (no vision stage, decode including vision).
    - Mitigation: Update `HOLISTIC_ANALYSIS.md`, `scripts/README.md`, and any relevant context docs; run all verification scripts and validate that printed stats align with expectations.
  - Risk: Introducing `start_vision` could tempt multiple workflows for “vision cost.”
    - Mitigation: Position `start_vision` as the **only supported entry point** for vision-stage analytics in docs; avoid adding alternative helper methods in the root class.
  - Risk: Static analyzer vs analytic model discrepancies for vision.
    - Mitigation: Use `DeepseekOCRStaticAnalyzer`’s `"sam"`, `"clip"`, `"projector"` stages as a reference and document acceptable tolerance/limitations in `HOLISTIC_ANALYSIS.md`.

**Expected Outcome**
- The analytic root model for DeepSeek-OCR will:
  - Expose a clear, three-stage story: `"vision"` → `"prefill"` → `"decode"`.
  - Provide accurate per-stage FLOPs/IO/memory estimates that mirror the real DeepSeek-OCR execution pipeline.
  - Keep the class API small and predictable, with a single way to obtain per-stage costs via `start_*` + `get_forward_cost()`.
- Static and dynamic tooling (static analyzer, hook-based runs, prefill/decode verification scripts) will be able to:
  - Attribute costs to vision vs decoder stages cleanly, and
  - Use analytic estimates as a trustworthy baseline when comparing against fvcore/pytorch flop counters.

**Implementation Summary**
- Implemented stage-aware aggregation in `DeepseekOCRModel`:
  - Added a new `start_vision(batch_size: int = 1)` method that switches the root model into `"vision"` mode and configures batch size for vision-only analytics.
  - Extended `operation_mode` semantics so it now returns `"vision"`, `"prefill"`, or `"decode"` based on the last `start_*` call.
- Updated `_iter_layers()` in `DeepseekOCRModel` so the contributing sublayers depend on the current stage:
  - `"vision"` → yields only the configured vision stack (`m_vision_layer`).
  - `"prefill"` → yields vision, the repeated decoder stack, and optional head (holistic prefill).
  - `"decode"` → yields only the repeated decoder stack and optional head (no vision cost during per-token decode).
- Tightened validation in all aggregate metric methods on `DeepseekOCRModel` (`forward_*`, `backward_*`, and memory/IO methods):
  - In `"vision"` mode, they require a configured vision layer but do not require a decoder.
  - In `"prefill"` / `"decode"` modes, they require a configured decoder layer and treat the vision stack as optional (included only in `"prefill"`).
- Kept the existing prefill/decode APIs (`start_prefill`, `start_decode`, `decode_one_token`) and `StageCostMixin` helpers unchanged at the call-site level so existing analytic workflows remain compatible while gaining explicit vision-stage support.

**References**
- Root analytic model:
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
- Vision analytic layers:
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/image_encoder_vit.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/vit_model.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/mlp_projector.py`
- Real model implementation:
  - `models/deepseek-ocr/modeling_deepseekocr.py` (vision pipeline and projector wiring)
- Static analysis:
  - `src/llm_perf_opt/runners/dsocr_analyzer.py` (`DeepseekOCRStaticAnalyzer.m_stage_module_map`)
- Analytic docs:
  - `extern/modelmeter/models/deepseek_ocr/HOLISTIC_ANALYSIS.md`
  - `extern/modelmeter/models/deepseek_ocr/TENSOR_OR_CUDA_CORE_GUIDE.md`
- Third-party libraries:
  - Hugging Face Transformers – `/huggingface/transformers`
  - PyTorch – `/pytorch/pytorch`
