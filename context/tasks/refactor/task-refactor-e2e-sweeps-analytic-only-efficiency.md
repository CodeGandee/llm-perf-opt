# Refactor Plan: DeepSeek-OCR E2E Sweeps Analytic-Only Efficiency

## What to Refactor

- Refactor the analytic-only code paths in the following sweep scripts so they avoid unnecessary Hydra config cloning and repeated analytic model instantiation:
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-decode.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-vision-prefill.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-vision-crops.py`
- For the decode sweep, decouple analytic decode StageCost computation from vision shape configuration, and reuse analytic models across all crop grids.
- For the vision+prefill sweep, prepare the codebase for a future step where a single analytic model instance can be reused across workloads by updating its vision workload instead of re-instantiating per grid.
- For the vision-stage sweep, eliminate repeated Hydra `compose` calls inside the per-grid loop by cloning a preloaded base config.

## Why Refactor

- Analytic-only runs are much slower than necessary because each sweep:
  - Clones a large Hydra configuration per candidate crop grid.
  - Instantiates new `DeepseekOCRModel` objects (normal and flash attention) per grid via Hydra and OmegaConf, incurring heavy deep-copy and resolution overhead.
- Profiling evidence from `tmp/debug/e2e-vision-prefill-analytic-only.prof` shows:
  - `analytic_modes.py:build_analytic_model_for_mode` dominates cumulative time for the vision+prefill sweep.
  - Hydra instantiation (`_instantiate2.instantiate`) and OmegaConf resolution (`omegaconf.resolve`, `OmegaConf.create`) plus `copy.deepcopy` account for most of the runtime.
- The decode sweep does not depend on per-grid vision shapes for analytic FLOPs, but still recomputes vision workloads and rebuilds models for each entry.
- Reducing redundant work will:
  - Shorten end-to-end sweep wall time significantly in analytic-only mode.
  - Make it cheaper to iterate on candidate grids and image-shape ranges.
  - Simplify reasoning about where time is spent, which is important for profiling and optimization work.

## How to Refactor

### 1) Decode Sweep: Reuse Analytic Models and Drop Vision-Shape Work

**Goal**

- In `sweep-e2e-decode.py`, reuse one analytic model pair across all crop grids and remove unnecessary vision workload configuration from the analytic decode path.

**Before (simplified)**

Per entry, we clone `cfg_base`, build vision shapes, and instantiate models:

```python
cfg_point_raw = OmegaConf.create(OmegaConf.to_container(cfg_base, resolve=False))
cfg_point: DictConfig = cast(DictConfig, cfg_point_raw)

(
    workload,
    _w_eff,
    _h_eff,
    num_crops,
    tokens_global,
    tokens_crops,
    seq_len_clip_global,
    seq_len_clip_crop,
) = build_vision_workload_and_tokens(
    cfg=cfg_point,
    base_size=base_size,
    image_size=crop_img_size,
    crop_mode=True,
    w_crop_vendor=(width_crop_num if (analytic_only_entry and w_crop_vendor == 0) else w_crop_vendor),
    h_crop_vendor=(height_crop_num if (analytic_only_entry and h_crop_vendor == 0) else h_crop_vendor),
    image_path=None,
    image_width=image_width,
    image_height=image_height,
)

apply_vision_overrides_for_workload(cfg_point, workload=workload, num_crops=num_crops, tokens_global=tokens_global, tokens_crops=max(tokens_crops, 1), seq_len_clip_global=seq_len_clip_global, seq_len_clip_crop=seq_len_clip_crop)

analytic_model_eager = build_analytic_model_for_mode(cfg_point, mode=AnalyticFlopMode.FULL_NORMAL_ATTENTION)
analytic_model_flash = build_analytic_model_for_mode(cfg_point, mode=AnalyticFlopMode.FULL_FLASH_ATTENTION)
decode_eager_cost = _compute_analytic_decode_stage_cost(analytic_model_eager, context_len=context_len, batch_size=batch_size, num_decode_steps=int(decode_steps), ignore_torch_unseen=False)
decode_flash_cost = _compute_analytic_decode_stage_cost(analytic_model_flash, context_len=context_len, batch_size=batch_size, num_decode_steps=int(decode_steps), ignore_torch_unseen=False)
```

**After (proposed)**

- Instantiate analytic models once at the top of `_sweep_decode_crops` using `cfg_base`.
- Remove vision workload and `apply_vision_overrides_for_workload` from the analytic path.
- Reuse the same models for all entries by varying only `context_len` and `num_decode_steps`.

```python
analytic_model_eager = build_analytic_model_for_mode(
    cfg_base,
    mode=AnalyticFlopMode.FULL_NORMAL_ATTENTION,
)
analytic_model_flash = build_analytic_model_for_mode(
    cfg_base,
    mode=AnalyticFlopMode.FULL_FLASH_ATTENTION,
)

for entry in selected:
    # ... infer context_len and batch_size as today ...
    decode_eager_cost = _compute_analytic_decode_stage_cost(
        analytic_model_eager,
        context_len=context_len,
        batch_size=batch_size,
        num_decode_steps=int(decode_steps),
        ignore_torch_unseen=False,
    )
    decode_flash_cost = _compute_analytic_decode_stage_cost(
        analytic_model_flash,
        context_len=context_len,
        batch_size=batch_size,
        num_decode_steps=int(decode_steps),
        ignore_torch_unseen=False,
    )
```

**Steps**

1. In `_sweep_decode_crops`, after obtaining `cfg_base`, create `analytic_model_eager` and `analytic_model_flash` from `cfg_base` instead of per-entry `cfg_point`.
2. Remove per-entry `cfg_point` creation and the `build_vision_workload_and_tokens` plus `apply_vision_overrides_for_workload` calls from the analytic decode path.
3. Keep vendor vision workload computation unchanged for vendor FLOPs and KV metrics; only the analytic path is simplified.
4. Ensure `_compute_analytic_decode_stage_cost` is stateless between calls beyond its own internal use of `start_prefill`/`start_decode` and that reusing a `DeepseekOCRModel` instance is safe (it already resets `m_stage` and KV cache per prefill call).
5. Add a small regression check to confirm that analytic decode FLOPs for a known grid (e.g., first entry) match pre-refactor values within tolerance.

### 2) Vision+Prefill Sweep: Prepare for Model Reuse

**Goal**

- Reduce per-entry overhead where possible and prepare for an eventual refactor that allows reusing a single analytic model instance across workloads.

**Constraints**

- Prefill StageCost depends on both the decoder and vision shapes, so we cannot trivially reuse a single model without giving the analytic vision stack a way to update its workload at runtime.

**Incremental plan**

1. Introduce a helper that encapsulates the repeated pattern:
   - Clone `cfg_base` to `cfg_point`.
   - Call `build_vision_workload_and_tokens` and `apply_vision_overrides_for_workload`.
   - Build both analytic models via `build_analytic_model_for_mode`.
   - Return the models plus the `VisionWorkload` metadata.
2. Replace the inlined per-entry code in `_sweep_vision_prefill_crops` with calls to this helper to make future changes more localized.
3. In a second stage (separate change):
   - Explore adding an API on `DeepseekOCRModel` such as `set_vision_workload(workload, tokens_global, tokens_crops, seq_len_clip_global, seq_len_clip_crop)` that updates the internal vision sublayers without re-instantiation.
   - Once this exists and is stable, adjust the helper so it:
     - Instantiates analytic models once per FLOP mode.
     - Updates their vision workloads per entry instead of rebuilding them.
4. Add comments and tests to ensure that any caching or reuse honors the current behavior of `start_prefill` and `get_forward_cost`.

### 3) Vision-Stage Sweep: Avoid Repeated Hydra Compose

**Goal**

- In `sweep-vision-crops.py`, avoid calling `compose(config_name="deepseek_ocr")` inside the per-grid loop.

**Before (simplified)**

```python
for entry in selected:
    # inside loop
    cfg_crops: DictConfig = compose(config_name="deepseek_ocr")
    if "model" in cfg_crops:
        cfg_crops.model.vision_mode = "crops"
    # build vision workload and analytic models
```

**After (proposed)**

```python
base_cfg: DictConfig = compose(config_name="deepseek_ocr")

for entry in selected:
    cfg_crops = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    if "model" in cfg_crops:
        cfg_crops.model.vision_mode = "crops"
    # build vision workload and analytic models as before
```

**Steps**

1. Ensure `base_cfg` is composed once at the beginning of `_sweep_vision_crops`.
2. Replace the inner-loop `compose` call with an `OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))` clone.
3. Keep the rest of the per-entry logic intact so behavior remains identical.
4. Optionally, if profiling still shows `build_analytic_model` as a hotspot, consider reusing normal/flash analytic models for pure vision StageCost by adding a `set_vision_workload` API similar to the prefill plan.

### 4) All Sweeps: Keep Vendor and Analytic Paths Clearly Separated

**Goal**

- Ensure that analytic-only refactors cannot accidentally introduce vendor work, and vice versa.

**Steps**

1. Factor vendor-only code into small, clearly named helpers where feasible, for example:
   - `_measure_vendor_prefill_workload(...)`
   - `_measure_vendor_decode_workload(...)`
   - `_measure_vendor_vision_workload(...)`
2. For each sweep:
   - Keep a top-level `if analytic_only:` branch that bypasses vendor helpers entirely and constructs an empty `VendorVisionContext`.
3. Add lightweight asserts or logging that confirm vendor context fields are `None` when `analytic_only=True`.

## Impact Analysis

**Functional impact**

- Decode sweep:
  - Analytic decode FLOPs and StageCost should remain unchanged because the decoder and head configurations do not depend on the vision workload.
  - Vendor decode FLOPs remain unaffected because vendor measurement and KV logic is untouched.
- Vision+prefill sweep:
  - Short-term changes (helper extraction, minor config cloning tweaks) are refactors only and should not change numerical results.
  - Long-term changes (adding a vision workload setter and reusing models) require careful validation but are conceptually equivalent to the current pattern of “config → model → prefill”.
- Vision-stage sweep:
  - Using `base_cfg` clones instead of fresh `compose` calls should be behaviorally identical as long as no global mutable state is hidden in Hydra composition (current configs appear static).

**Performance impact**

- Decode sweep with model reuse:
  - Expect substantial reduction in runtime for analytic-only decode sweeps because we eliminate 2× model instantiations and heavy OmegaConf cloning per grid.
- Vision-stage sweep with compose reuse:
  - Moderate reduction in runtime; Hydra composition cost per grid is replaced with cheaper deep-copy of an in-memory config.
- Vision+prefill sweep:
  - Short-term: only minor improvements from shared helper and possible base config flattening.
  - Long-term: reusing models per FLOP mode could bring similar benefits to decode, but will require more work.

**Risks and mitigations**

- Risk: Reusing `DeepseekOCRModel` instances could leak state between grids if `start_prefill`, `start_decode`, or internal layers retain workload-specific state.
  - Mitigation:
    - Audit `start_prefill` and `start_decode` to ensure they fully reset internal state for the new workload.
    - Add a small test that calls prefill twice with different context lengths and verifies the second cost matches a fresh model run.
- Risk: Using `base_cfg` clones instead of fresh `compose` calls could miss dynamic config changes injected elsewhere.
  - Mitigation:
    - Ensure sweeps always start from a clean process run.
    - Document the assumption that the Hydra config for DeepSeek-OCR is static within a sweep invocation.

## Expected Outcome

- Analytic-only sweeps become significantly faster, especially:
  - Decode sweep, due to model reuse and removal of unnecessary vision-shape work.
  - Vision-stage sweep, due to avoiding repeated Hydra composition.
- The code paths for analytic-only vs vendor-aligned runs become clearer and easier to maintain.
- The refactor establishes a clean foundation for future improvements where the analytic model can be reused across workloads by updating its vision workload at runtime.
- Profiling after refactor should show:
  - Fewer calls to `build_analytic_model_for_mode` and `instantiate`.
  - Reduced time spent in OmegaConf `create`, `resolve`, and deep-copy.

## References

- Sweep scripts:
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-decode.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-vision-prefill.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-vision-crops.py`
- Analytic model and vision workload helpers:
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/core/vision_shape_config.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/vision_workload.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/analytic_modes.py`
- Vendor utility helpers:
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/_vision_utils.py`
- Profiling evidence:
  - `tmp/debug/e2e-vision-prefill-analytic-only.prof`
  - `context/logs/code-review/20251127-230315-e2e-sweep-analytic-only-efficiency.md`
- Third-party libraries involved:
  - Hydra and OmegaConf (configuration and model instantiation).
  - PyTorch and related analytic infrastructure used indirectly via `DeepseekOCRModel`.

