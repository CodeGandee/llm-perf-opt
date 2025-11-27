# Code Review: DeepSeek-OCR E2E Sweeps – Analytic-Only Efficiency

**Date**: 2025-11-27 23:03:15  
**Scope**: `sweep-vision-crops.py`, `sweep-e2e-vision-prefill.py`, `sweep-e2e-decode.py`, `generate-image-shapes.py` integration  
**Reviewer**: GPT-5.1 (Codex CLI)  
**Issue**: Analytic-only all-stage sweeps are slower than expected; confirm that real images are not used for analytic runs and identify code-level efficiency improvements.

---

## 1. Executive Summary

Analytic-only runs of the DeepSeek-OCR sweep scripts are still relatively slow, but not because of image I/O. In `--analytic-only` mode, vendor paths and image resizing are correctly skipped, and analytic vision workloads are derived purely from integer shapes (`image_width`, `image_height`). The primary inefficiency is repeated model instantiation and Hydra config cloning inside per-point loops, especially in the prefill and decode sweeps.

Key findings:

- ✅ Analytic-only sweeps do **not** instantiate or resize real images; all vision workloads are shape-only.
- ⚠️ Each sweep repeatedly clones large Hydra configs and constructs new `DeepseekOCRModel` instances per candidate grid, which dominates runtime.
- 🔧 Decode sweep can be significantly optimized by reusing a single analytic model pair and dropping unnecessary vision-shape work.
- 🔧 Vision-stage sweep can avoid repeated Hydra `compose` calls by cloning a preloaded config instead.
- 🟡 Vision+prefill sweep is inherently more expensive; deeper refactor (model-level workload updates) would be required for large gains.

---

## 2. Image Handling in Analytic-Only Mode

**Files reviewed**

- `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-vision-crops.py`
- `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-vision-prefill.py`
- `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-decode.py`
- `extern/modelmeter/models/deepseek_ocr/layers/core/vision_shape_config.py`
- `extern/modelmeter/models/deepseek_ocr/scripts/sweep/_vision_utils.py`

**Behavior verification**

- In all three sweeps, analytic-only mode is controlled by the `--analytic-only` CLI flag and threaded into the internal sweep helpers.
- When `analytic_only=True`:
  - `build_vendor_vision_context` is not called; instead a `VendorVisionContext` with all fields `None` is constructed.
  - All vendor measurement blocks are gated by:
    - `not analytic_only`, and
    - per-entry `not analytic_only_entry` (YAML `extra_crops`).
  - `build_vision_workload_and_tokens` is always invoked with `image_path=None` and explicit `image_width` / `image_height`, so the PIL-based `Image.open` path is bypassed.

**Conclusion**

- The current analytic-only implementation already avoids instantiating or resizing the actual image. All workloads are derived from analytic shapes only.
- Performance issues are not due to image I/O; they are due to repeated model/config work per sample.

---

## 3. Profiling Evidence for Vision+Prefill Sweep

To confirm what dominates runtime in analytic-only mode, I profiled the vision+prefill sweep:

```bash
pixi run -e rtx5090 python -m cProfile \
  -o tmp/debug/e2e-vision-prefill-analytic-only.prof \
  extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-vision-prefill.py \
  --analytic-only \
  --image datasets/manual/example.png \
  --output-root tmp/sweep-debug/e2e_vision_prefill
```

Then inspected the stats:

```bash
pixi run -e rtx5090 python - << 'PY'
import pstats
from pstats import SortKey

stats = pstats.Stats("tmp/debug/e2e-vision-prefill-analytic-only.prof")
stats.strip_dirs()
stats.sort_stats(SortKey.CUMULATIVE).print_stats(30)
PY
```

Key numbers:

- Total time: ~114.7 seconds for 53 points.
- The dominant callers by cumulative time are:
  - `analytic_modes.py:build_analytic_model_for_mode` – ~93.0s cumtime over 106 calls (two analytic models per point).
  - `hydra._instantiate2.instantiate` – ~73.5s cumtime (model construction via Hydra).
  - `omegaconf` resolution and deep-copy:
    - `OmegaConf.resolve`, `OmegaConf.create`, and related `_resolve*` helpers – ~53.7s cumtime.
  - Python `copy.deepcopy` and associated helpers (`_reconstruct`, `_deepcopy_dict`, `dictconfig.__deepcopy__`, etc.) – ~50.5s cumtime.

Filtered stats (only Hydra/omegaconf/copy paths):

```text
ncalls  tottime  percall  cumtime  percall  function
  106    0.002    0.000   93.018   0.878    analytic_modes.py:34(build_analytic_model_for_mode)
  106    0.003    0.000   73.548   0.694    _instantiate2.py:148(instantiate)
22660874/11774  15.106    0.000   50.527   0.004    copy.py:118(deepcopy)
  106    0.000    0.000   53.660   0.506    omegaconf.py:769(resolve)
```

Interpretation:

- **Vendor code paths are not present** in the hot list when `--analytic-only` is used:
  - No calls into `_import_deepseek_ocr` or `_build_reference_ocr_model`.
  - No GPU model instantiation in the profile.
- The runtime is dominated by:
  - Per-point Hydra/OmegaConf cloning and resolution of the analytic config (`cfg_point`).
  - Repeated instantiation of `DeepseekOCRModel` via `build_analytic_model_for_mode` for both normal and flash attention modes.

This profiling confirms that:

- Vendor model loading does not happen at all in analytic-only sweeps (0 loads).
- The primary source of slowness is repeated analytic model construction and config processing per candidate crop grid, which aligns with the optimizations recommended in Sections 3–5.

---

## 4. Decode Sweep (`sweep-e2e-decode.py`) – High-Impact Optimization

**Current pattern**

- `_sweep_decode_crops`:
  - For each selected crop grid:
    - Clones the full analytic config:
      - `cfg_point = OmegaConf.create(OmegaConf.to_container(cfg_base, resolve=False))`
    - Computes vision shapes:
      - `build_vision_workload_and_tokens(...)`
      - `apply_vision_overrides_for_workload(cfg_point, ...)`
    - Instantiates **two** analytic models per entry:
      - `analytic_model_eager = build_analytic_model_for_mode(cfg_point, FULL_NORMAL_ATTENTION)`
      - `analytic_model_flash = build_analytic_model_for_mode(cfg_point, FULL_FLASH_ATTENTION)`
    - Runs `_compute_analytic_decode_stage_cost` on each model.

**Observation**

- The decode StageCost only depends on:
  - Decoder and (optional) head layers.
  - `context_len`, `batch_size`, and `num_decode_steps`.
- When `m_stage == "decode"`, `_iter_layers` in `DeepseekOCRModel` only yields decoder + head; the vision stack does not contribute. Vision configuration (`cfg_point.vision`) is therefore irrelevant to decode FLOPs.

**Recommended changes**

1. **Reuse analytic models across all points**
   - At the top of `_sweep_decode_crops` (after loading `cfg_base`):
     - Build analytic models once:
       ```python
       analytic_model_eager = build_analytic_model_for_mode(
           cfg_base,
           mode=AnalyticFlopMode.FULL_NORMAL_ATTENTION,
       )
       analytic_model_flash = build_analytic_model_for_mode(
           cfg_base,
           mode=AnalyticFlopMode.FULL_FLASH_ATTENTION,
       )
       ```
     - Use these same instances for all entries by calling `_compute_analytic_decode_stage_cost` with the appropriate `context_len` and `num_decode_steps`.

2. **Drop vision-shape recomputation for analytic decode**
   - For analytic-only decode (and arguably even with vendor present), `build_vision_workload_and_tokens` and `apply_vision_overrides_for_workload` are not necessary for the analytic model:
     - `_compute_analytic_decode_stage_cost` only uses decoder/head configuration, which is static across crop grids.
   - Recommended:
     - Remove the per-entry `cfg_point` clone and vision-shape overrides from the analytic path.
     - Keep vendor shape computation solely for context length and metadata (e.g., `image_tokens_total`, `text_tokens`).

**Expected impact**

- For ~50–100 candidate grids, this removes:
  - 2× `DeepseekOCRModel` instantiations per grid.
  - 1× Hydra config clone + 1× vision-shape override per grid.
- Decode sweeps in analytic-only mode should become significantly faster with no change in computed decode FLOPs.

---

## 4. Vision+Prefill Sweep (`sweep-e2e-vision-prefill.py`) – Constraints and Options

**Current pattern**

- `_sweep_vision_prefill_crops`:
  - Clones `cfg_base → cfg_point` for each grid.
  - Computes `VisionWorkload` and token counts via `build_vision_workload_and_tokens`.
  - Applies those shapes to `cfg_point.vision` with `apply_vision_overrides_for_workload`.
  - Instantiates two analytic models per grid (normal / flash) via `build_analytic_model_for_mode(cfg_point, ...)`.
  - Computes prefill StageCost for each model.

**Why reuse is harder here**

- Prefill mode aggregates all stages:
  - Vision stack (SAM + CLIP + projector).
  - Decoder stack.
  - Optional head.
- Vision shapes (base vs crop branch, token counts, sequence lengths) are tightly baked into the instantiated analytic vision composite via the Hydra configuration.
- The current design assumes that per-workload shapes are baked into the instantiated layers, not provided as dynamic runtime arguments.

**Short-term recommendations**

- Accept that prefill is the heaviest sweep and avoid micro-optimizations that risk correctness without deeper refactoring.
- Low-risk incremental improvement:
  - Replace direct `cfg_base` cloning with a pre-flattened base config:
    ```python
    cfg_base_flat = OmegaConf.create(OmegaConf.to_container(cfg_base, resolve=False))
    ...
    cfg_point = OmegaConf.create(OmegaConf.to_container(cfg_base_flat, resolve=False))
    ```
  - This can reduce nested interpolation work in some Hydra setups, but gains may be modest.

**Long-term refactor (if warranted)**

- Introduce explicit workload setters on analytic vision layers:
  - For example, a `set_vision_workload(workload: VisionWorkload, tokens_global, tokens_crops, ...)` method that updates internal shapes without re-instantiating the model.
- Then:
  - Instantiate `DeepseekOCRModel` once per FLOP mode.
  - For each workload:
    - Call `set_vision_workload(...)`.
    - Then call `start_prefill(...)`.
- This would substantially reduce per-grid overhead but requires coordinated changes across vision encoder, CLIP, projector, and config glue code.

---

## 6. Vision-Stage Sweep (`sweep-vision-crops.py`) – Moderate Optimization

**Current pattern**

- Inside `_sweep_vision_crops`, for each selected entry:
  - Re-composes Hydra config from disk:
    - `cfg_crops: DictConfig = compose(config_name="deepseek_ocr")`
  - Clones into normal/flash variants (`copy.deepcopy`).
  - Calls `build_vision_workload_and_tokens` and `apply_vision_overrides_for_workload`.
  - Instantiates two analytic models for normal vs flash attention (`build_analytic_model` with different vision composites).

**Recommendations**

1. **Avoid repeated Hydra `compose` calls**
   - At the top of `_sweep_vision_crops` you already have `base_cfg = compose(config_name="deepseek_ocr")`.
   - Instead of calling `compose` inside the loop, clone `base_cfg`:
     ```python
     cfg_crops = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
     ```
   - Then proceed with the per-entry overrides and model instantiation as today.

2. **Optional: share some common subtrees**

   - If profiling shows `build_analytic_model` as a hotspot, consider:
     - Extracting the decoder/head parts once and only varying the vision config per workload.
     - This gets closer to the “vision workload setter” design discussed for prefill, but could be smaller in scope for vision-only StageCost.

---

## 7. Image Shapes and `generate-image-shapes.py` Integration

**Context**

- `generate-image-shapes.py` builds a SQLite DB of canonical shapes by crop count:
  - For each crop count `n` in `[min_crops, max_crops]`, it picks the most square factor pair `(h, w)`, then:
    - `image_height = h * image_size`
    - `image_width = w * image_size`
  - Computes projector-token counts using the same formulas as `VisionWorkload`:
    - `q_base`, `q_crop` via `_compute_num_queries`.
    - `tokens_crops = (h * q_crop) * (w * q_crop + 1)`.
    - `tokens_total = global_tokens + tokens_crops`.
  - Marks vendor support via `2 <= crop_count <= 9`.

**Relevance to analytic sweeps**

- The sweeps themselves still source crop grids and token counts from `candidate-input-shapes.yaml`, not from the SQLite DB.
- For analytic-only use cases where vendor alignment is not required, the `image-shapes` DB could:
  - Provide a denser set of candidate grids or extended ranges (e.g., up to 1000 crops).
  - Feed directly into analytic sweeps without needing vendor-backed YAML entries.

**Note**

- Current slowdowns are not from DB integration; the DB is not in the hot path. The main cost is model/config work per grid.

---

## 8. Summary of Recommended Changes

1. **Decode sweep**
   - Reuse a single pair of analytic models (`FULL_NORMAL_ATTENTION`, `FULL_FLASH_ATTENTION`) across all crop grids.
   - Remove vision-shape recomputation (`build_vision_workload_and_tokens` and `apply_vision_overrides_for_workload`) from the analytic decode path.

2. **Vision-stage sweep**
   - Replace per-entry Hydra `compose` with clones of a preloaded `base_cfg`.

3. **Prefill sweep**
   - Accept current cost as expected for now; consider a future refactor that allows dynamic workload updates on a single analytic model instance instead of per-grid instantiation.

4. **Image usage**
   - No changes needed for analytic-only mode: real images are already not instantiated or resized; this behavior is correct and should be preserved.
