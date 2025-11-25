# Refactor Plan: DeepSeek-OCR Vision Shape-Accurate Workload Modeling

## What to Refactor

- **Vision workload modeling (global vs crops)**
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/image_encoder_vit.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/vit_model.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/mlp_projector.py`
  - Composite vision wiring:
    - `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`
    - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py` (`_CompositeLayer` usage)
- **Vision FLOP verification**
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end_vision.py`
  - Supporting debug helpers under `tmp/vision-flops-debug/`
- **End-to-end FLOP verification linkage**
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end.py`
  - Ensure its analytic vision usage consumes the same shape-accurate workload description.

The goal is to replace the current “single vision stack + effective batch_size” approach with a **shape-accurate decomposition** that explicitly models:

- One **global** SAM + CLIP + projector pass at `base_size=1024`
- `w_crop * h_crop` **crop** passes at `image_size=640`
- Correct CLIP/projector token counts for both branches, derived from the same formulas the vendor uses.

## Why Refactor

- **Root cause of current mismatch**
  - The end-to-end vision verifier (`run_verify_end2end_vision.py`) currently:
    - Derives an `effective_views` scalar from CLIP token counts:
      ```python
      effective_views = _compute_effective_views(
          base_size=base_size,
          image_size=image_size,
          crop_mode=crop_mode,
          w_crop=w_crop,
          h_crop=h_crop,
      )
      ```
    - Applies this to **all** analytic vision sub-configs via `batch_size`:
      ```python
      for key in ("image_encoder_vit", "clip_embeddings", "notp_attention",
                  "notp_feedforward", "vit_model", "projector"):
          if key in cfg.vision:
              cfg.vision[key]["batch_size"] = effective_views
      ```
  - This makes `VitModel` and `MlpProjector` scale roughly by `effective_views` (≈3.34× for the 2×3 crop case), but it makes `ImageEncoderViT` scale by ~7.3× because its analytic implementation multiplies both:
    - `batch_size`
    - the number of attention windows (`num_windows * batch_size`)
  - The vendor SAM runs once at 1024 and `num_crops` times at 640, not “`effective_views` batches of a single 1024 config”, so the SAM FLOPs are structurally overcounted.

- **Structural vs scalar problem**
  - The mismatch is not just a bad scalar; it is that we are trying to fold a **heterogeneous workload** (global view + smaller crops) into a **homogeneous model** via a single batch multiplier.
  - This breaks both FLOPs and StageCost metrics (`io_tb`, `activations_gb`) for SAM and partially for the projector.

- **Faithfulness to vendor behavior**
  - DeepSeek-OCR’s vision stack is fully shape-driven:
    - Global padded view: 1024×1024
    - Crops: layout computed by `dynamic_preprocess` using `image_size` and aspect ratio
    - CLIP/projector token counts derived from `(base_size, image_size, w_crop, h_crop, patch_size, downsample_ratio)`
  - Our analytic model should mirror that, not introduce synthetic “view multipliers” that couple unrelated pieces (e.g., SAM windows and CLIP tokens).

## How to Refactor

### 1. Introduce an explicit VisionWorkload and view layout helper

**Objective:** Move all shape logic into a reusable helper that mirrors vendor `dynamic_preprocess` and token layout.

- Add a small helper module (e.g. `extern/modelmeter/models/deepseek_ocr/layers/vision/vision_workload.py`) that defines:

  ```python
  @dataclass
  class VisionWorkload:
      base_size: int
      image_size: int
      crop_mode: bool
      patch_size: int
      downsample_ratio: int
      min_num: int = 2
      max_num: int = 9
      image_width: int = 0
      image_height: int = 0
  ```

  and:

  ```python
  def compute_view_layout(workload: VisionWorkload) -> tuple[int, int, int]:
      """
      Mirror modeling_deepseekocr.dynamic_preprocess:
      - Search target aspect ratios between min_num and max_num.
      - Pick the best (w_crop, h_crop) given the image aspect ratio and area.
      - Return (w_crop, h_crop, num_crops = w_crop * h_crop).
      """
  ```

- Add helpers for CLIP / projector token counts:

  ```python
  def compute_num_queries(image_size: int, patch_size: int, downsample_ratio: int) -> int:
      return math.ceil((image_size // patch_size) / downsample_ratio)

  def compute_projector_tokens(
      base_size: int, image_size: int, w_crop: int, h_crop: int,
      patch_size: int, downsample_ratio: int,
  ) -> tuple[int, int, int]:
      # Return (tokens_global, tokens_crops, tokens_total).
  ```

**Before (scattered logic in verifier):**
```python
num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)
num_queries_crop = math.ceil((image_size // patch_size) / downsample_ratio)
tokens_base = num_queries_base * num_queries_base
tokens_crops = num_crops * num_queries_crop * num_queries_crop
effective_views = (tokens_base + tokens_crops) / float(tokens_base)
```

**After (centralized helper):**
```python
workload = VisionWorkload(
    base_size=base_size,
    image_size=image_size,
    crop_mode=crop_mode,
    patch_size=16,
    downsample_ratio=4,
    image_width=orig_width,
    image_height=orig_height,
)
w_crop, h_crop, num_crops = compute_view_layout(workload)
tokens_global, tokens_crops, tokens_total = compute_projector_tokens(
    workload.base_size,
    workload.image_size,
    w_crop,
    h_crop,
    workload.patch_size,
    workload.downsample_ratio,
)
```

### 2. Split the analytic vision stack into global and crop branches

**Objective:** Represent SAM/CLIP/projector for global and crop views as separate analytic sub-layers with their own shapes and batch sizes.

- Update `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`:
  - Keep existing definitions for a single-view configuration.
  - Introduce composite configs for:
    - `vision_global` – SAM + CLIP + projector for a **single** 1024×1024 view.
    - `vision_crops` – SAM + CLIP + projector for `num_crops` 640×640 views.
  - Example sketch:

    ```yaml
    vision_global:
      _target_: modelmeter.models.deepseek_ocr.layers.core.deepseek_ocr_model._CompositeLayer
      layers:
        - ${vision.image_encoder_vit}   # img_size: 1024, batch_size: 1
        - ${vision.vit_model}          # seq_len: tokens_global
        - ${vision.projector}          # num_tokens: tokens_global

    vision_crops:
      _target_: modelmeter.models.deepseek_ocr.layers.core.deepseek_ocr_model._CompositeLayer
      layers:
        - ${vision.image_encoder_vit_crops}   # img_size: 640, batch_size: num_crops
        - ${vision.vit_model_crops}          # seq_len: tokens_crop
        - ${vision.projector_crops}          # num_tokens: tokens_crops
    ```

- Modify `vision.vision` to be a composite over both branches:

  ```yaml
  vision:
    _target_: modelmeter.models.deepseek_ocr.layers.core.deepseek_ocr_model._CompositeLayer
    layers:
      - ${vision_global}
      - ${vision_crops}
  ```

- Add matching analytic layer parameters:
  - `image_encoder_vit_crops` with:
    - `img_size: ${hf.vision.crop_img_size}` (or override at runtime to 640)
    - `batch_size` driven by `num_crops`
  - `vit_model_crops` with `seq_len` set to `num_queries_crop**2 + 1` and `batch_size` = `num_crops`.
  - `projector_crops` with `num_tokens` = `tokens_crops`.

**Before (single homogeneous vision stack):**
```yaml
vision:
  _target_: modelmeter.models.deepseek_ocr.layers.core.deepseek_ocr_model._CompositeLayer
  layers:
    - ${vision.image_encoder_vit}
    - ${vision.vit_model}
    - ${vision.projector}
```

**After (explicit global + crop branches):**
```yaml
vision:
  _target_: modelmeter.models.deepseek_ocr.layers.core.deepseek_ocr_model._CompositeLayer
  layers:
    - ${vision_global}
    - ${vision_crops}
```

### 3. Wire VisionWorkload into the analytic model instantiation

**Objective:** Let verification scripts compute a `VisionWorkload`, then override the vision configs to match the actual image and crop layout.

- In `run_verify_end2end_vision.py`:
  - After building vendor inputs, construct `VisionWorkload` with real image width/height and `(base_size, image_size, crop_mode)`.
  - Call `compute_view_layout` to get `(w_crop, h_crop, num_crops)` and `compute_projector_tokens` for token counts.
  - Override relevant Hydra nodes before instantiating the analytic model:

    ```python
    workload = VisionWorkload(
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=bool(args.crop_mode),
        patch_size=16,
        downsample_ratio=4,
        image_width=image.width,
        image_height=image.height,
    )
    w_crop, h_crop, num_crops = compute_view_layout(workload)
    tokens_global, tokens_crops, tokens_total = compute_projector_tokens(...)

    # Global branch: batch_size=1, img_size=base_size
    cfg.vision.image_encoder_vit.batch_size = 1

    # Crop branch: batch_size=num_crops, img_size=image_size
    cfg.vision.image_encoder_vit_crops.img_size = workload.image_size
    cfg.vision.image_encoder_vit_crops.batch_size = num_crops

    # CLIP/projector seq_len / num_tokens per branch
    cfg.vision.vit_model.seq_len = tokens_global
    cfg.vision.vit_model_crops.seq_len = tokens_crops_per_view
    cfg.vision.projector.num_tokens = tokens_global
    cfg.vision.projector_crops.num_tokens = tokens_crops
    ```

  - Remove the `effective_views`-based batch scaling from `run_verify_end2end_vision.py`.

**Before:**
```python
effective_views = _compute_effective_views(...)
for key in ("image_encoder_vit", "clip_embeddings", "notp_attention",
            "notp_feedforward", "vit_model", "projector"):
    if key in cfg.vision:
        cfg.vision[key]["batch_size"] = effective_views
analytic_model: DeepseekOCRModel = instantiate(cfg.model)
analytic_model.start_vision(batch_size=int(cfg.runtime.batch_size))
```

**After:**
```python
workload = build_vision_workload_from_vendor_inputs(...)
w_crop, h_crop, num_crops = compute_view_layout(workload)
tokens_global, tokens_crops, _ = compute_projector_tokens(...)
override_vision_configs_from_workload(cfg, workload, w_crop, h_crop, num_crops,
                                      tokens_global, tokens_crops)

analytic_model: DeepseekOCRModel = instantiate(cfg.model)
analytic_model.start_vision(batch_size=int(cfg.runtime.batch_size))
```

### 4. Keep BaseLayer-derived stats consistent with inference

**Objective:** Ensure all `BaseLayer`-derived metrics (not just FLOPs) reflect the new global + crops structure.

- For each analytic vision layer (`ImageEncoderViT`, `VitModel`, `MlpProjector` and any new `*_crops` variants):
  - Re-check `forward_cal_io`, `backward_cal_io`, `forward_memory_weight`, `forward_memory_activation`, and related methods so they:
    - Account for **both** global and crop branches with their correct batch sizes and spatial/token shapes.
    - Scale contributions by `num_crops` where the vendor actually runs per-crop work (e.g., SAM passes at 640, CLIP/projector passes per crop).
  - When adding the composite `vision_global` and `vision_crops` stacks, verify that `_CompositeLayer` aggregation still matches “run global once + run crops `num_crops` times” semantics for:
    - FLOPs (`forward_tensor_core_flops` / `forward_cuda_core_flops`)
    - I/O (`forward_cal_io`)
    - Memory (`forward_memory_weight`, `forward_memory_activation`, `forward_memory_kvcache`)
- Add a small debug mode (or extend existing debug scripts) to print these stats per branch so we can sanity-check that:
  - Increasing `w_crop * h_crop` increases both FLOPs and non-FLOP stats in line with vendor behavior.
  - Removing crops (crop_mode = 0 or 1×1 grid) reduces the workload back to the global-only case.

### 5. Update end-to-end verifier to reuse the same workload

**Objective:** Ensure `_compute_analytic_total_tflops` uses the same shape-accurate vision workload as the vision-only verifier.

- In `run_verify_end2end.py`:
  - Reuse `VisionWorkload`, `compute_view_layout`, and `override_vision_configs_from_workload(...)` before instantiating the analytic model.
  - Remove any residual assumptions that vision FLOPs are `batch_size * “single-view cost”`; they should now be the sum over global + crops.
  - Keep the FlashAttention toggle logic (`_set_ignore_torch_unsupported_flop_count(True)`) as-is for decoder alignment.

### 6. Re-run verification and adjust as needed

- Use:
  - `tmp/vision-flops-debug/run_vision_shape_debug.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end_vision.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end.py`
- Targets:
  - Vision-only FLOPs within ~1–5% of vendor across several OmniDocBench pages with different aspect ratios / crop grids.
  - End-to-end FLOPs (vision + prefill + short decode) within similar tolerance.
  - StageCost metrics (`io_tb`, `activations_gb`) scale reasonably with `(w_crop, h_crop)` and image sizes.

## Impact Analysis

- **Positive impacts**
  - Vision FLOPs become structurally aligned with the vendor model:
    - SAM is modeled as 1× global (1024) + `num_crops`× crops (640), not as `effective_views` copies of a single 1024 workload.
    - CLIP/projector token counts reflect the true mix of global and crop tokens, including the additional “newline” and “view separator” semantics.
  - StageCost metrics become interpretable: changes in crop grid directly affect the number of SAM passes and token counts, not an opaque multiplier.

- **Risks**
  - The Hydra config for vision becomes more complex (global + crops branches); misconfigurations could lead to inconsistent shapes if overrides are incomplete.
  - Verification scripts now depend on `VisionWorkload` helpers; mistakes in those helpers can propagate to both vision-only and end-to-end checks.

- **Mitigations**
  - Keep `VisionWorkload` and view-layout helpers small, well-documented, and unit-tested (e.g., small tests that assert `(w_crop, h_crop)` for a few synthetic aspect ratios).
  - Maintain a small regression set of OmniDocBench pages and log:
    - Vendor FLOPs
    - Analytic FLOPs
    - Crop grids and token counts
  - Use assertions in verification scripts to ensure:
    - `num_crops` derived analytically matches `w_crop * h_crop` read from `images_spatial_crop`.

## Expected Outcome

- The end-to-end vision verifier reports:
  - Analytic vision FLOPs within a few percent of vendor across representative documents.
  - No reliance on `effective_views`-style batch scaling for SAM; instead, multi-view workload comes from explicit global + crops branches.
- The analytic vision model:
  - Uses only vendor-equivalent parameters and image geometry to determine FLOPs and StageCost.
  - Has a clear mapping from vendor operations (global SAM pass + multiple crop passes + CLIP + projector) to analytic sub-layers.

## References

- **Vendor implementation**
  - Dynamic crops and infer path: `models/deepseek-ocr/modeling_deepseekocr.py`
    - `dynamic_preprocess`, `DeepseekOCRModel.forward`, CLIP + projector fusion.
  - Vision modules: `models/deepseek-ocr/deepencoder.py`
    - `ImageEncoderViT`, `VitModel`, `MlpProjector`.
- **Analytic implementation**
  - Root model aggregator: `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - Vision layers:
    - `extern/modelmeter/models/deepseek_ocr/layers/vision/image_encoder_vit.py`
    - `extern/modelmeter/models/deepseek_ocr/layers/vision/vit_model.py`
    - `extern/modelmeter/models/deepseek_ocr/layers/vision/mlp_projector.py`
  - Vision config: `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`
  - Analytic master config: `extern/modelmeter/models/deepseek_ocr/configs/deepseek_ocr.yaml`
- **Verification scripts**
  - Vision-only FLOPs: `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end_vision.py`
  - End-to-end FLOPs: `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end.py`
  - Debug helper: `tmp/vision-flops-debug/run_vision_shape_debug.py`
- **3rd-party libraries (Context7 IDs)**
  - PyTorch: `/pytorch/pytorch`
  - Hugging Face Transformers: `/huggingface/transformers`
