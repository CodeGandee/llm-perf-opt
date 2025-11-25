# Refactor Plan: DeepSeek-OCR Analytic Vision Workload Modeling

## What to Refactor

- **Analytic vision workload modeling**
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/image_encoder_vit.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/vit_model.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/mlp_projector.py`
  - Vision config: `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`
- **Analytic root aggregation**
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - Remove/avoid root-level “vision workload” scaling knobs that don’t exist in the vendor model.
- **Verification scripts**
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end_vision.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end.py`
  - Ensure FLOP and StageCost computations derive workload purely from vendor-like parameters:
    - `base_size`, `image_size`, `crop_mode`
    - `patch_size`, `downsample_ratio`
    - Dynamic crop grid `(w_crop, h_crop)` mirroring `dynamic_preprocess`.

## Why Refactor

- **Avoid non-vendor parameters**
  - The temporary `vision_workload_multiplier` at the root is not present in the vendor implementation and acts as a calibration “knob” rather than a real workload parameter.
  - This breaks the goal of analytically mimicking the vendor model: FLOPs and StageCost should be determined only by the same knobs the vendor code uses (image sizes, crop_mode, dynamic_preprocess).
- **Shape-driven behavior**
  - Vendor DeepSeek-OCR derives the number of views and tokens **only** from image geometry and a fixed tiling heuristic; our analytic model should do the same.
  - A separate multiplier decouples FLOPs from those shapes and can hide modeling errors (e.g., miscounted views or tokens).
- **Maintainability & transparency**
  - Modeling views directly via shapes makes it easier to reason about changes (e.g., different crop grids, resolutions) and avoids “magic constants” in verification.
  - Future contributors can inspect configs and see how many views/tokens are modeled without hunting for hidden multipliers.

## How to Refactor

### 1. Remove root-level vision multipliers and re-center on shapes

- **DeepseekOCRModel**
  - Remove the `m_vision_workload_multiplier` state and `set_vision_workload_multiplier(...)` API from `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`.
  - Ensure `_iter_layers()` continues to represent:
    - `"vision"` → vision composite only.
    - `"prefill"` → vision + decoder (+ optional head).
    - `"decode"` → decoder (+ optional head).
  - Keep `forward_*` aggregation purely as:

    **Before (root-level scaling idea):**
    ```python
    for layer in self._iter_layers():
        val = layer.forward_tensor_core_flops()
        if layer is self.m_vision_layer and self.m_stage in {"vision", "prefill"}:
            val = (val or 0.0) * self.m_vision_workload_multiplier
        total += val or 0.0
    ```

    **After (shape-driven only):**
    ```python
    for layer in self._iter_layers():
        val = layer.forward_tensor_core_flops()
        total += val or 0.0
    ```

  - Rationale: all multi-view multiplicity should be encoded in the **vision stack configuration** (batch, image_size, seq_len) derived from vendor parameters, not in a separate scalar.

- **Verification scripts**
  - Remove all calls to `set_vision_workload_multiplier(...)` in:
    - `run_verify_end2end_vision.py`
    - `run_verify_end2end.py`
  - Instead, rely on analytic vision layers whose batch/sizes already encode the correct number of views for the workload being verified.

### 2. Derive effective views from vendor-like parameters

- **Define an explicit “vision workload” object (internal only)**
  - Add a lightweight dataclass or plain struct within the analytic vision module (or a helper file), e.g.:

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
    ```

  - Provide a helper that mirrors vendor `dynamic_preprocess` logic but returns only the **grid** and view counts:

    ```python
    def compute_view_layout(workload: VisionWorkload, image_w: int, image_h: int) -> tuple[int, int, int]:
        """
        Returns (num_views_total, w_crop, h_crop) using the same
        aspect-ratio search and min/max grid constraints as
        modeling_deepseekocr.dynamic_preprocess, but purely from
        shape and workload parameters.
        """
    ```

  - This helper should:
    - Enumerate `(i, j)` grids with `min_num <= i*j <= max_num`.
    - Pick `(w_crop, h_crop)` by aspect-ratio matching with the same tie-break rules.
    - Compute:
      - `num_crops = w_crop * h_crop` if `crop_mode` and grid is non-trivial.
      - `num_views_total = 1 + num_crops` (global + crops) or just `1` if `crop_mode=False`.

### 3. Wire shapes into analytic SAM / CLIP / projector

- **ImageEncoderViT (SAM)**
  - Extend `ImageEncoderViT` to accept **image_height** and **image_width**, not just a square `img_size`, or document and enforce square behavior as in the vendor static run.
  - Configure `batch_size` and `img_size` using `VisionWorkload` and `compute_view_layout`:
    - For canonical verification:
      - Treat SAM as processing `num_views_total` views with shape approximated by:
        - Global: `(base_size, base_size)`
        - Crops: `(image_size, image_size)`
      - Option A (simple, statically approximated):
        - Use a single `img_size` representative (e.g. 640) and `batch_size = num_views_total`.
        - This ensures **all BaseLayer-derived stats** (FLOPs, `forward_cal_io`, `forward_memory_*`) scale with the number of views via the batch dimension, exactly as in vendor code.
      - Option B (richer model):
        - Split SAM into two analytic layers: one for global view (`batch=1, img_size=base_size`), one for crops (`batch=num_crops, img_size=image_size`), both included in the composite vision stack.

    **Before (one static batch):**
    ```yaml
    image_encoder_vit:
      img_size: ${hf.vision.img_size}      # 1024
      batch_size: ${runtime.batch_size}    # 1
    ```

    **After (example, canonical workload approximated):**
    ```yaml
    image_encoder_vit:
      img_size: 640
      batch_size: 4   # effective views from vendor workload
    ```

    or, using two explicit SAM layers in the composite for global + crops.

- **CLIP ViT (VitModel + NoTPTransformer)**
  - Set `batch_size` for CLIP to match the **effective number of CLIP sequences** dictated by the view layout:
    - Option A: treat all views symmetrically → `batch_size = num_views_total`.
    - Option B: split CLIP into global vs crop CLIP layers if we want to distinguish token counts.
  - Ensure `seq_len` approximates the combined token count (e.g., 256 global + 100 per crop) or keep the current representative `257` and treat per-seq FLOPs as averaged.
  - In all cases, use the same `batch_size` and `seq_len` when computing FLOPs, I/O and memory so that **StageCost (via `forward_cal_io`, `forward_memory_*`) reflects the total number of CLIP sequences including crops**.

- **MlpProjector**
  - Replace the fixed `num_tokens` with a shape-derived value:

    ```yaml
    projector:
      num_tokens: ${vision_tokens.total}  # computed from base_size, image_size, w_crop, h_crop, downsample_ratio
    ```

  - Where `vision_tokens.total` is derived using the same `num_queries_base` and `num_queries` formulas we already mirror in `dsocr_analyzer` / verification scripts.
  - Because projector FLOPs and memory scale with both `batch_size` (views) and `num_tokens` (patches per view), this shape-driven config ensures **all BaseLayer metrics (FLOPs, I/O, activations) automatically include the contribution from all crops**.

### 4. Update verification scripts to construct the same VisionWorkload

- **run_verify_end2end_vision.py**
  - When building vendor inputs via `_build_end_to_end_inputs(...)`, also call `compute_view_layout(...)` with:
    - `base_size`, `image_size`, `crop_mode` from CLI/Hydra.
    - `image.width`, `image.height` from the actual PIL image (shape only).
  - Use this layout to:
    - Instantiate the analytic vision stack with matching `batch_size` and `img_size` / `num_tokens`.
    - Avoid any extra multipliers; FLOPs should fall out of the configured shapes.

- **run_verify_end2end.py**
  - Reuse the same `VisionWorkload` / `compute_view_layout` when constructing the analytic model before `_compute_analytic_total_tflops(...)`.
  - Ensure the analytic **prefill** view count matches the view count used in the vendor prefill (SAM+CLIP + projector).

**Before (hacky scaling idea):**
```python
analytic_model: DeepseekOCRModel = instantiate(cfg.model)
if hasattr(analytic_model, "set_vision_workload_multiplier"):
    analytic_model.set_vision_workload_multiplier(4.0)
```

**After (shape-based only, pseudo-code):**
```python
workload = VisionWorkload(
    base_size=base_size,
    image_size=image_size,
    crop_mode=crop_mode,
    patch_size=16,
    downsample_ratio=4,
)
num_views, w_crop, h_crop = compute_view_layout(workload, image.width, image.height)

cfg_vision = build_vision_cfg_from_workload(workload, num_views=num_views)
analytic_model: DeepseekOCRModel = instantiate(cfg.model, vision_layer=instantiate(cfg_vision))
```

### 5. Recalibrate against static torchinfo and end-to-end scripts

- Re-run:
  - `tmp/dsocr-debug/run_vision_flops_debug.py` (or equivalent) to print per-layer analytic FLOPs vs vendor FLOPs under:
    - Canonical 1024/640 crop-mode.
    - A couple of other aspect ratios → different `(w_crop, h_crop)`.
  - `run_verify_end2end_vision.py` on several OmniDocBench pages, capturing not only FLOPs but also:
    - Analytic vs measured activation sizes where possible (e.g., via intermediate tensor shapes or additional debug logging) to sanity-check `forward_memory_activation`.
  - `run_verify_end2end.py` with a small decode length (e.g., 10 tokens).
- Adjust the effective SAM/CLIP/projector batches and token counts until:
  - Vision-only FLOPs within ~1–5% across representative pages.
  - End-to-end prefill + decode FLOPs within similar tolerance (decoder is already in good shape).
  - **StageCost components** (`io_tb`, `weights_gb`, `activations_gb`) stable and proportional when crop grids change (e.g., switching from 1×1 to 2×3 should scale stats roughly with view/token counts).

## Impact Analysis

- **Removal of hackish knobs**
  - Eliminating `vision_workload_multiplier` and its usages will slightly reshape the verification code, but improves conceptual consistency: FLOPs and StageCost are derived solely from **shapes and vendor-like parameters**.
- **Config changes**
  - Vision config will encode effective view count via `batch_size` and possibly additional fields (e.g., `num_views`, `vision_tokens.total`).
  - Existing callers relying on `runtime.batch_size` inside analytic vision layers will no longer implicitly control view count; they should instead pass desired workloads via Hydra overrides or dedicated workload helpers.
- **Risk of temporary mismatch**
  - While we rewire the analytic vision stack to be fully shape-driven, FLOP alignment may temporarily worsen on some workloads.
  - Mitigation: keep the debug script and vision-only verifier as ground truth checks; iterate until alignment is restored across a small battery of images.
- **Clarity and future extensions**
  - Once refactored, supporting new crop strategies or resolutions will mean:
    - Adjusting `VisionWorkload`/`compute_view_layout` and tokens formulas.
    - No need to hunt for hidden multipliers in root models or scripts.

## Expected Outcome

- The analytic vision model:
  - Uses only vendor-equivalent parameters (`base_size`, `image_size`, `crop_mode`, `patch_size`, `downsample_ratio`, dynamic crop grid) to determine FLOPs and StageCost.
  - Does not rely on ad-hoc multipliers unrelated to vendor configuration.
- Vision-only and end-to-end FLOP verifiers:
  - Match vendor FLOPs within a few percent across representative OmniDocBench pages and crop grids.
- The codebase:
  - Has a clear separation between **workload description** (VisionWorkload) and **analytic layer structure**.
  - Remains faithful to vendor behavior, making future debugging and profiling more straightforward.

## References

- **Vendor code / behavior**
  - Dynamic crops & infer path: `models/deepseek-ocr/modeling_deepseekocr.py`
  - SAM/CLIP/projector modules: `models/deepseek-ocr/deepencoder.py`
  - Static shapes / FLOPs: `reports/20211117-dsorc-op-analysis/static-20251118-130533/torchinfo-summary.txt`
- **Analytic model & configs**
  - Root model: `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - Vision layers: `extern/modelmeter/models/deepseek_ocr/layers/vision/*.py`
  - Vision config: `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`
  - Analytic master config: `extern/modelmeter/models/deepseek_ocr/configs/deepseek_ocr.yaml`
- **Verification scripts**
  - Vision-only: `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end_vision.py`
  - End-to-end: `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end.py`
- **3rd-party libraries (Context7 IDs)**
  - PyTorch: `/pytorch/pytorch`
  - Hugging Face Transformers: `/huggingface/transformers`
