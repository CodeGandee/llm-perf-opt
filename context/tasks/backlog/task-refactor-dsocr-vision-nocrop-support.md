# Refactor Plan: DeepSeek‑OCR Analytic Vision No‑Crop Support

## What to Refactor

- **Analytic vision stack configuration and layers** under:
  - `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/vision_workload.py`
- **Verification and sweep scripts** that reason about vision FLOPs:
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end_vision.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-vision-input-shape.py`

The refactor should introduce a clean way for the analytic model to support both:

- **Crop mode**: dynamic crops enabled (current behavior), and
- **No‑crop mode**: only the global view contributes to vision workloads, mirroring `crop_mode=False` and the vendor small‑image shortcut (`width <= 640 and height <= 640`).

This must be reflected consistently in:

- FLOP estimates,
- IO and memory estimates from `BaseLayer`/`StageCostMixin` (`forward_*` functions),
- and any higher‑level aggregations (e.g., `_CompositeLayer`, `DeepseekOCRModel.start_vision`).

---

## Why Refactor

1. **Match vendor behavior for small images and `crop_mode=False`**  
   The vendor code treats images with `width <= 640` and `height <= 640` (or `crop_mode=False`) as **no crop**: it uses only a single global view and never calls `dynamic_preprocess`. Our analytic model currently always models a crop branch when `crop_mode=True`, which overestimates vision FLOPs in no‑crop scenarios (e.g., 640×640).

2. **Enable clean comparisons for “global‑only” workloads**  
   Some analyses (e.g., FLOPs vs base resolution without dynamic crops) require a pure global view. Right now we can only approximate this by manually zeroing crop token counts in scripts, not by configuring the analytic model in a principled way.

3. **Clarify StageCost semantics for vision**  
   `DeepseekOCRModel` aggregates vision cost via an internal `_CompositeLayer`, but the concept of “vision” is currently hard‑wired to “global + crops”. Adding explicit no‑crop support forces us to clarify how StageCost is composed, making future analyses less error‑prone.

4. **Reduce ad‑hoc logic in sweeps and verify scripts**  
   `sweep-vision-input-shape.py` currently has to special‑case vendor no‑crop behavior without a matching analytic knob. Supporting no‑crop at the model level will simplify sweep scripts and make comparisons more transparent.

---

## How to Refactor

### 1. Introduce an explicit crop mode in `VisionWorkload`

File: `extern/modelmeter/models/deepseek_ocr/layers/vision/vision_workload.py`

- **Current**: `VisionWorkload` already has `crop_mode: bool`, but `compute_view_layout(...)` only treats `crop_mode=False` as “return 1×1 grid” (which we currently still treat as a global + crops configuration in higher‑level configs).
- **Change**:
  - Clarify in the docstring that:
    - `crop_mode=True` → dynamic crops enabled (2..9 tiles when large enough).
    - `crop_mode=False` → strictly no dynamic crops; only the global view contributes.
  - Ensure `compute_view_layout(...)` returns `(1, 1, 1)` and `tokens_crops=0` when `crop_mode=False`, independent of image size.

This solidifies the shape‑level semantics we need downstream.

### 2. Extend analytic vision configs to have two modes

File: `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`

- **Current**:
  - `vision_global`: composite of `image_encoder_vit`, `vit_model`, `projector` for the padded base view.
  - `vision_crops`: composite of `image_encoder_vit_crops`, `vit_model_crops`, `projector_crops`.
  - `vision`: `_CompositeLayer` over `[vision_global, vision_crops]` with both branches always active.
- **Change**:
  - Introduce an explicit vision‑mode flag in config, e.g.:

    ```yaml
    vision_mode: "crops"  # or "nocrop"
    ```

    or a more structured node:

    ```yaml
    vision:
      mode: "crops"  # or "nocrop"
      ...
    ```

  - Add a new composite definition for **no‑crop**:

    ```yaml
    vision_nocrop:
      _target_: modelmeter.models.deepseek_ocr.layers.core.deepseek_ocr_model._CompositeLayer
      layers:
        - ${vision.image_encoder_vit}
        - ${vision.vit_model}
        - ${vision.projector}
    ```

  - Keep the existing `vision` composite as the **crops** variant, but make which composite is bound into `DeepseekOCRModel` configurable via `cfg.model`:

    ```yaml
    model:
      _target_: modelmeter.models.deepseek_ocr.layers.core.deepseek_ocr_model.DeepseekOCRModel
      vision_layer: ${vision.vision_global_crops_or_nocrop}
    ```

  - Alternatively, add an enum field to `DeepseekOCRModel` that selects between `vision_global` and `[vision_global + vision_crops]` at construction time.

This allows Hydra to specify “no‑crop analytic vision” via config composition without touching code.

### 3. Add a runtime vision mode selector to `DeepseekOCRModel`

File: `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`

- **Current**:
  - `DeepseekOCRModel` has a single `m_vision_layer: _CompositeLayer` and `start_vision(batch_size=...)` simply switches operation mode to `"vision"` and uses that layer.
- **Change**:
  - Add an optional **secondary vision layer**:

    ```python
    self.m_vision_layer_nocrop: Optional[_CompositeLayer] = None
    self.m_vision_mode: str = "crops"  # or "nocrop"
    ```

  - Add setters:

    ```python
    def set_vision_layer_nocrop(self, layer: _CompositeLayer) -> None: ...
    def set_vision_mode(self, mode: str) -> None:
        if mode not in {"crops", "nocrop"}:
            raise ValueError("vision_mode must be 'crops' or 'nocrop'")
        self.m_vision_mode = mode
    ```

  - Update `start_vision(...)` to select the appropriate analytic vision layer:

    ```python
    def start_vision(self, *, batch_size: int = 1) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.m_stage = "vision"
        self.m_batch_size = batch_size

        if self.m_vision_mode == "crops":
            if self.m_vision_layer is None:
                raise ValueError("Vision layer must be configured for crops mode")
        else:
            if self.m_vision_layer_nocrop is None:
                raise ValueError("Vision no-crop layer must be configured for nocrop mode")
    ```

  - Adjust `_iter_layers` to yield only the active vision layer in `"vision"` mode:

    ```python
    if self.m_stage == "vision":
        if self.m_vision_mode == "crops" and self.m_vision_layer is not None:
            yield self.m_vision_layer
        elif self.m_vision_mode == "nocrop" and self.m_vision_layer_nocrop is not None:
            yield self.m_vision_layer_nocrop
        return
    ```

  - Ensure `_set_ignore_torch_unsupported_flop_count` propagates to both `m_vision_layer` and `m_vision_layer_nocrop` if they are present.

This cleanly encapsulates the notion of “vision mode” inside the analytic root model.

### 4. Wire Hydra configs into `DeepseekOCRModel`

File: `extern/modelmeter/models/deepseek_ocr/configs/deepseek_ocr.yaml` (or equivalent)

- Update the `model` node instantiation to pass both composites:

```yaml
model:
  _target_: modelmeter.models.deepseek_ocr.layers.core.deepseek_ocr_model.DeepseekOCRModel
  vision_layer: ${vision.vision}          # crops mode composite
  vision_layer_nocrop: ${vision.vision_nocrop}
  vision_mode: "crops"                    # default; can override with Hydra CLI
```

- Extend `DeepseekOCRModel.__init__` or a `from_layers` factory to accept these additional parameters and call `set_vision_layer`, `set_vision_layer_nocrop`, and `set_vision_mode`.

This lets scripts do:

```bash
pixi run -e rtx5090 python -m ... \
  model.vision_mode=nocrop
```

to switch analytic vision to global‑only mode.

### 5. Update sweep and verify scripts to use vision modes

#### 5.1 `run_verify_end2end_vision.py`

- Instead of manually approximating no‑crop behavior in scripts, use `vision_mode`:
  - For the “crops” verification, keep `vision_mode="crops"` and derive `VisionWorkload` with `crop_mode=True`.
  - For a “no‑crop” variant (if desired), instantiate the analytic model with `vision_mode="nocrop"` and `crop_mode=False` in `VisionWorkload`.
- This aligns the analytic model with vendor `crop_mode` flags and small‑image shortcut.

#### 5.2 `sweep-vision-input-shape.py`

- Simplify the no‑crop handling:
  - Remove any attempts to hack batch sizes to zero for crop layers.
  - For analytic **crops** curve:

    ```python
    cfg_crops.model.vision_mode = "crops"
    workload_crops.crop_mode = True
    ```

  - For analytic **no‑crop** curve:

    ```python
    cfg_nocrop.model.vision_mode = "nocrop"
    workload_nocrop.crop_mode = False
    ```

- Plot both analytic curves (`vision_crops`, `vision_nocrop`) and both vendor curves (`vision_vendor`, `vision_vendor_nocrop`) where applicable.

This reduces script complexity and ensures consistency with the analytic model’s notion of vision mode.

### 6. Keep BaseLayer/StageCost functions consistent

Because vision modes are implemented by selecting different analytic composites, existing `BaseLayer` and `StageCostMixin` functions remain valid:

- `forward_tensor_core_flops`, `forward_cal_io`, `forward_memory_*` on individual vision layers do not need to know about crop mode.
- `DeepseekOCRModel.get_forward_cost()` in `"vision"` mode will aggregate over the currently active vision composite (crops or no‑crop) and expose a consistent `StageCost` for each mode.

No changes are required to the StageCost structure itself; we are only changing how we choose which vision sublayers participate.

---

## Impact Analysis

### Functional impact

- **Analytic vision cost** becomes configurable between:
  - “global + crops” (current behavior), and
  - “global only” (no‑crop mode), matching vendor `crop_mode` semantics and small‑image shortcut.
- Verification and sweeps can now:
  - Compare analytic vs vendor FLOPs in both modes explicitly.
  - Inspect how much FLOP increase is attributable to crops versus global tokens alone.

### Risks

- Mis‑wiring Hydra configs for `vision_layer` / `vision_layer_nocrop` could break model instantiation.
- `_iter_layers` and `_set_ignore_torch_unsupported_flop_count` logic must be carefully updated to avoid double‑counting or skipping vision layers.
- Existing scripts that assume a single `vision_layer` may need minor updates to respect `vision_mode`.

### Mitigation

- Add small unit tests (or manual checks) to verify:
  - Instantiating the model with `vision_mode="crops"` vs `"nocrop"` yields different `get_forward_cost()` values for `"vision"` mode.
  - No exceptions are thrown when switching modes.
  - For `crop_mode=False` + `vision_mode="nocrop"`, the analytic vision FLOPs match the vendor `crop_mode=False` measurement closely at representative image sizes (e.g., 640×640, 1024×1024).
- Use existing `run_verify_vision.py` and `run_verify_end2end_vision.py` as benchmarks to confirm no regressions in the crops mode.

---

## Expected Outcome

After this refactor:

- The analytic DeepSeek‑OCR model will have a **first‑class no‑crop mode** that:
  - Mirrors vendor `crop_mode=False` and the small‑image rule (`width <= 640` and `height <= 640`).
  - Produces StageCost metrics for vision that correspond to “global view only”.
- Sweep and verify scripts will be able to:
  - Plot analytic and vendor curves for both “global only” and “global + crops” workloads.
  - Cleanly attribute FLOP differences to the presence or absence of dynamic crops.
- BaseLayer/StageCost APIs remain unchanged; only how we compose vision layers at the root changes, preserving consistency across the analytic model.

---

## References

- Vendor code:
  - `models/deepseek-ocr/modeling_deepseekocr.py` (vision + cropping logic)
  - `models/deepseek-ocr/deepencoder.py` (SAM-B, CLIP, projector)
- Analytic model:
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/image_encoder_vit.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/vision_workload.py`
  - `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`
- Verification and sweep scripts:
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_vision.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end_vision.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-vision-input-shape.py`

