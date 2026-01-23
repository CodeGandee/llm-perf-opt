# Refactor Plan: Wan2.1 precision options (FP8 vs FP16) for analytic sizing

## What to Refactor
- The Wan2.1 analytic model and sizing scripts currently mix assumptions: Wan2.1 configs use `bits: 16` (memory / IO / VRAM accounting), while NGU800P sizing uses `Tensor (FP8) peak` for compute timing.
- There is no first-class, end-to-end “precision” knob for analysis runs (FP8 vs FP16) that controls both (1) compute peak selection and (2) storage-bit assumptions used for IO/VRAM.
- Reports do not explicitly state the precision assumptions (compute peak selection + storage bits), which makes results hard to compare across runs and easy to misinterpret.

## Why Refactor
- Correctness of interpretation: “FP8 compute peak + 16-bit storage” is a mixed assumption; we want a single precision knob so analysis is either fully FP8 or fully FP16.
- Flexibility: we need to run the same workload sizing under different deployment assumptions (fully FP16 vs fully FP8) and compare bottlenecks.
- Reproducibility: analysis artifacts (CSV/JSON/summary.md) should carry the precision assumptions so downstream readers don’t need to infer them from device names.
- Maintainability: precision logic should live in one place (device peak mapping + Hydra config), not duplicated across scripts.

## How to Refactor
1) Define a precision contract (single source of truth)
- Introduce a single `precision` choice for Wan2.1 analysis: `fp8` or `fp16`.
- The precision choice controls both:
  - **Compute precision** (what tensor-core peak is used for timing): `fp8` or `fp16`.
  - **Storage bits** (what bitwidth is used for IO + VRAM accounting): `8` for `fp8`, and `16` for `fp16`.
- Ship two standard profiles for Wan2.1 analysis:
  - `fp8`: `compute_precision=fp8`, `storage_bits=8`.
  - `fp16`: `compute_precision=fp16`, `storage_bits=16`.

2) Add a device API for tensor peak selection
- Add a small helper in ModelMeter devices, e.g. `get_tensor_peak_tflops(device, tensor_dtype: Literal["fp8","fp16"])`.
- Update `NGU800P` to provide an FP16 tensor peak value (`fp16_tflops`) based on the official spec (or document a conservative estimate if the spec is not final).
- Keep backward compatibility: existing code can still read `device.fp8_tflops`, but new sizing/reporting should call the helper.

3) Make storage bitwidth explicit in Wan2.1 configs (without breaking existing overrides)
- Today Wan2.1 uses `bits` in multiple config groups (transformer/text_encoder/vae).
- Refactor config naming to reduce confusion:
  - Prefer `storage_bits` in config and in constructors for analytic layers/models.
  - Keep reading `bits` as an alias for one release cycle (Hydra interpolation or code-side fallback) to avoid breaking existing overrides.
- Ensure full pipeline config routes all three stage bitwidths through the precision choice (transformer/text_encoder/vae), while still allowing stage-specific overrides for debugging.

4) Add a Wan2.1 config group for precision profiles
- Add `extern/modelmeter/models/wan2_1/configs/precision/` with YAMLs like:
  - `fp8.yaml`
  - `fp16.yaml`
- Update entry configs (`wan2_1_t2v_14b*.yaml`) to include a default precision profile via Hydra defaults, while allowing users to override it from CLI.

5) Plumb precision into sizing scripts and metadata
- Update `modelmeter.models.wan2_1.scripts.sizing.run_ngu800p_concurrency_sweep` to:
  - Load the selected precision profile (via Hydra config, or via a new `--precision` CLI arg that becomes a Hydra override).
  - Select device tensor peak based on `compute_precision` (FP8 vs FP16).
  - Use `storage_bits` to set the Wan2.1 model’s IO/VRAM accounting bits (transformer/text_encoder/vae) unless explicitly overridden.
  - Record the precision profile in `results.json` metadata and in `summary.md` (“Key settings” section).

6) Reporting updates
- Update `extern/modelmeter/models/wan2_1/docs/contracts/req-reporting.md` to require that reports state:
  - Compute precision used for peak tensor throughput selection (FP8 vs FP16).
  - Storage bitwidth used for IO/VRAM accounting (8 vs 16).
- Update stakeholder reports (EN/CN) to include a one-line “Precision assumptions” statement to avoid confusion when presented externally.

7) Validation
- Add a lightweight unit/integration check that:
  - For the same workload, switching `compute_precision` changes predicted compute-bound latency components (tensor_cost_s) but does not change model FLOPs totals.
  - Switching `storage_bits` changes `model_io_tb`, `model_weights_gb`, and `model_act_peak_gb` as expected.
- Add a golden-file sanity check that `summary.md` includes the precision line and that tables remain deterministically ordered.

## Impact Analysis
- Behavior changes
  - Default behavior should become a single consistent mode (recommend defaulting to `fp16` to match current `bits: 16` configs).
  - New analyses can switch between fully FP16 and fully FP8 with one override.
- Risks
  - Adding/renaming config keys (`bits` -> `storage_bits`) can break external overrides if not handled carefully.
  - Some devices may not have reliable FP16 peak numbers; estimates must be documented to avoid misleading results.
- Mitigations
  - Keep `bits` as an alias and emit a deprecation note in docs (and optionally a warning in scripts).
  - Gate FP16 selection on device support; fail fast with a clear error if `fp16_tflops` is missing.
  - Record the precision profile in every artifact (CSV/JSON/summary) so results are self-describing.

## Expected Outcome
- Wan2.1 analysis runs can explicitly choose fully FP8 vs fully FP16 assumptions (compute peak + storage bits) with a single, consistent knob.
- Sizing scripts produce comparable, reproducible results with clear precision metadata.
- Reports no longer require readers to infer precision from device names or from config fragments.

## TODO
- [ ] Add `configs/precision/` group for Wan2.1 (`fp8`, `fp16`).
- [ ] Add device helper for tensor peak selection (FP8/FP16) and document assumptions for NGU800P FP16 peak.
- [ ] Refactor Wan2.1 configs to prefer `storage_bits` while keeping `bits` compatibility.
- [ ] Update Wan2.1 pipeline/core constructors to accept `storage_bits` (or map config to existing `bits` field without ambiguity).
- [ ] Update NGU800P sizing sweep script to select tensor peak by `compute_precision` and to record precision metadata in outputs.
- [ ] Update reporting contract to require explicit precision assumptions.
- [ ] Add a small test/sanity check for precision switching (compute vs storage).

## Example Code Snippets

Before: FP8 compute peak is hard-coded in sizing, independent of model `bits`
```python
# modelmeter.models.wan2_1.scripts.sizing.run_ngu800p_concurrency_sweep (current pattern)
device = NGU800P()
device_tensor_tflops = float(device.fp8_tflops) * util.tensor_util  # always FP8

# Wan2.1 config uses bits=16 (memory accounting), but that does not affect compute peak selection here.
model, cfg = _load_model(config_dir=config_dir, config_name="wan2_1_t2v_14b_pipeline", overrides=[...])
```

After: precision profile controls both compute-peak selection and storage bits (fully FP8 or fully FP16)
```python
# Pseudocode: precision profile is loaded via Hydra (defaults + overrides)
precision = cfg.precision.name  # "fp8" or "fp16"
compute_precision = precision
storage_bits = 8 if precision == "fp8" else 16

device = NGU800P()
device_tensor_tflops = get_tensor_peak_tflops(device, compute_precision) * util.tensor_util

# Model instantiation uses storage bits consistently (transformer/text/vae)
overrides = [
    f"transformer.storage_bits={storage_bits}",
    f"text_encoder.storage_bits={storage_bits}",
    f"vae.storage_bits={storage_bits}",
    ...
]
model, cfg = _load_model(..., overrides=overrides)
```

## References
- Code
  - extern/modelmeter/models/wan2_1/configs/transformer/wan2_1_dit.yaml
  - extern/modelmeter/models/wan2_1/configs/model/wan2_1_pipeline.yaml
  - extern/modelmeter/models/wan2_1/scripts/sizing/run_ngu800p_concurrency_sweep.py
  - extern/modelmeter/devices/gpu.py (NGU800P peaks)
  - extern/modelmeter/models/wan2_1/docs/contracts/req-reporting.md
- Third-party libraries (Context7)
  - Hydra: /facebookresearch/hydra
  - mdutils: /didix21/mdutils
