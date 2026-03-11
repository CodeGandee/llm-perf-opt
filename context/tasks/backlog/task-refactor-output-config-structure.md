# Refactor Plan: Per‑Pipeline Output Configuration (Structure + Paths + Schema)

This revision scopes output settings to each pipeline stage (e.g., torch_profiler), adds per‑stage save_dir semantics, and completes the schema renames (`outputs`→`output`, `predictions`→`prediction`).

## What to Refactor
- Configure outputs within each pipeline stage:
  - `pipeline.torch_profiler.output.prediction.{enable, save_dir}`
  - `pipeline.torch_profiler.output.visualization.{enable, save_dir}`
  - `pipeline.torch_profiler.output.extra.<model>.*` for model‑specific options
- Save‑dir resolution: when a `save_dir` is relative, resolve it against the stage’s output dir, not the run root.
- Apply the same pattern to other stages as needed (e.g., `pipeline.nsys.output.*` in the future), but this refactor focuses on `torch_profiler` where predictions/visualization are produced today.
- Keep a backward‑compat layer for legacy keys during a migration window:
  - Top‑level `outputs.*` and prior `output.*` reads map to `pipeline.torch_profiler.output.*`.

Scope
- Config files: `conf/config.yaml`, new `conf/output/torch.default.yaml`, new `conf/model/deepseek_ocr/output/torch.default.yaml`.
- Code: Stage‑1 runner (`llm_profile_runner.py`) to read from `pipeline.torch_profiler.output.*` and resolve dirs relative to the stage dir; Stage‑2 runner (`deep_profile_runner.py`) to pass stage‑scoped overrides for its workload.
- Docs: `docs/configuration.md`, quickstarts, README examples.

## Why Refactor
- Accuracy: outputs are created by stages; configuration should live with the stage to avoid confusion.
- Predictable paths: `save_dir` relative to the stage dir matches user expectations and keeps artifacts tidy.
- Extensibility: other stages can introduce their own `output.*` blocks without clashing with torch_profiler.

## How to Refactor

1) Add per‑stage general output preset
- Create `conf/output/torch.default.yaml` (mounted under `pipeline.torch_profiler.output`):

```yaml
# General (model‑agnostic) output controls for the torch_profiler stage
prediction:
  enable: false
  # Relative resolves under the stage output directory (Artifacts.out_dir('torch_profiler')).
  # Default subdirectory when enabled: 'pred'. If save_dir is omitted/null while
  # enable=true, use 'pred'.
  save_dir: pred
visualization:
  enable: true
  # predictions.md and viz/* location; relative to the stage output directory.
  # Default subdirectory when enabled: 'viz'. If save_dir is omitted/null while
  # enable=true, use 'viz'.
  save_dir: viz
```

2) Add per‑stage model‑specific output preset (DeepSeek‑OCR example)
- Create `conf/model/deepseek_ocr/output/torch.default.yaml` (mounted under `pipeline.torch_profiler.output.extra.deepseek_ocr`):

```yaml
# DeepSeek‑OCR specific output knobs for the torch_profiler stage
prediction:
  strip_special_tokens: false
visualization:
  max_images: 16
  thumbnail_width: 480
```

3) Mount presets in the unified entry config
- Update `conf/config.yaml` defaults:

```yaml
defaults:
  # ...existing defaults...
  - output/torch@pipeline.torch_profiler.output: default
  - model/deepseek_ocr/output/torch@pipeline.torch_profiler.output.extra.deepseek_ocr: default
  - _self_
```

4) Update Stage‑1 runner (torch_profiler) to read per‑stage keys with fallback
- Feature toggles and directories:

Before (legacy reads)
```python
save_preds = bool(getattr(cfg, "outputs", {}).get("save_predictions", False))
viz_cfg = getattr(cfg.outputs, "visualization", {}) if hasattr(cfg, "outputs") else {}
pj = artifacts_dir / "predictions.jsonl"  # bound to torch_profiler dir
```

After (per‑stage reads + stage‑relative path resolution)
```python
stage_dir = artifacts_dir  # Artifacts.out_dir('torch_profiler')

tp_out = getattr(getattr(cfg.pipeline, "torch_profiler", {}), "output", {})
pred = getattr(tp_out, "prediction", {})
vis  = getattr(tp_out, "visualization", {})

save_preds = bool(pred.get("enable", False))
make_gallery = bool(vis.get("enable", True))

# Default subdirs when enabled and save_dir is omitted: 'pred' and 'viz'
pred_dir_cfg = str(pred.get("save_dir", "pred" if save_preds else "."))
vis_dir_cfg  = str(vis.get("save_dir", "viz" if make_gallery else "."))

from pathlib import Path
pred_dir = Path(pred_dir_cfg) if Path(pred_dir_cfg).is_absolute() else (stage_dir / pred_dir_cfg)
vis_dir  = Path(vis_dir_cfg)  if Path(vis_dir_cfg).is_absolute()  else (stage_dir / vis_dir_cfg)

pj = pred_dir / "predictions.jsonl"
md_path = vis_dir / "predictions.md"

# Back‑compat: if not present, consult legacy keys under cfg.outputs or cfg.output
```

- Model‑specific knobs (DeepSeek‑OCR), with legacy fallback:

```python
strip_special = False
extra = getattr(tp_out, "extra", {})
dso = getattr(extra, "deepseek_ocr", {})
strip_special = bool(getattr(getattr(dso, "prediction", {}), "strip_special_tokens", False))

max_images = getattr(getattr(dso, "visualization", {}), "max_images", 16)
thumb_w = int(getattr(getattr(dso, "visualization", {}), "thumbnail_width", 480))

# Legacy fallbacks (one release):
# - outputs.save_predictions → pipeline.torch_profiler.output.prediction.enable
# - outputs.visualization.enable → pipeline.torch_profiler.output.visualization.enable
# - outputs.predictions.strip_special_tokens → pipeline.torch_profiler.output.extra.deepseek_ocr.prediction.strip_special_tokens
# - outputs.visualization.{max_images, thumbnail_width} → pipeline.torch_profiler.output.extra.deepseek_ocr.visualization.{max_images, thumbnail_width}
```

- Wire directories into writers as before, using `pred_dir` and `vis_dir` bases.

5) Update Stage‑2 runner (workload separation)
- When building the Stage‑1 workload argv, use per‑stage overrides:
  - `pipeline.torch_profiler.output.prediction.save_dir=tmp/workload/torch_profiler`
  - `pipeline.torch_profiler.output.visualization.save_dir=tmp/workload/torch_profiler`
- Continue to disable `pipeline.torch_profiler.enable=false` to avoid CUPTI conflicts under NSYS.

6) Documentation
- Update docs/examples to use:
  - `pipeline.torch_profiler.output.prediction.enable=true`
  - `pipeline.torch_profiler.output.prediction.save_dir=pred` (default; saved under `<run>/torch_profiler/pred/`)
  - `pipeline.torch_profiler.output.extra.deepseek_ocr.prediction.strip_special_tokens=true`
  - `pipeline.torch_profiler.output.extra.deepseek_ocr.visualization.max_images=8`

7) Backward compatibility and deprecation
- For one release window:
  - Prefer per‑stage keys; still read legacy top‑level keys if present.
  - Emit a single INFO when legacy keys are used, pointing to the new per‑stage path.

8) Validation
- Stage‑1:
  - `pixi run stage1-run -- pipeline.torch_profiler.output.prediction.enable=true pipeline.torch_profiler.output.visualization.enable=false`
  - `pixi run stage1-run -- pipeline.torch_profiler.output.prediction.save_dir=preds`
  - `pixi run stage1-run -- pipeline.torch_profiler.output.extra.deepseek_ocr.prediction.strip_special_tokens=true`
- Stage‑2:
  - `pixi run stage2-profile` → workload predictions/gallery under `<run>/tmp/workload/torch_profiler/`.

## Impact Analysis
- Risk: low. Changes are confined to per‑stage config and read‑paths with fallbacks.
- Path behavior: absolute `save_dir` honored; relative resolved under the stage dir.
- Default behavior remains: files written under the stage dir when `save_dir="."`.

## Expected Outcome
- Output controls live with the pipeline stages that produce them.
- Clear, stage‑relative path semantics for predictions and visualization artifacts.
- Smooth migration via back‑compat reads.

## References
- Hydra configuration (package mounting): /facebookresearch/hydra
- Code touch points:
  - Stage‑1 reads/writes: src/llm_perf_opt/runners/llm_profile_runner.py
  - Stage‑2 workload overrides: src/llm_perf_opt/runners/deep_profile_runner.py
- Current legacy keys: `outputs.*` in conf/config.yaml (to be mapped to `pipeline.torch_profiler.output.*`)
