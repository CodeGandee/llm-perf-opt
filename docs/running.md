# Running

Pixi tasks
- `bench-stage1`: subprocess Hydra run via `tests/manual/manual_stage1_benchmark.py`
- `bench-stage1-inproc`: in-process Hydra run (notebook-friendly)
- `dsocr-infer-one`: run vendor `infer()` across inputs for parity

Examples
```
# Subprocess Hydra runner (default configs)
pixi run bench-stage1

# In-process variant (suitable for Jupyter)
pixi run bench-stage1-inproc

# Save predictions + viz
pixi run python -m llm_perf_opt.runners.llm_profile_runner \
  model/deepseek_ocr/arch@model=deepseek_ocr.default \
  model/deepseek_ocr/infer@infer=deepseek_ocr.default \
  outputs.save_predictions=true repeats=1 device=cuda:0

# Vendor parity on a single image (writes result_with_boxes.jpg/result.mmd)
pixi run python scripts/deepseek-ocr-infer-one.py \
  -i /abs/path/to/image.png -o tmp/vendor-parity
```

Hydra overrides
- Use `@model` and `@infer` when selecting model groups, e.g., `model/deepseek_ocr/arch@model=deepseek_ocr.default`.
- Common toggles:
  - `repeats=<int>`
  - `device=cuda:0`
  - `outputs.save_predictions=true`
  - `model.preprocess.enable=false` to skip preprocessing (debug mode)

Where outputs go
- Hydra run dir is set to `tmp/stage1/<run_id>/`.
- We write: `report.md`, `operators.md`, `metrics.json`, optional predictions + `viz/` and `llm_profile_runner.log`.

