# Running

Pixi tasks
- `stage1-run`: run pipelines `torch_profiler` + `static_analysis` and produce full artifacts (report/operators/metrics + reproducibility)
- `stage1-run-no-static`: same as `stage1-run` but disables static analyzer (skips `static_analysis/*`)
- `stage2-profile`: run Nsight Systems (`nsys`) capture (workload profiling/static disabled during capture)
- `stage-all-run`: run Stage 1 first, then `nsys` to the same output dir
- `bench-stage1`, `bench-stage1-inproc`: manual benchmark variants
- `dsocr-infer-one`: run vendor `infer()` across inputs for parity

Examples
```
# Subprocess Hydra runner (default configs)
pixi run bench-stage1

# In-process variant (suitable for Jupyter)
pixi run bench-stage1-inproc

# Save predictions + viz (direct Hydra)
pixi run python -m llm_perf_opt.runners.llm_profile_runner \
  model/deepseek_ocr/arch@model=deepseek_ocr.default \
  model/deepseek_ocr/infer@infer=deepseek_ocr.default \
  pipeline.torch_profiler.output.prediction.enable=true \
  pipeline.torch_profiler.output.visualization.enable=true \
  dataset/sampling@dataset.sampling=default \
  dataset.sampling.num_epochs=1 \
  dataset.sampling.num_samples_per_epoch=1 \
  device=cuda:0 \
  infer.max_new_tokens=8192

# One-liners with Pixi tasks
pixi run stage1-run                      # torch_profiler + static_analysis
pixi run stage1-run-no-static            # torch_profiler only
pixi run stage2-profile                  # nsys only (ncu disabled)
pixi run stage-all-run                   # stage1 + nsys into same run dir

# Vendor parity on a single image (writes result_with_boxes.jpg/result.mmd)
pixi run python scripts/deepseek-ocr-infer-one.py \
  -i /abs/path/to/image.png -o tmp/vendor-parity
```

Hydra overrides
- Use `@model` and `@infer` when selecting model groups, e.g., `model/deepseek_ocr/arch@model=deepseek_ocr.default`.
- Common toggles:
  - `device=cuda:0`
  - `pipeline.torch_profiler.output.prediction.enable=true`
  - `pipeline.torch_profiler.output.visualization.enable=true`
  - `infer.max_new_tokens=<int>` (integer only; e.g., 8192)
  - `model.preprocess.enable=false` to skip preprocessing (debug mode)
 - Disable static analyzer:
   - `pipeline.static_analysis.enable=false`

Where outputs go
- Hydra run dir defaults to `tmp/profile-output/<run_id>/`.
- Pipeline outputs:
  - `torch_profiler/`: `report.md`, `operators.md`, `metrics.json`, `llm_profile_runner.log`
  - `static_analysis/`: `static_compute.{json,md}` (when enabled)
  - `nsys/`: `run.nsys-rep`, `run.sqlite`, `summary_*.csv`, `cmd.txt` (when enabled)
  - `ncu/`: `raw.csv`, `.ncu-rep`, `sections_report.txt`, `cmd*.txt` (when enabled)
- Reproducibility at run root: `env.json`, `config.yaml`, `inputs.yaml`
- Optional predictions + viz: `torch_profiler/pred/predictions.jsonl`, `torch_profiler/viz/<hash>/{result_with_boxes.jpg,result.mmd,images/*,info.json}`
