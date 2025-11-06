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
  - `use_flash_attn=true|false` (FlashAttention 2 toggling for supported GPUs)
  - `pipeline.torch_profiler.output.prediction.enable=true`
  - `pipeline.torch_profiler.output.visualization.enable=true`
  - `infer.max_new_tokens=<int>` (integer only; e.g., 8192)
  - `model.preprocess.enable=false` to skip preprocessing (debug mode)
 - Disable static analyzer:
   - `pipeline.static_analysis.enable=false`

Model and dataset paths (Stage 2)
- Stage 2 (deep profiling) no longer hard‑codes model or dataset paths. It resolves absolute paths from Hydra config values:
  - `cfg.model.path` → absolute via `${hydra:runtime.cwd}` if relative
  - `cfg.dataset.root` → absolute via `${hydra:runtime.cwd}` if relative
- Override paths directly on the command line as needed:
  ```bash
  pixi run -e rtx5090 python -m llm_perf_opt.runners.deep_profile_runner \
    model.path=/abs/models/deepseek-ocr \
    dataset.root=/abs/datasets/OmniDocBench \
    device=cuda:0 pipeline.nsys.enable=true pipeline.ncu.enable=false
  ```

NSYS with FlashAttention
- Ensure you run on GPU (e.g., `device=cuda:0`). FlashAttention may emit warnings during module registration but is supported.
  ```bash
  pixi run -e rtx5090 python -m llm_perf_opt.runners.deep_profile_runner \
    device=cuda:0 use_flash_attn=true pipeline.nsys.enable=true pipeline.ncu.enable=false \
    dataset.sampling.num_epochs=1 dataset.sampling.num_samples_per_epoch=1 \
    dataset.subset_filelist=datasets/omnidocbench/subsets/dev-3.txt
  ```

Where outputs go
- Hydra run dir defaults to `tmp/profile-output/<run_id>/`.
- Pipeline outputs:
  - `torch_profiler/`: `report.md`, `operators.md`, `metrics.json`, `llm_profile_runner.log`
  - `static_analysis/`: `static_compute.{json,md}` (when enabled)
- `nsys/`: `run.nsys-rep`, `run.sqlite`, `summary_*.csv`, `cmd.txt` (when enabled)
- `ncu/`: `raw.csv`, `.ncu-rep`, `sections_report.txt`, `cmd*.txt` (when enabled)
- Reproducibility at run root: `env.json`, `config.yaml`, `inputs.yaml`
- Optional predictions + viz: `torch_profiler/pred/predictions.jsonl`, `torch_profiler/viz/<hash>/{result_with_boxes.jpg,result.mmd,images/*,info.json}`

NCU Kernel Profiling Workflow
The project includes production scripts for detailed per-kernel profiling with Nsight Compute.

Typical workflow:
1. Run Stage 1 or Stage 2 with `pipeline.nsys.enable=true` to generate timeline
2. Extract top kernels from Nsys summary:
   ```bash
   python scripts/ncu/release/extract-top-kernels.py \
     tmp/profile-output/<run_id>/nsys/summary_cuda_gpu_kern_sum.csv \
     -o top-kernels.yaml --topk 30
   ```
3. Profile specific kernels (bash recommended for production):
   ```bash
   ./scripts/ncu/release/ncu-profile-kernels.sh \
     --kernel-config top-kernels.yaml \
     --topk 5 \
     --output-dir tmp/ncu-detailed \
     -- pixi run python -m llm_perf_opt.runners.llm_profile_runner device=cuda:0
   ```
   Or Python variant:
   ```bash
   python scripts/ncu/release/ncu-profile-kernels.py \
     --kernel-config top-kernels.yaml \
     --topk 5 \
     -- python -m llm_perf_opt.runners.llm_profile_runner device=cuda:0
   ```
   Or via vendor builder integration:
   ```bash
   ./scripts/ncu/release/ncu-profile-kernels-via-runner.sh \
     --kernel-config top-kernels.yaml \
     --topk 3 \
     -- python -m llm_perf_opt.runners.llm_profile_runner \
          pipeline.torch_profiler.enable=false pipeline.static_analysis.enable=false
   ```

See `scripts/ncu/release/README.md` for detailed usage, options, and examples.

## NVTX Range Replay (stub)

- Use dummy model configs via Hydra overrides for quick verification:
  - `model/dummy_shallow_resnet/arch@model=dummy_shallow_resnet.default`
  - `model/dummy_shallow_resnet/infer@infer=dummy_shallow_resnet.default`
- Generate NVTX ranges with `tests/manual/ncu/manual_nvtx_regions.py` and profile with Nsight Compute using `pipeline.ncu.ncu_cli.replay_mode=range`.
