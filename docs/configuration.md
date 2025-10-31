# Configuration

Hydra is used for all configuration. Top-level defaults are in `conf/config.yaml` and now compose all pipelines from a single entrypoint:

```
defaults:
  - dataset: omnidocbench
  - model/deepseek_ocr/arch@model: deepseek_ocr.default
  - model/deepseek_ocr/infer@infer: deepseek_ocr.default
  - profiling/torch@pipeline.torch_profiler: torch-profiler.default
  - profiling/nsys@pipeline.nsys: nsys.default
  - profiling/ncu@pipeline.ncu: ncu.default
  - _self_

repeats: 3
device: cuda:0
use_flash_attn: true

run:
  mode: deep
  stage1_repeats: 1
  dataset_subset_filelist: null
  top_n_kernels: 30

pipeline:
  static_analysis:
    enable: true
    write_reports: true
    use_analytic_fallback: true
    use_synthetic_inputs: true
  torch_profiler:
    enable: ${pipeline.torch_profiler.enabled}
  nsys:
    enable: false
    gating_nvtx: true
  ncu:
    enable: false
    gating_nvtx: true

hydra:
  run:
    dir: ${hydra:runtime.cwd}/tmp/profile-output/${now:%Y%m%d-%H%M%S}
  output_subdir: null
  job:
    chdir: true

outputs:
  save_predictions: false
  predictions:
    strip_special_tokens: false
  visualization:
    enable: true
    max_images: 16
    thumbnail_width: 480
```

Config groups
- Dataset: `conf/dataset/omnidocbench.yaml` (root, subset_filelist, fallback_patterns)
- Model arch: `conf/model/deepseek_ocr/arch/deepseek_ocr.default.yaml`
  - keys: `path`, `dtype`, `preprocess.{enable,base_size,image_size,crop_mode,patch_size,downsample_ratio}`
- Model infer: `conf/model/deepseek_ocr/infer/deepseek_ocr.default.yaml`
  - keys: `temperature`, `max_new_tokens`, `no_repeat_ngram_size`, `do_sample`
- Pipeline presets:
  - PyTorch profiler: `conf/profiling/torch/torch-profiler.{min,default,max}.yaml`
  - Nsight Systems: `conf/profiling/nsys/nsys.default.yaml` (supports `capture_range`, `nvtx_capture`, `capture_range_end`)
  - Nsight Compute: `conf/profiling/ncu/*.yaml`

PyTorch profiler preset keys
- `enabled`: Master on/off for the representative profiling pass.
- `activities`: List of profilers to enable, values from {`cpu`, `cuda`}.
- `record_shapes`, `profile_memory`, `with_stack`, `group_by_input_shape`
- `rep_max_new_tokens`: Cap profiled decode length to bound trace size.

Notes
- Use `model/deepseek_ocr/infer@infer=deepseek_ocr.fast` to swap fast infer.
- Static analyzer toggle: `pipeline.static_analysis.enable` (default true).
- Stage‑oriented configs under `conf/runner/` have been removed. Use pipeline toggles in `conf/config.yaml` instead.

Nsight Systems (nvtx gating)
- `pipeline.nsys.capture_range` mirrors the CLI (`nvtx|cudaProfilerApi|hotkey|none`).
- If `capture_range=nvtx`, you must set `pipeline.nsys.nvtx_capture` (e.g., `prefill`, `decode`, or `name@*`). Omitting it results in no trigger and an empty report; the runner will error.
- Optional: `pipeline.nsys.capture_range_end` supports values like `stop`, `repeat[:N]`, etc. If omitted/empty → not passed.
