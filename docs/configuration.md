# Configuration

Hydra is used for all configuration. Top-level defaults are in `conf/config.yaml`:

```
defaults:
  - dataset: omnidocbench
  - model/deepseek_ocr/arch@model: deepseek_ocr.default
  - model/deepseek_ocr/infer@infer: deepseek_ocr.default
  - _self_

experiment: stage1
repeats: 3
device: cuda:0
use_flash_attn: true

hydra:
  run:
    dir: ${hydra:runtime.cwd}/tmp/stage1/${now:%Y%m%d-%H%M%S}
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

PyTorch profiler presets (profiling/torch)
- Files: `torch-profiler.{min,default,max}.yaml`
- Common keys:
  - `enabled`: Master on/off for the representative profiling pass.
  - `activities`: List of profilers to enable, values from {`cpu`, `cuda`}.
  - `record_shapes`: Record operator input shapes (adds CPU overhead).
  - `profile_memory`: Collect memory stats/timeline (slower, larger traces).
  - `with_stack`: Capture Python stacks for events (very slow; huge traces).
  - `group_by_input_shape`: Aggregate key_averages by input shape (more CPU).
  - `rep_max_new_tokens`: Hard cap on decode tokens during the profiled run to bound trace size.

Switching presets
```
profiling/torch/torch-profiler@profiling=torch-profiler.min      # fastest
profiling/torch/torch-profiler@profiling=torch-profiler.default  # balanced (default)
profiling/torch/torch-profiler@profiling=torch-profiler.max      # most detailed
```

Notes
- We no longer use a flat `conf/model/deepseek_ocr.yaml`; it was replaced by the arch/infer groups.
- To swap fast inference, use `model/deepseek_ocr/infer@infer=deepseek_ocr.fast`.
