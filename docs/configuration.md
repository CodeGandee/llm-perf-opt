# Configuration

Hydra is used for all configuration. Top‑level defaults are in `conf/config.yaml` and compose all pipelines from a single entrypoint:

```
defaults:
  - dataset: omnidocbench
  - dataset/sampling@dataset.sampling: default
  - model/deepseek_ocr/arch@model: deepseek_ocr.default
  - model/deepseek_ocr/infer@infer: deepseek_ocr.default
  - profiling/torch@pipeline.torch_profiler: torch-profiler.default
  - output/torch@pipeline.torch_profiler.output: default
  - model/deepseek_ocr/output/torch@pipeline.torch_profiler.output.extra.deepseek_ocr: default
  - profiling/nsys@pipeline.nsys: nsys.default
  - profiling/ncu@pipeline.ncu: ncu.default
  - _self_

experiment: stage1
device: cuda:0
use_flash_attn: true

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
    # Nsight Compute args are provided by the preset under `ncu_cli.*`

hydra:
  run:
    dir: ${hydra:runtime.cwd}/tmp/profile-output/${now:%Y%m%d-%H%M%S}
  output_subdir: null
  job:
    chdir: true
```

Config groups
- Dataset: `conf/dataset/omnidocbench.yaml` (root, subset_filelist, fallback_patterns)
- Dataset sampling: `conf/dataset/sampling/{default,random}.yaml` (epochs, samples per epoch, randomize, seed)
- Model arch: `conf/model/deepseek_ocr/arch/deepseek_ocr.default.yaml`
  - keys: `path`, `dtype`, `preprocess.{enable,base_size,image_size,crop_mode,patch_size,downsample_ratio}`
- Model infer: `conf/model/deepseek_ocr/infer/deepseek_ocr.default.yaml`
  - keys: `temperature`, `max_new_tokens` (integer only), `no_repeat_ngram_size`, `do_sample`
- Pipeline presets:
  - PyTorch profiler: `conf/profiling/torch/torch-profiler.{min,default,max}.yaml`
  - Nsight Systems: `conf/profiling/nsys/nsys.default.yaml` (supports `capture_range`, `nvtx_capture`, `capture_range_end`)
  - Nsight Compute: `conf/profiling/ncu/*.yaml`
- Output (per‑pipeline):
  - General: `conf/output/torch/default.yaml` → `pipeline.torch_profiler.output.prediction|visualization`
  - Model‑specific: `conf/model/deepseek_ocr/output/torch/default.yaml` → `pipeline.torch_profiler.output.extra.deepseek_ocr`
  - Defaults: when `prediction.enable=true` and `prediction.save_dir` is omitted/null, outputs write to `pred/`. When `visualization.enable=true` and `visualization.save_dir` is omitted/null, outputs write to `viz/`. Paths are relative to the stage output dir unless absolute.

Model and dataset paths
- Stage 2 runner resolves `model.path` and `dataset.root` from Hydra config to absolute paths using the Hydra runtime cwd. There are no hard‑coded repo paths.
- Override at runtime as needed:
  ```bash
  pixi run -e rtx5090 python -m llm_perf_opt.runners.deep_profile_runner \
    model.path=/abs/models/deepseek-ocr \
    dataset.root=/abs/datasets/OmniDocBench \
    device=cuda:0 pipeline.nsys.enable=true pipeline.ncu.enable=false
  ```

PyTorch profiler preset keys
- `enabled`: Master on/off for the representative profiling pass.
- `activities`: List of profilers to enable, values from {`cpu`, `cuda`}.
- `record_shapes`, `profile_memory`, `with_stack`, `group_by_input_shape`

Inference: max_new_tokens policy
- `infer.max_new_tokens` must be an explicit integer. The special value `inf` is not accepted. If you want near‑unbounded decoding, pass a large value (e.g., 8192) and let EOS stop generation.

Notes
- Swap infer preset: `model/deepseek_ocr/infer@infer=deepseek_ocr.fast`.
- Static analyzer toggle: `pipeline.static_analysis.enable` (default true).
- Stage‑oriented configs under `conf/runner/` are deprecated/removed; use pipeline toggles in `conf/config.yaml` instead.

Nsight Compute (`ncu_cli`) fields
- Each preset under `conf/profiling/ncu/*.yaml` defines an `ncu_cli` map:
  - `target_processes`: string (e.g., `all`) → `--target-processes`.
  - `nvtx.include`: string (e.g., `decode*`) → `--nvtx --nvtx-include` when NVTX gating is enabled.
  - `set`: string (e.g., `roofline`) → `--set`.
  - `metrics`: list of metric names, or `null` to omit `--metrics`.
  - `sections`: list of sections (e.g., SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats).
  - `kernel_name_base`: kernel name format (e.g., `demangled`) → `--kernel-name-base`.
  - `export.csv`: boolean to indicate CSV-friendly exports for tooling.
  - `replay_mode`: `kernel|range|app-range|application` → `--replay-mode` (for per‑range aggregation or per‑kernel).
  - Defaults: `ncu.default` aligns with the scripts' defaults but hard-codes both sections and a concise metrics set:
    - sections: [SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats]
    - metrics: [flop_count_hp, flop_count_sp, gpu__time_duration.sum, sm__throughput.avg.pct_of_peak_sustained_elapsed, dram__throughput.avg.pct_of_peak_sustained_elapsed]
- Available presets:
  - `ncu.default` — Standard 4-section preset (SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats)
  - `ncu.high` — High-coverage preset with union of sections from device-specific presets (adds ComputeWorkloadAnalysis, WorkloadDistribution, WarpStateStats, LaunchStats, InstructionStats)
  - `ncu.rtx3090` — GA102-specific balanced preset
  - `ncu.rtx3090.compute` — Compute-focused preset for RTX 3090
  - `ncu.rtx3090.memory` — Memory-focused preset for RTX 3090

Nsight Systems (NVTX gating)
- `pipeline.nsys.capture_range` mirrors the CLI (`nvtx|cudaProfilerApi|hotkey|none`).
- If `capture_range=nvtx`, you must set `pipeline.nsys.nvtx_capture` (e.g., `prefill`, `decode`, or `name@*`). Omitting it results in no trigger and an empty report; the runner errors to prevent this.
- Optional: `pipeline.nsys.capture_range_end` supports values like `stop`, `repeat[:N]`, etc. If omitted/empty → not passed.

## NCU CLI Config Mapping (stub)

Quick reference for key Hydra → CLI mappings (see section above for details):
- `pipeline.ncu.ncu_cli.replay_mode` → `--replay-mode`
- `pipeline.ncu.ncu_cli.nvtx.include` → `--nvtx --nvtx-include <expr>`
- `pipeline.ncu.ncu_cli.kernel_name` → `--kernel-name <name|regex:...>`
- `pipeline.ncu.ncu_cli.kernel_name_base` → `--kernel-name-base <demangled|mangled>`
