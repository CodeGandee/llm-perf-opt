Hydra Configuration Tree

This directory holds Hydra/OmegaConf configuration groups used to compose runs.

- `config.yaml`: top-level defaults list that selects config group options.
- `hydra/`: settings for run directories and job behavior.
- `model/`: model identity and locations (e.g., symlink or submodule paths, dtypes).
- `dataset/`: dataset roots and optional variant names.
- `runtime/`: runtime parameters (e.g., PyTorch, vLLM, TensorRT-LLM).
- `hardware/`: device selection and hardware-specific options.
- `profiling/`: profiler toggles (nsys, ncu, nvml) and options.

Use `hydra` CLI overrides to swap options, e.g., `model=qwen2_5_7b profiling=full`.
