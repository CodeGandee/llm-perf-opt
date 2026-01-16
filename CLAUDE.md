# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A comprehensive framework for evaluating Large Language Model (LLM) runtime performance and investigating optimization strategies for specific hardware platforms. The project focuses on benchmarking LLM inference performance, identifying bottlenecks, and developing hardware-specific optimization techniques.

## Environment Management

**Pixi** is the primary package manager. Pixi environments are defined in `pyproject.toml`.

### Available Environments

1. **default** - PyTorch 2.6.0 + CUDA 12.6 (up to sm_90)
2. **rtx5090** - PyTorch nightly + CUDA 12.8 (sm_120 support for Blackwell architecture)

### RTX 5090 Setup

For RTX 5090 (Blackwell sm_120) development:
```bash
pixi install -e rtx5090
pixi run -e rtx5090 setup-rtx5090  # Installs PyTorch nightly, builds Triton and Flash Attention
pixi run -e rtx5090 verify-rtx5090  # Verifies the setup
```

## Common Commands

### Bootstrap Assets
```bash
./bootstrap.sh --yes  # Bootstrap both models and datasets
models/bootstrap.sh --yes  # Models only
datasets/omnidocbench/bootstrap.sh --yes  # Datasets only
```

Models are symlinked from `$HF_SNAPSHOTS_ROOT`, datasets from `$DATASETS_ROOT`.

### Stage 1 Profiling (PyTorch Profiler)

Collects prefill/decode timings, operator summaries, MFU, and writes artifacts under `tmp/profile-output/<run_id>/torch_profiler/` and `tmp/profile-output/<run_id>/static_analysis/`.

```bash
# Quick run with defaults (20 samples, 3 repeats per sample)
pixi run stage1-run

# Skip static analysis for faster runs
pixi run stage1-run-no-static

# Custom run with Hydra overrides
pixi run python -m llm_perf_opt.runners.llm_profile_runner \
  'hydra.run.dir=tmp/profile-output/${now:%Y%m%d-%H%M%S}' \
  device=cuda:0 infer.max_new_tokens=64 \
  'pipeline.torch_profiler.activities=[cpu,cuda]' \
  pipeline.nsys.enable=false pipeline.ncu.enable=false
```

Artifacts include `report.md`, `operators.md`, `metrics.json`, `stakeholder_summary.md`, and reproducibility files (`env.json`, `inputs.yaml`, `assumptions.md`).

### Stage 2 Profiling (Nsight Systems/Nsight Compute)

Deep kernel profiling with Nsight Systems (timeline) and Nsight Compute (per-kernel metrics):

```bash
pixi run stage2-profile
```

**Important**: Stage 2 automatically disables torch_profiler and static_analysis for the workload to avoid overhead. Artifacts are written under `tmp/profile-output/<run_id>/nsys/` and `tmp/profile-output/<run_id>/ncu/`.

### NCU Kernel Profiling Workflow

For detailed per-kernel profiling:

1. Run Stage 1 or Stage 2 with `pipeline.nsys.enable=true` to generate Nsight Systems report
2. Extract top kernels from the Nsys CSV summary:
```bash
python scripts/ncu/release/extract-top-kernels.py \
  tmp/profile-output/<run_id>/nsys/summary_cuda_gpu_kern_sum.csv \
  -o top-kernels.yaml --topk 30
```
3. Profile specific kernels using the generated YAML (manual step; see `scripts/ncu/release/README.md`)

### Direct Inference

Run inference without profiling, outputting predictions and visualizations:

```bash
pixi run direct-infer-dev20
```

### Testing
```bash
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
python tests/manual/<test>.py  # Manual tests
```

### Code Quality
```bash
ruff check src/             # Lint with Ruff
ruff format src/            # Format with Ruff
mypy src/                   # Type check with mypy
```

### Documentation
```bash
pixi run docs-serve         # Serve docs at 127.0.0.1:8000
pixi run docs-build         # Build docs
```

## Architecture

### Hydra Configuration System

Configuration is managed via Hydra with a hierarchical structure under `conf/`:

- `conf/config.yaml` - Top-level defaults that compose config groups
- `conf/model/` - Model identity, paths, dtypes (e.g., `deepseek_ocr/`)
- `conf/dataset/` - Dataset roots, variants, sampling presets
- `conf/runtime/` - Runtime parameters (PyTorch, vLLM, TensorRT-LLM)
- `conf/hardware/` - Device selection and hardware-specific options
- `conf/profiling/` - Profiler presets (nsys, ncu, torch)
- `conf/output/` - Output artifact configurations per pipeline stage

**Hydra overrides** are used to swap configs and control pipeline stages. Example: `model=qwen2_5_7b profiling=full pipeline.nsys.enable=true`.

**Important**: The config uses a `pipeline.*` structure to control different profiling stages:
- `pipeline.torch_profiler.*` - PyTorch profiler settings (Stage 1)
- `pipeline.static_analysis.*` - Static model analysis
- `pipeline.nsys.*` - Nsight Systems profiler
- `pipeline.ncu.*` - Nsight Compute profiler
- `pipeline.direct_inference.*` - Direct inference without profiling

### Core Package Structure (`src/llm_perf_opt/`)

The codebase uses a unified package structure:

- **`runners/`** - Execution orchestrators that compose profiling harnesses with workloads
  - `llm_profile_runner.py` - Stage 1 entry point (Hydra main)
  - `deep_profile_runner.py` - Stage 2 entry point for Nsight profiling
  - `direct_inference_runner.py` - Inference without profiling
  - `dsocr_session.py` - DeepSeek-OCR model session wrapper
  - `inference_engine.py` - Generic inference loop with NVTX annotations

- **`profiling/`** - Profiling harnesses, parsers, and analysis
  - `harness.py` - NVTX range context managers
  - `mfu.py` - Model FLOPs Utilization (MFU) computation
  - `hw.py` - Hardware detection (GPU names, peak TFLOPS, NVML)
  - `aggregate.py` - Timing aggregation (mean/std across repeats)
  - `export.py` - Report generation (markdown, JSON artifacts)
  - `vendor/` - Nsight Systems/Compute wrappers and subprocess launchers
  - `parsers/` - CSV/JSON parsers for profiler outputs

- **`data/`** - Dataset utilities and preprocessors
  - `models.py` - Dataset schema definitions

- **`contracts/`** - Shared data models and type conversions
  - `models.py` - Pydantic models for internal APIs
  - `convert.py` - Converters between formats

- **`visualize/`** - Visualization utilities
  - `annotations.py` - Render vendor-style annotations on images

### Multi-Stage Profiling Architecture

The project uses a **unified pipeline architecture** where different profiling stages can be enabled/disabled via Hydra config:

1. **Stage 1** (`llm_profile_runner.py`): PyTorch Profiler for operator-level analysis and MFU estimation. Lightweight, quick turnaround. Produces operator summaries and timing statistics.

2. **Stage 2** (`deep_profile_runner.py`): Nsight Systems/Compute for kernel-level analysis and GPU timeline. Automatically disables torch_profiler and static_analysis for the workload subprocess to avoid measurement overhead.

The runners share a common workload implementation but wrap it with different profiling harnesses. All runners use the same Hydra config (`conf/config.yaml`) with pipeline-specific overrides.

### Profiling Workflow Pattern

Runners follow this general pattern:

1. Load Hydra config and resolve paths (models, datasets)
2. Initialize profiling harness (PyTorch profiler, Nsight Systems, or NCU)
3. Create model session wrapper (e.g., `DeepSeekOCRSession`)
4. Run inference loop with NVTX annotations (`nvtx_range()`)
5. Aggregate results across repeats
6. Write artifacts (reports, CSVs, JSONs) to unified output directory

**Key abstraction**: `inference_engine.run_stage_dataset()` provides a reusable inference loop that accepts a session object and emits timing metrics.

### NVTX Range Gating

When `gating_nvtx=true`, Nsight profilers use NVTX ranges to selectively capture specific workload phases (e.g., "prefill", "decode"). This reduces overhead and file sizes. The workload code must emit NVTX ranges using `nvtx_range()` context managers.

## System Prerequisites

**NVIDIA GPU profiling requires system tools:**

1. **NVIDIA Driver** - Check with `nvidia-smi` for CUDA compatibility
2. **Nsight Systems (`nsys`)** - Install via NVIDIA APT repo or Developer site
3. **CUDA Toolkit (`nvcc`)** - Required to build CUDA dependencies (e.g., flash-attn)
   - Automated script: `./scripts/install-cuda-toolkit-12-8.sh`
   - Manual: See README.md for APT installation steps
4. **Nsight Compute (`ncu`)** - Kernel-level profiler
   - Preferred: `pixi global install nsight-compute --channel nvidia --channel conda-forge`
   - Alternative: `sudo apt-get install -y nsight-compute`
5. **NVTX (optional)** - For readable CUDA timeline ranges: `uv pip install -U nvtx`

See README.md "System Prerequisites" section for detailed installation commands.

## Development Patterns

### Hydra Configuration Changes

When modifying configs:
1. Check `conf/config.yaml` for defaults list composition
2. New config groups go under appropriate subdirectory (e.g., `conf/model/new_model/`)
3. Use `@` notation for mounting configs to nested keys (e.g., `profiling/torch@pipeline.torch_profiler`)
4. Test overrides with `--cfg job` to print resolved config

### Adding New Profiling Stages

To add a new profiling stage:
1. Create config preset under `conf/profiling/` or `conf/output/`
2. Mount preset into `pipeline.<stage_name>` in `conf/config.yaml`
3. Add enable/disable toggle to `pipeline.<stage_name>.enable`
4. Implement harness wrapper in `src/llm_perf_opt/profiling/`
5. Create or extend runner in `src/llm_perf_opt/runners/`
6. Add Pixi task to `pyproject.toml` for easy invocation

### Working with Models

Models are symlinked from external storage (`$HF_SNAPSHOTS_ROOT`). To add a new model:
1. Update `models/bootstrap.yaml` with the model path
2. Run `models/bootstrap.sh --yes` to create symlink
3. Add model config group under `conf/model/<model_name>/`
4. Update defaults list in `conf/config.yaml` if making it default

### Dataset Sampling

Dataset sampling is controlled via `dataset.sampling.*` overrides:
- `dataset.sampling.num_epochs` - Number of passes through the dataset
- `dataset.sampling.num_samples_per_epoch` - Samples per epoch (null = all)
- `dataset.sampling.randomize` - Whether to shuffle samples
- `dataset.subset_filelist` - Path to file list for subset selection (e.g., `datasets/omnidocbench/subsets/dev-20.txt`)

## Speckit Integration

This project uses **Speckit** for structured feature development. Templates are under `.specify/`:

- `.specify/templates/` - Templates for specs, plans, tasks, checklists
- `.specify/memory/constitution.md` - Project principles and conventions

Available slash commands (defined in `.claude/`):
- `/speckit.specify` - Create/update feature specifications
- `/speckit.plan` - Generate implementation plan
- `/speckit.tasks` - Generate actionable tasks
- `/speckit.implement` - Execute implementation
- `/speckit.analyze` - Cross-artifact consistency analysis
- `/speckit.clarify` - Identify underspecified areas

Use these commands when developing new features or making architectural changes.

## Important Notes

- **Pixi over pip/conda**: Always use `pixi run` or activate Pixi environment for development
- **CUDA architecture targeting**: Default env supports up to sm_90 (Ada/Hopper). Use rtx5090 env for sm_120 (Blackwell)
- **Flash Attention builds**: Requires `no-build-isolation` and `TORCH_CUDA_ARCH_LIST` set correctly
- **Hydra chdir**: By default, Hydra changes to run directory. Use `hydra:runtime.cwd` for repo-relative paths
- **Output directories**: All profiling stages write to unified `tmp/profile-output/<run_id>/` with stage-specific subdirectories
- **Stage interop**: When profiling Stage 1 as a workload in Stage 2, disable extra profiling stages to avoid overhead
