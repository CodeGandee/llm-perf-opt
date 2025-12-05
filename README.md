# LLM Performance Optimization

A comprehensive framework for evaluating Large Language Model (LLM) runtime performance and investigating optimization strategies for specific hardware platforms.

## Overview

This project provides tools and methodologies for:
- Benchmarking LLM inference performance across different hardware configurations
- Identifying performance bottlenecks in LLM execution through multi-stage profiling
- Developing and validating hardware-specific optimization techniques
- Providing actionable insights for deploying LLMs efficiently

**Key Features:**
- **Multi-stage profiling**: PyTorch Profiler (Stage 1) for operator analysis, Nsight Systems/Compute (Stage 2) for kernel-level profiling
- **Hydra-based configuration**: Flexible, composable configs for models, datasets, hardware, and profiling workflows
- **Hardware-specific environments**: Separate Pixi environments for different GPU architectures (sm_90, sm_120)
- **Comprehensive reporting**: Automated generation of profiling reports, roofline analysis, and performance visualizations
- **Reproducibility**: All profiling runs capture configuration, environment, and input data for full reproducibility

## Quick Start

### Prerequisites

**System Requirements:**
- NVIDIA GPU (CUDA 12.0+)
- Linux (tested on Ubuntu 20.04, 22.04, 24.04)
- Python 3.11 or 3.12
- [Pixi](https://pixi.sh/) package manager

**Install Pixi** (if not already installed):
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/CodeGandee/llm-perf-opt.git
cd llm-perf-opt
```

2. Install dependencies with Pixi:
```bash
pixi install  # Installs default environment (CUDA 12.4, sm_90)
```

3. Bootstrap assets (models and datasets):
```bash
./bootstrap.sh --yes
```

This will:
- Create symlinks to models from `$HF_SNAPSHOTS_ROOT` (default: `~/.cache/huggingface/hub`)
- Create symlinks to datasets from `$DATASETS_ROOT` (default: `~/datasets`)
- Extract dataset archives if needed

You can also bootstrap individually:
- Models only: `models/bootstrap.sh --yes`
- Datasets only: `datasets/omnidocbench/bootstrap.sh --yes`

### Run Your First Profile

**Stage 1 Profiling** (PyTorch Profiler - operator-level analysis):
```bash
pixi run stage1-run
```

This runs a quick profile with defaults:
- 3 samples from the dev-20 subset
- 64 max new tokens
- Outputs: `tmp/profile-output/<timestamp>/`

**View the results:**
- `report.md`: Comprehensive profiling report with prefill/decode timings
- `operators.md`: Operator-level summaries
- `metrics.json`: Machine-readable metrics
- `stakeholder_summary.md`: Executive summary

## Environment Management

**Pixi** is the primary package manager. Pixi environments are defined in `pyproject.toml`.

### Available Environments

1. **default** (default) - PyTorch 2.5.1 + CUDA 12.4
   - Supports up to sm_90 (Ada/Hopper architectures)
   - Includes Flash Attention 2.7.4.post1
   - Use for RTX 3090, RTX 4090, A100, H100

2. **rtx5090** - PyTorch nightly + CUDA 12.8
   - Supports sm_120 (Blackwell architecture)
   - Builds Flash Attention and Triton from source
   - Required for RTX 5090 and newer Blackwell GPUs

### RTX 5090 Setup

For RTX 5090 (Blackwell sm_120) development:

```bash
# Install the rtx5090 environment
pixi install -e rtx5090

# Run the full setup (installs PyTorch nightly, builds Triton and Flash Attention)
pixi run -e rtx5090 setup-rtx5090

# Verify the setup
pixi run -e rtx5090 verify-rtx5090
```

**Note:** RTX 5090 setup requires:
- CUDA Toolkit 12.8+ installed system-wide (see [System Prerequisites](#system-prerequisites))
- Significant build time for Flash Attention (~10-30 minutes depending on CPU)

### Switching Environments

```bash
# Use default environment
pixi run stage1-run

# Use rtx5090 environment
pixi run -e rtx5090 stage1-run
```

## Usage

### Stage 1 Profiling (PyTorch Profiler)

Collects prefill/decode timings, operator summaries, Model FLOPs Utilization (MFU), and writes artifacts under `tmp/profile-output/<run_id>/torch_profiler/` and `tmp/profile-output/<run_id>/static_analysis/`.

**Quick run with defaults:**
```bash
pixi run stage1-run  # 3 samples, 3 repeats per sample
```

**Skip static analysis for faster runs:**
```bash
pixi run stage1-run-no-static
```

**Custom run with Hydra overrides:**
```bash
pixi run python -m llm_perf_opt.runners.llm_profile_runner \
  'hydra.run.dir=tmp/profile-output/${now:%Y%m%d-%H%M%S}' \
  device=cuda:0 infer.max_new_tokens=64 \
  dataset.sampling.num_samples_per_epoch=10 \
  'pipeline.torch_profiler.activities=[cpu,cuda]' \
  pipeline.nsys.enable=false pipeline.ncu.enable=false
```

**Artifacts generated:**
- `report.md` - Comprehensive profiling report
- `operators.md` - Top operators by time
- `metrics.json` - Machine-readable metrics
- `stakeholder_summary.md` - Executive summary
- `env.json`, `inputs.yaml`, `assumptions.md` - Reproducibility data
- `static_compute.json`, `static_compute.md` - Static model analysis (if enabled)

### Stage 2 Profiling (Nsight Systems/Nsight Compute)

Deep kernel profiling with Nsight Systems (timeline) and Nsight Compute (per-kernel metrics).

**Run Stage 2 profiling:**
```bash
pixi run stage2-profile
```

**Important:** Stage 2 automatically disables `torch_profiler` and `static_analysis` for the workload to avoid overhead. Artifacts are written under `tmp/profile-output/<run_id>/nsys/` and `tmp/profile-output/<run_id>/ncu/`.

**Nsight Systems output:**
- `*.nsys-rep` - Binary report (open with Nsight Systems GUI)
- `summary_cuda_gpu_kern_sum.csv` - Kernel summary CSV
- `cmd.txt` - Command used for profiling

**Nsight Compute output:**
- Per-kernel CSV files with metrics (occupancy, throughput, roofline, etc.)
- `command.yaml` - NCU profiling configuration

### NCU Kernel Profiling Workflow

For detailed per-kernel profiling:

1. **Run Stage 1 or Stage 2** with `pipeline.nsys.enable=true` to generate Nsight Systems report

2. **Extract top kernels** from the Nsys CSV summary:
```bash
pixi run python scripts/ncu/release/extract-top-kernels.py \
  tmp/profile-output/<run_id>/nsys/summary_cuda_gpu_kern_sum.csv \
  -o top-kernels.yaml --topk 30
```

3. **Profile specific kernels** using the generated YAML:
```bash
# See scripts/ncu/release/README.md for detailed instructions
pixi run python scripts/ncu/release/profile-from-yaml.py \
  --config top-kernels.yaml \
  --output-dir tmp/ncu-kernels/
```

4. **Analyze kernel metrics**:
```bash
pixi run python scripts/ncu/analysis/analyze_ncu_dir.py \
  tmp/ncu-kernels/ \
  --output-dir tmp/ncu-analysis/
```

This generates:
- Roofline plots (normalized and physical)
- Metric histograms (occupancy, throughput, bandwidth, etc.)
- Classification summaries (memory-bound, compute-bound, balanced)
- Per-kernel CSV exports

### Direct Inference

Run inference without profiling, outputting predictions and visualizations:

```bash
pixi run direct-infer-dev20
```

**Custom direct inference:**
```bash
pixi run python -m llm_perf_opt.runners.direct_inference_runner \
  dataset.subset_filelist=datasets/omnidocbench/subsets/dev-20.txt \
  device=cuda:0 \
  infer.max_new_tokens=8192 \
  pipeline.direct_inference.enable=true \
  pipeline.direct_inference.output.prediction.enable=true \
  pipeline.direct_inference.output.visualization.enable=true
```

Outputs:
- `predictions/` - JSON predictions for each sample
- `visualizations/` - Images with OCR bounding boxes overlaid

### Testing

```bash
# Unit tests
pixi run pytest tests/unit/

# Integration tests
pixi run pytest tests/integration/

# Manual tests
pixi run python tests/manual/<test>.py
```

### Code Quality

```bash
# Lint with Ruff
pixi run ruff check src/

# Format with Ruff
pixi run ruff format src/

# Type check with mypy
pixi run mypy src/
```

### Documentation

```bash
# Serve docs locally at 127.0.0.1:8000
pixi run docs-serve

# Build docs
pixi run docs-build
```

## Architecture

### Hydra Configuration System

Configuration is managed via Hydra with a hierarchical structure under `conf/`:

```
conf/
├── config.yaml                 # Top-level defaults that compose config groups
├── model/                      # Model identity, paths, dtypes (e.g., deepseek_ocr/)
├── dataset/                    # Dataset roots, variants, sampling presets
├── runtime/                    # Runtime parameters (PyTorch, vLLM, TensorRT-LLM)
├── hardware/                   # Device selection and hardware-specific options
├── profiling/                  # Profiler presets (nsys, ncu, torch)
└── output/                     # Output artifact configurations per pipeline stage
```

**Hydra overrides** are used to swap configs and control pipeline stages:
```bash
# Example: Change model, enable full profiling with Nsight Systems
pixi run python -m llm_perf_opt.runners.llm_profile_runner \
  model=qwen2_5_7b \
  profiling=full \
  pipeline.nsys.enable=true
```

**Pipeline structure:**
The config uses a `pipeline.*` structure to control different profiling stages:
- `pipeline.torch_profiler.*` - PyTorch profiler settings (Stage 1)
- `pipeline.static_analysis.*` - Static model analysis
- `pipeline.nsys.*` - Nsight Systems profiler (Stage 2)
- `pipeline.ncu.*` - Nsight Compute profiler (Stage 2)
- `pipeline.direct_inference.*` - Direct inference without profiling

### Core Package Structure

The codebase uses a unified package structure under `src/llm_perf_opt/`:

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

- **`dnn_models/`** - Model architectures for testing
  - `shallow_resnet.py` - Simple ResNet for profiling validation

### Multi-Stage Profiling Architecture

The project uses a **unified pipeline architecture** where different profiling stages can be enabled/disabled via Hydra config:

1. **Stage 1** (`llm_profile_runner.py`): PyTorch Profiler for operator-level analysis and MFU estimation
   - Lightweight, quick turnaround
   - Produces operator summaries and timing statistics
   - Optional static model analysis for refined FLOPs counting

2. **Stage 2** (`deep_profile_runner.py`): Nsight Systems/Compute for kernel-level analysis and GPU timeline
   - Automatically disables `torch_profiler` and `static_analysis` for the workload subprocess to avoid measurement overhead
   - Provides detailed kernel metrics, roofline analysis, and timeline visualization

The runners share a common workload implementation but wrap it with different profiling harnesses. All runners use the same Hydra config (`conf/config.yaml`) with pipeline-specific overrides.

### NVTX Range Gating

When `gating_nvtx=true`, Nsight profilers use NVTX ranges to selectively capture specific workload phases (e.g., "prefill", "decode"). This reduces overhead and file sizes. The workload code emits NVTX ranges using `nvtx_range()` context managers from `src/llm_perf_opt/profiling/harness.py`.

## Project Structure

```
llm-perf-opt/
├── conf/                 # Hydra config groups (model, dataset, runtime, hardware, profiling)
│   ├── model/           # Model configs (deepseek_ocr, qwen2_5_7b, etc.)
│   ├── dataset/         # Dataset configs (omnidocbench, etc.)
│   ├── runtime/         # Runtime configs (PyTorch, vLLM, TensorRT-LLM)
│   ├── hardware/        # Hardware configs (CUDA device selection)
│   ├── profiling/       # Profiler presets (torch, nsys, ncu)
│   └── output/          # Output artifact configurations
├── models/              # Model weights/tokenizers (symlinks from $HF_SNAPSHOTS_ROOT)
├── datasets/            # Datasets (symlinks from $DATASETS_ROOT)
│   └── omnidocbench/   # OmniDocBench dataset with bootstrap scripts
├── src/
│   └── llm_perf_opt/   # Unified project package
│       ├── profiling/  # Profiling harnesses, parsers, MFU computation
│       ├── runners/    # Execution orchestrators (Stage 1, Stage 2, direct inference)
│       ├── data/       # Dataset utilities
│       ├── contracts/  # Shared data models (Pydantic)
│       ├── visualize/  # Visualization utilities
│       └── dnn_models/ # Model architectures for testing
├── scripts/            # Utility scripts
│   ├── ncu/           # NCU profiling and analysis scripts
│   │   ├── release/   # Top-kernel extraction and profiling
│   │   └── analysis/  # Roofline plots, histograms, classification
│   └── install-*.sh   # Installation scripts (CUDA toolkit, vLLM, etc.)
├── reports/            # Profiling reports and artifacts
│   └── 20251107-dsocr/ # DeepSeek-OCR profiling report (example)
│       ├── final-report-v2.md         # Technical report
│       ├── final-report-v2-chinese.md # Chinese translation
│       ├── ncu/                       # NCU raw data (per-kernel CSVs)
│       ├── ncu-v2/                    # NCU analysis (roofline, histograms)
│       └── nsys/                      # Nsys reports (per-stage, all-stage)
├── tests/              # Tests (manual/unit/integration)
├── docs/               # Documentation (MkDocs)
├── context/            # Knowledge base and development hints
├── extern/             # Read-only upstream references (submodules, snapshots)
├── .specify/           # Speckit constitution and templates
└── pyproject.toml      # Pixi environments and project metadata
```

## System Prerequisites

**NVIDIA GPU profiling requires system tools:**

### 1. NVIDIA Driver

Check with `nvidia-smi` for CUDA compatibility:
```bash
nvidia-smi
```

Note the "CUDA Version" shown - this is the maximum CUDA version your driver supports.

### 2. Nsight Systems (`nsys`)

Install via NVIDIA APT repository (Ubuntu):

```bash
# Replace ubuntu2204 with your release (e.g., ubuntu2004, ubuntu2404)
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install Nsight Systems CLI + target components
sudo apt-get install -y nsight-systems nsight-systems-target

# Verify
nsys --version
```

**Alternative:** Download from https://developer.nvidia.com/nsight-systems

### 3. CUDA Toolkit (`nvcc`)

Required to build CUDA-based dependencies (e.g., flash-attn). Use the official NVIDIA repository for system-wide installation.

**Automated installation:**
```bash
./scripts/install-cuda-toolkit-12-8.sh
```

**Manual installation:**
```bash
# 1. Download and install CUDA keyring (Ubuntu 24.04 example)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 2. Update package list
sudo apt-get update

# 3. Install CUDA Toolkit 12.8 (toolkit only, no driver update)
sudo apt-get install -y cuda-toolkit-12-8

# 4. Add to PATH and environment
echo '' >> ~/.bashrc
echo '# CUDA 12.8 Toolkit' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 5. Verify installation
nvcc --version
```

**Important Notes:**
- Replace `ubuntu2404` with your Ubuntu version: `ubuntu2004`, `ubuntu2204`, or `ubuntu2404`
- Pick a CUDA version that matches your driver (`nvidia-smi` shows max supported "CUDA Version")
- System-wide installation ensures compilers can find CUDA headers at `/usr/local/cuda-12.8/include`
- For other distributions (RHEL, Fedora, SLES), see [official installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

### 4. Nsight Compute (`ncu`)

Per-kernel profiler for detailed metrics.

**Preferred (newer versions):** Pixi Global
```bash
pixi global install nsight-compute --channel nvidia --channel conda-forge
~/.pixi/bin/ncu --version   # or add ~/.pixi/bin to PATH
```

**Alternative (root):** APT via NVIDIA repository
```bash
sudo apt-get install -y nsight-compute
```

### 5. NVTX (Optional)

For readable CUDA timeline ranges:
```bash
uv pip install -U nvtx || pip install -U nvtx
```

### Example Nsight Systems Capture

```bash
nsys profile --trace=cuda,nvtx,osrt -o tmp/nsys/deepseek \
  pixi run python tests/manual/deepseek_ocr_hf_manual.py
```

## Configuration

### Dataset Sampling

Dataset sampling is controlled via `dataset.sampling.*` overrides:

```bash
pixi run python -m llm_perf_opt.runners.llm_profile_runner \
  dataset.sampling.num_epochs=1 \
  dataset.sampling.num_samples_per_epoch=10 \
  dataset.sampling.randomize=true
```

**Parameters:**
- `dataset.sampling.num_epochs` - Number of passes through the dataset
- `dataset.sampling.num_samples_per_epoch` - Samples per epoch (`null` = all)
- `dataset.sampling.randomize` - Whether to shuffle samples
- `dataset.subset_filelist` - Path to file list for subset selection (e.g., `datasets/omnidocbench/subsets/dev-20.txt`)

### Hydra Configuration Changes

When modifying configs:
1. Check `conf/config.yaml` for defaults list composition
2. New config groups go under appropriate subdirectory (e.g., `conf/model/new_model/`)
3. Use `@` notation for mounting configs to nested keys (e.g., `profiling/torch@pipeline.torch_profiler`)
4. Test overrides with `--cfg job` to print resolved config:
   ```bash
   pixi run python -m llm_perf_opt.runners.llm_profile_runner --cfg job
   ```

### Adding New Models

Models are symlinked from external storage (`$HF_SNAPSHOTS_ROOT`). To add a new model:

1. Update `models/bootstrap.yaml` with the model path
2. Run `models/bootstrap.sh --yes` to create symlink
3. Add model config group under `conf/model/<model_name>/`
4. Update defaults list in `conf/config.yaml` if making it default

## Reports and Analysis

Recent profiling reports and artifacts are stored in `reports/`:

### DeepSeek-OCR Profiling Report (Example)

Located in `reports/20251107-dsocr/`:
- **`final-report-v2.md`** - Comprehensive technical report with roofline analysis, kernel classification, and optimization recommendations
- **`final-report-v2-chinese.md`** - Chinese translation
- **`ncu-v2/analysis/`** - Roofline plots, histograms, and per-kernel metrics
- **`nsys/`** - Nsight Systems reports for each pipeline stage
- **`kernel-info.yaml`** - Kernel metadata for top 20 kernels

This report demonstrates the full profiling workflow from Stage 1 through NCU kernel analysis, providing insights into memory-bound vs compute-bound kernels, occupancy, throughput, and optimization opportunities.

## Development Patterns

### Working with Models

Models are symlinked from external storage to avoid duplicating large files:

```bash
# Set custom model root (default: ~/.cache/huggingface/hub)
export HF_SNAPSHOTS_ROOT=/path/to/models

# Bootstrap models
models/bootstrap.sh --yes
```

### Adding New Profiling Stages

To add a new profiling stage:

1. Create config preset under `conf/profiling/` or `conf/output/`
2. Mount preset into `pipeline.<stage_name>` in `conf/config.yaml`
3. Add enable/disable toggle to `pipeline.<stage_name>.enable`
4. Implement harness wrapper in `src/llm_perf_opt/profiling/`
5. Create or extend runner in `src/llm_perf_opt/runners/`
6. Add Pixi task to `pyproject.toml` for easy invocation

## Contributing

Contributions are welcome! This project is actively developed and we're building out the foundation.

**Areas for contribution:**
- Additional model support (encoder-decoder, multimodal)
- Hardware platform support (AMD GPUs, Intel GPUs, ARM processors)
- Optimization techniques (quantization, kernel fusion, memory optimization)
- Benchmark datasets and scenarios
- Documentation and tutorials

## License

*To be determined*

## Roadmap

- [x] Define benchmark methodology and metrics
- [x] Implement core benchmarking framework
- [x] Multi-stage profiling architecture (PyTorch Profiler, Nsight Systems/Compute)
- [x] Roofline analysis and kernel classification
- [x] Comprehensive reporting and visualization
- [ ] Add support for major hardware platforms (AMD, Intel, ARM)
- [ ] Develop optimization toolkit (quantization, kernel fusion)
- [ ] Expand model coverage (encoder-decoder, multimodal)
- [ ] Build interactive performance dashboards
- [ ] Publish benchmark results and optimization case studies

---

**Note**: This project focuses on NVIDIA GPU profiling and optimization. Support for additional hardware platforms is planned for future releases.

For detailed development guidelines and advanced usage, see [`CLAUDE.md`](CLAUDE.md).
