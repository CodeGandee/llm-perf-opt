# LLM Performance Optimization

A comprehensive framework for evaluating Large Language Model (LLM) runtime performance and investigating optimization strategies for specific hardware platforms.

## Overview

This project aims to provide tools and methodologies for:
- Benchmarking LLM inference performance across different hardware configurations
- Identifying performance bottlenecks in LLM execution
- Developing and validating hardware-specific optimization techniques
- Providing actionable insights for deploying LLMs efficiently

## Objectives

1. **Performance Evaluation**: Establish comprehensive benchmarking suite for LLM runtime metrics
2. **Hardware Analysis**: Characterize performance across diverse hardware platforms (GPUs, CPUs, NPUs, custom accelerators)
3. **Optimization Research**: Investigate and validate optimization techniques including:
   - Model quantization (INT8, INT4, FP16, etc.)
   - Kernel optimization
   - Memory management strategies
   - Batching and scheduling algorithms
   - Hardware-specific optimizations
4. **Best Practices**: Document hardware-specific optimization guidelines and deployment recommendations

## Scope

### Hardware Platforms
- NVIDIA GPUs (various architectures)
- AMD GPUs
- Intel CPUs and GPUs
- ARM processors
- Custom AI accelerators (TPUs, AWS Inferentia, etc.)
- Mobile and edge devices

### LLM Categories
- Decoder-only models (GPT-style)
- Encoder-decoder models (T5-style)
- Various model sizes (from small edge models to large server models)

### Performance Metrics
- Throughput (tokens/second)
- Latency (time to first token, inter-token latency)
- Memory utilization (GPU VRAM, system RAM)
- Power consumption
- Cost efficiency (performance per dollar)

## Planned Components

### 1. Benchmarking Framework
- Standardized benchmark suite
- Performance profiling tools
- Metric collection and reporting
- Comparative analysis utilities

### 2. Optimization Toolkit
- Quantization tools and evaluation
- Kernel optimization utilities
- Memory profiling and optimization
- Hardware-specific optimization templates

### 3. Analysis and Visualization
- Performance dashboards
- Bottleneck identification tools
- Hardware utilization analysis
- Optimization impact visualization

### 4. Documentation and Guidelines
- Hardware-specific optimization guides
- Best practices for different deployment scenarios
- Performance tuning tutorials
- Case studies and benchmarks

## Project Structure

```
llm-perf-opt/
├── conf/                 # Hydra config groups (defaults, model, dataset, runtime, hardware, profiling)
│   ├── hydra/
│   ├── model/
│   ├── dataset/
│   ├── runtime/
│   ├── hardware/
│   └── profiling/
├── models/               # Weights/tokenizers (symlinks or submodules; prefer symlinks)
├── datasets/             # Dataset roots, variants, metadata and docs
├── third_party/          # Read-only upstream references (symlinks or submodules)
│   ├── github/
│   └── hf/
├── src/
│   └── llm_perf_opt/     # Unified project package
│       ├── profiling/
│       │   └── parsers/
│       ├── runners/
│       └── data/
├── scripts/              # Utility scripts (symlinks, dataset prep, snapshots)
├── tests/                # Tests (manual/unit/integration as needed)
├── docs/                 # Documentation and guides
├── context/              # Knowledge base and hints
├── magic-context/        # Speckit templates and guidance
└── .specify/             # Speckit constitution/templates
```

Each subdirectory contains a README describing its purpose.

## Getting Started

Bootstrap assets (models + datasets)
- Run all: `./bootstrap.sh --yes`
  - Models config: `models/bootstrap.yaml` (uses `$HF_SNAPSHOTS_ROOT` by default)
  - Datasets config: `datasets/omnidocbench/bootstrap.yaml` (uses `$DATASETS_ROOT` by default)
  - The dataset bootstrap can extract `images.zip` / `pdfs.zip` into place when needed.
- Or run individually:
  - Models: `models/bootstrap.sh --yes`
  - Dataset: `datasets/omnidocbench/bootstrap.sh --yes`

Quick run
- With symlinks in place, defaults point to `models/deepseek-ocr` and `datasets/omnidocbench/source-data`.
- Example (Pixi): `pixi run stage1-run` or run the module directly:

```bash
pixi run python -m llm_perf_opt.runners.llm_profile_runner \
  'hydra.run.dir=tmp/profile-output/${now:%Y%m%d-%H%M%S}' \
  device=cuda:0 infer.max_new_tokens=64 \
  'pipeline.torch_profiler.activities=[cpu,cuda]' \
  pipeline.nsys.enable=false pipeline.ncu.enable=false
```

## CLI Usage (Stage 1 Profiling)

Run the Stage 1 profiling runner with Pixi. This collects prefill/decode timings, operator summaries, and MFU, and writes artifacts under `tmp/profile-output/<run_id>/torch_profiler/` and `tmp/profile-output/<run_id>/static_analysis/`.

```bash
pixi run python -m llm_perf_opt.runners.llm_profile_runner \
  model.path=/data2/huangzhe/code/llm-perf-opt/models/deepseek-ocr \
  dataset.root=/data2/huangzhe/code/llm-perf-opt/data/samples \
  repeats=3 device=cuda:0 infer.max_new_tokens=64 \
  'torch_profiler.activities=[cpu,cuda]'
```

Artifacts include `report.md`, `operators.md`, `metrics.json`, `stakeholder_summary.md`, and reproducibility files (`env.json`, `inputs.yaml`, `assumptions.md`).

Static analyzer (optional)
- By default, the runner performs a static model analysis to refine MFU and writes `static_compute.json` / `static_compute.md`.
- To disable for faster runs, either:
  - Use the Pixi task: `pixi run stage1-run-no-static`
  - Or pass a Hydra override while running: `pipeline.static_analysis.enable=false`

## Profiling Requirements (Nsight Systems)

To capture end‑to‑end GPU/CPU timelines while running DeepSeek‑OCR and other workloads, we use NVIDIA Nsight Systems (`nsys`). Please install it so profiling commands work locally.

Preferred install order (where applicable): uv (PyPI) > pixi/conda‑forge > apt > direct .deb

- Nsight Systems (nsys)
  - Ubuntu (recommended):
    - `sudo apt-get update && sudo apt-get install -y nsight-systems nsight-systems-target`
  - Or download from NVIDIA Developer if apt isn’t available: https://developer.nvidia.com/nsight-systems
  - Verify: `nsys --version`

- Optional: Nsight Compute (ncu) for per‑kernel analysis
  - pixi/conda‑forge: `pixi add nsight-compute -c conda-forge`
  - Verify: `ncu --version`

- Optional: NVTX annotations for readable timelines
  - `uv pip install -U nvtx`

Example capture:

```
nsys profile --trace=cuda,nvtx,osrt -o tmp/nsys/deepseek \
  pixi run python tests/manual/deepseek_ocr_hf_manual.py
```

## Contributing

Contributions are welcome! This project is in early stages and we're building the foundation.

## License

*To be determined*

## Roadmap

- [ ] Define benchmark methodology and metrics
- [ ] Implement core benchmarking framework
- [ ] Add support for major hardware platforms
- [ ] Develop optimization toolkit
- [ ] Create visualization and analysis tools
- [ ] Build comprehensive documentation
- [ ] Publish benchmark results and findings

---

**Note**: This project is in the planning and initial development phase. Implementation details will be added iteratively.
