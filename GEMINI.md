# Gemini Project Context: LLM Performance Optimization

This `GEMINI.md` file provides context and instructions for AI agents working with the `llm-perf-opt` codebase.

## Project Overview

`llm-perf-opt` is a comprehensive framework for benchmarking and profiling Large Language Model (LLM) inference performance on NVIDIA GPUs. It enables multi-stage profiling to identify bottlenecks at both the operator (PyTorch) and kernel (CUDA) levels.

**Key Technologies:**
- **Language:** Python 3.11+
- **Environment & Tasks:** [Pixi](https://pixi.sh/)
- **Configuration:** [Hydra](https://hydra.cc/) (`conf/`)
- **Profiling:** PyTorch Profiler, Nsight Systems (`nsys`), Nsight Compute (`ncu`)
- **Frameworks:** PyTorch (vLLM and TensorRT-LLM support planned/referenced)

**Architecture:**
- **`src/llm_perf_opt/`**: The main Python package.
    - `runners/`: Orchestrators for different execution modes (Stage 1, Stage 2, Inference).
    - `profiling/`: Harnesses for profilers and MFU calculation.
    - `data/` & `contracts/`: Data models and dataset utilities.
- **`conf/`**: Hydra configuration hierarchy (models, datasets, hardware, profiling pipelines).
- **`scripts/`**: Utility scripts for analysis, visualization, and installation.
- **`models/` & `datasets/`**: Symlinked directories for large assets (managed via `bootstrap.sh`).

## Building and Running

The project relies heavily on `pixi` for dependency management and task execution.

### Setup
1.  **Install Dependencies:**
    ```bash
    pixi install
    ```
2.  **Bootstrap Assets (Models/Datasets):**
    ```bash
    ./bootstrap.sh --yes
    ```
    *Creates symlinks from `$HF_SNAPSHOTS_ROOT` and `$DATASETS_ROOT`.*

### Profiling Workflows
*defined in `pyproject.toml`*

*   **Stage 1 (PyTorch Profiler - Operator Level):**
    ```bash
    pixi run stage1-run
    ```
    *Generates `report.md`, `operators.md`, and static analysis in `tmp/profile-output/`.*

*   **Stage 2 (Nsight Systems/Compute - Kernel Level):**
    ```bash
    pixi run stage2-profile
    ```
    *Disables torch profiler overhead. Outputs to `tmp/profile-output/<run_id>/nsys/` and `ncu/`.*

*   **Direct Inference (No Profiling):**
    ```bash
    pixi run direct-infer-dev20
    ```
    *Runs inference on a subset of data for validation/visualization.*

### Environments
*   **Default:** `pixi install` (PyTorch 2.5.1, CUDA 12.4).
*   **RTX 5090 (Blackwell):** `pixi install -e rtx5090` (PyTorch Nightly, CUDA 12.8).
    *   Setup: `pixi run -e rtx5090 setup-rtx5090`

## Development Conventions

*   **Configuration:** Use Hydra overrides for runtime changes. Do not hardcode parameters.
    *   *Example:* `pixi run python -m ... device=cuda:0 infer.max_new_tokens=128`
*   **Code Style:**
    *   **Lint:** `pixi run ruff check src/`
    *   **Format:** `pixi run ruff format src/`
    *   **Type Check:** `pixi run mypy src/`
*   **Testing:**
    *   Unit: `pixi run pytest tests/unit/`
    *   Integration: `pixi run pytest tests/integration/`
*   **Submodules:** The project uses git submodules (`magic-context`, `extern/modelmeter`). Ensure they are initialized if needed.

## Key Files & Directories

*   `pyproject.toml`: Defines dependencies, environments, and run tasks. **Read this to understand available commands.**
*   `conf/config.yaml`: Root Hydra configuration.
*   `src/llm_perf_opt/runners/llm_profile_runner.py`: Main entry point for Stage 1 profiling.
*   `src/llm_perf_opt/runners/deep_profile_runner.py`: Main entry point for Stage 2 profiling.
*   `bootstrap.sh`: Script to set up model and dataset symlinks.
