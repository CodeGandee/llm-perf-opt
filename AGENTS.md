# Repository Guidelines

This document is a concise contributor guide for llm-perf-opt. Follow it for any changes within this repository.

## Project Structure & Module Organization
- src/llm_perf_opt: Python package (runners, profiling, data, visualize)
- conf/: Hydra config groups (model, dataset, runtime, hardware, profiling)
- tests/: unit/, integration/, manual/ (manual tests use manual_*.py)
- scripts/: tooling and profiling helpers; bootstrap.sh at root for assets
- datasets/, models/, extern/: symlinks or local mounts; do not commit large artifacts
- docs/, context/, magic-context/, specs/, tmp/: docs, knowledge, plans, and run outputs

## Build, Test, and Development Commands
- Environment: install Pixi, then run `pixi install` (default CUDA 12.6 env)
- Python: run scripts via `pixi run -e rtx5090 python ...` so they execute in the RTX 5090 environment
- Lint: `pixi run ruff check .`  |  Types: `pixi run mypy src`
- Unit tests: `pixi run pytest tests/unit/`
- Integration tests: `pixi run pytest tests/integration/`
- Manual run (stage‑1): `pixi run stage1-run`
- Deep profiling (stage‑2): `pixi run stage2-profile`
- Docs: `pixi run docs-serve` (dev) | `pixi run docs-build`
- Bootstrap assets: `./bootstrap.sh --yes`

## Coding Style & Naming Conventions
- Use type hints; keep functions small and single‑purpose
- Naming: modules/functions snake_case; Classes CamelCase; constants UPPER_CASE
- Run `ruff` (PEP8 + rules in pyproject) and `mypy` before pushing
- For docs and markdown in this repo, DO NOT hard break long lines; keep each logical sentence or item on a single physical line and let tools handle wrapping

## Testing Guidelines
- Framework: pytest. Place fast, deterministic tests in tests/unit/
- Integration tests may touch filesystem or external tools; mark/select via `-m integration` as needed
- Manual tests live in tests/manual/ and must be prefixed with `manual_` to avoid pytest collection
- Aim for meaningful coverage; add tests beside the feature you modify

## Commit & Pull Request Guidelines
- Commits: imperative, concise, and scoped (e.g., `runners: fix nsys gating`)
- PRs: include summary, motivation, linked issue, before/after notes, and any performance impact
- Attach or reference artifacts under tmp/profile-output/<run_id>/ when relevant
- Keep diffs focused; update docs/config if behavior changes

## Security & Configuration Tips
- Do not commit datasets, weights, profiler traces, or large binaries; prefer symlinks and local paths
- Use Hydra overrides for reproducibility (e.g., `hydra.run.dir=tmp/profile-output/${now:%Y%m%d-%H%M%S}`)

## Active Technologies
- Python 3.11 + Hydra (omegaconf), mdutils, attrs, nvtx (runtime), Nsight Systems/Compute CLIs (nsys/ncu) (003-nvtx-ncu-profiling)
- Filesystem artifacts under `/workspace/code/llm-perf-opt/tmp/profile-output/<run_id>/` (nsys/, ncu/, torch_profiler/, static_analysis/) (003-nvtx-ncu-profiling)
- Python 3.11 (Pixi-managed environment, CUDA 12.6 toolchain) + PyTorch, ModelMeter (`extern/modelmeter`), Hydra/omegaconf, attrs, TorchInfo, NVIDIA Nsight (001-deepseek-ocr-modelmeter)
- Filesystem-only artifacts under `reports/` and `tmp/profile-output/` (no external database or message bus) (001-deepseek-ocr-modelmeter)
- Python 3.11 + Hydra (omegaconf), attrs, PyTorch (reference flop counter), ModelMeter analytic layers (`modelmeter.layers.*`) (004-wan2-1-analytic-model)
- Filesystem artifacts under `/data1/huangzhe/code/llm-perf-opt/tmp/profile-output/<run_id>/` (static_analysis/, nsys/, ncu/, torch_profiler/) (004-wan2-1-analytic-model)

## Recent Changes
- 003-nvtx-ncu-profiling: Added Python 3.11 + Hydra (omegaconf), mdutils, attrs, nvtx (runtime), Nsight Systems/Compute CLIs (nsys/ncu)
