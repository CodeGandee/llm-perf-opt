Stage 2 — Deep LLM Profiling (Manual Tests)
==========================================

Purpose
- Validate environment and scaffolding for Stage 2 profiling before adding the runner.

Checklist
- Python environment active (Pixi or venv)
- NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`) available in PATH
- Repo root contains `conf/runner/stage2.yaml`

Quick Checks
- pixi run python -c "import llm_perf_opt; print('ok')"
- pixi run python -c "print('stage2-profile' in open('pyproject.toml').read())"

Notes
- The actual deep profiling runner will be added in later phases.
- Artifacts will be written under `tmp/stage2/<run_id>/` once the runner lands.

