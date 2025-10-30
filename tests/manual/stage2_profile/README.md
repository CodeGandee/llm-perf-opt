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

Workload selection and runner interop
- When profiling the Stage 1 runner with Nsight, disable its static analyzer to avoid extra overhead:
  - Add Hydra override `runner@stage1_runner=stage1.no-static` to your workload argv
  - Example `work_argv` passed to `build_nsys_cmd`/`build_ncu_cmd`:
    - `['python', '-m', 'llm_perf_opt.runners.llm_profile_runner', 'dataset.subset_filelist=/abs/dev-20.txt', 'device=cuda:0', 'repeats=1', 'infer.max_new_tokens=64', 'runner@stage1_runner=stage1.no-static']`

Notes
- The actual deep profiling runner will be added in later phases.
- Artifacts will be written under `tmp/stage2/<run_id>/` once the runner lands.
