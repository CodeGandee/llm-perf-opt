Stage 2 — Deep LLM Profiling (Manual Tests)
==========================================

Purpose
- Validate environment and scaffolding for Stage 2 profiling before adding the runner.

Checklist
- Python environment active (Pixi or venv)
- NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`) available in PATH

Quick Checks
- pixi run python -c "import llm_perf_opt; print('ok')"
- pixi run python -c "print('stage2-profile' in open('pyproject.toml').read())"

Workload selection and runner interop
- When profiling the Stage 1 runner as the workload, disable extra stages to avoid overhead by using unified pipeline toggles:
  - Add Hydra overrides like `pipeline.static_analysis.enable=false` and `pipeline.torch_profiler.enable=false` to your workload argv (the deep profiler does this automatically).
  - Example `work_argv` passed to `build_nsys_cmd`/`build_ncu_cmd`:
    - `['python', '-m', 'llm_perf_opt.runners.llm_profile_runner', 'dataset.subset_filelist=/abs/dev-20.txt', 'device=cuda:0', 'repeats=1', 'infer.max_new_tokens=64', 'pipeline.static_analysis.enable=false', 'pipeline.torch_profiler.enable=false']`

Notes
- Use `pixi run stage2-profile` to drive Nsight Systems with the unified entry config `conf/config.yaml`.
- Artifacts are written under `tmp/profile-output/<run_id>/` with subfolders `nsys/`, `ncu/`, and workload outputs under `tmp/workload/`.
