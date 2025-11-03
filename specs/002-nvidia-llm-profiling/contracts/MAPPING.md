# Contract ↔ CLI Mapping (Stage 2)

This document maps hypothetical API endpoints to local CLI/task behavior for Stage 2 profiling.

- POST `/profile/run` → `pixi run stage2-profile`
  - Launches a Stage 2 deep profiling session, writing artifacts to `tmp/stage2/<run_id>/`.
  - When profiling the Stage 1 runner as the workload, the runner should be invoked with `runners=stage1.no-static` to avoid extra overhead.

- GET `/profile/{run_id}/artifacts` → list files under `tmp/stage2/<run_id>/`
  - Returns a simple listing or JSON manifest of artifacts (traces, kernel metrics, provenance files).

Notes
- No new runtime models are introduced; this is a mapping note only.
- Artifact layout is managed by `llm_perf_opt.profiling.artifacts.Artifacts`.
