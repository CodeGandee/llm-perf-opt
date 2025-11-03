# Tasks: Basic Profiling for DeepSeek‑OCR (Stage 1)

Feature Dir: /data2/huangzhe/code/llm-perf-opt/specs/001-profile-deepseek-ocr
Plan: /data2/huangzhe/code/llm-perf-opt/specs/001-profile-deepseek-ocr/plan.md
Spec: /data2/huangzhe/code/llm-perf-opt/specs/001-profile-deepseek-ocr/spec.md

## Phase 1 — Setup

- [X] T001 Ensure packaging exports only unified package in /data2/huangzhe/code/llm-perf-opt/pyproject.toml
- [X] T002 [P] Create contracts module with attrs models in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/contracts/models.py
- [X] T003 [P] Scaffold profiling harness skeleton (NVTX/profiler context) in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/profiling/harness.py
- [X] T004 [P] Add MFU estimation utilities (decode/prefill) in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/profiling/mfu.py
- [X] T005 [P] Add hardware peak TFLOPs lookup and env logging in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/profiling/hw.py
- [X] T006 [P] Implement operator summary export (Markdown/JSON) in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/profiling/export.py
- [X] T007 [P] Implement aggregation helpers (mean/std across repeats) in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/profiling/aggregate.py
- [X] T008 Scaffold LLMProfileRunner CLI entry in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/runners/llm_profile_runner.py
- [X] T009 Update quickstart examples to reference llm_perf_opt runner in /data2/huangzhe/code/llm-perf-opt/specs/001-profile-deepseek-ocr/quickstart.md

## Phase 2 — Foundational

- [X] T010 Define domain data models (attrs) from data-model.md in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/data/models.py
- [X] T011 [P] Add cattrs converters between domain and contracts in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/contracts/convert.py
- [X] T012 [P] Implement NVTX helpers (prefill/decode push/pop) in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/profiling/nvtx_utils.py
- [X] T013 Implement DeepSeekOCRSession wrapper (load-once model/tokenizer; NVTX in run_inference) in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/runners/dsocr_session.py
- [X] T014 Wire session + harness + helpers into LLMProfileRunner CLI skeleton in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/runners/llm_profile_runner.py

## Phase 3 — User Story 1 (P1): Stage 1 Profiling Report
Goal: Stage segmentation, operator summary, MFU (model-level and per-stage) present for a representative run; repeated runs aggregate.

- [X] T020 [US1] Integrate NVTX segmentation (prefill/decode) around DeepSeek‑OCR calls in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/runners/llm_profile_runner.py
- [X] T021 [P] [US1] Add PyTorch profiler (CPU+CUDA) context and collect operator stats in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/runners/llm_profile_runner.py
- [X] T022 [P] [US1] Compute per-stage timings and throughput in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/runners/llm_profile_runner.py
- [X] T023 [P] [US1] Implement MFU estimator (model-level and per-stage) using /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/profiling/mfu.py
- [X] T024 [P] [US1] Export operator top‑K summary (Markdown/JSON) via /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/profiling/export.py
- [X] T025 [US1] Support repeated passes, aggregate mean/std via /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/profiling/aggregate.py
- [X] T026 [US1] Write report.md with stage timings, operator summary, MFU to /data2/huangzhe/code/llm-perf-opt/tmp/stage1/<run_id>/report.md
- [X] T027 [US1] Persist raw metrics (JSON) for reproducibility to /data2/huangzhe/code/llm-perf-opt/tmp/stage1/<run_id>/metrics.json

## Phase 4 — User Story 2 (P2): Stakeholder Summary
Goal: Concise summary of top cost centers and stage attribution; plain language recommendations.

- [X] T030 [US2] Generate stakeholder_summary.md from metrics to /data2/huangzhe/code/llm-perf-opt/tmp/stage1/<run_id>/stakeholder_summary.md
- [X] T031 [P] [US2] Extract top‑N operators and stage attribution using /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/profiling/export.py
- [X] T032 [US2] Add recommendations section template to /data2/huangzhe/code/llm-perf-opt/tmp/stage1/<run_id>/stakeholder_summary.md

## Phase 5 — User Story 3 (P3): Reproducible Inputs & Notes
Goal: Identify input set and assumptions to repeat Stage 1 profiling and compare results.

- [X] T040 [US3] Emit input list (absolute paths) and image metadata to /data2/huangzhe/code/llm-perf-opt/tmp/stage1/<run_id>/inputs.yaml
- [X] T041 [P] [US3] Capture environment info (GPU, CUDA, torch, transformers) to /data2/huangzhe/code/llm-perf-opt/tmp/stage1/<run_id>/env.json using /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/profiling/hw.py
- [X] T042 [US3] Persist run assumptions (batch size, decoding params) to /data2/huangzhe/code/llm-perf-opt/tmp/stage1/<run_id>/assumptions.md
- [X] T043 [US3] Update quickstart with reproducibility notes in /data2/huangzhe/code/llm-perf-opt/specs/001-profile-deepseek-ocr/quickstart.md

## Final Phase — Polish & Cross-Cutting

- [X] T050 Add NumPy‑style docstrings and examples to public APIs in /data2/huangzhe/code/llm-perf-opt/src/llm_perf_opt/**
- [X] T051 Ensure full typing; mypy & ruff clean in /data2/huangzhe/code/llm-perf-opt/pyproject.toml
- [X] T052 [P] Add CLI usage snippet to README in /data2/huangzhe/code/llm-perf-opt/README.md
- [X] T053 [P] Write manual validation steps documenting independent tests in /data2/huangzhe/code/llm-perf-opt/specs/001-profile-deepseek-ocr/quickstart.md

---

## Dependencies (Story Order)

1) US1 (P1) → 2) US2 (P2) → 3) US3 (P3)
- US2 depends on metrics/artifacts produced by US1
- US3 depends on basic run capability (US1), but environment/inputs capture (T040–T042) can be developed in parallel with US1 after T014

## Parallel Execution Examples

- After T008 (runner skeleton) and T014 (wiring), run in parallel:
  - T021 (profiler), T023 (MFU estimator), T024 (operator export), T022 (timings)
- While US1 metrics work proceeds, in parallel:
  - T031 (top‑N extraction) for US2
  - T041 (env capture) for US3

## Implementation Strategy (MVP First)

- MVP scope: Complete US1 minimal path (T020–T027) to produce a Stage 1 report with stage timings, operator summary, and MFU on a small image set.
- Incremental delivery: Add US2 stakeholder summary (T030–T032), then US3 reproducibility artifacts (T040–T043).
- Hardening: Polish phase (T050–T053) to meet constitution gates and improve docs.

## Validation Checklist

- Each user story phase yields independently reviewable artifacts under /data2/huangzhe/code/llm-perf-opt/tmp/stage1/<run_id>/
- Repeated runs (repeats ≥ 2) produce aggregated metrics with reduced variance (per spec SC‑005)
- All tasks follow checklist format with unique Task IDs and absolute file paths
