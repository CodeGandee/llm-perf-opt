# Tasks: Stage 2 — NVIDIA-Backed Deep LLM Profiling

Input: Design documents from `specs/002-nvidia-llm-profiling/`
Prerequisites: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

Tests: Per the constitution, provide manual tests for major functionality under `tests/manual/…`. Automated tests are OPTIONAL; research.md requests minimal pytest for table generation in exporters.

Organization: Tasks are grouped by user story to enable independent implementation and testing of each story.

Format: `[ID] [P?] [Story] Description`
- [P]: Can run in parallel (different files, no dependencies)
- [Story]: User story label (US1, US2, …) for story phases only
- All tasks include exact file paths

Path Conventions
- Single project: `src/`, `tests/` at repository root
- This feature extends: `src/llm_perf_opt/` runners and profiling utilities

---

## Phase 1: Setup (Shared Infrastructure)

Purpose: Project initialization and basic structure for Stage 2

- [ ] T001 Create vendor wrapper module for Nsight Systems in `src/llm_perf_opt/profiling/vendor/nsys.py`
- [ ] T002 [P] Create vendor wrapper module for Nsight Compute in `src/llm_perf_opt/profiling/vendor/ncu.py`
- [ ] T003 Add Pixi task `stage2-profile` in `pyproject.toml` under `[tool.pixi.tasks]`
- [ ] T004 [P] Add Hydra config for Stage 2 modes in `conf/profiling/stage2.yaml` (keys: `run.mode={deep,light}`, `run.top_n_kernels`, `artifacts.stage2_dir`)
- [ ] T005 [P] Create manual tests directory and seed doc in `tests/manual/stage2_profile/README.md`
- [ ] T006 [P] Add `tmp/stage2/` ignore rule in `.gitignore`

---

## Phase 2: Foundational (Blocking Prerequisites)

Purpose: Core building blocks that MUST be complete before user stories

⚠️ CRITICAL: No user story work can begin until this phase is complete

- [ ] T007 Introduce only new kernel attribution types per reuse-first model: add `KernelRecord` (attrs or TypedDict) in `src/llm_perf_opt/data/models.py` and optionally extend `LLMProfileReport` with `kernels_topk`
- [ ] T008 [P] Implement artifacts manager in `src/llm_perf_opt/profiling/artifacts.py` (run_id creation, dir layout `tmp/stage2/<run_id>/`, write `env.json`, `config.yaml`, `inputs.yaml`)
- [ ] T009 [P] Implement kernel results parser in `src/llm_perf_opt/profiling/kernels.py` (parse `ncu` CSV/JSON → `KernelRecord[]`)
- [ ] T010 [P] Implement vendor tool checks in `src/llm_perf_opt/profiling/vendor/checks.py` (`ensure_nsys()`, `ensure_ncu()`, friendly errors)
- [ ] T011 [P] Add contract→CLI mapping note in `specs/002-nvidia-llm-profiling/contracts/MAPPING.md` (map `/profile/run` → Pixi `stage2-profile`, `/profile/{run_id}/artifacts` → list `tmp/stage2/<run_id>`) — no new runtime models introduced

Checkpoint: Foundation ready — user story implementation can now begin

---

## Phase 3: User Story 1 — Run Deep Profiling Session (Priority: P1) 🎯 MVP

Goal: Run a deep profiling session for DeepSeek-OCR using vendor tools; capture kernel-level metrics and produce a complete artifact bundle.

Independent Test: One command runs a profiling session and writes artifacts under `tmp/stage2/<run_id>/` including trace (`.qdrep`), kernel metrics (ncu CSV/JSON), `operators.md`, `kernels.md` placeholder, `stakeholder_summary.md`, and provenance files.

### Implementation for User Story 1

- [ ] T012 [P] [US1] Implement Hydra entrypoint for Stage 2 in `src/llm_perf_opt/runners/deep_profile_runner.py` (reads `conf/profiling/stage2.yaml`)
- [ ] T013 [P] [US1] Implement `build_nsys_cmd()` in `src/llm_perf_opt/profiling/vendor/nsys.py` (trace `cuda,nvtx,osrt`, NVTX capture range, output to `tmp/stage2/<run_id>/nsys.qdrep`)
- [ ] T014 [US1] Write provenance artifacts from runner in `tmp/stage2/<run_id>/` (`env.json`, `inputs.yaml`, `config.yaml`) via `artifacts.py`
- [ ] T015 [US1] Integrate Nsight Compute in `src/llm_perf_opt/profiling/vendor/ncu.py` and call from runner (auto-select top-N kernels strategy stub; persist CSV/JSON under `tmp/stage2/<run_id>/ncu/`)
- [ ] T016 [P] [US1] Add manual run script `tests/manual/stage2_profile/manual_stage2_profile.py` (invokes Pixi `stage2-profile`, asserts expected files exist)

Checkpoint: US1 runnable end-to-end; artifacts produced consistently

---

## Phase 4: User Story 2 — Export Top GPU Kernels (Priority: P2)

Goal: Generate `kernels.md` table sorted by total time, including kernel name, device, total ms, calls, mean ms.

Independent Test: Running the profiling session generates `kernels.md` alongside `operators.md` with correctly formatted columns and sorting by total time.

### Tests for User Story 2 (requested in research.md)

- [ ] T017 [P] [US2] Unit test for kernel table export in `tests/unit/test_kernels_export.py` (headers, sort by `total_ms`, mean calculation)

### Implementation for User Story 2

- [ ] T018 [P] [US2] Implement `top_n_kernels()` and `write_kernel_markdown()` in `src/llm_perf_opt/profiling/export.py`
- [ ] T019 [US2] Generate `kernels.md` in runner `src/llm_perf_opt/runners/deep_profile_runner.py` after parsing ncu outputs

Checkpoint: `kernels.md` renders correctly and matches top-time attribution in traces

---

## Phase 5: User Story 3 — Stakeholder Report with Actionable Tables (Priority: P3)

Goal: Produce a concise stakeholder report including Environment, Aggregates, Per-Stage Timings, MFU, Top Operators, Top Kernels + executive summary.

Independent Test: The report renders with all mandatory sections and tables and a brief narrative of bottlenecks and recommendations.

### Tests for User Story 3 (OPTIONAL)

- [ ] T020 [P] [US3] Unit test for summary structure in `tests/unit/test_stakeholder_summary.py` (sections present, minimal schema)

### Implementation for User Story 3

- [ ] T021 [P] [US3] Extend `write_stakeholder_summary()` in `src/llm_perf_opt/profiling/export.py` to reference/include Top Kernels table
- [ ] T022 [US3] Ensure `report.md` includes Per-Stage Timings table and MFU per spec in `src/llm_perf_opt/runners/deep_profile_runner.py` (reusing Stage 1 logic where possible)
- [ ] T023 [US3] Verify final `stakeholder_summary.md` written under `tmp/stage2/<run_id>/` contains all required sections

Checkpoint: Stakeholder report complete with actionable tables and narrative

---

## Phase 6: User Story 4 — Apply to Other LLMs (Priority: P4)

Goal: Run the same profiling workflow on different LLMs without code changes; include model metadata in outputs.

Independent Test: Point to a different model config and generate complete artifacts with correct model identifiers and stage coverage.

### Implementation for User Story 4

- [ ] T024 [P] [US4] Add model target config(s) for alternates in `conf/model/llm/alt.yaml` (name/variant/params/dtype)
- [ ] T025 [US4] Update `src/llm_perf_opt/runners/deep_profile_runner.py` to accept `+model.target=...` and embed model metadata in provenance
- [ ] T026 [P] [US4] Add manual script `tests/manual/stage2_profile/manual_stage2_alt_model.py` (runs with alternate model target)

Checkpoint: Cross-model run produces comparable artifact set

---

## Phase 7: Polish & Cross-Cutting Concerns

Purpose: Improve robustness, docs, and performance toggles

- [ ] T027 [P] Documentation updates for Stage 2 command and artifacts in `specs/002-nvidia-llm-profiling/quickstart.md`
- [ ] T028 Ensure `ruff` and `mypy` pass clean; adjust types/docs in `src/**` as needed
- [ ] T029 [P] Implement light-mode parameters in `conf/profiling/stage2.yaml` and `src/llm_perf_opt/profiling/vendor/ncu.py` (reduced counters/sampling)

---

## Dependencies & Execution Order

Phase Dependencies
- Setup (Phase 1): No dependencies
- Foundational (Phase 2): Depends on Setup; BLOCKS all user stories
- User Stories (Phase 3+): Depend on Foundational; execute in priority order (P1 → P2 → P3 → P4) or in parallel if artifacts/contracts are stubbed
- Polish (Final): After all desired user stories complete

User Story Dependencies
- User Story 1 (P1): After Phase 2; no other story deps
- User Story 2 (P2): Depends on US1 artifacts to finalize; development can start after Phase 2 using sample ncu CSVs
- User Story 3 (P3): Depends on US1 (summary inputs) and US2 (kernels table)
- User Story 4 (P4): Depends on US1; independent of US2/US3

Within Each User Story
- Tests (if any) first → Models → Services/Vendor wrappers → Runner/Endpoints → Integration
- Ensure each story is independently testable before moving on

---

## Parallel Examples

Parallel Example: User Story 1
- [P] T013 in `src/llm_perf_opt/profiling/vendor/nsys.py`
- [P] T016 in `tests/manual/stage2_profile/manual_stage2_profile.py`

Parallel Example: User Story 2
- [P] T017 in `tests/unit/test_kernels_export.py`
- [P] T018 in `src/llm_perf_opt/profiling/export.py`

Parallel Example: User Story 3
- [P] T020 in `tests/unit/test_stakeholder_summary.py`
- [P] T021 in `src/llm_perf_opt/profiling/export.py`

Parallel Example: User Story 4
- [P] T024 in `conf/model/llm/alt.yaml`
- [P] T026 in `tests/manual/stage2_profile/manual_stage2_alt_model.py`

---

## Implementation Strategy

MVP First (User Story 1 Only)
1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Implement Phase 3: US1 deep profiling run
4. STOP and VALIDATE: Verify artifacts, trace size, and runtime bound

Incremental Delivery
1. Add US2 (kernels export) → Validate independently
2. Add US3 (stakeholder report) → Validate independently
3. Add US4 (cross-model) → Validate independently
4. Each story adds value without breaking previous stories

---

## Notes
- [P] tasks = different files, no dependency on incomplete tasks
- [Story] label maps task to a user story for traceability
- Each user story is independently completable/testable per its acceptance criteria
- Keep trace size and runtime within bounds; switch to light mode if needed
- Prefer `pixi run` for all commands

