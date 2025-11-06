---

description: "Task list for NVTX-based NCU Regional Profiling"
---

# Tasks: NVTX-based NCU Regional Profiling

**Input**: Design documents from `/workspace/code/llm-perf-opt/specs/003-nvtx-ncu-profiling/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Manual scenario under `tests/manual/…` is required per plan. Unit tests are included for critical parsers/aggregators.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Single project layout rooted at `/workspace/code/llm-perf-opt`
- Artifacts under `/workspace/code/llm-perf-opt/tmp/profile-output/<run_id>/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare configuration, docs, and manual test entrypoint.

- [ ] T001 Update Hydra config to add region mode toggle `pipeline.ncu.region_mode.enable` in `/workspace/code/llm-perf-opt/conf/config.yaml`
- [ ] T002 [P] Seed region-mode defaults (include patterns, replay mode) under `ncu_cli` in `/workspace/code/llm-perf-opt/conf/profiling/ncu/ncu.default.yaml`
- [ ] T003 [P] Create manual test entrypoint file at `/workspace/code/llm-perf-opt/tests/manual/ncu/manual_nvtx_regions.py`
- [ ] T004 [P] Add NVTX region-mode examples to `/workspace/code/llm-perf-opt/docs/running.md`
- [ ] T005 [P] Document new config keys (region mode, selection) in `/workspace/code/llm-perf-opt/docs/configuration.md`

Checkpoint: Config and docs prepared; manual test file exists (can be empty placeholder).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core models and utilities required by all user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T006 Add `NCUProfileRegion` and `NCUProfileRegionReport` data models (attrs) in `/workspace/code/llm-perf-opt/src/llm_perf_opt/data/models.py`
- [ ] T007 [P] Extend Nsight Compute builder to support `--replay-mode` via `replay_mode` param in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/vendor/ncu.py`
- [ ] T008 [P] Add `sanitize_region_name(name: str) -> str` utility in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/nvtx_utils.py`
- [ ] T009 Implement region report writers: `write_region_report_markdown(...)` and `write_region_report_json(...)` in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/export.py`
- [ ] T010 [P] Ensure artifacts layout includes `ncu/regions/` subdir creation in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/artifacts.py`

Checkpoint: Models/utilities available; builder and export helpers ready.

---

## Phase 3: User Story 1 - Profile NVTX-marked regions only (Priority: P1) 🎯 MVP

**Goal**: Limit Nsight Compute profiling to NVTX-marked regions and produce per‑region reports, including nested ranges with inclusive parent totals.

**Independent Test**: Run profiling on a sample with three ranges (A, B, and A::A1 nested) and verify outputs include one section per region (A, A::A1, B), metrics populated, and no kernels outside ranges appear.

### Tests for User Story 1 (requested in plan)

- [ ] T011 [P] [US1] Implement manual scenario with 3 NVTX ranges (A, A::A1, B) in `/workspace/code/llm-perf-opt/tests/manual/ncu/manual_nvtx_regions.py`
- [ ] T012 [P] [US1] Unit test: region model serialization/parent inference in `/workspace/code/llm-perf-opt/tests/unit/test_ncu_region_models.py`

### Implementation for User Story 1

- [ ] T013 [US1] Extend runner to support region-mode orchestration loop in `/workspace/code/llm-perf-opt/src/llm_perf_opt/runners/deep_profile_runner.py` (reads `pipeline.ncu.region_mode.enable` and iterates include patterns)
- [ ] T014 [US1] Add per-region NCU invocation with `--nvtx --nvtx-include <expr> --replay-mode=range` in `/workspace/code/llm-perf-opt/src/llm_perf_opt/runners/deep_profile_runner.py`
- [ ] T015 [US1] Write per-region artifacts under `/workspace/code/llm-perf-opt/tmp/profile-output/<run_id>/ncu/regions/<sanitized>/` (cmd, .ncu-rep, optional CSV/sections) in `/workspace/code/llm-perf-opt/src/llm_perf_opt/runners/deep_profile_runner.py`
- [ ] T016 [US1] Parse NCU outputs and assemble `NCUProfileRegion` objects (kernels + metrics/sections) in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/aggregate.py`
- [ ] T017 [US1] Implement inclusive nesting (parent includes child totals) during aggregation in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/aggregate.py`
- [ ] T018 [US1] Emit consolidated JSON at `/workspace/code/llm-perf-opt/tmp/profile-output/<run_id>/ncu/regions/report.json` in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/export.py`
- [ ] T019 [US1] Emit consolidated Markdown at `/workspace/code/llm-perf-opt/tmp/profile-output/<run_id>/ncu/regions/report.md` in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/export.py`
- [ ] T020 [US1] Graceful no-match handling: write baseline section and exit 0 in `/workspace/code/llm-perf-opt/src/llm_perf_opt/runners/deep_profile_runner.py`

Checkpoint: User Story 1 independently functional. Run manual scenario and verify outputs.

---

## Phase 4: User Story 2 - Select kernels within each region (Priority: P2)

**Goal**: Apply name-based include/exclude patterns to constrain which kernels are summarized per region.

**Independent Test**: Run on a sample with 20+ kernel calls; applying include/exclude patterns should reduce per‑region kernel tables to the intended subset while keeping totals aligned with the subset.

### Tests for User Story 2 (requested in plan for parsers)

- [ ] T021 [P] [US2] Unit test: kernel selection (include/exclude precedence) in `/workspace/code/llm-perf-opt/tests/unit/test_kernel_selection.py`

### Implementation for User Story 2

- [ ] T022 [P] [US2] Add kernel selection keys to preset (e.g., `ncu_cli.kernel_name`, `ncu_cli.kernel_name_base`) in `/workspace/code/llm-perf-opt/conf/profiling/ncu/ncu.default.yaml`
- [ ] T023 [US2] Pass kernel selection from config to builder (`kernel_regex`, base) in `/workspace/code/llm-perf-opt/src/llm_perf_opt/runners/deep_profile_runner.py`
- [ ] T024 [US2] Reflect selection in region aggregation (only selected kernels contribute) in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/aggregate.py`
- [ ] T025 [US2] Annotate selection patterns in report outputs (JSON + Markdown) in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/export.py`

Checkpoint: User Stories 1 and 2 both work and are testable independently.

---

## Phase 5: User Story 3 - Configure reported metrics/sections (Priority: P3)

**Goal**: Choose which metric groups/sections appear in per‑region reports via Hydra config, consistent with existing presets.

**Independent Test**: Changing `ncu_cli.sections` or `ncu_cli.metrics` in the config should deterministically change report content on next run; verify alignment across two different configurations.

### Implementation for User Story 3

- [ ] T026 [P] [US3] Ensure region-mode path applies existing `ncu_cli.sections` and `ncu_cli.metrics` in `/workspace/code/llm-perf-opt/src/llm_perf_opt/runners/deep_profile_runner.py`
- [ ] T027 [P] [US3] Add section export for each region using import command in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/vendor/ncu.py` and runner wiring in `/workspace/code/llm-perf-opt/src/llm_perf_opt/runners/deep_profile_runner.py`
- [ ] T028 [US3] Update docs to demonstrate section/metric overrides for region mode in `/workspace/code/llm-perf-opt/docs/running.md`
- [ ] T029 [US3] Update docs to document config keys and examples in `/workspace/code/llm-perf-opt/docs/configuration.md`

Checkpoint: All user stories independently functional; reports reflect configured sections/metrics.

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Repo-wide quality and finishing touches.

- [ ] T030 [P] Add/refresh example commands for region mode in `/workspace/code/llm-perf-opt/specs/003-nvtx-ncu-profiling/quickstart.md`
- [ ] T031 Type hints + mypy clean for all touched modules in `/workspace/code/llm-perf-opt/src/`
- [ ] T032 Ruff clean (PEP8 + project rules) for all touched modules in `/workspace/code/llm-perf-opt/src/`
- [ ] T033 [P] Update `/workspace/code/llm-perf-opt/scripts/ncu/release/README.md` with region-mode notes
- [ ] T034 Validate manual quickstart run end-to-end and attach artifacts under `/workspace/code/llm-perf-opt/tmp/profile-output/<run_id>/`

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup (Phase 1) → Foundational (Phase 2) → User Stories (Phase 3+) → Polish
- User stories can proceed in parallel after Foundational, or sequentially P1 → P2 → P3

### User Story Dependencies

- User Story 1 (P1): No dependency on other stories (depends on Foundational)
- User Story 2 (P2): Independent of US1 but integrates selection into aggregation
- User Story 3 (P3): Independent of US1/US2; configuration wiring applies to region-mode path

### Within Each User Story

- Tests (if included) before implementation; ensure they fail first
- Models before services/aggregation
- Aggregation before export
- Story complete before starting next priority

---

## Parallel Examples

### User Story 1

Tasks runnable in parallel:
- T011 — implement manual scenario (`/workspace/code/llm-perf-opt/tests/manual/ncu/manual_nvtx_regions.py`)
- T012 — unit tests for models (`/workspace/code/llm-perf-opt/tests/unit/test_ncu_region_models.py`)

### User Story 2

Tasks runnable in parallel:
- T021 — unit tests for selection (`/workspace/code/llm-perf-opt/tests/unit/test_kernel_selection.py`)
- T022 — preset update (`/workspace/code/llm-perf-opt/conf/profiling/ncu/ncu.default.yaml`)

### User Story 3

Tasks runnable in parallel:
- T026 — ensure section/metric wiring in runner (`/workspace/code/llm-perf-opt/src/llm_perf_opt/runners/deep_profile_runner.py`)
- T027 — region section export wiring (`/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/vendor/ncu.py` and runner)

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Setup (T001–T005)
2. Complete Foundational (T006–T010)
3. Implement US1 (T011–T020)
4. Validate US1 independently using manual scenario and reports

### Incremental Delivery

1. Deliver US1 (MVP) → demo/report
2. Add US2 (selection) → demo/report
3. Add US3 (metrics/sections) → demo/report

---

## Validation Checklist

- All tasks follow the required format: `- [ ] T### [P?] [US?] Description with absolute file path`
- Each user story has clear, independent test criteria and can be validated in isolation
- Dependencies are explicit; [P] marks identify safe parallel work
- MVP scope = User Story 1 only

