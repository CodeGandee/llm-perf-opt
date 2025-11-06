---

description: "Task list for NVTX-based NCU Regional Profiling"
---

# Tasks: NVTX-based NCU Regional Profiling

**Input**: Design documents from `/specs/003-nvtx-ncu-profiling/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Provide manual tests for major functionality under `tests/manual/…`. Automated tests are OPTIONAL unless requested.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Global Testing Model Assumption

All phases (Setup, Foundational, User Stories, Polish) MUST use a dummy model as the workload for testing and verification. We will ship minimal Hydra config groups for these dummy models under:

- `conf/model/dummy_<what-model>/arch/dummy_<what-model>.default.yaml`
- `conf/model/dummy_<what-model>/infer/dummy_<what-model>.default.yaml`

Example for the shallow ResNet dummy:

- `conf/model/dummy_shallow_resnet/arch/dummy_shallow_resnet.default.yaml`
- `conf/model/dummy_shallow_resnet/infer/dummy_shallow_resnet.default.yaml`

Select the dummy model via Hydra overrides in all manual tests and examples, e.g.:

```
model/dummy_shallow_resnet/arch@model=dummy_shallow_resnet.default \
model/dummy_shallow_resnet/infer@infer=dummy_shallow_resnet.default
```

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initial scaffolding including a reusable dummy model for testing/debugging

- [ ] T001 [P] Create package scaffold for dummy models in `/workspace/code/llm-perf-opt/src/llm_perf_opt/dnn_models/__init__.py`; add Hydra config groups for the dummy model under:
      - `/workspace/code/llm-perf-opt/conf/model/dummy_shallow_resnet/arch/dummy_shallow_resnet.default.yaml`
      - `/workspace/code/llm-perf-opt/conf/model/dummy_shallow_resnet/infer/dummy_shallow_resnet.default.yaml`
- [ ] T002 [P] Implement `ShallowResNet` dummy model in `/workspace/code/llm-perf-opt/src/llm_perf_opt/dnn_models/shallow_resnet.py` (few layers, CPU/GPU compatible)
- [ ] T003 [P] Add model factory `get_model(name)` in `/workspace/code/llm-perf-opt/src/llm_perf_opt/dnn_models/factory.py` (returns `ShallowResNet` by name)
- [ ] T004 [P] Create manual test scaffold in `/workspace/code/llm-perf-opt/tests/manual/ncu/manual_nvtx_regions.py`
- [ ] T005 [P] Add "NVTX Range Replay" section stub to `/workspace/code/llm-perf-opt/docs/running.md`
- [ ] T006 [P] Add "NCU CLI Config Mapping" section stub to `/workspace/code/llm-perf-opt/docs/configuration.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data models and config/CLI plumbing required by all stories

- [ ] T007 [P] Create attrs data models `NCUProfileRegion` and `NCUProfileRegionReport` in `/workspace/code/llm-perf-opt/src/llm_perf_opt/data/ncu_regions.py`
- [ ] T008 [P] Extend NCU command builder with `replay_mode` (map to `--replay-mode`) in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/vendor/ncu.py`
- [ ] T009 [P] Add `ncu_cli.replay_mode: kernel` default to `/workspace/code/llm-perf-opt/conf/profiling/ncu/ncu.default.yaml`
- [ ] T010 [P] Add `ncu_cli.replay_mode: kernel` default to `/workspace/code/llm-perf-opt/conf/profiling/ncu/ncu.high.yaml`
- [ ] T011 [P] Add `ncu_cli.replay_mode: kernel` default to `/workspace/code/llm-perf-opt/conf/profiling/ncu/ncu.rtx3090.yaml` (and related variants)

**Checkpoint**: Foundation ready – user story implementation can begin

---

## Phase 3: User Story 1 - Profile NVTX-marked regions only (Priority: P1) 🎯 MVP

**Goal**: Restrict profiling to NVTX-marked regions and aggregate results per range (including nested regions)

**Independent Test**: Use the dummy model via Hydra overrides:

```
model/dummy_shallow_resnet/arch@model=dummy_shallow_resnet.default \
model/dummy_shallow_resnet/infer@infer=dummy_shallow_resnet.default
```

Run the manual script with 3 ranges (A, B, A::A1) and `pipeline.ncu.ncu_cli.replay_mode=range` + `pipeline.ncu.ncu_cli.nvtx.include`, then verify one section per region in `ncu/regions/` and consolidated `report.{md,json}`.

### Implementation for User Story 1

- [ ] T012 [US1] Implement NVTX range replay and per-range outputs in `/workspace/code/llm-perf-opt/src/llm_perf_opt/runners/deep_profile_runner.py` (handles `replay_mode=range|app-range`, writes to `ncu/regions/<region>/` and consolidated `ncu/regions/`)
- [ ] T013 [P] [US1] Implement region assembler to build `NCUProfileRegionReport` in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/regions.py`
- [ ] T014 [P] [US1] Implement region report exporters (Markdown + JSON) in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/export_regions.py`
- [ ] T015 [P] [US1] Add filesystem-safe region path helper to `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/artifacts.py` for `ncu/regions/<sanitized_name>`
- [ ] T016 [US1] Implement manual test with 3 ranges in `/workspace/code/llm-perf-opt/tests/manual/ncu/manual_nvtx_regions.py` (validate expected files exist)
- [ ] T017 [US1] Update NVTX range replay examples in `/workspace/code/llm-perf-opt/docs/running.md` (commands + expected outputs under `ncu/regions/`)

Parallel execution example: T013, T014, T015 can run in parallel after T007–T008

---

## Phase 4: User Story 2 - Select kernels within each region (Priority: P2)

**Goal**: Constrain per-region reporting to selected kernels via name patterns (include/exclude)

**Independent Test**: Use the dummy model via the same Hydra overrides as US1. Run the manual script with `pipeline.ncu.ncu_cli.nvtx.include='decode*'` and a regex kernel filter; verify only matching kernels appear in per-region listings and selection metadata is recorded.

### Implementation for User Story 2

- [ ] T018 [P] [US2] Read `ncu_cli.kernel_name` and `kernel_name_base` in `/workspace/code/llm-perf-opt/src/llm_perf_opt/runners/deep_profile_runner.py` and pass through to NCU builder
- [ ] T019 [P] [US2] Add optional `ncu_cli.kernel_exclude` (list) placeholders to `/workspace/code/llm-perf-opt/conf/profiling/ncu/ncu.default.yaml` and `/workspace/code/llm-perf-opt/conf/profiling/ncu/ncu.high.yaml`
- [ ] T020 [P] [US2] Apply include/exclude selection to per-region kernel lists in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/regions.py` and persist `selection` metadata
- [ ] T021 [US2] Extend manual test in `/workspace/code/llm-perf-opt/tests/manual/ncu/manual_nvtx_regions.py` to add kernel regex within NVTX range and verify subset
- [ ] T022 [US2] Update kernel filter examples in `/workspace/code/llm-perf-opt/docs/configuration.md` (exact and `regex:` forms)

Parallel execution example: T018–T020 can run in parallel; T021–T022 follow

---

## Phase 5: User Story 3 - Configure reported metrics/sections (Priority: P3)

**Goal**: Allow users to select `sections` and `metrics` via config; reflect in region reports

**Independent Test**: With the dummy model selected via Hydra overrides, override `pipeline.ncu.ncu_cli.sections` and `metrics`; run the manual script and confirm Markdown/JSON reflect selected groups and NCU importer outputs.

### Implementation for User Story 3

- [ ] T023 [P] [US3] Ensure `sections`/`metrics` apply to range flows in `/workspace/code/llm-perf-opt/src/llm_perf_opt/runners/deep_profile_runner.py` and import per-region sections into `ncu/regions/<region>/sections.txt`
- [ ] T024 [P] [US3] Add sample `sections`/`metrics` for region runs in `/workspace/code/llm-perf-opt/conf/profiling/ncu/ncu.default.yaml`
- [ ] T025 [P] [US3] Include selected sections/metrics summaries in Markdown in `/workspace/code/llm-perf-opt/src/llm_perf_opt/profiling/export_regions.py`
- [ ] T026 [US3] Document `sections`/`metrics` overrides and replay mode interactions in `/workspace/code/llm-perf-opt/docs/configuration.md`

Parallel execution example: T023–T025 can run in parallel after T012–T014

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Validation, error handling, and documentation hardening

- [ ] T027 [P] Add JSON schema validator for region report at `/workspace/code/llm-perf-opt/scripts/ncu/validate_regions_json.py` (schema: `/workspace/code/llm-perf-opt/specs/003-nvtx-ncu-profiling/contracts/openapi.yaml`)
- [ ] T028 Improve empty-range handling and messaging in `/workspace/code/llm-perf-opt/src/llm_perf_opt/runners/deep_profile_runner.py` (emit "no matching regions" note)
- [ ] T029 [P] Update nesting semantics and artifact layout docs in `/workspace/code/llm-perf-opt/docs/internals.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup (Phase 1): No dependencies – can start immediately
- Foundational (Phase 2): Depends on Setup completion – BLOCKS all user stories
- User Stories (Phase 3+): Depend on Foundational; proceed in priority order (P1 → P2 → P3) or in parallel if staffed
- Polish (Final Phase): Depends on desired user stories being complete

### User Story Dependencies

- User Story 1 (P1): Starts after Foundational; no dependency on other stories
- User Story 2 (P2): Starts after Foundational; leverages US1 assembler if available (T010, T011)
- User Story 3 (P3): Starts after Foundational; leverages US1 runner hooks (T009)

### Within Each User Story

- Models/helpers before runner wiring
- Runner wiring before exporters/docs
- Manual test after core implementation

### Parallel Opportunities

- Setup: T001–T003 in parallel
- Foundational: T004–T008 in parallel
- US1: T010–T012 in parallel (after T004–T005); T013–T014 follow
- US2: T015–T017 in parallel; T018–T019 follow
- US3: T020–T022 in parallel; T023 follows
