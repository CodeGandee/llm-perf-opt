---

description: "Task list for Wan2.1 Analytic FLOP Model"
---

# Tasks: Wan2.1 Analytic FLOP Model

**Input**: Design documents from `/data1/huangzhe/code/llm-perf-opt/specs/004-wan2-1-analytic-model/`  
**Prerequisites**: `/data1/huangzhe/code/llm-perf-opt/specs/004-wan2-1-analytic-model/plan.md`, `/data1/huangzhe/code/llm-perf-opt/specs/004-wan2-1-analytic-model/spec.md`, `/data1/huangzhe/code/llm-perf-opt/specs/004-wan2-1-analytic-model/research.md`, `/data1/huangzhe/code/llm-perf-opt/specs/004-wan2-1-analytic-model/data-model.md`, `/data1/huangzhe/code/llm-perf-opt/specs/004-wan2-1-analytic-model/contracts/`

**Tests**: Include test tasks by default. Major behavior MUST have a manual test under `/data1/huangzhe/code/llm-perf-opt/tests/manual/` (prefixed `manual_*.py`), and critical invariants SHOULD have pytest coverage under `/data1/huangzhe/code/llm-perf-opt/tests/unit/` and/or `/data1/huangzhe/code/llm-perf-opt/tests/integration/`.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Implementation Guides

- `context/tasks/working/004-wan2-1-analytic-model/impl-phase-1-setup.md`
- `context/tasks/working/004-wan2-1-analytic-model/impl-phase-2-foundational.md`
- `context/tasks/working/004-wan2-1-analytic-model/impl-phase-3-us1-report.md`
- `context/tasks/working/004-wan2-1-analytic-model/impl-phase-4-us2-verify.md`
- `context/tasks/working/004-wan2-1-analytic-model/impl-phase-5-us3-hotspots.md`
- `context/tasks/working/004-wan2-1-analytic-model/impl-phase-6-polish.md`
- `context/tasks/working/004-wan2-1-analytic-model/impl-integrate-phases.md`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initial scaffolding and local model reference validation

- [ ] T001 Ensure `/data1/huangzhe/code/llm-perf-opt/models/wan2.1-t2v-14b/source-data/config.json` exists by running `/data1/huangzhe/code/llm-perf-opt/models/wan2.1-t2v-14b/bootstrap.sh` (uses `LLM_MODELS_ROOT` or `WAN21_T2V_14B_PATH`)
- [ ] T002 [P] Create Wan2.1 ModelMeter package scaffold in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/__init__.py` and `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/layers/__init__.py`
- [ ] T003 [P] Create Wan2.1 Hydra config scaffold under `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/configs/wan2_1_t2v_14b.yaml` and `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/configs/README.md`
- [ ] T004 [P] Create verification scaffold in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/verify/README.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared report schema + Wan domain models that all user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 Create shared analytic schema module in `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/data/analytic_common.py` (move reusable nodes/categories/metrics helpers from `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/data/deepseek_ocr_analytic.py` without breaking DeepSeek-OCR consumers)
- [ ] T006 Refactor `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/data/deepseek_ocr_analytic.py` to import/re-export shared types from `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/data/analytic_common.py` (keep API compatibility for existing imports)
- [ ] T007 [P] Update DeepSeek-OCR call sites to stay compatible with the refactor in `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/runners/dsocr_analyzer.py` and `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/contracts/models.py`
- [ ] T008 [P] Add unit tests for shared schema validators/serialization in `/data1/huangzhe/code/llm-perf-opt/tests/unit/data/test_analytic_common.py`
- [ ] T009 [P] Add unit tests for DeepSeek-OCR analytic schema backward-compatibility in `/data1/huangzhe/code/llm-perf-opt/tests/unit/data/test_deepseek_ocr_analytic_compat.py`
- [ ] T010 Implement Wan domain models in `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/data/wan2_1_analytic.py` (Wan2.1 model spec + workload profile + report wrapper reusing shared schema types)
- [ ] T011 [P] Implement Wan2.1 contract models in `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/contracts/models.py` (mirror `/data1/huangzhe/code/llm-perf-opt/specs/004-wan2-1-analytic-model/contracts/python-contracts.md`)
- [ ] T012 Add Wan2.1 static-analysis output path helpers to `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/utils/paths.py` (e.g., `tmp/profile-output/<run_id>/static_analysis/wan2_1/`)
- [ ] T013 [P] Add unit tests for Wan2.1 path helpers in `/data1/huangzhe/code/llm-perf-opt/tests/unit/utils/test_paths_wan2_1.py`

**Checkpoint**: Foundation ready – user story implementation can begin

---

## Phase 3: User Story 1 - Generate Wan2.1 analytic report (Priority: P1) 🎯 MVP

**Goal**: Generate a structured report with per-layer metrics and end-to-end totals for Wan2.1-T2V-14B under a chosen workload, written under `tmp/profile-output/<run_id>/static_analysis/wan2_1/`.

**Independent Test**: Run `/data1/huangzhe/code/llm-perf-opt/tests/manual/wan2_1/manual_wan2_1_static_analysis.py` and verify it writes `report.json` with non-negative metrics, stable module ids, and totals equal to the sum of per-layer values.

### Tests for User Story 1 (recommended) ⚠️

- [ ] T014 [P] [US1] Add token-geometry unit tests (frames/resolution/steps scaling invariants) in `/data1/huangzhe/code/llm-perf-opt/tests/unit/wan2_1/test_geometry.py`
- [ ] T015 [P] [US1] Add report-invariants unit tests (totals=sum, stable ids, non-negative finite values) in `/data1/huangzhe/code/llm-perf-opt/tests/unit/wan2_1/test_report_invariants.py`
- [ ] T016 [P] [US1] Add integration test for report generation on the CI tiny workload in `/data1/huangzhe/code/llm-perf-opt/tests/integration/wan2_1/test_wan2_1_analyzer_report.py`
- [ ] T017 [P] [US1] Add manual test script in `/data1/huangzhe/code/llm-perf-opt/tests/manual/wan2_1/manual_wan2_1_static_analysis.py` (writes artifacts to a user-provided `tmp/profile-output/<run_id>/` directory)

### Implementation for User Story 1

- [ ] T018 [P] [US1] Implement video/latent token geometry helper in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/layers/geometry.py` (single canonical helper used by analytic layers and tests)
- [ ] T019 [P] [US1] Implement analytic attention and MLP sublayers in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/layers/transformer/wan2_1_attention.py` and `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/layers/transformer/wan2_1_mlp.py`
- [ ] T020 [P] [US1] Implement Wan transformer block analytic layer with stable child/module ids in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/layers/transformer/wan2_1_transformer_block.py`
- [ ] T021 [P] [US1] Implement diffusion-core analytic layer (stack blocks) and step scaling in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/layers/core/wan2_1_dit_model.py`
- [ ] T022 [P] [US1] Add hf architecture config mirroring `/data1/huangzhe/code/llm-perf-opt/models/wan2.1-t2v-14b/source-data/config.json` to `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/configs/hf/wan2_1_t2v_14b.yaml`
- [ ] T023 [P] [US1] Add runtime/workload and transformer config groups in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/configs/runtime/analytic_defaults.yaml` and `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/configs/transformer/wan2_1_dit.yaml`
- [ ] T024 [US1] Add model root config and top-level config composition in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/configs/model/wan2_1_root.yaml` and `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/configs/wan2_1_t2v_14b.yaml`
- [ ] T025 [US1] Implement the Wan2.1 static analyzer runner in `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/runners/wan2_1_analyzer.py` (compose Hydra config, build module tree, write `report.json` + optional `summary.md`)
- [ ] T026 [P] [US1] Implement contract-oriented CLI wrapper in `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/runners/wan2_1_analyzer_main.py` (parse args to `Wan2_1AnalyticRequest`, call analyzer, print artifact dir)

**Checkpoint**: User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Verify FLOP accuracy vs a reference (Priority: P2)

**Goal**: Provide automated verification that layer-by-layer and end-to-end FLOPs match a reference measurement within ≤5% for the standard workload set.

**Independent Test**: Run `pixi run -e rtx5090 python -m modelmeter.models.wan2_1.scripts.verify.run_verify_layers --workload wan2-1-ci-tiny` and confirm it fails if any compared layer exceeds 5% error.

### Tests for User Story 2 (recommended) ⚠️

- [ ] T027 [P] [US2] Add unit tests for FLOP diff computation and tolerance enforcement in `/data1/huangzhe/code/llm-perf-opt/tests/unit/wan2_1/test_verify_utils.py`
- [ ] T028 [P] [US2] Add integration test that runs CI-tiny verification when available in `/data1/huangzhe/code/llm-perf-opt/tests/integration/wan2_1/test_verify_layers_ci_tiny.py`

### Implementation for User Story 2

- [ ] T029 [P] [US2] Add verification tolerances/config to `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/configs/wan2_1_t2v_14b.yaml` (device + accept_rel_diff defaults aligned to ≤5%)
- [ ] T030 [P] [US2] Implement PyTorch reference modules for FLOP measurement in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/verify/_reference_modules.py`
- [ ] T031 [US2] Implement per-layer verification script in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/verify/run_verify_layers.py` (per-block + subcomponent checks, prints per-layer % error)
- [ ] T032 [US2] Implement end-to-end verification script in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/verify/run_verify_end2end.py` (diffusion core across steps for the standard workload set)
- [ ] T033 [P] [US2] Implement analytic aggregation invariant check in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/verify/run_verify_core.py` (analytic root equals sum of sublayers)
- [ ] T034 [P] [US2] Document verification commands and prerequisites in `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/verify/README.md`

**Checkpoint**: Verification scripts fail on >5% regressions and pass when aligned

---

## Phase 5: User Story 3 - Compare hotspots across workloads (Priority: P3)

**Goal**: Make the report usable for identifying dominant layers/categories and comparing how costs shift across workloads (frames, resolution, steps).

**Independent Test**: Generate two reports for different workloads and confirm the summary lists top-10 layers and top-5 categories by FLOPs for each report, and totals scale monotonically with increased workload.

### Tests for User Story 3 (recommended) ⚠️

- [ ] T035 [P] [US3] Add unit tests for top-k extraction and stable ordering in `/data1/huangzhe/code/llm-perf-opt/tests/unit/wan2_1/test_hotspots.py`
- [ ] T036 [P] [US3] Add integration test for monotonic scaling and hotspot attribution across workloads in `/data1/huangzhe/code/llm-perf-opt/tests/integration/wan2_1/test_hotspot_scaling.py`

### Implementation for User Story 3

- [ ] T037 [US3] Implement summary markdown generator in `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/visualize/wan2_1_analytic_summary.py` and call it from `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/runners/wan2_1_analyzer.py`
- [ ] T038 [P] [US3] Implement report tooling CLI (load report.json, print top-k, compare two reports) in `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/runners/wan2_1_report_tools.py`

**Checkpoint**: Hotspot summaries are generated and usable for side-by-side workload comparison

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, gates, and end-to-end validation across all stories

- [ ] T039 [P] Update repo docs to reference Wan2.1 static analysis and verification in `/data1/huangzhe/code/llm-perf-opt/docs/running.md`
- [ ] T040 Ensure all new public APIs have NumPy-style docstrings and type hints in `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/` and `/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/`
- [ ] T041 Run lint gate `pixi run ruff check .` and fix any issues in changed files under `/data1/huangzhe/code/llm-perf-opt/pyproject.toml`
- [ ] T042 Run type gate `pixi run mypy src` and fix any issues in `/data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/`
- [ ] T043 Run tests `pixi run pytest tests/unit/` and `pixi run pytest tests/integration/` and fix failures in `/data1/huangzhe/code/llm-perf-opt/tests/`
- [ ] T044 Validate `/data1/huangzhe/code/llm-perf-opt/specs/004-wan2-1-analytic-model/quickstart.md` commands and update the document if behavior/paths change

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies – can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion – BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational – delivers the MVP
- **User Story 2 (Phase 4)**: Depends on User Story 1 – verification requires an implemented analytic model
- **User Story 3 (Phase 5)**: Depends on User Story 1 – hotspot tooling requires report generation
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies (graph)

`Phase 1 → Phase 2 → US1 → {US2, US3} → Polish`

### Within Each User Story

- Tests (if included) should be written first and fail before implementation
- Shared helpers before runners/scripts that depend on them
- Story is complete only when its independent test passes

### Parallel Opportunities

- Setup: T002–T004 can run in parallel after T001
- Foundational: T008, T009, T011, T013 can run in parallel after T005–T007
- US1: T014–T017 can run in parallel; T018–T023 can run in parallel once tests exist; T024–T026 follow
- US2: T027–T030 can run in parallel; T031–T034 follow
- US3: T035–T036 can run in parallel; T037–T038 follow

---

## Parallel Example: User Story 1

```bash
Task: "T014 Add token-geometry unit tests in /data1/huangzhe/code/llm-perf-opt/tests/unit/wan2_1/test_geometry.py"
Task: "T015 Add report-invariants unit tests in /data1/huangzhe/code/llm-perf-opt/tests/unit/wan2_1/test_report_invariants.py"
Task: "T018 Implement token geometry helper in /data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/layers/geometry.py"
Task: "T020 Implement transformer block analytic layer in /data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/layers/transformer/wan2_1_transformer_block.py"
```

## Parallel Example: User Story 2

```bash
Task: "T030 Implement reference modules in /data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/verify/_reference_modules.py"
Task: "T033 Implement aggregation invariant check in /data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/verify/run_verify_core.py"
Task: "T034 Document verification commands in /data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/verify/README.md"
```

## Parallel Example: User Story 3

```bash
Task: "T035 Add hotspot unit tests in /data1/huangzhe/code/llm-perf-opt/tests/unit/wan2_1/test_hotspots.py"
Task: "T037 Implement summary generator in /data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/visualize/wan2_1_analytic_summary.py"
Task: "T038 Implement report tooling CLI in /data1/huangzhe/code/llm-perf-opt/src/llm_perf_opt/runners/wan2_1_report_tools.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. STOP and validate with `/data1/huangzhe/code/llm-perf-opt/tests/manual/wan2_1/manual_wan2_1_static_analysis.py`

### Incremental Delivery

1. Setup + Foundational → foundation ready
2. US1 report generation → validate artifacts
3. US2 verification scripts → enforce ≤5% budget
4. US3 hotspot summaries → validate workload comparisons
5. Polish phase → run gates and update docs
