---

description: "Task list for DeepSeek-OCR analytic modeling feature"
---

# Tasks: DeepSeek-OCR Analytic Modeling in ModelMeter

**Input**: Design documents from `/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/`  
**Prerequisites**: `plan.md` (required), `spec.md` (required for user stories), `research.md`, `data-model.md`, `contracts/`

**Tests**: Per the constitution, provide manual tests for major functionality
under `tests/manual/…`. Automated tests (unit/integration) are OPTIONAL and only
included where explicitly requested in the spec or for critical paths.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Single project with Python package in `src/` and tests under `tests/`
- Static DeepSeek-OCR artifacts under `reports/20211117-dsorc-op-analysis/static-20251118-130533/`
- Analytic artifacts under `tmp/profile-output/<run_id>/static_analysis/analytic_model/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Environment and static-analysis prerequisites for all user stories

- [X] T001 [P] Verify Pixi `rtx5090` Python 3.11 environment and required analytic dependencies (`torch`, `attrs`, `cattrs`, `omegaconf`, `modelmeter`) are present in `pyproject.toml`
- [X] T002 [P] Confirm DeepSeek-OCR TorchInfo static artifacts exist or regenerate them using `scripts/analytical/dsocr_find_static_components.py` with output directory `reports/20211117-dsorc-op-analysis/static-20251118-130533`
- [X] T003 Review and adjust `specs/001-deepseek-ocr-modelmeter/quickstart.md` to ensure setup and CLI examples reference the analytic mode and artifact paths under `tmp/profile-output/<run_id>/static_analysis/analytic_model/`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core analytic data models, layer package skeletons, and path helpers required by all user stories

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 [P] Create DeepSeek-OCR analytic layer package structure under `extern/modelmeter/models/deepseek_ocr/layers/` (add `__init__.py`, `core/`, `vision/`, `decoder/`, `llama/` subpackages and stub `BaseLayer` subclasses for each file proposed in `specs/001-deepseek-ocr-modelmeter/plan.md`)
- [X] T005 [P] Implement core analytic domain models `DeepSeekOCRModelSpec`, `OCRWorkloadProfile`, and `AnalyticModelReport` in `src/llm_perf_opt/data/deepseek_ocr_analytic.py` and export them from `src/llm_perf_opt/data/__init__.py` according to `specs/001-deepseek-ocr-modelmeter/data-model.md`
- [X] T006 [P] Implement supporting analytic domain models (`AnalyticModuleNode`, `OperatorCategory`, `ModuleMetricsSnapshot`, `OperatorMetrics`, `TargetOperatorList`, `OperatorSpec`) in `src/llm_perf_opt/data/deepseek_ocr_analytic.py` with attrs-based validation
- [X] T007 [P] Add analytic artifact path helpers to `src/llm_perf_opt/utils/paths.py` for `AnalyticModelReport` outputs and `layer_docs_dir`, ensuring all returned paths are absolute within `tmp/profile-output/<run_id>/static_analysis/analytic_model/`
- [X] T008 [P] Implement a `load_target_operator_list` helper in `src/llm_perf_opt/utils/dsocr_callgraph_parse.py` that parses `reports/20211117-dsorc-op-analysis/static-20251118-130533/torchinfo-unique-layers.{json,md}` into a populated `TargetOperatorList`

**Checkpoint**: Foundational models, layer skeletons, and helper utilities are ready; user story implementation can now begin

---

## Phase 3: User Story 1 - Vision analytic layers (Priority: P1) 🎯 MVP (part 1)

**Goal**: Implement analytic `BaseLayer` subclasses for all DeepSeek-OCR vision encoder/projector modules under `extern/modelmeter/models/deepseek_ocr/layers/vision/`, so we can compute theoretical FLOPs/I/O/memory for the vision stack.

**Independent Test**: For a fixed synthetic workload (e.g., `dsocr-standard-v1` image parameters), construct each vision analytic layer with representative shapes and verify that forward/backward FLOPs and memory metrics are non-negative and scale correctly with sequence length, channels, and image size.

### Implementation for User Story 1 – Vision

- [X] T009 [P] [US1] Implement `Attention(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/attention.py` using TorchInfo shapes and call counts from `reports/20211117-dsorc-op-analysis/static-20251118-130533/torchinfo-unique-layers.json` (docs: context/hints/dsocr-kb/ops/op-Attention.md)
- [X] T010 [P] [US1] Implement `Block(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/block.py` to aggregate attention and MLP vision costs (docs: context/hints/dsocr-kb/ops/op-Block.md)
- [X] T011 [P] [US1] Implement `CLIPVisionEmbeddings(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/clip_vision_embeddings.py` modeling patch + positional embeddings (docs: context/hints/dsocr-kb/ops/op-CLIPVisionEmbeddings.md)
- [X] T012 [P] [US1] Implement `ImageEncoderViT(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/image_encoder_vit.py` as the main ViT vision encoder analytic model (docs: context/hints/dsocr-kb/ops/op-ImageEncoderViT.md)
- [X] T013 [P] [US1] Implement `LayerNorm2d(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/layer_norm2d.py` including forward/backward FLOPs and memory (docs: context/hints/dsocr-kb/ops/op-LayerNorm2d.md)
- [X] T014 [P] [US1] Implement `MLPBlock(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/mlp_block.py` for the vision MLP sub-block FLOPs/I/O (docs: context/hints/dsocr-kb/ops/op-MLPBlock.md)
- [X] T015 [P] [US1] Implement `MlpProjector(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/mlp_projector.py` to capture the projection from vision to decoder input space (docs: context/hints/dsocr-kb/ops/op-MlpProjector.md)
- [X] T016 [P] [US1] Implement `NoTPAttention(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/notp_attention.py` with attention FLOPs/I/O formulas (docs: context/hints/dsocr-kb/ops/op-NoTPAttention.md)
- [X] T017 [P] [US1] Implement `NoTPFeedForward(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/notp_feedforward.py` modeling the NoTP MLP stack (docs: context/hints/dsocr-kb/ops/op-NoTPFeedForward.md)
- [X] T018 [P] [US1] Implement `NoTPTransformer(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/notp_transformer.py` aggregating per-block NoTP costs (docs: context/hints/dsocr-kb/ops/op-NoTPTransformer.md)
- [X] T019 [P] [US1] Implement `NoTPTransformerBlock(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/notp_transformer_block.py` as a single NoTP block analytic layer (docs: context/hints/dsocr-kb/ops/op-NoTPTransformerBlock.md)
- [X] T020 [P] [US1] Implement `PatchEmbed(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/patch_embed.py` capturing patch projection FLOPs/I/O (docs: context/hints/dsocr-kb/ops/op-PatchEmbed.md)
- [X] T021 [P] [US1] Implement `VitModel(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/vision/vit_model.py` aggregating encoder blocks into a full vision backbone analytic model (docs: context/hints/dsocr-kb/ops/op-VitModel.md)

**Checkpoint**: Vision analytic layers are implemented and sanity-checked in isolation; LLaMA, decoder, and core aggregation remain to be implemented for the full User Story 1 flow.

---

## Phase 4: User Story 1 - LLaMA analytic layers (Priority: P1) 🎯 MVP (part 2)

**Goal**: Implement analytic `BaseLayer` subclasses for the LLaMA attention primitives used inside DeepSeek-OCR so they can be referenced consistently by decoder and core analytic layers.

**Independent Test**: For representative sequence length and head configuration, construct the LLaMA analytic layers and verify that FLOPs and memory metrics are non-negative and scale with sequence length and number of heads.

### Implementation for User Story 1 – LLaMA

- [X] T022 [P] [US1] Implement `LlamaFlashAttention2(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/llama/llama_flash_attention2.py`, reusing generic FlashAttention math where possible (docs: context/hints/dsocr-kb/ops/op-LlamaFlashAttention2.md)
- [X] T023 [P] [US1] Implement `LlamaRotaryEmbedding(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/llama/llama_rotary_embedding.py`, modeling parameter and activation memory plus any compute overhead (docs: context/hints/dsocr-kb/ops/op-LlamaRotaryEmbedding.md)

**Checkpoint**: LLaMA analytic layers are implemented and ready to be composed into DeepSeek-OCR decoder and core models.

---

## Phase 5: User Story 1 - Decoder analytic layers (Priority: P1) 🎯 MVP (part 3)

**Goal**: Implement analytic `BaseLayer` subclasses for the DeepSeek-V2 decoder stack and MoE components so we can attribute FLOPs/I/O/memory across decoder layers.

**Independent Test**: For a fixed `dsocr-standard-v1` workload profile, construct decoder analytic layers with representative hidden size, intermediate size, and MoE configuration and confirm that FLOPs and memory metrics are non-negative and increase with layer width/depth and expert counts.

### Implementation for User Story 1 – Decoder

- [X] T024 [P] [US1] Implement `DeepseekV2DecoderLayer(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/decoder/deepseek_v2_decoder_layer.py` aggregating attention + MLP costs (docs: context/hints/dsocr-kb/ops/op-DeepseekV2DecoderLayer.md)
- [X] T025 [P] [US1] Implement `DeepseekV2MLP(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/decoder/deepseek_v2_mlp.py` with analytic FLOPs/I/O formulas for the decoder MLP (docs: context/hints/dsocr-kb/ops/op-DeepseekV2MLP.md)
- [X] T026 [P] [US1] Implement `DeepseekV2MoE(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/decoder/deepseek_v2_moe.py` modeling active experts, gating, and expert FLOPs/I/O (docs: context/hints/dsocr-kb/ops/op-DeepseekV2MoE.md)
- [X] T027 [P] [US1] Implement `DeepseekV2RMSNorm(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/decoder/deepseek_v2_rms_norm.py` including forward/backward FLOPs and memory (docs: context/hints/dsocr-kb/ops/op-DeepseekV2RMSNorm.md)
- [X] T028 [P] [US1] Implement `MoEGate(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/decoder/moe_gate.py` capturing gating compute and associated I/O (docs: context/hints/dsocr-kb/ops/op-MoEGate.md)

**Checkpoint**: Decoder analytic layers are implemented, enabling full DeepSeek-OCR model-level aggregation in the next phase.

---

## Phase 6: User Story 1 - Core aggregation & analytic pipeline (Priority: P1) 🎯 MVP (part 4)

**Goal**: Wire the vision, LLaMA, and decoder analytic layers into a root `DeepseekOCRModel` aggregator, expose an analytic mode in `DeepseekOCRStaticAnalyzer`, and generate both structured JSON/YAML artifacts and Markdown layer documentation for DeepSeek-OCR.

**Independent Test**: Run the analytic measurement tool with DeepSeek-OCR and the standard OCR workload and confirm that (a) structured analysis data (JSON/YAML) is emitted with per-component analytic time and memory estimates, and (b) human-readable Markdown documentation is generated explaining each analyzed layer/operator, its definition, and how and why the formulas are applied.

### Tests for User Story 1 (Core)

- [X] T029 [P] [US1] Create manual performance-report script `tests/manual/deepseek_ocr/manual_deepseek_ocr_performance_report.py` that runs `python -m llm_perf_opt.runners.dsocr_analyzer --mode analytic` and checks for JSON/YAML and Markdown outputs for a `dsocr-standard-v1` workload under `tmp/profile-output/<run_id>/static_analysis/analytic_model/`
- [X] T030 [P] [US1] Add unit tests in `tests/unit/deepseek_ocr/test_analytic_layers_scaling.py` to validate that selected DeepSeek analytic layers in `extern/modelmeter/models/deepseek_ocr/layers/` produce non-negative metrics and scale FLOPs/I/O monotonically with sequence length and hidden size

### Implementation for User Story 1 – Core & Pipeline

- [X] T031 [US1] Implement a root `DeepseekOCRModel(BaseLayer)` aggregator in `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py` that composes vision, decoder, and LLaMA layers to compute per-module FLOPs, I/O, and memory metrics (docs: context/hints/dsocr-kb/ops/op-DeepseekOCRModel.md)
- [X] T032 [US1] Extend `DeepseekOCRStaticAnalyzer` in `src/llm_perf_opt/runners/dsocr_analyzer.py` with an analytic code path that instantiates ModelMeter layers, builds an `AnalyticModelReport`, and writes structured artifacts via helpers in `src/llm_perf_opt/utils/paths.py`
- [X] T033 [US1] Implement Markdown generation utilities in `src/llm_perf_opt/visualize/analytic_layers.py` that render per-layer and summary docs from `AnalyticModelReport` into the `layer_docs_dir` directory
- [X] T034 [US1] Update `extern/modelmeter/models/deepseek_ocr/README.md` and `specs/001-deepseek-ocr-modelmeter/quickstart.md` to include the analytic mode CLI invocation, expected JSON/YAML outputs, and pointers to the generated Markdown layer docs

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently (analytic CLI + JSON/YAML + Markdown docs for DeepSeek-OCR)

---

## Phase 7: User Story 2 - Build analytic models for planning (Priority: P2)

**Goal**: Provide a machine-readable breakdown of DeepSeek-OCR down to underlying operation types, including module hierarchy, call counts, and basic memory estimates, so capacity-planning engineers can plug the data into separate analytic tools and run “what-if” scenarios.

**Independent Test**: Export a structured representation of DeepSeek-OCR’s modules, operation counts, and call relationships from the analytic tool and verify it can be consumed by a separate modeling script or spreadsheet without manual edits or runtime traces.

### Tests for User Story 2

- [ ] T035 [P] [US2] Implement manual analytic-model export script `tests/manual/deepseek_ocr/manual_deepseek_ocr_model_export.py` that runs the analytic CLI and writes a standalone `AnalyticModelReport` JSON/YAML file suitable for downstream planning tools
- [ ] T036 [P] [US2] Add unit tests in `tests/unit/deepseek_ocr/test_analytic_model_report_io.py` to verify round-trip JSON/YAML serialization of `AnalyticModelReport` in `src/llm_perf_opt/data/deepseek_ocr_analytic.py`

### Implementation for User Story 2

- [ ] T037 [P] [US2] Implement operator category mapping logic in `src/llm_perf_opt/data/deepseek_ocr_analytic.py` that assigns TorchInfo `class_name_qualified` values from `TargetOperatorList` to `OperatorCategory` ids for use in capacity-planning exports
- [ ] T038 [P] [US2] Extend the analytic aggregation pipeline in `src/llm_perf_opt/runners/dsocr_analyzer.py` to populate `ModuleMetricsSnapshot` and `OperatorMetrics` records for each module when building an `AnalyticModelReport`
- [ ] T039 [P] [US2] Add DeepSeek-OCR analytic contract models (request, accepted, summary, full model) to `src/llm_perf_opt/contracts/models.py` and register `cattrs` hooks in `src/llm_perf_opt/contracts/convert.py` per `specs/001-deepseek-ocr-modelmeter/contracts/python-contracts.md`
- [ ] T040 [US2] Implement a small CLI wrapper `src/llm_perf_opt/runners/dsocr_analyzer_main.py` (or `__main__` handler in `src/llm_perf_opt/runners/dsocr_analyzer.py`) that accepts `DeepSeekOCRAnalyticRequest` fields and dispatches analytic runs, returning the `report_id` and artifacts directory
- [ ] T041 [P] [US2] Create an example “what-if” planning script `scripts/analytical/dsocr_analytic_what_if.py` that loads an `AnalyticModelReport` JSON from `tmp/profile-output/<run_id>/static_analysis/analytic_model/` and recomputes projected FLOPs/time for alternative batch sizes or device assumptions
- [ ] T042 [US2] Validate that implemented analytic outputs match field names and shapes described in `specs/001-deepseek-ocr-modelmeter/contracts/openapi.yaml` and `specs/001-deepseek-ocr-modelmeter/contracts/MAPPING.md`, updating docs or code where discrepancies arise

**Checkpoint**: At this point, User Stories 1 and 2 should both work independently (analytic CLI + exportable analytic model files for planning)

---

## Phase 8: User Story 3 - Reuse DeepSeek-OCR definitions (Priority: P3)

**Goal**: Ensure DeepSeek-OCR is described using the same abstractions and naming as other models in the measurement system so its analytic outputs can be reused in existing comparison and reporting workflows without DeepSeek-specific glue code.

**Independent Test**: Review the DeepSeek-OCR analytic model and verify that its module and operation abstractions align with existing measurement concepts and can be plugged into other workflows (e.g., dashboards or comparison reports) without custom handling.

### Tests for User Story 3

- [ ] T043 [P] [US3] Implement manual reuse-workflows script `tests/manual/deepseek_ocr/manual_deepseek_ocr_reuse_workflows.py` that loads existing reporting utilities (e.g., `src/llm_perf_opt/profiling/export.py`) and verifies DeepSeek-OCR analytic reports can be displayed alongside other models without DeepSeek-specific branching

### Implementation for User Story 3

- [ ] T044 [P] [US3] Align stage and operator naming in `src/llm_perf_opt/data/deepseek_ocr_analytic.py` and `AnalyticModelReport` with existing profiling concepts in `src/llm_perf_opt/data/models.py` and `src/llm_perf_opt/profiling/aggregate.py` so DeepSeek-OCR fits existing dashboards
- [ ] T045 [P] [US3] Adapt existing summary/report generation utilities in `src/llm_perf_opt/profiling/export.py` and `src/llm_perf_opt/visualize/annotations.py` to optionally consume `AnalyticModelReport` and `DeepSeekOCRAnalyticReportSummary` without DeepSeek-specific conditionals
- [ ] T046 [US3] Add documentation page `docs/deepseek_ocr_analytic_reuse.md` describing how DeepSeek-OCR analytic abstractions map to the generic measurement model and how other teams can plug analytic outputs into their workflows

**Checkpoint**: All three user stories should now be independently functional and usable in existing measurement workflows

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final documentation, testing depth, and quality gates across all user stories

- [ ] T047 [P] Add rich docstrings and type hints for all new analytic models and runners in `extern/modelmeter/models/deepseek_ocr/layers/` and `src/llm_perf_opt/runners/dsocr_analyzer.py` to satisfy repository style and `mypy` expectations
- [ ] T048 [P] Add additional edge-case unit tests in `tests/unit/deepseek_ocr/test_analytic_edge_cases.py` covering extreme sequence lengths, image sizes, and MoE configurations for selected analytic layers
- [ ] T049 Run `ruff` and `mypy` over updated modules in `extern/modelmeter/` and `src/llm_perf_opt/` and fix any style or typing issues before merging
- [ ] T050 [P] Update top-level documentation (for example, `docs/index.md` or `README.md`) to link to the DeepSeek-OCR analytic modeling quickstart and reference the generated Markdown layer docs

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **User Stories (Phases 3–8)**: All depend on Foundational phase completion
  - User Story 1 (US1) should be implemented first as the MVP (Phases 3–6: vision → LLaMA → decoder → core)
  - User Story 2 (US2) builds on US1’s analytic pipeline and artifacts (Phase 7)
  - User Story 3 (US3) builds on US1+US2 and focuses on reuse/integration (Phase 8)
- **Polish (Phase 9)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Phase 2; provides the core analytic CLI, ModelMeter layers, and JSON/YAML + Markdown outputs
- **User Story 2 (P2)**: Depends on US1; extends the analytic model into exportable planning artifacts and contract-aligned structures
- **User Story 3 (P3)**: Depends on US2; focuses on aligning abstractions and plugging DeepSeek-OCR analytic outputs into existing workflows

### Within Each User Story

- Tests (where included) should be written and exercised before core implementation tasks
- Models and data structures before services/runners
- Runners and scripts before integration with external tools or dashboards
- Story complete (tests + implementation) before moving to the next priority story

---

## Parallel Opportunities

- All Setup tasks marked `[P]` (T001–T002) can run in parallel
- Foundational tasks T004–T008 are parallelizable across different files once the repo is checked out and environment is ready
- Within **User Story 1**, layer implementations for vision (T009–T021), LLaMA (T022–T023), and decoder (T024–T028), plus core tests and wiring (T029–T034), can be developed in parallel where dependencies allow
- Within **User Story 2**, export-related tasks (T037–T038, T041) and contract wiring (T039–T040) can proceed in parallel after foundational models exist
- Within **User Story 3**, alignment work (T044–T045) and the reuse manual script (T043) can run in parallel
- Polish tasks marked `[P]` (T047, T048, T050) can be executed in parallel once core implementations are stable

---

## Implementation Strategy (MVP First, Incremental Delivery)

- **MVP Scope**: Complete Phases 1–2 and all User Story 1 tasks (T001–T034) to deliver a working analytic CLI that produces JSON/YAML reports and Markdown docs for DeepSeek-OCR.
- **Next Increment**: Implement User Story 2 (T035–T042) to expose a richer analytic model suitable for capacity planning and “what-if” analysis via scripts and contracts.
- **Final Increment**: Implement User Story 3 (T043–T046) to align abstractions with existing measurement workflows and ensure DeepSeek-OCR analytic outputs can be reused without special-casing.
- **Hardening**: Finish Phase 9 tasks (T047–T050) to raise documentation quality, test coverage, and typing/linting hygiene to repository standards.
