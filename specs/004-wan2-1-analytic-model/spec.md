# Feature Specification: Wan2.1 Analytic FLOP Model

**Feature Branch**: `[004-wan2-1-analytic-model]`  
**Created**: 2026-01-16  
**Status**: Draft  
**Input**: User description: "we are going to implement context/plans/done/plan-wan2-1-analytic-model.md , read it and create the spec, make sure you understand extern/modelmeter and the pattern of extern/modelmeter/models, we will be extending that, and make sure the flop count can match actual model layer by layer, and end to end, within 5% margin of error, new branch should be named 004-<what>"

## Constitution Constraints *(mandatory)*

- The feature MUST comply with `.specify/memory/constitution.md` (Python-first usability, ruff+mypy gates, Pixi-based execution, tests + evidence, and tmp/ artifacts).
- The spec MUST name the intended test approach (manual/unit/integration) and how success will be verified via `pixi run ...` commands.
- The delivered functionality MUST fit the existing “static analysis” workflow used by this repository (human-inspectable artifacts under `tmp/profile-output/<run_id>/` and automated checks via pytest).
- Verification commands for completion: `pixi run ruff check .`, `pixi run mypy src`, `pixi run pytest tests/unit/`, and `pixi run pytest tests/integration/`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate Wan2.1 analytic report (Priority: P1)

A developer runs a Wan2.1 “static analysis” workflow to generate a structured report that estimates inference-time compute and memory costs for Wan2.1-T2V-14B under a chosen workload (frames, resolution, steps, text length, batch size), using FLOPs (floating-point operations) as the primary compute proxy.

**Why this priority**: This is the minimum useful outcome: it produces an inspectable artifact that lets the team reason about Wan2.1 performance and compare workloads without running the full model.

**Independent Test**: Can be tested by running the analyzer and verifying that it writes a report artifact with totals and a per-layer breakdown (no reference comparison required for this story).

**Acceptance Scenarios**:

1. **Given** the Wan2.1-T2V-14B model metadata is available locally, **When** the developer runs the Wan2.1 static analyzer in the Pixi environment, **Then** a new report is written under `tmp/profile-output/<run_id>/static_analysis/` and includes model metadata, workload parameters, totals, and a per-layer breakdown.
2. **Given** the developer provides workload overrides (for example, changing frames/resolution/steps), **When** they rerun the analyzer, **Then** the report reflects the new workload values and produces different totals consistent with the change (for example, increasing steps increases total compute).

---

### User Story 2 - Verify FLOP accuracy vs a reference (Priority: P2)

A developer (or CI) runs a verification workflow that compares analytic FLOP counts against a reference measurement for the same model layers and workload, and clearly reports pass/fail along with per-layer error.

**Why this priority**: The analytic model is only useful if it stays aligned with the real model; verification prevents silent drift and builds trust in the estimates.

**Independent Test**: Can be tested by running automated tests that produce a reference FLOP table for a fixed small workload and asserting layer-by-layer and end-to-end error are within the defined tolerance.

**Acceptance Scenarios**:

1. **Given** a supported reference measurement path is available, **When** the developer runs `pixi run pytest tests/unit/` and `pixi run pytest tests/integration/`, **Then** tests fail if any compared layer or end-to-end total exceeds the 5% error budget and pass otherwise.
2. **Given** a verification run completes, **When** a developer inspects its output, **Then** it shows per-layer percent error and an end-to-end percent error for the workload(s) under test.

---

### User Story 3 - Compare hotspots across workloads (Priority: P3)

A developer uses the report to understand “what dominates compute” and how costs shift as workload knobs change (frames, resolution, steps), enabling faster iteration on profiling and optimization plans.

**Why this priority**: Hotspot visibility is the reason to invest in layer-by-layer analytics; it informs what to measure, what to optimize, and what to ignore.

**Independent Test**: Can be tested by generating two reports with different workloads and verifying that totals and per-category breakdowns change in the expected direction (monotonic scaling) and remain internally consistent (totals equal sum of components).

**Acceptance Scenarios**:

1. **Given** two workload configurations, **When** the developer generates two reports, **Then** the reports include enough detail to identify the top-k layers/categories by FLOPs and the ordering is stable for identical workloads.
2. **Given** a workload change that increases token count (for example, higher resolution or more frames), **When** the report is generated, **Then** the end-to-end FLOPs increase and the change is attributable to specific layers in the breakdown.

---

### Edge Cases

- Workload parameters are invalid (non-positive frames/steps/batch size, unsupported resolution, non-integer inputs).
- Model metadata is missing or incomplete (for example, missing a required dimension), and the system must fail with a clear, actionable error.
- A user requests a component that is not modeled in v1 (for example, optional submodules); the report must still be produced with explicit “not modeled” disclosure rather than silently omitting costs.
- Numerical overflow/underflow in intermediate calculations for large workloads; the system must avoid producing NaN/Inf in the report (or fail loudly with context).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide an analytic cost model for Wan2.1-T2V-14B that estimates inference-time FLOPs and memory for a workload defined by batch size, frames, height, width, number of inference steps, and text length.
- **FR-002**: System MUST generate a structured report artifact that includes (a) model metadata, (b) workload parameters, (c) per-layer metrics, and (d) end-to-end totals.
- **FR-003**: The report MUST include a stable hierarchical identifier per layer/module to enable “layer by layer” comparisons across runs of the same model version.
- **FR-004**: The system MUST ensure end-to-end totals are the sum of the per-layer values (within floating point tolerance) for each reported metric.
- **FR-005**: System MUST provide a verification workflow that compares analytic FLOP estimates against a reference measurement for the same layers and workload.
- **FR-006**: Verification MUST enforce an absolute percent error budget of ≤ 5% (computed as `abs(analytic - reference) / reference`) for (a) each compared layer and (b) the end-to-end total, for the standard workload set defined by this feature.
- **FR-007**: System MUST validate and document scaling invariants: FLOPs scale linearly with batch size and number of inference steps, and increase monotonically with higher frame count and higher spatial resolution for supported ranges.
- **FR-008**: The analytic model MUST report zero KV-cache memory for diffusion transformer layers unless explicitly justified in the report metadata.
- **FR-009**: The report schema MUST be designed for reuse across models so that Wan2.1 and DeepSeek-OCR can share common report structures without breaking existing DeepSeek-OCR report consumers.
- **FR-010**: The feature MUST include a manual run path that produces human-inspectable artifacts under `tmp/profile-output/<run_id>/` and a deterministic automated test path runnable via `pixi run pytest`.
- **FR-011**: The per-layer breakdown MUST be granular enough to support “layer by layer” verification for the diffusion transformer core (at minimum: per transformer block and major subcomponents such as attention and feed-forward).

### Standard Workload Set

The feature defines a bounded “standard workload set” used for automated verification and success criteria checks.

- **W-001 (CI / tiny)**: batch size 1, 1 inference step, 4 frames, 256×256 resolution, text length 64.
- **W-002 (Representative / 512p)**: batch size 1, 50 inference steps, 16 frames, 512×512 resolution, text length 512.
- **W-003 (Representative / 720p)**: batch size 1, 50 inference steps, 16 frames, 720×1280 resolution, text length 512.

### Key Entities *(include if feature involves data)*

- **Wan2.1 Model Spec**: A versioned description of the model architecture relevant to analytic accounting (for example, layer counts, widths, and operator shapes).
- **Workload Profile**: A set of workload knobs (batch size, frames, resolution, steps, text length) that fully determines the analytic evaluation.
- **Analytic Report**: A structured, machine-readable artifact containing the model spec, workload profile, per-layer metrics, and totals.
- **Layer/Module Node**: A hierarchical node in the report representing one analytic “layer” with its own metrics and children.
- **Metrics Snapshot**: A consistent set of metrics captured per node (for example FLOPs, memory, and I/O) that can be aggregated.
- **Reference Measurement**: The data source used as the “actual” FLOP baseline for verification, captured for the same workload and layer mapping.
- **Verification Result**: Per-layer and end-to-end error metrics plus an overall pass/fail decision for the defined tolerance.

### Assumptions

- v1 targets Wan2.1-T2V-14B and treats the diffusion transformer core as the primary source of layer-by-layer accuracy requirements.
- A reference measurement method for Wan2.1 layers is available in the project environment (for example, via a locally available reference implementation or an equivalent measurement artifact) so the ≤ 5% error budget is objectively testable.
- FLOPs use a single, explicit counting convention that is consistent between analytic estimates and the reference measurement (for example, counting one multiply-add as 2 FLOPs).
- This feature is focused on inference-time accounting; training-time backward pass accuracy is out of scope unless explicitly added later.

### Out of Scope

- Delivering or distributing model weights or datasets as part of this feature.
- Optimizing runtime performance; this feature produces analytic estimates and verification, not speedups.
- Guaranteeing accuracy outside the declared “standard workload set” (those workloads may be expanded later, but v1 will declare a bounded set).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For each workload in the standard workload set, the analytic end-to-end FLOP estimate differs by no more than 5% from the reference measurement.
- **SC-002**: For each compared layer in the standard workload set, the analytic per-layer FLOP estimate differs by no more than 5% from the reference measurement.
- **SC-003**: A developer can generate a Wan2.1 analytic report artifact end-to-end and locate it under `tmp/profile-output/<run_id>/` without manual file editing.
- **SC-004**: For each workload in the standard workload set, the report includes per-layer FLOP values and layer/category labels that allow identifying the top 10 layers and top 5 categories by FLOPs.
