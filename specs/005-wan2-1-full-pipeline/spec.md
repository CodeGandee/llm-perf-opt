# Feature Specification: Wan2.1 Full Pipeline Analytic Model (UMT5 + DiT + Wan-VAE)

**Feature Branch**: `005-wan2-1-full-pipeline`  
**Created**: 2026-01-22  
**Status**: Draft  
**Input**: User description: "Implement context/plans/plan-wan2-1-full-pipeline-analytic-model.md and add FLOP count verifications"

## Constitution Constraints *(mandatory)*

- The feature MUST comply with `.specify/memory/constitution.md` (Python-first usability, ruff+mypy gates, Pixi-based execution, tests + evidence, and tmp/ artifacts).
- The spec MUST name the intended test approach (manual/unit/integration) and how success will be verified via `pixi run ...` commands.
- Intended test approach: fast unit tests for invariants + script-based FLOP verification on small reference shapes; no large weights or long-running workloads required for automated verification.
- Verification commands (expected):
  - `pixi run pytest tests/unit/`
  - `pixi run -e rtx5090 python extern/modelmeter/models/wan2_1/scripts/verify/run_verify_text_encoder.py`
  - `pixi run -e rtx5090 python extern/modelmeter/models/wan2_1/scripts/verify/run_verify_vae_decode.py`
  - `pixi run -e rtx5090 python extern/modelmeter/models/wan2_1/scripts/verify/run_verify_pipeline_invariants.py`

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Full-Pipeline Cost Breakdown (Priority: P1)

As a user doing hardware sizing for Wan2.1 text-to-video (T2V) workloads, I want a single entrypoint that produces an end-to-end cost estimate with a clear per-stage breakdown (text encoder, diffusion model core, VAE decode), so I can understand which stage dominates and size hardware accordingly.

**Why this priority**: Without an end-to-end view, sizing decisions can be incorrect because non-core stages (text encoding and VAE decode) may be non-trivial for certain workloads.

**Independent Test**: Run a single command that generates a report for a default T2V workload and verify the report contains all stages and a total.

**Acceptance Scenarios**:

1. **Given** a valid Wan2.1 T2V workload configuration, **When** I run the full-pipeline cost command, **Then** I receive a report with stage entries for `text_encoder`, `diffusion_core`, and `vae_decode`, plus an end-to-end total.
2. **Given** a workload configuration with overridden resolution/frames/steps, **When** I run the full-pipeline cost command, **Then** the report reflects those overrides and remains internally consistent (totals and subtotals agree).

---

### User Story 2 - FLOP Convention Verification for New Stages (Priority: P2)

As a maintainer, I want verification that the analytic compute-cost estimates (measured in FLOPs, a standard unit of “how much math work”) for the text encoder and VAE decode follow the project’s FLOP conventions on small reference shapes, so changes do not silently drift and comparisons remain meaningful.

**Why this priority**: The end-to-end pipeline is only useful if FLOP accounting is consistent across stages and aligned with the repository’s reference counter conventions.

**Independent Test**: Run verification scripts that compare analytic counts vs a reference counter on small shapes, and confirm they exit successfully and write a small summary artifact under `tmp/`.

**Acceptance Scenarios**:

1. **Given** a small reference text length and batch size, **When** I run the text encoder verification, **Then** the reported relative error for FLOPs is within the allowed tolerance.
2. **Given** a small reference latent shape for VAE decode, **When** I run the VAE decode verification, **Then** the reported relative error for FLOPs is within the allowed tolerance.

---

### User Story 3 - Stage-Aware Sizing Reports (Priority: P3)

As a user preparing a sizing report for stakeholders, I want plots or tabular summaries that show how each stage scales with workload knobs (resolution, frames, diffusion steps), so tradeoffs are easy to communicate.

**Why this priority**: Stage-level scaling clarifies whether optimizations should focus on text encoder, diffusion, or VAE, and helps avoid over-indexing on only one component.

**Independent Test**: Generate a report for a small grid of workloads and verify the output includes separate curves/series per stage and an overall curve/series.

**Acceptance Scenarios**:

1. **Given** two workloads that differ only in resolution (higher vs lower), **When** I generate a stage-aware report, **Then** each stage’s cost is non-decreasing with resolution and the report shows the breakdown.

### Edge Cases

- Missing or unavailable local weight files for metadata extraction.
- Workloads that exceed modeled assumptions (e.g., attention pattern differences) still produce outputs but include an explicit warning that results are extrapolations.
- Extremely large workloads (4k, long videos, large step counts) produce very large numbers without overflowing or crashing report generation.
- Invalid workload overrides (negative sizes, zero steps, etc.) are rejected with a clear error message.

## Requirements *(mandatory)*

### Scope Boundaries

**In scope**:

- End-to-end cost estimates for a Wan2.1 text-to-video workload that include: text encoder, diffusion core, and VAE decode.
- Clear stage breakdowns and an end-to-end total suitable for hardware sizing and bottleneck identification.
- Verification of FLOP accounting conventions for the newly modeled stages on small reference shapes.

**Out of scope (v1)**:

- Exact modeling of diffusion cross-attention when query length differs from key/value length.
- Exact modeling of kernel fusion, overlap, scheduling, or vendor-specific runtime behavior (the model remains a first-order estimator).
- Scheduler overheads and non-model components (e.g., prompt extension) beyond the defined stages.

### Assumptions & Dependencies

**Assumptions**:

- Text encoder runs once per prompt, diffusion core runs per diffusion step, and VAE decode runs once per generated output.
- The FLOP conventions match the repository’s reference counting conventions used in verification.

**Dependencies**:

- Local, non-committed access to the relevant model configuration and weight artifacts for metadata extraction.
- Existing diffusion-core-only analytic functionality remains available and is not regressed by introducing the pipeline model.

### Functional Requirements

- **FR-001**: System MUST provide an end-to-end pipeline cost estimate for a Wan2.1 T2V workload that includes the stages: `text_encoder`, `diffusion_core`, and `vae_decode`.
- **FR-002**: System MUST expose per-stage outputs and an overall total in a single report, including numeric compute cost per stage (FLOPs) and any warnings/assumptions applied.
- **FR-003**: System MUST support workload overrides for at least: resolution, frames, diffusion steps, batch size, and text length.
- **FR-004**: System MUST keep the existing diffusion-core-only analytic path available and unchanged for users who rely on it.
- **FR-005**: System MUST provide verification scripts for FLOP accounting for the text encoder and VAE decode on small reference shapes.
- **FR-006**: System MUST provide an invariant check that the end-to-end total matches the sum of the configured stage totals.
- **FR-007**: System MUST include automated unit tests that validate stage aggregation and monotonic scaling for a small set of representative workloads.
- **FR-008**: System MUST document model limitations and clearly distinguish “modeled” vs “unmodeled” behaviors for the full pipeline.
- **FR-009**: System MUST provide a reproducible way to run verification and tests in the managed environment.

### Acceptance Criteria

- **FR-001 / FR-002**: A single run produces a report that contains `text_encoder`, `diffusion_core`, `vae_decode`, and an end-to-end total (User Story 1, Scenario 1).
- **FR-003**: Changing workload knobs changes the reported stage costs while preserving internal consistency (User Story 1, Scenario 2).
- **FR-004**: Existing diffusion-core-only flows continue to run without behavior changes (validated via existing checks plus a regression test added in this feature).
- **FR-005**: Verification runs for text encoder and VAE decode pass within the defined tolerance (User Story 2, Scenarios 1–2; Success Criteria SC-001 and SC-002).
- **FR-006**: A pipeline invariant check confirms totals equal stage sums and fails loudly on mismatch (Success Criteria SC-003).
- **FR-007**: Unit tests assert stage aggregation and monotonic scaling for representative workloads (Success Criteria SC-003 and SC-004).
- **FR-008**: Documentation includes a section that lists modeled vs unmodeled behaviors and warns when results are extrapolations (Edge Cases and Scope Boundaries).
- **FR-009**: The repository documents and provides commands to run verification and tests in the managed environment (Constitution Constraints).

### Key Entities *(include if feature involves data)*

- **Workload**: A description of the requested T2V generation job (e.g., resolution, frames, diffusion steps, batch size, text length).
- **Stage Cost**: A per-stage estimate (FLOPs and related sizing metrics) for `text_encoder`, `diffusion_core`, and `vae_decode`.
- **Pipeline Report**: A human- and machine-readable artifact that contains stage costs, totals, and warnings/assumptions applied.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Text encoder FLOP verification passes on defined small reference shapes with relative error ≤ 1e-6.
- **SC-002**: VAE decode FLOP verification passes on defined small reference shapes with relative error ≤ 1e-6.
- **SC-003**: For each tested workload, the pipeline total equals the sum of stage totals (no discrepancy) and unit tests enforce this invariant.
- **SC-004**: For each tested scaling knob (resolution, frames, steps), increasing the knob does not decrease the estimated FLOPs for any stage, validated by unit tests.
- **SC-005**: Linting, type checking, and unit tests all pass in the managed environment.
