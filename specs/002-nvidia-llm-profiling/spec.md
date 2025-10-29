# Feature Specification: Stage 2 — NVIDIA-Backed Deep LLM Profiling

**Feature Branch**: `002-nvidia-llm-profiling`  
**Created**: 2025-10-29  
**Status**: Draft  
**Input**: User description: "in-depth profiling LLM with nvidia tools, with deepseek-ocr being our current profiling target, but the tools and framework we develop shall be able to applied to other LLMs. check context/plans/plan-deep-profile-llm.md , we already done with stage 1, now we move to stage 2, we will be using nvidia tools to inspect the runtime computational resource utilization, and identify the bottleneck of performance, in order to find out how to optimize it. Stage 1 implementation records are in context/tasks/001-profile-deepseek-ocr/, this is an iterative development, so we shall follow previous framework and add functionalities."

<!-- Constitution Compliance (author must ensure):
  - Public APIs/classes documented with NumPy-style docstrings and examples
  - All functions/classes fully type-annotated (mypy-clean), ruff-clean
  - Runtime environment declared (Pixi preferred; else virtualenv)
  - Manual test plan and file paths for major functionality under tests/manual/
  - Data models use attrs (default) or pydantic (for web schemas), no business logic
-->

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

### User Story 1 - Run Deep Profiling Session (Priority: P1)

As a performance engineer, I want to run a deep profiling session for the current LLM target (DeepSeek-OCR) using hardware-vendor GPU profiling tools so that I can collect kernel-level utilization, memory throughput, and occupancy metrics and pinpoint bottlenecks.

**Why this priority**: Identifying true runtime bottlenecks is the most valuable outcome of Stage 2 and directly informs optimization priorities.

**Independent Test**: A single command triggers a profiling run against a known input set and produces a self-contained artifact set (trace, kernel table, stage timings, stakeholder report) that can be reviewed independently.

<!-- Include at least one manual test script path for major functionality,
     e.g., tests/manual/<feature_area>/test_<name>.py. Automated tests are
     optional unless requested; if included, place under tests/unit/… and
     tests/integration/… -->

**Acceptance Scenarios**:

1. **Given** a configured model target and sample inputs, **When** a deep profiling run is executed, **Then** kernel-level metrics (compute utilization, memory throughput, achieved occupancy, kernel durations, calls) are captured and exported as tables.
2. **Given** the same configuration and inputs, **When** the run is repeated, **Then** outputs are reproducible within expected variance and include a provenance bundle (environment snapshot, inputs manifest, configuration).

---

### User Story 2 - Apply to Other LLMs (Priority: P2)

As an ML engineer, I want to apply the profiling workflow to different LLMs without code changes so that I can compare models and architectures under the same methodology.

**Why this priority**: Cross-model comparability allows broader optimization impact and reuse of Stage 2 workflows beyond the initial target.

**Independent Test**: Point the profiler to a different model path/configuration and generate the same set of artifacts with correct model metadata and stage timing coverage.

**Acceptance Scenarios**:

1. **Given** a different LLM target with valid inputs, **When** a profiling run is executed, **Then** artifacts are produced with accurate model identifiers and stage coverage consistent with the framework’s schema.

---

### User Story 3 - Stakeholder Report with Actionable Tables (Priority: P3)

As a stakeholder, I want a concise report that summarizes environment details, aggregate timings, per-stage timing, top operators and top kernels in tables, plus an executive summary, so that I can quickly see where time is spent and what to fix.

**Why this priority**: Non-technical stakeholders need clear tables and brief summaries to drive decisions.

**Independent Test**: The report renders correctly with at least one table per section and a short narrative highlighting top bottlenecks and recommendations.

**Acceptance Scenarios**:

1. **Given** completed profiling artifacts, **When** the report is generated, **Then** it contains environment, aggregates, per-stage timings, MFU, top operators, and top kernels as tables, followed by a concise narrative summary.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- Profiler tools not present or unsupported driver/runtime versions
- CPU-only environment or incompatible kernels; skip deep GPU mode gracefully
- Excessive trace size or memory pressure during profiling; provide a reduced-detail mode
- Multi-GPU or MIG environments; ensure device selection is explicit
- Asynchronous kernels and 0 ms at higher-level ops; kernel-level attribution must still be captured
- Missing datasets or model weights; provide clear errors and a dry-run validation

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST enable deep runtime profiling sessions that capture kernel-level metrics, including compute utilization, memory throughput, achieved occupancy, kernel wall time, and call counts.
- **FR-002**: The system MUST record per-stage timings using range annotations (e.g., prefill, decode) and present them as a table in the “Aggregates” section; vision processing (sam+clip+projector) MUST be documented as a note, not as a separate stage row.
- **FR-003**: The system MUST export a reproducible artifact set per run: environment snapshot, inputs manifest, configuration used, per-stage timings, operator table (with total, calls, mean ms), and kernel table (with total, calls, mean ms), plus a stakeholder report.
- **FR-004**: The system MUST support targeting different LLMs via configuration without code changes and include model metadata in outputs.
- **FR-005**: The stakeholder report MUST contain at minimum the following tables: Environment, Aggregates, Per-Stage Timings, MFU, Top Operators, Top Kernels; followed by a concise narrative summary.
- **FR-006**: Profiling overhead MUST be bounded and disclosed; default profiling mode MUST complete within an acceptable overhead (target ≤ 25% over non-profiled runs) or clearly instruct users to switch to a lighter mode.
- **FR-007**: The system MUST provide utilization estimates (e.g., MFU) leveraging available static compute estimations and measured timings and document assumptions used.
- **FR-008**: The system MUST handle missing tooling gracefully, emitting actionable guidance and partial artifacts when full profiling cannot run.
- **FR-009**: The system MUST include at least one manual test flow under `tests/manual/stage2_profile/` that guides users through a complete profiling session and verification of artifacts.

### Key Entities *(include if feature involves data)*

- **ProfilingSession**: A single execution context capturing inputs, configuration, environment, timings, and generated artifacts.
- **ModelTarget**: Identifies the profiled model family, variant, parameters, and relevant metadata for cross-run comparison.
- **StageTiming**: Aggregated timing per annotated stage (prefill, decode), including totals, means, and counts.
- **OperatorRecord**: Aggregated operator-level metrics (total time, calls, mean ms), sorted by total time.
- **KernelRecord**: Aggregated kernel-level metrics (name, device, total time, calls, mean ms) used to attribute GPU time.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A profiling session can be executed end-to-end by a single command and produces all required artifacts in under 30 minutes on a standard single-GPU workstation (baseline inputs).
- **SC-002**: Reports include all mandatory tables (Environment, Aggregates, Per-Stage Timings, MFU, Top Operators, Top Kernels) and render correctly in markdown previews 100% of the time.
- **SC-003**: In at least 90% of runs on supported hardware, kernel-level metrics are captured with non-zero device time and clearly identify the top-3 bottlenecks by total time.
- **SC-004**: Re-running the same session with unchanged inputs and configuration reproduces metrics within ±10% variance for totals and means.
- **SC-005**: Cross-model applicability is demonstrated by successfully running the same workflow on at least one additional LLM target with complete artifacts and comparable report structure.

## Assumptions

- Access to a compatible NVIDIA GPU and driver with vendor profiling tools installed on the target machine.
- Stage 1 outputs (NVTX ranges, baseline runtime) exist and can be reused as the foundation for Stage 2.
- Datasets and model checkpoints are locally accessible and valid for the selected LLM target.
- Long, exhaustive traces can be substituted with reduced-detail modes to respect overhead constraints when needed.
