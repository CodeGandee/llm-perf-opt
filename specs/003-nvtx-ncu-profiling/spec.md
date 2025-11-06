# Feature Specification: NVTX-based NCU Regional Profiling

**Feature Branch**: `003-nvtx-ncu-profiling`  
**Created**: 2025-11-06  
**Status**: Draft  
**Input**: User description: "nvtx-based ncu profiling, extend current ncu profiling facility to let to support nvtx markers, so that only those nvtx-marked regions are profiled by ncu, and the reporting should aggregate the results by nvtx ranges. Specifically, given nvtx-marked regions in the code, the ncu profiling should be able to: - report the ncu metrics/sections for each region, including nested regions - report ncu metrics/sections for selected kernels within each region - metrics/sections must be configurable by user, through hydra configs (similar to current ncu profiling facility) Note: - this is an extension of current ncu profiling facility, which already supports kernel-level profiling and reporting, so we must retain the overall structure of current ncu profiling facility and usage pattern."

<!-- Constitution Compliance (author must ensure):
  - Public APIs/classes documented with NumPy-style docstrings and examples
  - All functions/classes fully type-annotated (mypy-clean), ruff-clean
  - Runtime environment declared (Pixi preferred; else virtualenv)
  - Manual test plan and file paths for major functionality under tests/manual/
  - Data models use attrs (default) or pydantic (for web schemas), no business logic
-->

## User Scenarios & Testing *(mandatory)*
## Clarifications

### Session 2025-11-06

- Q: Should parent region totals include child-region activity (inclusive) or show exclusive values only? → A: Inclusive parent totals.
- Q: What kernel selection method should be used within each region for reporting? → A: Name-based patterns (include/exclude).
- Q: How should results be scoped for multi-process/multi-device runs? → A: Both per-scope sections and an aggregate summary.

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

### User Story 1 - Profile NVTX-marked regions only (Priority: P1)

A performance engineer can limit per-kernel profiling to only code regions explicitly marked with profiling ranges, and view a report that aggregates results by those ranges, including visibility into nested ranges.

**Why this priority**: Enables focused, low-noise profiling on business-critical sections, reducing data volume and analysis time while improving signal-to-noise.

**Independent Test**: Can be fully verified by running the profiling workflow on a sample program with 3 distinct marked regions (one nested), and inspecting the generated report to confirm one section per region with the expected metrics and correct inclusion of nested ranges.

Manual test entry point: `tests/manual/ncu/manual_nvtx_regions.py`

**Acceptance Scenarios**:

1. **Given** a program with 3 named ranges (A, B, and A::A1 nested), **When** the user runs the profiling workflow with region-only capture enabled, **Then** the output contains separate report sections for A, A::A1, and B with configured metrics populated for each section.
2. **Given** overlapping or back-to-back ranges, **When** profiling is executed, **Then** only kernels occurring within the active range are attributed to that range and no kernels outside ranges appear in region reports.

---

### User Story 2 - Select kernels within each region (Priority: P2)

A performance engineer can constrain which kernels are summarized within each region using name-based include/exclude patterns to focus the report on the most relevant operations.

**Why this priority**: Large regions can contain many kernels; selection makes reports actionable and concise.

**Independent Test**: Run the profiling on a sample with 20+ kernel invocations per region; verify that applying include/exclude patterns reduces the report to the intended subset per region while keeping totals consistent with the subset definition.

**Acceptance Scenarios**:

1. **Given** include/exclude patterns, **When** profiling is executed, **Then** the per-region kernel table includes only kernels that match the include rules minus any excluded kernels, and the reported metrics correspond to that subset.

---

### User Story 3 - Configure reported metrics/sections (Priority: P3)

A performance engineer can choose which metric groups/sections appear in the per-region report using the project’s configuration system, leveraging existing configuration patterns used elsewhere in the profiling workflow.

**Why this priority**: Different analyses require different metrics; configurability avoids code changes and supports reproducible workflows.

**Independent Test**: Change the configuration to include/exclude specific metric groups; run profiling and confirm the report reflects the selected sections only.

**Acceptance Scenarios**:

1. **Given** a configuration specifying a subset of metric groups, **When** profiling is executed, **Then** only those groups appear for each region and kernels.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right edge cases.
-->

- What happens when [boundary condition]?
- How does system handle [error scenario]?

- No ranges present: profiling runs and produces a clear message and a baseline report without region sections.
- Nested ranges: parent vs. child attribution and roll-up semantics are consistent and documented. Parent totals are inclusive by default (parents include child-region activity), and exclusive values may be derived where useful.
- Overlapping or improperly nested ranges: deterministic attribution order and conflict resolution are defined in documentation and applied consistently.
- Very large numbers of ranges: reporting remains usable (pagination or truncation rules) and run completes without excessive overhead.
- Empty regions (no matching kernels): sections appear with zeroed/empty metrics and an explicit note.
- Kernel selection rule matches nothing: per-region kernel table renders with an informative note and does not fail the run.
- Multi-process or multi-device runs: provide per-process/per-device sections and an aggregate summary; the aggregate can be toggled via configuration.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: The profiling workflow MUST restrict per-kernel collection and reporting to code regions explicitly marked by profiling ranges.
- **FR-002**: The report MUST include a distinct section for each marked region encountered during execution, including nested regions as separate sections.
- **FR-003**: For each region section, the system MUST present configured metric groups/sections and summarize results for that region’s kernels.
- **FR-004**: The system MUST support kernel selection within each region via name-based include/exclude pattern rules and reflect the selection in the report. If both include and exclude rules are provided, exclude rules take precedence. If no patterns are provided, include all kernels by default.
- **FR-005**: Region and kernel names MUST be preserved in the report to enable cross-referencing with the program’s instrumentation.
- **FR-006**: When no regions are present, the system MUST complete successfully and clearly state that no region-specific profiling was performed.
- **FR-007**: The configuration system MUST allow users to select which metric groups/sections are included in the report without code changes.
- **FR-008**: The report MUST indicate nesting (e.g., parent/child relationships) and use inclusive parent totals by default (parent includes child regions); where applicable, the report MAY also present exclusive values for clarity.
- **FR-009**: The system MUST provide deterministic attribution when ranges overlap or are improperly nested and document the applied rules.
- **FR-010**: The workflow MUST maintain the existing usage pattern and entry points of the current kernel-level profiling feature (no breaking changes to how users run it).
- **FR-011**: The system MUST generate machine-readable outputs for region-level results in addition to any human-readable report to enable downstream analysis.
- **FR-012**: The system MUST provide both per-process/per-device reporting and an aggregate summary. Per-scope sections MUST always be generated; the aggregate summary is enabled by default and can be disabled via configuration.

<!-- Add language and quality gates where appropriate, e.g.,
  - Code MUST be type-annotated and pass mypy
  - Code MUST pass ruff linting/formatting
  - Public APIs MUST include NumPy-style docstrings and examples
-->

 

Quality gates:

- Code MUST be type-annotated and pass static type checks and linting required by the repository.
- Public-facing commands or configuration MUST be documented in the project docs and examples updated accordingly.

### Key Entities *(include if feature involves data)*

- **NCUProfileRegion**: A named range in the program’s execution with start/end, optional parent (nesting), and attributes used for attribution and display.
- **KernelRecord**: Reused kernel aggregation model (identifier, calls, timing metrics) from `llm_perf_opt.data.models`.
- **NCUProfileRegionReport**: Aggregated metrics and selected kernel summaries for a single region, including indicators of nesting and any scope qualifiers (e.g., process/device).

<!-- Data model guidance:
  - Prefer attrs (@define kw_only=True); use pydantic for web schemas
  - Keep business logic out of data models; use services/helpers for behavior
-->

### Assumptions

- The project’s existing configuration system is used to declare metric groups/sections and selection rules; defaults mirror current profiling defaults to avoid breaking changes.
- Existing profiling entry points and outputs remain; this feature adds region-scoped views and machine-readable artifacts alongside existing kernel-level outputs.
- Users have instrumented their code with profiling ranges where they want targeted analysis.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: When run on a program with at least 3 marked regions (including one nested), the generated report contains distinct sections for each region with the configured metric groups populated; no kernels outside marked regions appear in those sections.
- **SC-002**: Changing the configuration to include/exclude specific metric groups changes the report content accordingly on the next run, with 100% alignment between configuration and rendered sections.
- **SC-003**: Applying a kernel selection rule reduces the per-region kernel listings to the intended subset and the reported metrics reflect only that subset, confirmed across at least two distinct pattern strategies (include-only and include+exclude) in the manual test.
- **SC-004**: Nested-region attribution uses inclusive semantics (parent includes child regions), and totals reconcile within ±1% of the sum of constituent parts in the sample run.
- **SC-005**: In a multi-process/multi-device run, the report includes per-scope sections and an aggregate summary by default; disabling the aggregate via configuration removes only the summary while retaining per-scope sections.
