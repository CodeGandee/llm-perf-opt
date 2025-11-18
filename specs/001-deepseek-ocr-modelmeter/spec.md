# Feature Specification: DeepSeek-OCR Analytic Modeling in ModelMeter

**Feature Branch**: `[001-deepseek-ocr-modelmeter]`  
**Created**: 2025-11-17  
**Status**: Draft  
**Input**: User description: "analytically analyze the deepseek ocr model, and implement modelmeter layers for it - Figure out the main model components (modules) used in DeepSeek-OCR, its call relationships, calling counts, and per-module operator breakdowns. - The module-level analysis goes down until we reach pytorch builtin operators (e.g., `torch.nn.Conv2d`, `torch.nn.LayerNorm`, etc.) or well-known custom layers (e.g., FlashAttention). - This information will be used to build accurate analytic performance and memory models, which will be implemented under `extern/modelmeter/models/deepseek_ocr/`, according to contracts given in `extern/modelmeter/layers/base.py` - `context/hints/dsocr-kb/about-dynamic-tracing-deepseek-ocr.md`, approaches to dynamically trace DeepSeek-OCR model execution, we prefer the recommended approach metioned there. note that this is the 004 feature"

<!-- Constitution Compliance (author must ensure):
  - Public APIs/classes documented with NumPy-style docstrings and examples
  - All functions/classes fully type-annotated (mypy-clean), ruff-clean
  - Runtime environment declared (Pixi preferred; else virtualenv)
  - Manual test plan and file paths for major functionality under tests/manual/
  - Data models use attrs (default) or pydantic (for web schemas), no business logic
-->

## Clarifications

### Session 2025-11-17

- Q: What granularity should DeepSeek-OCR modules use in the analytic model? → A: Medium-granularity modules with leaf nodes at framework built-in layers and no deeper analysis.

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

### User Story 1 - Understand DeepSeek-OCR performance profile (Priority: P1)

An internal performance engineer wants to generate a performance and memory report for the DeepSeek-OCR model so they can see where time and memory are spent across the major components of the model and identify optimization opportunities.

**Why this priority**: Without a clear breakdown of how DeepSeek-OCR spends time and memory, it is difficult to plan optimizations, hardware budgets, and service-level expectations; this report is the primary value of the feature.

**Independent Test**: This story is independently testable by selecting DeepSeek-OCR in the measurement tool, running a standard analysis, and confirming that a structured report appears showing per-component time and memory usage without requiring any other new features.

Manual test script path (planned): `tests/manual/deepseek_ocr/manual_deepseek_ocr_performance_report.py`

**Acceptance Scenarios**:

1. **Given** the measurement tool is configured with DeepSeek-OCR and a standard OCR workload, **When** the performance engineer requests a performance report, **Then** the tool produces a human-readable summary of runtime and memory usage for each major DeepSeek-OCR component (for example, visual encoder, sequence model, and output head).
2. **Given** the same configuration and workload, **When** the engineer runs the analysis multiple times, **Then** the top-level per-component metrics are stable within an agreed variance range so they can be used for planning.

---

### User Story 2 - Build analytic models for planning (Priority: P2)

An internal modeling or capacity-planning engineer wants to access a machine-readable breakdown of DeepSeek-OCR down to underlying operation types so they can plug those numbers into analytic performance and memory models and run “what-if” scenarios (for example, different batch sizes or hardware targets).

**Why this priority**: Accurate analytic models depend on detailed counts of lower-level operations and their grouping into meaningful modules; without this, any capacity or cost estimates for DeepSeek-OCR are unreliable.

**Independent Test**: This story is independently testable by exporting a structured representation of DeepSeek-OCR’s modules, operation counts, and call relationships from the measurement tool and verifying it can be consumed by a separate modeling script or spreadsheet without manual editing.

Manual test script path (planned): `tests/manual/deepseek_ocr/manual_deepseek_ocr_model_export.py`

**Acceptance Scenarios**:

1. **Given** DeepSeek-OCR is selected and the standard workload is defined, **When** the modeling engineer exports the analytic model, **Then** they receive a structured artifact that includes module hierarchy, per-module operation counts, call counts, and basic memory estimates suitable for downstream modeling.
2. **Given** the exported data for DeepSeek-OCR, **When** the engineer loads it into a separate analytic tool, **Then** they can compute total projected runtime and memory use without needing additional information about the implementation.

---

### User Story 3 - Reuse DeepSeek-OCR definitions (Priority: P3)

An internal ML or tools engineer wants DeepSeek-OCR to be described using the same abstractions and naming as other models in the measurement system so they can reuse components, compare models, and extend the system with minimal extra work.

**Why this priority**: Consistent abstractions make it easier to maintain the measurement system, onboard new models, and share insights across teams; it also reduces the chance of DeepSeek-OCR becoming a one-off integration.

**Independent Test**: This story is independently testable by reviewing the DeepSeek-OCR analytic model and confirming that its module and operation abstractions align with existing measurement concepts and can be plugged into other workflows (for example, dashboards or comparison reports) without custom handling.

Manual test script path (planned): `tests/manual/deepseek_ocr/manual_deepseek_ocr_reuse_workflows.py`

**Acceptance Scenarios**:

1. **Given** DeepSeek-OCR analytic data has been generated, **When** an engineer uses existing comparison or reporting workflows that already support other models, **Then** DeepSeek-OCR appears alongside those models without needing DeepSeek-OCR-specific code changes in those workflows.
2. **Given** an engineer inspects the DeepSeek-OCR model definition, **When** they compare its structure to other supported models, **Then** they find consistent naming, module granularity, and operation categories that follow the same conventions.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- Very long or complex documents (for example, multi-page scans with dense text and graphics) may significantly change call counts and memory usage; the analysis should either remain valid under these workloads or clearly document any supported workload limits.
- Different DeepSeek-OCR variants (for example, “base” versus “large” configurations) may have different module structures; the system should either distinguish between them or make explicit which variant the analytic model describes.
- Some operations inside DeepSeek-OCR may not map cleanly to known operation categories; the system should group such operations into a clearly labeled “other” or “unclassified” bucket rather than silently ignoring them.
- If a required trace or measurement cannot be collected (for example, due to missing dependencies or misconfiguration), the system should fail with a clear, actionable message rather than producing a partial or misleading analytic model.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: The measurement system MUST expose DeepSeek-OCR as a named model option that users can select when requesting performance and memory analysis.
- **FR-002**: The system MUST represent DeepSeek-OCR as a hierarchy of medium-granularity modules that reflects its main architectural components and key sub-blocks (for example, visual encoder, sequence or token model, and output head). Leaf modules MUST correspond to individual low-level operations that are not further decomposed, each with a stable identifier used in reports and exported data.
- **FR-003**: For each module in the DeepSeek-OCR hierarchy, the system MUST provide aggregated metrics for a standard OCR workload, including execution time, number of calls, and an estimate of memory usage.
- **FR-004**: The system MUST provide, for each module, a breakdown into underlying operation categories (for example, convolutions, normalizations, linear transformations, attention-like mechanisms, and activation functions) with associated counts and relative cost contributions.
- **FR-005**: The system MUST capture and expose the call relationships between modules in DeepSeek-OCR, including parent–child links and call counts, so that users can understand the overall call graph and module fan-out/fan-in.
- **FR-006**: The system MUST support exporting the DeepSeek-OCR analytic model and associated metrics as a machine-readable artifact that can be consumed by downstream analytic tools without manual editing.
- **FR-007**: The system MUST define a reproducible “standard OCR workload” for DeepSeek-OCR (for example, a small set of representative document images) and use it consistently when generating analytic models and reports for this feature.
- **FR-008**: The implementation of this feature MUST follow existing project quality practices, including clear documentation for public interfaces, automated tests for core behaviors, and adherence to the repository’s static analysis and style checks.

### Key Entities *(include if feature involves data)*

- **DeepSeek-OCR model definition**: Conceptual description of the DeepSeek-OCR model as a hierarchy of named modules with their relationships, covering the main components relevant for performance and memory analysis.
- **Module metrics snapshot**: A collection of metrics for each DeepSeek-OCR module under a specific workload, including execution time, call count, memory usage estimates, and categorized operation counts.
- **OCR workload profile**: Description of the input documents used to characterize DeepSeek-OCR (for example, number of pages, typical content density, and layout complexity), referenced by the analytic model so that results are interpreted in context.
- **Target operator list**: The current list of DeepSeek-OCR operator and module types targeted by the analytic model is captured in `reports/20211117-dsorc-op-analysis/static-20251118-130533/torchinfo-unique-layers.md`, with a machine-readable version in `reports/20211117-dsorc-op-analysis/static-20251118-130533/torchinfo-unique-layers.json`.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: Internal users can generate a DeepSeek-OCR performance and memory report from the measurement tool in a single guided flow, and the report is available within 5 minutes for the defined standard workload.
- **SC-002**: For the standard OCR workload on a chosen target environment, the total processing time predicted by the analytic model for DeepSeek-OCR is within 15% of the measured processing time in at least 90% of test runs.
- **SC-003**: At least 90% of DeepSeek-OCR’s measured execution time for the standard workload is attributed to named modules and operation categories in the report, with any remaining time clearly labeled as “other” or “unclassified.”
- **SC-004**: In feedback from at least two internal performance or ML engineers after using the feature, at least 80% report that the DeepSeek-OCR analytic reports are clear, actionable, and sufficient to guide optimization or capacity-planning decisions.
