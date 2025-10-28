# Feature Specification: Basic Profiling for DeepSeek‑OCR (Stage 1)

**Feature Branch**: `001-profile-deepseek-ocr`  
**Created**: 2025-10-28  
**Status**: Draft  
**Input**: User description: "basic profiling of deepseek ocr, see stage 1 in context/plans/plan-deep-profile-llm.md"

<!-- Constitution Compliance (author must ensure):
  - Public APIs/classes documented with NumPy-style docstrings and examples
  - All functions/classes fully type-annotated (mypy-clean), ruff-clean
  - Runtime environment declared (Pixi preferred; else virtualenv)
  - Manual test plan and file paths for major functionality under tests/manual/
  - Data models use attrs (default) or pydantic (for web schemas), no business logic
-->

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Stage 1 Profiling Report (Priority: P1)

As a performance engineer, I can produce a Stage 1 profiling report for
DeepSeek‑OCR that segments the run into major stages, summarizes operator‑level
time/memory, and provides early model‑level and per‑stage MFU estimates.

**Why this priority**: This creates immediate visibility into where time is spent
and establishes directional MFU numbers that guide deeper investigation.

**Independent Test**: Running the profiling workflow on a representative image
set yields a Stage 1 report with stage segmentation, an operator summary, and
MFU (model‑level and per‑stage) estimates that can be reviewed standalone.

**Acceptance Scenarios**:

1. Given a prepared model and a small image set, when the Stage 1 profiling run
   completes, then the report includes stage segmentation (prefill/decode),
   top operators by time, and MFU estimates.
2. Given two repeated runs on the same inputs, when the report is generated,
   then key metrics (stage timings and MFU estimates) are consistent within a
   reasonable tolerance (e.g., ±10%).

---

### User Story 2 - Stakeholder Summary (Priority: P2)

As a stakeholder, I can read a concise summary of Stage 1 findings that explains
the dominant cost centers and utilization patterns in plain language and
recommends where to focus next.

**Why this priority**: Ensures non‑technical decision‑makers understand the key
takeaways and can prioritize further work.

**Independent Test**: The summary can be consumed without technical tooling and
clearly communicates the biggest contributors to time and the implications.

**Acceptance Scenarios**:

1. Given the Stage 1 artifacts, when the summary is prepared, then it lists the
   top cost centers and the corresponding stage where each is most pronounced.

---

### User Story 3 - Reproducible Inputs & Notes (Priority: P3)

As a contributor, I can identify the input set and assumptions used so I can
repeat the Stage 1 profiling and compare my results.

**Why this priority**: Reproducibility reduces ambiguity and speeds iteration.

**Independent Test**: Another contributor following the notes can reproduce the
Stage 1 run and obtain metrics within tolerance.

**Acceptance Scenarios**:

1. Given the documented inputs and assumptions, when another person executes the
   workflow, then they obtain a Stage 1 report within the stated tolerances.

---

### Edge Cases

- Missing or inaccessible model assets
- No available GPU or insufficient device memory
- Empty or unsupported input images
- Significant variance between runs due to background system load

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The profiling run MUST segment inference into at least two
  user‑visible stages (prefill and decode) and include their elapsed times.
  Prefill is defined as the first forward pass after inputs are prepared
  (tokenized, tensors on device) up to first logits. Decode is defined as the
  token‑by‑token generation loop until the final token. Input reading and
  tokenization are excluded from both stages.
- **FR-002**: The profiling output MUST include an operator‑level summary of the
  heaviest operations by time and memory footprint.
- **FR-003**: The profiling output MUST include an early model‑level MFU
  estimate derived from measured throughput and an analytical FLOPs‑per‑token
  model of the network.
- **FR-004**: The profiling output MUST include per‑stage MFU estimates based on
  stage‑specific throughput and estimated compute.
- **FR-005**: The deliverable MUST include a short stakeholder summary that
  explains top cost centers and where optimization is likely to pay off.
- **FR-006**: Inputs and assumptions (e.g., input set description, batch size
  assumptions) MUST be documented to enable reproducibility.
 - **FR-007**: The input set MUST contain 10–20 diverse pages spanning
  text‑heavy, mixed‑layout, and image‑rich documents.
 - **FR-008**: The workflow MUST support multiple passes over the input set and
  produce aggregated metrics (e.g., average and variance) when repetitions are
  performed.

### Key Entities *(include if feature involves data)*

- **Model Under Test**: The specific model configuration being analyzed, with
  dimensions relevant to FLOPs‑per‑token estimation.
- **Input Set**: A small, representative set of images used for the run,
  described sufficiently for reproducibility.
- **Stage 1 Report**: The human‑readable results including stage timing,
  operator summary, and MFU estimates.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Stage segmentation, operator summary, and MFU (model‑level and
  per‑stage) are present in the report for a representative run.
- **SC-002**: Repeated runs on the same inputs yield Stage 1 metrics within ±10%
  variance for timings and MFU estimates.
- **SC-003**: The stakeholder summary clearly identifies top‑3 cost centers and
  specifies which stage they predominantly impact.
- **SC-004**: Inputs and assumptions are documented such that another
  contributor can reproduce a Stage 1 report within the stated tolerances.
 - **SC-005**: When multiple passes are run, the report includes aggregated
  metrics (e.g., mean and variance) across passes and shows reduced variance
  versus single‑pass measurements.
## Clarifications

### Session 2025-10-28

- Q: Define NVTX stage boundaries (prefill vs decode) → A: Prefill = first
  forward pass after inputs are prepared (tokenized, tensors on device) up to
  first logits; Decode = token-by-token generation loop until final token;
  exclude IO/tokenization.
 - Q: Define input set size/composition and repetition → A: Use 10–20 diverse
  pages (text‑heavy, mixed‑layout, image‑rich). Runs may iterate over the
  dataset multiple times to improve metric stability.
