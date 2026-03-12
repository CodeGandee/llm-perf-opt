## Context

The follow-up change sits entirely on top of the completed `wan2-1-dv-hardware-sweeps` work. The underlying sweep math and device-selectable artifact layout were reviewed as internally consistent; the remaining issues are all in the reporting layer:

- The generated DV stakeholder report mixes two different concepts under `1s SLA`: batch=1 latency and a normalized 1-second resource-rate sizing heuristic.
- The stakeholder-facing at-a-glance table currently highlights MemIO gap for every tier even though `DV300` is compute-limited in the produced comparison output.
- The report should state that the DV device profiles are preliminary analytic inputs from the shared registry, not authoritative vendor-validated hardware claims.
- The markdown generator emits the first image in each device section without a guaranteed blank-line boundary, producing malformed-looking markdown.

The implementation boundary remains the Wan2.1-local reporting code under `extern/modelmeter/models/wan2_1/`. The analytic cost model, workload grid, and device registry stay unchanged in this follow-up.

## Goals / Non-Goals

**Goals:**
- Separate latency wording from normalized 1-second resource-gap wording throughout the DV stakeholder report and related figure titles.
- Make the stakeholder summary emphasize the dominant resource gap for each device tier while preserving the existing underlying numeric summaries.
- Add explicit disclosure that DV100/DV200/DV300 inputs are first-pass analytic assumptions from the shared device registry.
- Fix the markdown block formatting bug and regenerate the affected DV comparison artifacts.
- Add focused automated checks around the reporting helpers most likely to regress.

**Non-Goals:**
- Changing `multi_device_cost`, sweep execution semantics, or the selection of the “most demanding input”.
- Editing shared device definitions in `extern/modelmeter/devices/gpu.py`.
- Reworking the legacy NGU report in this follow-up.
- Introducing new device tiers or new comparison workflows beyond the existing DV consolidated report.

## Decisions

### 1. Keep the existing numeric model outputs, but rename and explain the normalized 1-second metric

Chosen approach:
- Preserve `latency_batch1_s` as the report’s direct latency metric.
- Preserve the existing normalized per-video resource-rate calculations derived from the batch=1 workload demand.
- Replace stakeholder-facing `1s SLA` wording in report tables, narrative text, and figure titles with terminology that makes the metric a capacity-gap heuristic rather than a latency claim.
- Add one explicit methodology note that comparing a per-video 1-second normalized demand against an 8-GPU peak describes a sizing gap under DP-across-requests, not a claim that multi-GPU DP reduces single-video latency to 1 second.

Rationale:
- The review found a wording problem, not a numerical inconsistency.
- Keeping the underlying numbers stable avoids needless churn in the validated sweep outputs while still fixing the stakeholder interpretation risk.

Alternatives considered:
- Remove the normalized 1-second metric entirely.
  Rejected because it is still useful for hardware sizing discussion.
- Reinterpret the metric as a throughput-at-saturation target.
  Rejected because the current calculation is derived from single-video workload demand, not queueing or node-level throughput modeling.

### 2. Use a dominant-resource gap in the stakeholder-facing summary table

Chosen approach:
- Keep `Primary bottleneck` in the report summary.
- Replace the single MemIO-only gap presentation in the report’s at-a-glance table with a dominant-resource-aligned `Required/peak gap` derived from `primary_bottleneck`.
- Keep the richer tensor and MemIO gap fields available in machine-readable summaries and internal computations so detailed analysis is still possible.

Rationale:
- The current stakeholder table is compact, but it is misleading for `DV300`.
- A dominant-resource-aligned gap preserves table readability while making the headline comparison consistent with the actual bottleneck classification.

Alternatives considered:
- Add both tensor and MemIO gaps as full columns in the markdown table.
  Rejected because it makes the executive summary table significantly wider and harder to scan.
- Continue using MemIO-only gap and rely on surrounding prose to clarify `DV300`.
  Rejected because the table is likely to be skimmed independently of the prose.

### 3. Put the provisional-device-input disclosure in the report header and shared methodology

Chosen approach:
- Add a short header note that DV100/DV200/DV300 values are sourced from the current shared device registry and should be treated as first-pass analytic assumptions pending external validation.
- Keep the existing caveat that the whole report is a first-order analytic model.

Rationale:
- The review correctly noted that the shared DV definitions include placeholder-style fields and a TBD-style device entry, so the report should not imply stronger certainty than the inputs support.

Alternatives considered:
- Put the disclosure in a footnote only.
  Rejected because stakeholders can miss it.
- Omit the disclosure and rely on internal context.
  Rejected because the generated report is intended to stand on its own.

### 4. Fix markdown rendering at the helper boundary

Chosen approach:
- Update the report-writing helper path so every inserted image starts on a fresh markdown block, regardless of the preceding paragraph emission path.
- Prefer a localized helper fix over sprinkling manual newline handling across call sites.

Rationale:
- The bug is mechanical and localized to markdown assembly.
- Fixing the helper boundary prevents the same defect from reappearing in future sections.

Alternatives considered:
- Add manual blank lines at each image call site.
  Rejected because it is brittle and easy to miss when the report grows.

### 5. Add narrow regression tests around the reporting helpers

Chosen approach:
- Add small tests for the dominant-resource summary logic and for markdown/image block separation in the report generator.
- Prefer deterministic helper-level tests over end-to-end report diffing for this follow-up.

Rationale:
- The reviewed artifact bundle already gave confidence in the math; the main regression risks are presentation logic and string assembly.

Alternatives considered:
- No tests in this follow-up.
  Rejected because the touched logic is easy to regress silently.
- Only golden-file test the full generated report.
  Rejected because it would be brittle and noisier than the targeted checks needed here.

## Risks / Trade-offs

- [Terminology drift from the legacy NGU report] -> Mitigation: keep the same underlying numbers and add explicit wording that maps the new names to the same sizing heuristic.
- [Dominant-resource gap can hide secondary constraints] -> Mitigation: keep `Primary bottleneck` visible in the summary and retain detailed tensor/MemIO gap values in machine-readable outputs and per-device sections.
- [Regenerated artifacts may change more text than expected] -> Mitigation: scope the generator edits narrowly and verify the affected report/table/figure titles against the review decisions.

## Migration Plan

1. Update the Wan2.1-local reporting helpers and DV stakeholder report generator.
2. Regenerate the consolidated DV comparison artifacts from the existing compatible sweep runs.
3. Verify the report wording, dominant-resource gap presentation, provisional-input disclosure, and markdown formatting.
4. Run the targeted reporting tests.

Rollback:
- Revert the reporting code and regenerated comparison artifacts together if the wording or table changes prove unacceptable.

## Open Questions

- No major open questions remain for artifact creation. The only implementation-level choice left is the exact final phrase used for the renamed `1s` metric, which should optimize for stakeholder readability while preserving the design intent above.
