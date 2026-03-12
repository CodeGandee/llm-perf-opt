## 1. Reporting Semantics Updates

- [x] 1.1 Update the Wan2.1 reporting helpers to rename the current `1s SLA` figure/title language so it clearly represents a normalized 1-second resource-gap metric rather than direct latency.
- [x] 1.2 Update the DV stakeholder report generator to separate batch=1 latency from the normalized 1-second gap narrative in the executive summary, methodology notes, per-device conclusions, and figure references.
- [x] 1.3 Change the stakeholder-facing at-a-glance comparison table to report a dominant-resource `Required/peak gap` aligned with each device’s primary bottleneck instead of a MemIO-only gap for all tiers.
- [x] 1.4 Add report disclosure that DV100/DV200/DV300 values are first-pass analytic assumptions sourced from the shared device registry.
- [x] 1.5 Fix markdown block emission so every generated figure reference starts on a fresh block line.

## 2. Verification Coverage

- [x] 2.1 Add focused tests for the summary logic that selects the dominant-resource gap from the existing tensor and MemIO metrics.
- [x] 2.2 Add a focused test for markdown generation that catches missing blank-line separation before inserted image blocks.

## 3. Artifact Regeneration and Validation

- [x] 3.1 Regenerate the consolidated DV comparison artifacts from the existing compatible DV sweep runs using the updated reporting code.
- [x] 3.2 Verify that the regenerated stakeholder report includes the provisional-device-input note, clarified metric wording, dominant-resource gap table, and clean markdown figure separation.
- [x] 3.3 Verify that the updated comparison figures and report text remain numerically consistent with the underlying sweep outputs and that existing machine-readable summary fields still support detailed analysis.
