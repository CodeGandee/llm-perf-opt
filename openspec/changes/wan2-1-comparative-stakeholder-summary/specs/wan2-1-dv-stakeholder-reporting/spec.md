## MODIFIED Requirements

### Requirement: Consolidated DV stakeholder report
The system SHALL produce one English detailed stakeholder report for each compatible precision-specific DV comparison group across `DV100`, `DV200`, and `DV300`, and SHALL disclose that the DV device-profile values are first-pass analytic assumptions sourced from the shared device registry.

#### Scenario: One detailed report covers all DV devices for one precision group
- **WHEN** stakeholder reporting is run on compatible sweep results for `DV100`, `DV200`, and `DV300` that share one precision profile
- **THEN** the system produces a single English markdown report named `stakeholder-report.en.md` with separate sections for each DV device profile
- **AND** the report header includes a note describing the provisional nature of the DV hardware inputs

### Requirement: Stakeholder report mirrors NGU report structure
Each detailed precision-specific DV stakeholder report SHALL include information similar in type to the existing NGU stakeholder report so the output is familiar and decision-useful, while also distinguishing batch=1 latency from normalized 1-second resource-gap metrics and aligning the headline cross-device gap summary with the dominant bottleneck for each device tier.

#### Scenario: Detailed report includes NGU-style section content with clarified gap semantics
- **WHEN** a detailed precision-specific DV stakeholder report is generated
- **THEN** it includes an executive summary, workload and device assumptions, single-request breakdown, 8-device DP serving analysis, conclusions, and appendix-style tables for each covered DV device
- **AND** the at-a-glance comparison section reports batch=1 latency separately from a dominant-resource `Required/peak gap` summary rather than using a MemIO-only gap for every device

### Requirement: Comparison input compatibility checks
The system SHALL validate that DV sweep results are comparable before combining them into one detailed precision-specific stakeholder report.

#### Scenario: Reject incompatible detailed DV report inputs
- **WHEN** a user attempts to combine DV sweep results that differ in workload structure or precision assumptions within one detailed DV comparison group
- **THEN** the system refuses to generate the detailed stakeholder report
- **AND** the system reports which dimensions are incompatible
