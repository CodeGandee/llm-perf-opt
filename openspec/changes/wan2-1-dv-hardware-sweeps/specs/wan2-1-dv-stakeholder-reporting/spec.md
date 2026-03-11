## ADDED Requirements

### Requirement: Generic per-device sizing figures
The system SHALL generate the standard Wan2.1 sizing figures from generalized sweep results for any supported device profile, without hard-coding NGU800P-specific names or labels.

#### Scenario: Generate per-device figures for a DV run
- **WHEN** a user runs the reporting flow for a completed DV sweep
- **THEN** the system produces device-labeled throughput, used-rate, and SLA-sizing figures for that device

### Requirement: Consolidated DV stakeholder report
The system SHALL produce one English stakeholder report that summarizes Wan2.1 behavior across `DV100`, `DV200`, and `DV300` for a common workload slice.

#### Scenario: One report covers all DV devices
- **WHEN** stakeholder reporting is run on compatible sweep results for `DV100`, `DV200`, and `DV300`
- **THEN** the system produces a single English markdown report with separate sections for each DV device profile

### Requirement: Stakeholder report mirrors NGU report structure
The consolidated DV stakeholder report SHALL include information similar in type to the existing NGU stakeholder report so the output is familiar and decision-useful.

#### Scenario: Report includes NGU-style section content
- **WHEN** the consolidated DV stakeholder report is generated
- **THEN** it includes an executive summary, workload and device assumptions, single-request breakdown, 8-device DP serving analysis, conclusions, and appendix-style tables for each covered DV device

### Requirement: Comparison input compatibility checks
The system SHALL validate that DV sweep results are comparable before combining them into the consolidated stakeholder report.

#### Scenario: Reject incompatible DV report inputs
- **WHEN** a user attempts to combine DV sweep results that differ in workload structure or precision assumptions
- **THEN** the system refuses to generate the consolidated stakeholder report and reports which dimensions are incompatible
