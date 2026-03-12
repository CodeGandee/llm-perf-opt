## MODIFIED Requirements

### Requirement: Generic per-device sizing figures
The system SHALL generate the standard Wan2.1 sizing figures from generalized sweep results for any supported device profile, without hard-coding NGU800P-specific names or labels, and SHALL use figure titles and labels that distinguish direct latency measurements from normalized 1-second resource-gap views.

#### Scenario: Generate per-device figures for a DV run
- **WHEN** a user runs the reporting flow for a completed DV sweep
- **THEN** the system produces device-labeled throughput, used-rate, and normalized 1-second resource-gap figures for that device

### Requirement: Consolidated DV stakeholder report
The system SHALL produce one English stakeholder report that summarizes Wan2.1 behavior across `DV100`, `DV200`, and `DV300` for a common workload slice, and SHALL disclose that the DV device-profile values are first-pass analytic assumptions sourced from the shared device registry.

#### Scenario: One report covers all DV devices
- **WHEN** stakeholder reporting is run on compatible sweep results for `DV100`, `DV200`, and `DV300`
- **THEN** the system produces a single English markdown report with separate sections for each DV device profile and a header note describing the provisional nature of the DV hardware inputs

### Requirement: Stakeholder report mirrors NGU report structure
The consolidated DV stakeholder report SHALL include information similar in type to the existing NGU stakeholder report so the output is familiar and decision-useful, while also distinguishing batch=1 latency from normalized 1-second resource-gap metrics and aligning the headline cross-device gap summary with the dominant bottleneck for each device tier.

#### Scenario: Report includes NGU-style section content with clarified gap semantics
- **WHEN** the consolidated DV stakeholder report is generated
- **THEN** it includes an executive summary, workload and device assumptions, single-request breakdown, 8-device DP serving analysis, conclusions, and appendix-style tables for each covered DV device
- **AND** the at-a-glance comparison section reports batch=1 latency separately from a dominant-resource `Required/peak gap` summary rather than using a MemIO-only gap for every device

## ADDED Requirements

### Requirement: Generated stakeholder markdown uses clean block formatting
The system SHALL emit consolidated DV stakeholder markdown with figure references on their own blocks so the report renders cleanly in standard Markdown viewers.

#### Scenario: Device-section figures start on fresh markdown blocks
- **WHEN** the report generator emits a device analysis section
- **THEN** each figure reference begins on a fresh markdown block after the preceding paragraph or list content
