### Requirement: Generic per-device sizing figures
The system SHALL generate the standard Wan2.1 sizing figures from generalized sweep results for any supported device profile and supported precision profile, without hard-coding NGU800P-specific names or labels, and SHALL use figure titles and labels that distinguish direct latency measurements from normalized 1-second resource-gap views.

#### Scenario: Generate per-device figures for a DV run
- **WHEN** a user runs the reporting flow for a completed DV sweep
- **THEN** the system produces device-labeled throughput, used-rate, and normalized 1-second resource-gap figures for that device

#### Scenario: Generate per-device figures for a DV fp4 run
- **WHEN** a user runs the reporting flow for a completed DV sweep under a supported precision profile such as `fp4`
- **THEN** the system produces device-labeled throughput, used-rate, and normalized 1-second resource-gap figures for that device and precision context

### Requirement: Consolidated DV stakeholder report
The system SHALL produce one English detailed stakeholder report for each compatible precision-specific DV comparison group across `DV100`, `DV200`, and `DV300`, and SHALL disclose both the provisional nature of the DV device-profile values and the precision profile used for that bundle.

#### Scenario: One detailed report covers all DV devices for one precision group
- **WHEN** stakeholder reporting is run on compatible sweep results for `DV100`, `DV200`, and `DV300` that share one precision profile
- **THEN** the system produces a single English markdown report named `stakeholder-report.en.md` with separate sections for each DV device profile
- **AND** the report header includes notes describing the provisional nature of the DV hardware inputs and the precision assumptions used for that bundle

### Requirement: Stakeholder report mirrors NGU report structure
Each detailed precision-specific DV stakeholder report SHALL include information similar in type to the existing NGU stakeholder report so the output is familiar and decision-useful, while also distinguishing batch=1 latency from normalized 1-second resource-gap metrics and aligning the headline cross-device gap summary with the dominant bottleneck for each device tier.

#### Scenario: Detailed report includes NGU-style section content with clarified gap semantics
- **WHEN** a detailed precision-specific DV stakeholder report is generated
- **THEN** it includes an executive summary, workload and device assumptions, single-request breakdown, 8-device DP serving analysis, conclusions, and appendix-style tables for each covered DV device
- **AND** the at-a-glance comparison section reports batch=1 latency separately from a dominant-resource `Required/peak gap` summary rather than using a MemIO-only gap for every device tier

### Requirement: Comparison input compatibility checks
The system SHALL validate that DV sweep results are comparable before combining them into one detailed precision-specific stakeholder report.

#### Scenario: Reject incompatible detailed DV report inputs
- **WHEN** a user attempts to combine DV sweep results that differ in workload structure or precision assumptions within one detailed DV comparison group
- **THEN** the system refuses to generate the detailed stakeholder report
- **AND** the system reports which dimensions are incompatible

### Requirement: Generated stakeholder markdown uses clean block formatting
The system SHALL emit consolidated DV stakeholder markdown with figure references on their own blocks so the report renders cleanly in standard Markdown viewers.

#### Scenario: Device-section figures start on fresh markdown blocks
- **WHEN** the report generator emits a device analysis section
- **THEN** each figure reference begins on a fresh markdown block after the preceding paragraph or list content

### Requirement: Precision-aware comparison artifact labeling
The system SHALL label DV comparison artifacts in a way that makes the covered precision context explicit and prevents fp8 and fp4 outputs from being confused or overwritten.

#### Scenario: fp8 and fp4 comparison outputs do not collide
- **WHEN** stakeholder reporting is generated for both fp8 and fp4 DV comparison inputs
- **THEN** the resulting artifacts are written to distinct, precision-explicit comparison outputs
- **AND** readers can identify from the artifact layout or bundle labeling which outputs are fp8 and which outputs are fp4
