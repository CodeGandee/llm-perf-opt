## ADDED Requirements

### Requirement: Precision-aware comparison artifact labeling
The system SHALL label DV comparison artifacts in a way that makes the covered precision context explicit and prevents fp8 and fp4 outputs from being confused or overwritten.

#### Scenario: fp8 and fp4 comparison outputs do not collide
- **WHEN** stakeholder reporting is generated for both fp8 and fp4 DV comparison inputs
- **THEN** the resulting artifacts are written to distinct, precision-explicit comparison outputs
- **AND** readers can identify from the artifact layout or bundle labeling which outputs are fp8, which are fp4, and which summarize both

## MODIFIED Requirements

### Requirement: Generic per-device sizing figures
The system SHALL generate the standard Wan2.1 sizing figures from generalized sweep results for any supported device profile and supported precision profile, without hard-coding NGU800P-specific names or labels, and SHALL use figure titles and labels that distinguish direct latency measurements from normalized 1-second resource-gap views.

#### Scenario: Generate per-device figures for a DV fp4 run
- **WHEN** a user runs the reporting flow for a completed DV sweep under a supported precision profile such as `fp4`
- **THEN** the system produces device-labeled throughput, used-rate, and normalized 1-second resource-gap figures for that device and precision context

### Requirement: Consolidated DV stakeholder report
The system SHALL produce one English stakeholder report for each compatible precision-specific DV comparison group across `DV100`, `DV200`, and `DV300`, and SHALL disclose both the provisional nature of the DV device-profile values and the precision profile used for that bundle. When compatible fp8 and fp4 DV comparison groups are available for the same workload slice, the system SHALL also produce a compact English cross-precision stakeholder summary.

#### Scenario: One detailed report covers all DV devices for one precision group
- **WHEN** stakeholder reporting is run on compatible sweep results for `DV100`, `DV200`, and `DV300` that share one precision profile
- **THEN** the system produces one English markdown report for that precision group with separate sections for each DV device profile
- **AND** the report header discloses the provisional nature of the DV hardware inputs and the precision assumptions used for that bundle

#### Scenario: Cross-precision summary is generated for shared fp8 and fp4 contexts
- **WHEN** stakeholder reporting is run on compatible fp8 and fp4 DV comparison groups for the same workload slice and device set
- **THEN** the system produces a compact English stakeholder summary that compares fp8 and fp4 side by side for `DV100`, `DV200`, and `DV300`

### Requirement: Stakeholder report mirrors NGU report structure
Each detailed precision-specific DV stakeholder report SHALL include information similar in type to the existing NGU stakeholder report so the output is familiar and decision-useful, while also distinguishing batch=1 latency from normalized 1-second resource-gap metrics and aligning the headline cross-device gap summary with the dominant bottleneck for each device tier. The compact cross-precision summary SHALL stay shorter and focus on side-by-side stakeholder metrics rather than repeating the full appendix structure.

#### Scenario: Detailed precision-specific report keeps the familiar section structure
- **WHEN** a detailed DV stakeholder report is generated for one precision group
- **THEN** it includes an executive summary, workload and device assumptions, single-request breakdown, 8-device DP serving analysis, conclusions, and appendix-style tables for each covered DV device
- **AND** the at-a-glance comparison section reports batch=1 latency separately from the report's sizing-gap metrics rather than collapsing them into one number

#### Scenario: Cross-precision summary stays compact
- **WHEN** the cross-precision DV stakeholder summary is generated
- **THEN** it presents the common workload and precision assumptions plus a compact side-by-side comparison across `(device, precision)` pairs
- **AND** it does not duplicate the full appendix tables from the detailed reports

### Requirement: Comparison input compatibility checks
The system SHALL validate that DV sweep results are comparable before combining them into detailed or summary stakeholder artifacts. Detailed precision-specific reports MUST require one shared workload structure and one shared precision profile within each DV comparison group. Cross-precision summaries MUST require the same workload structure, device coverage, and comparison semantics across the fp8 and fp4 groups while allowing the precision value itself to differ by group.

#### Scenario: Reject incompatible detailed DV report inputs
- **WHEN** a user attempts to combine DV sweep results for one detailed bundle that differ in workload structure or precision assumptions
- **THEN** the system refuses to generate that detailed stakeholder report and reports which dimensions are incompatible

#### Scenario: Reject cross-precision summary inputs with mismatched comparison context
- **WHEN** a user attempts to generate a cross-precision DV summary from fp8 and fp4 groups that do not cover the same workload slice, device set, or comparison semantics
- **THEN** the system refuses to generate the cross-precision summary and reports which group-level dimensions are incompatible
