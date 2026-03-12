## MODIFIED Requirements

### Requirement: Chinese DV stakeholder report is generated with the comparison bundle
The system SHALL generate a Chinese DV stakeholder markdown report alongside the corresponding English DV stakeholder markdown output for every generated DV comparison bundle, including each precision-specific detailed bundle and any compact cross-precision summary bundle.

#### Scenario: Precision-aware comparison reporting emits Chinese outputs
- **WHEN** stakeholder reporting is run on compatible DV comparison inputs for fp8, fp4, or a compatible fp8-vs-fp4 summary
- **THEN** the generated comparison outputs include a Chinese stakeholder markdown artifact alongside the corresponding English artifact

### Requirement: Chinese DV stakeholder report mirrors the English report scope and data
The system SHALL generate each Chinese DV stakeholder report from the same comparison summaries, tables, and figure references as its corresponding English DV stakeholder report so both artifacts describe the same workload slice, device set, precision context, and computed values for that bundle.

#### Scenario: Chinese report stays aligned with its corresponding English precision bundle
- **WHEN** English and Chinese DV stakeholder reports are generated for the same detailed fp8 or fp4 comparison bundle
- **THEN** both reports cover `DV100`, `DV200`, and `DV300` for the same validated workload and precision context
- **AND** the Chinese report references the same comparison figures and architecture image as the English report
- **AND** the Chinese report presents the same underlying numeric results, with only language-specific wording and labels differing

#### Scenario: Chinese report stays aligned with the English cross-precision summary
- **WHEN** English and Chinese cross-precision DV stakeholder summaries are generated for the same fp8-vs-fp4 comparison context
- **THEN** both summaries cover the same validated workload slice, device set, and per-precision numeric results
- **AND** the Chinese summary uses the same underlying comparison data as the English summary

### Requirement: Chinese DV stakeholder markdown is localized for stakeholder readability
The system SHALL localize the Chinese DV stakeholder markdown headings, explanatory prose, table headers, Q&A notes, and appendix framing into Chinese for both the detailed precision-specific reports and the compact cross-precision summary, while keeping figure generation shared with the English path for this change.

#### Scenario: Chinese outputs localize markdown but not figure assets
- **WHEN** the Chinese DV stakeholder artifacts are generated
- **THEN** the markdown section titles, table labels, and explanatory notes are written in Chinese
- **AND** the generated Chinese artifacts do not require a separate Chinese-localized figure set in order to be generated
