## MODIFIED Requirements

### Requirement: Chinese DV stakeholder report is generated with the comparison bundle
The system SHALL generate a Chinese detailed DV stakeholder markdown report alongside the English detailed DV stakeholder report whenever stakeholder reporting runs successfully on compatible precision-specific `DV100`, `DV200`, and `DV300` comparison inputs.

#### Scenario: Detailed comparison reporting emits both language variants
- **WHEN** stakeholder reporting is run on compatible DV sweep results for one precision-specific detailed comparison group
- **THEN** the comparison output directory contains both `stakeholder-report.en.md` and `stakeholder-report.cn.md`

### Requirement: Chinese DV stakeholder report mirrors the English report scope and data
The system SHALL generate the Chinese DV stakeholder report from the same comparison summaries, tables, and figure references as the English DV stakeholder report so both artifacts describe the same workload slice, device set, and computed values for one detailed precision-specific DV comparison bundle.

#### Scenario: Chinese detailed report covers the same comparison context
- **WHEN** English and Chinese DV stakeholder reports are generated for the same detailed comparison run
- **THEN** both reports cover `DV100`, `DV200`, and `DV300` for the same validated workload and precision context
- **AND** the Chinese report references the same comparison figures and architecture image as the English report
- **AND** the Chinese report presents the same underlying numeric results, with only language-specific wording and labels differing
