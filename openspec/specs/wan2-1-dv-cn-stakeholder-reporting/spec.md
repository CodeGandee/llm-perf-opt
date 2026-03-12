### Requirement: Chinese DV stakeholder report is generated with the comparison bundle
The system SHALL generate a Chinese DV stakeholder markdown report alongside the English DV stakeholder report whenever stakeholder reporting runs successfully on compatible `DV100`, `DV200`, and `DV300` comparison inputs.

#### Scenario: Comparison reporting emits both language variants
- **WHEN** stakeholder reporting is run on compatible DV sweep results
- **THEN** the comparison output directory contains both `stakeholder-report.en.md` and `stakeholder-report.cn.md`

### Requirement: Chinese DV stakeholder report mirrors the English report scope and data
The system SHALL generate the Chinese DV stakeholder report from the same comparison summaries, tables, and figure references as the English DV stakeholder report so both artifacts describe the same workload slice, device set, and computed values.

#### Scenario: Chinese report covers the same comparison context
- **WHEN** English and Chinese DV stakeholder reports are generated for the same comparison run
- **THEN** both reports cover `DV100`, `DV200`, and `DV300` for the same validated workload and precision context
- **AND** the Chinese report references the same comparison figures and architecture image as the English report
- **AND** the Chinese report presents the same underlying numeric results, with only language-specific wording and labels differing

### Requirement: Chinese DV stakeholder markdown is localized for stakeholder readability
The system SHALL localize the Chinese DV stakeholder markdown headings, explanatory prose, table headers, Q&A notes, and appendix framing into Chinese, while keeping figure generation shared with the English path for this change.

#### Scenario: Chinese report localizes markdown but not figure assets
- **WHEN** the Chinese DV stakeholder report is generated
- **THEN** the markdown section titles, table labels, and explanatory notes are written in Chinese
- **AND** the report artifact is named `stakeholder-report.cn.md`
- **AND** the report does not require a separate Chinese-localized figure set in order to be generated

### Requirement: Chinese DV stakeholder report reads like native Chinese and preserves critical English terminology
The system SHALL author the Chinese DV stakeholder report as native-sounding Chinese stakeholder prose rather than as a literal sentence-by-sentence translation of the English report, and SHALL retain critical technical terminology in English where that improves precision and recognizability.

#### Scenario: Chinese wording is localized rather than mechanically translated
- **WHEN** the Chinese DV stakeholder report is generated
- **THEN** the narrative reads as natural Chinese writing for a technical stakeholder audience
- **AND** key technical terms such as hardware/resource labels, parallelism terms, or model-stage names MAY remain in English when that usage is clearer than forced Chinese equivalents
