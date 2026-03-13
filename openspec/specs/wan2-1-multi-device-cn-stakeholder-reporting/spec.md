### Requirement: Chinese detailed multi-device stakeholder report is generated with the bundle
The system SHALL generate a Chinese detailed multi-device Wan2.1 stakeholder report alongside the English detailed multi-device stakeholder report whenever compatible detailed bundle generation succeeds.

#### Scenario: Detailed multi-device bundle emits both language variants
- **WHEN** detailed multi-device stakeholder reporting succeeds on compatible device runs
- **THEN** the output directory contains both `stakeholder-report.en.md` and `stakeholder-report.cn.md`

### Requirement: Chinese detailed multi-device report mirrors the English detailed bundle scope and data
The system SHALL generate the Chinese detailed multi-device stakeholder report from the same comparison rows, metadata, and figure references as the English detailed multi-device stakeholder report so both artifacts describe the same validated comparison context and computed values.

#### Scenario: Chinese detailed report stays numerically aligned with English
- **WHEN** English and Chinese detailed multi-device stakeholder reports are generated for the same bundle
- **THEN** both reports cover the same validated workload and precision context
- **AND** both reports reference the same generated figures
- **AND** both reports present the same underlying numeric results, with only language-specific wording and labels differing

### Requirement: Chinese detailed multi-device report reads like native Chinese and preserves critical English terminology
The system SHALL author the Chinese detailed multi-device stakeholder report as native-sounding Chinese stakeholder prose rather than as a literal sentence-by-sentence translation of the English detailed multi-device stakeholder report, and SHALL retain critical technical terminology in English where that is clearer.

#### Scenario: Chinese detailed report is localized rather than mechanically translated
- **WHEN** the Chinese detailed multi-device stakeholder report is generated
- **THEN** the narrative reads as natural Chinese for a technical stakeholder audience
- **AND** key technical terms such as device labels, precision labels, bottleneck names, or model-stage names MAY remain in English when that wording is clearer than forced Chinese equivalents
