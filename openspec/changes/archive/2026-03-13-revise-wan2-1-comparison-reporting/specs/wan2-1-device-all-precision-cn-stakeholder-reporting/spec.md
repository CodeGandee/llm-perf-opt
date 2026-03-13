## ADDED Requirements

### Requirement: Chinese per-device all-precision stakeholder report is generated with the bundle
The system SHALL generate a Chinese detailed per-device all-precision Wan2.1 stakeholder report alongside the English detailed per-device all-precision stakeholder report whenever compatible bundle generation succeeds.

#### Scenario: Per-device all-precision bundle emits both language variants
- **WHEN** per-device all-precision detailed stakeholder reporting succeeds on compatible source scenarios for one device
- **THEN** the output directory contains both `stakeholder-report.en.md` and `stakeholder-report.cn.md`

### Requirement: Chinese per-device all-precision report mirrors the English detailed bundle scope and data
The system SHALL generate the Chinese per-device all-precision stakeholder report from the same comparison rows, metadata, scenario ordering, and figure references as the English detailed stakeholder report so both artifacts describe the same validated comparison context and computed values.

#### Scenario: Chinese per-device detailed report stays numerically aligned with English
- **WHEN** English and Chinese per-device all-precision stakeholder reports are generated for the same bundle
- **THEN** both reports cover the same device, selected precision scenarios, and validated workload context
- **AND** both reports reference the same generated figures
- **AND** both reports present the same underlying numeric results, with only language-specific wording and labels differing

### Requirement: Chinese per-device all-precision report reads like native Chinese and preserves critical English terminology
The system SHALL author the Chinese per-device all-precision stakeholder report as native-sounding Chinese stakeholder prose rather than as a literal sentence-by-sentence translation of the English report, and SHALL retain critical technical terminology in English where that is clearer.

#### Scenario: Chinese per-device report is localized rather than mechanically translated
- **WHEN** the Chinese per-device all-precision stakeholder report is generated
- **THEN** the narrative reads as natural Chinese for a technical stakeholder audience
- **AND** key technical terms such as device labels, precision labels, bottleneck names, or model-stage names MAY remain in English when that wording is clearer than forced Chinese equivalents
