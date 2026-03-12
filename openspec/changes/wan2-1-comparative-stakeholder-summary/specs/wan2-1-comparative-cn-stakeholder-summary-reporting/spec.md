## ADDED Requirements

### Requirement: Chinese comparative stakeholder summary is generated with the bundle
The system SHALL generate a Chinese comparative stakeholder summary alongside the English comparative stakeholder summary whenever comparative summary reporting succeeds on compatible Wan2.1 detailed comparison bundles.

#### Scenario: Comparative summary emits both language variants
- **WHEN** comparative summary reporting is run on compatible detailed comparison bundles
- **THEN** the output directory contains both `stakeholder-summary.en.md` and `stakeholder-summary.cn.md`

### Requirement: Chinese comparative summary mirrors the English summary scope and data
The system SHALL generate the Chinese comparative stakeholder summary from the same scenario rows, metadata, and figure references as the English comparative stakeholder summary so both artifacts describe the same shared comparison context and computed values.

#### Scenario: Chinese comparative summary stays numerically aligned with English
- **WHEN** English and Chinese comparative stakeholder summaries are generated for the same comparison bundle
- **THEN** both reports cover the same validated workload and comparison context
- **AND** both reports reference the same generated comparison figures
- **AND** both reports present the same underlying numeric results, with only language-specific wording and labels differing

### Requirement: Chinese comparative summary reads like native Chinese and preserves critical English terminology
The system SHALL author the Chinese comparative stakeholder summary as native-sounding Chinese stakeholder prose rather than as a literal sentence-by-sentence translation of the English summary, and SHALL retain critical technical terminology in English where that is clearer.

#### Scenario: Chinese comparative summary is localized rather than mechanically translated
- **WHEN** the Chinese comparative stakeholder summary is generated
- **THEN** the narrative reads as natural Chinese for a technical stakeholder audience
- **AND** key technical terms such as hardware labels, precision labels, bottleneck names, or model-stage names MAY remain in English when that wording is clearer than forced Chinese equivalents
