## MODIFIED Requirements

### Requirement: Chinese comparative stakeholder summary is generated with the bundle
The system SHALL generate a Chinese comparative stakeholder summary markdown scaffold alongside the English comparative stakeholder summary scaffold whenever comparative summary reporting succeeds on a selected set of compatible Wan2.1 source scenarios.

#### Scenario: Comparative summary emits both language variants
- **WHEN** comparative summary reporting is run on a selected set of compatible source scenarios
- **THEN** the output directory contains both `stakeholder-summary.en.md` and `stakeholder-summary.cn.md`

### Requirement: Chinese comparative summary mirrors the English summary scope and data
The system SHALL generate the Chinese comparative stakeholder summary scaffold from the same scenario rows, metadata, scenario ordering, figure references, and placeholder structure as the English comparative stakeholder summary scaffold so both artifacts describe the same shared comparison context and computed values.

#### Scenario: Chinese comparative summary stays numerically aligned with English
- **WHEN** English and Chinese comparative stakeholder summaries are generated for the same comparison bundle
- **THEN** both reports cover the same validated workload and comparison context
- **AND** both reports reference the same generated comparison figures
- **AND** both reports present the same underlying numeric results, with only language-specific wording and labels differing

### Requirement: Chinese comparative summary reads like native Chinese and preserves critical English terminology
The system SHALL localize the generated factual text and placeholder instructions in the Chinese comparative stakeholder summary scaffold for a Chinese technical stakeholder audience, and SHALL retain critical technical terminology in English where that is clearer.

#### Scenario: Chinese comparative summary is localized rather than mechanically translated
- **WHEN** the Chinese comparative stakeholder summary scaffold is generated
- **THEN** the generated factual sections and placeholder instructions read as natural Chinese for a technical stakeholder audience
- **AND** key technical terms such as hardware labels, precision labels, bottleneck names, or model-stage names MAY remain in English when that wording is clearer than forced Chinese equivalents

## ADDED Requirements

### Requirement: Chinese comparative summary scaffold preserves editable boundaries
The Chinese comparative stakeholder summary scaffold SHALL preserve the same generated-region and editable-placeholder boundaries as the English scaffold so users can apply the same manual LLM-editing workflow in either locale.

#### Scenario: Chinese scaffold contains draft warning and placeholder blocks
- **WHEN** the Chinese comparative stakeholder summary scaffold is generated
- **THEN** the markdown contains a visible draft warning that rerunning generation may overwrite later manual edits
- **AND** the markdown marks generated factual regions and LLM-editable placeholder regions explicitly
- **AND** the placeholder blocks instruct the user or downstream LLM to base analysis only on facts already present in the file
