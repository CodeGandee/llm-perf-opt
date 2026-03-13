## MODIFIED Requirements

### Requirement: Comparative stakeholder summary bundle
The system SHALL generate one English comparative stakeholder summary markdown scaffold from an explicit selection of compatible Wan2.1 sweep scenarios that share one analytic comparison context, without requiring those scenarios to be preassembled into detailed subgroup comparison bundles.

#### Scenario: Generate comparative summary from selected scenarios
- **WHEN** a user runs comparative summary reporting on selected scenarios such as `DV100 fp8`, `DV100 fp4`, `DV300 fp8`, `B200 fp8`, and `H20 fp8` that share the same workload structure, model mode, utilization profile, and device-count semantics
- **THEN** the system produces one English comparative summary scaffold bundle for that shared context
- **AND** each compared case in the bundle corresponds to one selected `(device, precision)` source scenario
- **AND** the bundle metadata records the included source run identifiers

### Requirement: Comparative summary artifacts use generic scenario-oriented naming
The comparative summary bundle SHALL use generic artifact names and long-format scenario data shapes that are not hard-coded to one device family, one precision pair, or one intermediate subgroup bundle layout.

#### Scenario: Comparative summary output uses scenario-oriented artifacts
- **WHEN** a comparative summary is generated from scenarios such as `DV300 fp8`, `DV300 fp4`, `B200 fp8`, or `H20 fp8`
- **THEN** the bundle writes `stakeholder-summary.en.md`
- **AND** the bundle writes a `comparison-table.csv` whose rows represent individual scenarios rather than wide prefixed columns such as `fp8_*` and `fp4_*`
- **AND** the bundle metadata records the included source scenarios and their source run ids directly

### Requirement: Comparative summary stays compact and conclusion-focused
The English comparative stakeholder summary scaffold SHALL present only the shared context, deterministic factual comparison metrics, a small figure set for the selected scenarios, and explicit placeholder blocks for higher-level analysis text, and SHALL NOT duplicate appendix-style sweep tables from detailed reports.

#### Scenario: Comparative summary omits detailed appendix tables
- **WHEN** the English comparative stakeholder summary scaffold is generated
- **THEN** it includes a short context block, a factual digest section, a compact scenario comparison table, and a small number of headline comparison figures
- **AND** it includes designated placeholder blocks for user-authored or LLM-authored analysis text
- **AND** it does not include per-batch appendix tables copied from detailed reports

### Requirement: Comparative summary validates shared comparison context
The system SHALL validate that selected source scenarios are analytically comparable before combining them into one comparative stakeholder summary.

#### Scenario: Reject incompatible comparative summary inputs
- **WHEN** a user attempts to generate a comparative summary from selected scenarios that differ in workload structure, model mode, utilization profile, or device-count semantics
- **THEN** the system refuses to generate the comparative summary
- **AND** the system reports which shared-context dimensions are incompatible

## ADDED Requirements

### Requirement: Comparative summary orders scenarios predictably
The system SHALL order comparative summary scenarios deterministically so the `DV` series appears before non-`DV` scenarios, and scenarios for the same device remain grouped together in tables and headline figures.

#### Scenario: DV scenarios appear before non-DV scenarios
- **WHEN** a comparative summary includes `DV100 fp8`, `DV100 fp4`, `DV200 fp8`, `DV300 fp8`, `B200 fp8`, and `H20 fp8`
- **THEN** the comparison table lists the `DV` scenarios before the non-`DV` scenarios
- **AND** scenarios for the same device remain adjacent

### Requirement: Comparative summary scaffold marks generated and editable regions
The English comparative stakeholder summary scaffold SHALL clearly separate generated factual content from LLM-editable analysis placeholder regions so users can safely hand the markdown file to a reasoning model without rewriting tables or numeric facts unintentionally.

#### Scenario: Summary scaffold contains editable placeholder blocks and draft warning
- **WHEN** the English comparative stakeholder summary scaffold is generated
- **THEN** the markdown contains a visible draft warning that rerunning generation may overwrite later manual edits
- **AND** the markdown marks generated factual regions and LLM-editable placeholder regions explicitly
- **AND** the placeholder blocks instruct the user or downstream LLM to base analysis only on facts already present in the file
