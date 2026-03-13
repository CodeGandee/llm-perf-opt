### Requirement: Comparative stakeholder summary bundle
The system SHALL generate one English comparative stakeholder summary from multiple compatible detailed Wan2.1 comparison bundles that share one analytic comparison context.

#### Scenario: Generate comparative summary from compatible detailed bundles
- **WHEN** a user runs comparative summary reporting on two or more detailed comparison bundles that share the same workload structure, model mode, utilization profile, and device-count semantics
- **THEN** the system produces one English comparative summary bundle for that shared context
- **AND** each compared case in the bundle corresponds to one source scenario such as a hardware and precision combination

### Requirement: Comparative summary artifacts use generic scenario-oriented naming
The comparative summary bundle SHALL use generic artifact names and data shapes that are not hard-coded to one device family or one precision pair.

#### Scenario: Comparative summary output is not fp8-vs-fp4 specific
- **WHEN** a comparative summary is generated from scenarios such as `DV300 fp8`, `DV300 fp4`, `B200 fp8`, or `H200 fp8`
- **THEN** the bundle writes `stakeholder-summary.en.md`
- **AND** the bundle writes a `comparison-table.csv` whose rows represent individual scenarios rather than wide prefixed columns such as `fp8_*` and `fp4_*`
- **AND** the bundle metadata records the included source detailed bundles and scenario identities

### Requirement: Comparative summary stays compact and conclusion-focused
The English comparative stakeholder summary SHALL present only the shared context, conclusive comparison metrics, and a small figure set, and SHALL NOT duplicate appendix-style sweep tables from the detailed source bundles.

#### Scenario: Comparative summary omits detailed appendix tables
- **WHEN** the English comparative stakeholder summary is generated
- **THEN** it includes a short context block, executive takeaways, a compact scenario comparison table, and a small number of headline comparison figures
- **AND** it does not include per-batch appendix tables copied from the detailed source bundles

### Requirement: Comparative summary validates shared comparison context
The system SHALL validate that source detailed comparison bundles are analytically comparable before combining them into one comparative stakeholder summary.

#### Scenario: Reject incompatible comparative summary inputs
- **WHEN** a user attempts to generate a comparative summary from detailed bundles that differ in workload structure, model mode, utilization profile, or device-count semantics
- **THEN** the system refuses to generate the comparative summary
- **AND** the system reports which shared-context dimensions are incompatible
