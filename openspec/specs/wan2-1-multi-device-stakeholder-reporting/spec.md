### Requirement: Generic detailed multi-device stakeholder bundle
The system SHALL generate one English detailed Wan2.1 stakeholder-report bundle from two or more compatible device runs that share one analytic comparison context, without hard-coding the device family to the `DV100`/`DV200`/`DV300` set.

#### Scenario: Generate a detailed NVIDIA bundle from H20 and B200 runs
- **WHEN** a user runs detailed stakeholder reporting on compatible `H20 fp8` and `B200 fp8` Wan2.1 sweep runs
- **THEN** the system produces one English detailed comparison bundle for that shared context
- **AND** the bundle contains a device comparison table, per-device analysis sections, appendix-style sweep tables, and generated figures

### Requirement: Detailed multi-device bundle uses standard detailed-report artifacts
The English multi-device stakeholder bundle SHALL use the standard detailed artifact names so it remains a first-class source bundle for downstream reporting.

#### Scenario: Generic detailed bundle writes standard artifacts
- **WHEN** a compatible multi-device detailed bundle is generated
- **THEN** the output directory contains `bundle-metadata.json`
- **AND** the output directory contains `comparison-table.csv`
- **AND** the output directory contains `stakeholder-report.en.md`
- **AND** the output directory contains a `figures/` directory

### Requirement: Detailed multi-device reporting validates shared context and precision
The system SHALL validate that device runs are analytically compatible before combining them into one detailed stakeholder bundle, and compatibility MUST include one shared workload structure, model mode, utilization profile, device-count semantics, and precision profile.

#### Scenario: Reject incompatible detailed bundle inputs
- **WHEN** a user attempts to generate a detailed stakeholder bundle from device runs that differ in workload structure, model mode, utilization profile, device-count semantics, or precision assumptions
- **THEN** the system refuses to generate the detailed bundle
- **AND** the system reports which compatibility dimensions are mismatched

### Requirement: Detailed multi-device report discloses device-profile assumptions
The English detailed multi-device stakeholder report SHALL disclose that the compared hardware values come from the shared ModelMeter device registry and should be treated as first-pass analytic assumptions pending validation.

#### Scenario: Detailed report header marks hardware assumptions as analytic inputs
- **WHEN** an English detailed multi-device stakeholder report is generated
- **THEN** the report header states that the device-profile values come from the shared registry
- **AND** the report communicates that those values are analytic inputs rather than measured benchmark results
