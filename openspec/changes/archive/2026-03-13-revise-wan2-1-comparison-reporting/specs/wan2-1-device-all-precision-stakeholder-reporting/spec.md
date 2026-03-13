## ADDED Requirements

### Requirement: Per-device all-precision detailed stakeholder bundle
The system SHALL generate one English detailed Wan2.1 stakeholder-report bundle from two or more compatible sweep runs for the same device that share one analytic comparison context, without hard-coding the device family or the compared precision pair.

#### Scenario: Generate a per-device all-precision bundle
- **WHEN** a user runs per-device detailed stakeholder reporting on compatible `DV300 fp8` and `DV300 fp4` Wan2.1 sweep runs
- **THEN** the system produces one English detailed comparison bundle for `DV300`
- **AND** each compared case in the bundle corresponds to one precision scenario for that device

### Requirement: Per-device all-precision bundle uses standard detailed artifacts
The English per-device all-precision stakeholder bundle SHALL use standard detailed-report artifact names and SHALL record scenario-level source run metadata for the included precision scenarios.

#### Scenario: Per-device detailed bundle writes standard artifacts
- **WHEN** a compatible per-device all-precision bundle is generated
- **THEN** the output directory contains `bundle-metadata.json`
- **AND** the output directory contains `comparison-table.csv`
- **AND** the output directory contains `stakeholder-report.en.md`
- **AND** the output directory contains a `figures/` directory
- **AND** the bundle metadata records the device identity and the included source scenarios directly

### Requirement: Per-device all-precision reporting validates same-device shared context
The system SHALL validate that selected source runs belong to one device and share one workload structure, model mode, utilization profile, and device-count semantics before combining them into one per-device all-precision detailed bundle.

#### Scenario: Reject mismatched device inputs
- **WHEN** a user attempts to generate a per-device all-precision detailed bundle from `DV300 fp8` and `DV200 fp4`
- **THEN** the system refuses to generate the detailed bundle
- **AND** the system reports that the selected scenarios do not belong to the same device

#### Scenario: Reject incompatible context inputs for one device
- **WHEN** a user attempts to generate a per-device all-precision detailed bundle from `DV300` scenarios that differ in workload structure, model mode, utilization profile, or device-count semantics
- **THEN** the system refuses to generate the detailed bundle
- **AND** the system reports which shared-context dimensions are incompatible

#### Scenario: Reject a singleton selection
- **WHEN** a user selects fewer than two compatible scenarios for a per-device all-precision detailed bundle
- **THEN** the system refuses to generate the detailed bundle
- **AND** the system reports that at least two scenarios are required

### Requirement: Per-device detailed report stays comprehensive and precision-comparison oriented
The English per-device all-precision stakeholder report SHALL remain a detailed stakeholder artifact with shared device and workload assumptions, a precision comparison section, conclusion-oriented narrative, and appendix-style sweep tables for each included precision scenario.

#### Scenario: Detailed per-device report includes precision comparison and appendix content
- **WHEN** an English per-device all-precision stakeholder report is generated
- **THEN** it includes a shared context section, an at-a-glance precision comparison, detailed analysis for the included precision scenarios, and appendix-style tables
- **AND** it does not require prebuilt family-based subgroup bundles as source inputs
