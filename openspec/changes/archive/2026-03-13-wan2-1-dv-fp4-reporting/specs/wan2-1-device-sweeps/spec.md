## ADDED Requirements

### Requirement: Precision-selectable Wan2.1 sweep profiles
The system SHALL allow the Wan2.1 analytic hardware sweep to run with supported precision profiles, and supported profiles MUST include `fp4` in addition to the existing Wan2.1 precision options when the selected device profile exposes an fp4 tensor peak.

#### Scenario: Run a DV sweep with the fp4 profile
- **WHEN** a user runs the Wan2.1 sizing workflow for `dv100`, `dv200`, or `dv300` with `precision=fp4`
- **THEN** the system generates sweep artifacts using the fp4 tensor peak for that device
- **AND** the generated artifacts record `precision=fp4`, `compute_precision=fp4`, and `storage_bits=4`

#### Scenario: Reject fp4 on a device profile that lacks fp4 support
- **WHEN** a user requests `precision=fp4` for a device profile whose normalized metadata does not include an fp4 tensor peak
- **THEN** the system fails fast with an error that identifies the unsupported precision/device combination

## MODIFIED Requirements

### Requirement: Device-normalized sweep metadata
The system SHALL emit sweep metadata that records the normalized hardware assumptions and precision assumptions used for the run so reporting tools can interpret results consistently across device families and precision profiles.

#### Scenario: Sweep metadata records device and precision assumptions
- **WHEN** a sweep run completes successfully
- **THEN** the generated machine-readable artifacts include the selected device name and the normalized compute, memory-bandwidth, and interconnect values used for the run
- **AND** the generated machine-readable artifacts include the selected precision profile name, compute precision, storage bits, and the tensor-peak context implied by that precision
