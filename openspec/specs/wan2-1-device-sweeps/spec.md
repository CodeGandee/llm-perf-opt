### Requirement: Device-selectable Wan2.1 sweep
The system SHALL allow the Wan2.1 analytic hardware sweep to run against a named device profile instead of being limited to `NGU800P`. Supported profiles MUST include `ngu800p`, `dv100`, `dv200`, `dv300`, `h20`, and `b200`.

#### Scenario: Run sweep with a supported device profile
- **WHEN** a user runs the Wan2.1 sizing workflow with one of the supported device profile names
- **THEN** the system generates sweep artifacts using that device profile's hardware assumptions

#### Scenario: H20 and B200 are valid Wan2.1 sweep targets
- **WHEN** a user runs the Wan2.1 sizing workflow with `device=h20` or `device=b200`
- **THEN** the system accepts the selector
- **AND** the generated machine-readable artifacts record the normalized hardware assumptions for that device

#### Scenario: Reject an unsupported device profile
- **WHEN** a user requests a device profile that the sweep does not support
- **THEN** the system fails fast with an error that identifies the invalid device name and lists the supported device profiles

### Requirement: Model-local implementation boundary
The system SHALL implement the device-selectable Wan2.1 sweep without modifying code outside `extern/modelmeter/models/`.

#### Scenario: Sweep uses shared device specs without editing them
- **WHEN** the implementation consumes shared device aliases such as `DV100`, `DV200`, and `DV300`
- **THEN** any normalization or compatibility logic is implemented under `extern/modelmeter/models/` rather than by changing shared modules outside that subtree

### Requirement: Device-normalized sweep metadata
The system SHALL emit sweep metadata that records the normalized hardware assumptions and precision assumptions used for the run so reporting tools can interpret results consistently across device families and precision profiles.

#### Scenario: Sweep metadata records device and precision assumptions
- **WHEN** a sweep run completes successfully
- **THEN** the generated machine-readable artifacts include the selected device name and the normalized compute, memory-bandwidth, and interconnect values used for the run
- **AND** the generated machine-readable artifacts include the selected precision profile name, compute precision, storage bits, and the tensor-peak context implied by that precision

### Requirement: Device-scoped artifact layout
The system SHALL store sweep artifacts under a device-scoped path so results from different hardware families remain isolated and discoverable.

#### Scenario: Same run identifier on different devices does not collide
- **WHEN** two sweep runs use the same run identifier but different device profiles
- **THEN** the artifacts are written to separate device-scoped directories rather than overwriting one another

### Requirement: Comparable DP sweep semantics
The system SHALL preserve the current Wan2.1 DP concurrency semantics across supported device profiles so cross-device results remain directly comparable.

#### Scenario: Effective GPU count remains DP-limited by batch size
- **WHEN** the sweep evaluates a configuration where `batch_size` is smaller than the requested device count
- **THEN** the reported effective GPU count is limited to `min(batch_size, device_num)` for that configuration

### Requirement: Precision-selectable Wan2.1 sweep profiles
The system SHALL allow the Wan2.1 analytic hardware sweep to run with supported precision profiles, and supported profiles MUST include `fp4` in addition to the existing Wan2.1 precision options when the selected device profile exposes an fp4 tensor peak.

#### Scenario: Run a DV sweep with the fp4 profile
- **WHEN** a user runs the Wan2.1 sizing workflow for `dv100`, `dv200`, or `dv300` with `precision=fp4`
- **THEN** the system generates sweep artifacts using the fp4 tensor peak for that device
- **AND** the generated artifacts record `precision=fp4`, `compute_precision=fp4`, and `storage_bits=4`

#### Scenario: Reject fp4 on a device profile that lacks fp4 support
- **WHEN** a user requests `precision=fp4` for a device profile whose normalized metadata does not include an fp4 tensor peak
- **THEN** the system fails fast with an error that identifies the unsupported precision/device combination
