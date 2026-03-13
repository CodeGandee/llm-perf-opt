## MODIFIED Requirements

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
