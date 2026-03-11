## Why

The current Wan2.1 hardware sizing workflow is hard-coded around NGU800P, even though the repo already carries preliminary `DV100`/`DV200`/`DV300` device aliases. We need a repeatable way to run the same analytic sweep on multiple device profiles and produce apples-to-apples comparison artifacts for hardware planning.

## What Changes

- Generalize the Wan2.1 concurrency sweep flow so it can run against multiple device definitions instead of only `NGU800P`.
- Add support for `DV100`, `DV200`, and `DV300` as selectable sizing targets for the Wan2.1 analytic sweep.
- Standardize sweep outputs and metadata so downstream reporting can identify the device profile used for each run.
- Generalize the figure/report generation path so it can produce per-device figures and a cross-device comparison view, rather than NGU-only output names and titles.
- Add one stakeholder-facing English report that covers `DV100`, `DV200`, and `DV300` in separate sections, using a structure similar to the existing NGU800P stakeholder report.
- Keep the implementation self-contained under `extern/modelmeter/models/`; this change does not modify shared code outside that subtree.

## Capabilities

### New Capabilities
- `wan2-1-device-sweeps`: Run the existing Wan2.1 analytic hardware sweep against a selected device profile, including `DV100`/`DV200`/`DV300`, and emit device-tagged artifacts.
- `wan2-1-dv-stakeholder-reporting`: Generate per-device figures plus one consolidated DV stakeholder report from sweep results so stakeholders can compare bottlenecks, throughput saturation, and SLA gaps across `DV100`/`DV200`/`DV300`.

### Modified Capabilities
- None.

## Impact

- Affected code: `extern/modelmeter/models/wan2_1/scripts/sizing/`, `extern/modelmeter/models/wan2_1/scripts/reporting/`, `extern/modelmeter/models/wan2_1/reports/README.md`, and any new Wan2.1-local helpers created under `extern/modelmeter/models/`.
- Affected outputs: Wan2.1 sizing run directories, figure names/locations, summary metadata, and stakeholder reports.
- Dependencies/systems: Hydra-based Wan2.1 analytic model configs, `multi_device_cost`, and the existing device-spec registry in `modelmeter.devices.gpu` as a read-only input source.
