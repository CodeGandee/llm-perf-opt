## Why

The first `wan2-1-dv-hardware-sweeps` artifact bundle is numerically consistent, but the review found several stakeholder-facing report issues: the current `1s SLA` wording mixes latency and capacity concepts, the at-a-glance table overstates MemIO gap for the compute-limited `DV300` tier, and the generated markdown has an image-spacing bug. The same review also concluded that the DV device peaks should be presented as preliminary modeling inputs rather than authoritative hardware claims.

## What Changes

- Refine the consolidated DV stakeholder report wording so single-request latency and normalized 1-second capacity-gap metrics are clearly distinguished.
- Update the cross-device comparison table and summary text so the reported gap metric aligns with the primary bottleneck for each device tier instead of always emphasizing MemIO.
- Add explicit report disclosure that `DV100`, `DV200`, and `DV300` values are first-pass analytic assumptions sourced from the shared device registry.
- Fix markdown block formatting in the generated stakeholder report so figures always render on their own lines.
- Regenerate the consolidated DV stakeholder report and comparison artifacts after the reporting changes.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `wan2-1-dv-stakeholder-reporting`: Clarify report semantics, align gap reporting with the dominant resource constraint, disclose the provisional nature of DV device inputs, and ensure generated markdown formatting is clean.

## Impact

- Affected code: `extern/modelmeter/models/wan2_1/scripts/reporting/run_make_dv_stakeholder_report.py`, `extern/modelmeter/models/wan2_1/hardware_sweep_reporting.py`, and any Wan2.1-local reporting tests added for this follow-up.
- Affected outputs: `extern/modelmeter/models/wan2_1/reports/hardware_sweeps/comparisons/<comparison_run_id>/stakeholder-report.en.md`, `comparison-table.csv`, and related comparison/per-device figure titles or labels.
- Dependencies/systems: the existing Wan2.1 sweep result schema and the shared DV device aliases in `modelmeter.devices.gpu` remain inputs; this follow-up does not change the analytic cost model itself.
