## Why

Wan2.1 hardware reporting currently publishes DV-focused metrics, while `H20` and `B200` exist in the shared device registry but are not exposed as supported Wan2.1 sweep/report targets. We now need first-pass stakeholder-visible Wan2.1 metrics for these NVIDIA devices under the same analytic workload context so they can be compared fairly with existing DV bundles.

## What Changes

- Extend the Wan2.1 device-selectable sweep path to support `h20` and `b200` as normalized device profiles.
- Add a generic detailed multi-device stakeholder-report flow that can build one detailed comparison bundle from any compatible set of Wan2.1 device runs, rather than only from the hard-coded `DV100`/`DV200`/`DV300` trio.
- Generate first-pass Wan2.1 `fp8` sweep artifacts for `H20` and `B200` using the existing shared device-registry values, with report text that clearly marks those hardware assumptions as analytic inputs pending external validation.
- Publish English and Chinese detailed comparison bundles for the new NVIDIA runs and keep the Chinese version native-sounding while preserving critical English technical terms.
- Reuse the existing comparative stakeholder-summary flow to include the new H20/B200 detailed bundle(s) alongside existing DV detailed bundles under the same shared workload context.
- Do not introduce new `fp4` assumptions for `H20` or `B200` in this change; first-pass publication is limited to the precision profiles that the current normalized Wan2.1 device metadata can already support defensibly.

## Capabilities

### New Capabilities
- `wan2-1-multi-device-stakeholder-reporting`: Generate one English detailed Wan2.1 stakeholder report from a compatible set of device runs without hard-coding the device family to DV.
- `wan2-1-multi-device-cn-stakeholder-reporting`: Generate the matching Chinese detailed Wan2.1 stakeholder report from the same comparison bundle inputs and figures, using native Chinese prose with key technical terminology preserved in English where clearer.

### Modified Capabilities
- `wan2-1-device-sweeps`: Expand the supported Wan2.1 device selectors and normalized sweep metadata to include `h20` and `b200` so those devices can produce machine-readable sweep artifacts and downstream reports.

## Impact

- Affected code: `extern/modelmeter/models/wan2_1/hardware_sweep.py`, `extern/modelmeter/models/wan2_1/scripts/sizing/run_hardware_concurrency_sweep.py`, a new generic detailed comparison report generator under `extern/modelmeter/models/wan2_1/scripts/reporting/`, and related Wan2.1 reporting helpers/docs.
- Affected artifacts: new single-device sweep runs under `reports/hardware_sweeps/h20/` and `reports/hardware_sweeps/b200/`, new detailed comparison bundles, and regenerated comparative stakeholder-summary bundles that reuse those detailed outputs.
- Affected assumptions: `H20` and `B200` hardware values continue to come from the shared device registry and must be disclosed as first-pass analytic inputs pending validation.
