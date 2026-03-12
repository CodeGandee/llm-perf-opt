## Why

The current DV stakeholder bundle only covers one precision context, `fp8`, even though the DV device registry already carries `fp4` tensor peak assumptions. That leaves stakeholders without a first-class way to compare DV100, DV200, and DV300 under the fp4 scenario they are likely to ask about, and it also makes the current report easy to misread as a complete precision story when it is only one slice.

## What Changes

- Extend the Wan2.1 DV sweep contract so the analytic hardware sweep can model `fp4` in addition to the existing precision profiles.
- Generate fp4 sweep artifacts for `DV100`, `DV200`, and `DV300` with the same workload grid and DP semantics currently used for the fp8 DV runs.
- Update the DV stakeholder reporting flow so fp8 and fp4 are both represented as first-class report outputs rather than forcing fp4 readers to infer results from fp8-only artifacts.
- Add a compact cross-precision stakeholder view that makes fp8 vs fp4 differences easy to compare without duplicating every detailed appendix table into one oversized report.
- Update the bilingual reporting docs so the new precision-specific artifacts and their intended interpretation are explicit.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `wan2-1-device-sweeps`: expand the sweep precision contract and emitted metadata so DV runs can be generated and interpreted under `fp4`.
- `wan2-1-dv-stakeholder-reporting`: change DV reporting from a single fp8-only stakeholder bundle to precision-aware reporting that covers fp8 and fp4 explicitly.
- `wan2-1-dv-cn-stakeholder-reporting`: keep the Chinese stakeholder path aligned with the English precision-aware reporting outputs and scope.

## Impact

Affected areas include the Wan2.1 precision configs, the device-peak selection helpers, the hardware sweep CLI and metadata, the DV stakeholder report generator, the generated DV comparison bundles, and the Wan2.1 reporting documentation under `extern/modelmeter/models/wan2_1/`.
