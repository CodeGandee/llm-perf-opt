## Context

Wan2.1 already has the core pieces needed for this change, but they stop one layer short of being useful for `H20` and `B200`.

- The shared device registry already defines `H20` and `B200` in [extern/modelmeter/devices/gpu.py](/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/devices/gpu.py).
- The Wan2.1 sweep CLI is generic once a device selector is exposed through [extern/modelmeter/models/wan2_1/hardware_sweep.py](/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/hardware_sweep.py).
- The compact comparative stakeholder summary is already generic across device families and precision scenarios, but the detailed stakeholder-report generator remains hard-coded to the `DV100`/`DV200`/`DV300` trio.

So the real missing pieces are:
- exposing `h20` and `b200` as supported Wan2.1 sweep selectors
- generating first-pass Wan2.1 sweep runs for those devices
- replacing the DV-only detailed comparison generator with a generic detailed bundle path that can also produce a focused NVIDIA bundle

This change should keep the existing layered reporting model intact:

```text
single-device sweeps
        |
        v
detailed comparison bundles
        |
        v
comparative stakeholder summaries
```

## Goals / Non-Goals

**Goals:**
- Support `h20` and `b200` as first-class Wan2.1 sweep targets.
- Introduce one generic detailed multi-device stakeholder-report generator that can compare any compatible set of device runs under one shared context.
- Keep detailed bundle artifacts named `stakeholder-report.en.md` and `stakeholder-report.cn.md`.
- Publish first-pass Wan2.1 `fp8` metrics for `H20` and `B200` and make them reusable by the existing comparative-summary flow.
- Preserve native-sounding Chinese report prose with critical technical terminology left in English where that is clearer.

**Non-Goals:**
- Add `fp4` support for `H20` or `B200` in this change.
- Redesign the Wan2.1 analytic model or change its DP concurrency semantics.
- Replace the existing comparative summary generator.
- Produce Chinese-localized figure text.

## Decisions

### 1. Add `h20` and `b200` only through Wan2.1-local selector normalization

The Wan2.1 model layer will expose `h20` and `b200` by extending the selector map in `hardware_sweep.py`, while continuing to consume the shared registry classes as inputs.

Rationale:
- The shared registry already contains these devices, so the missing behavior is local selector exposure, not a new hardware source.
- This stays consistent with the existing `wan2-1-device-sweeps` boundary that Wan2.1 normalization logic lives under `extern/modelmeter/models/`.

Alternative considered:
- Edit the shared device registry or create separate Wan2.1-specific device classes.
Rejected because it duplicates hardware definitions or widens the change beyond the Wan2.1 model layer.

### 2. Introduce a generic detailed comparison generator and keep the DV entrypoint as a wrapper

The current `run_make_dv_stakeholder_report.py` flow will be generalized into a device-agnostic detailed comparison generator that accepts repeated device-run inputs for a shared context. The existing DV script can remain as a compatibility wrapper that calls the generic path with the fixed DV selectors.

Rationale:
- The present DV-specific script is the architectural bottleneck; adding an `H20`/`B200`-specific copy would hard-code another family-specific reporting path.
- A generic detailed bundle path lets future device sets such as `DV`, `NVIDIA`, or mixed-family comparisons reuse one reporting contract.
- Keeping the DV entrypoint preserves existing workflows and published bundle naming.

Alternative considered:
- Extend the DV-specific script directly with optional NVIDIA branches.
Rejected because it keeps the wrong abstraction and will become harder to maintain as more hardware families are added.

### 3. The generic detailed bundle will require one shared precision profile and one shared comparison context

The detailed comparison generator will only combine device runs that share:
- `input_struct`
- `model_mode`
- `util_profile`
- `device_num`
- `precision.name`
- `compute_precision`
- `storage_bits`

Variation is allowed in:
- device family
- device selector
- hardware peak values and resulting bottlenecks

Rationale:
- A detailed stakeholder report is meant to tell one coherent story for one workload slice and one precision assumption.
- Cross-precision comparison is already the job of the comparative summary layer.

Alternative considered:
- Allow mixed precisions inside one detailed bundle.
Rejected because it would blur the detailed-report contract and duplicate the role of the summary bundle.

### 4. First-pass `H20` and `B200` publication will be `fp8` only

This change will generate and publish `H20 fp8` and `B200 fp8` Wan2.1 metrics using the current shared registry values, but it will not add new `fp4` assumptions for these devices.

Rationale:
- The Wan2.1 precision-selection logic already supports `fp4`, but the current normalized `H20` and `B200` device metadata does not expose `fp4_tflops` in the shared registry.
- Publishing unsupported or improvised `fp4` peaks would weaken trust in the report more than it helps scope.

Alternative considered:
- Add provisional `fp4` numbers for `H20` and `B200` in the same change.
Rejected because the current user ask is about getting metrics published, not inventing new hardware assumptions that the registry does not yet carry.

### 5. The new NVIDIA bundle will be a detailed source bundle for the existing comparative summary flow

After generating `H20 fp8` and `B200 fp8` sweeps and building a detailed NVIDIA comparison bundle, the existing comparative stakeholder-summary generator will reuse that bundle alongside existing compatible detailed bundles such as `DV fp8`.

Rationale:
- The comparative summary layer is already designed to consume multiple detailed bundles under one shared context.
- Reusing it avoids reopening the summary architecture in this change.

Alternative considered:
- Build a separate one-off H20/B200 summary without going through detailed bundles.
Rejected because it bypasses the layered reporting model and creates redundant logic.

### 6. Device-profile disclosure stays explicit in the detailed report header

The generic detailed report will include a header note that the compared device-profile values come from the shared ModelMeter registry and should be treated as first-pass analytic assumptions pending validation.

Rationale:
- This matches the current DV reporting stance.
- It is especially important for newly published H20/B200 results because the reported metrics are theoretical outputs derived from registry peaks, not measured benchmarks.

Alternative considered:
- Omit the disclosure for NVIDIA devices because they are more familiar hardware names.
Rejected because the issue is not device popularity; it is whether the specific registry values have been externally validated for this analytic path.

## Risks / Trade-offs

- [Risk] The shared `H20` and `B200` registry values may be incomplete or too optimistic for stakeholder use. → Mitigation: keep explicit analytic-input disclosure in the report header and avoid expanding scope to unsupported precisions.
- [Risk] Generalizing the detailed report generator may accidentally break the current DV workflow. → Mitigation: keep the existing DV entrypoint as a wrapper and preserve current bundle names and report filenames.
- [Risk] Mixed-family detailed bundles could become too broad if users pass many devices at once. → Mitigation: keep the generator context validation strict and use the short comparative-summary layer for broader cross-bundle comparisons.
- [Risk] First-pass H20/B200 publication at only `fp8` may prompt immediate requests for `fp4`. → Mitigation: document that `fp4` requires explicit normalized hardware support and keep that as a follow-on change rather than a guessed extension.

## Migration Plan

1. Extend Wan2.1-local device selector normalization to support `h20` and `b200`.
2. Implement the generic detailed multi-device stakeholder-report generator and keep the DV generator as a thin compatibility wrapper.
3. Generate new Wan2.1 single-device sweep runs for `H20 fp8` and `B200 fp8`.
4. Publish a detailed NVIDIA comparison bundle from those runs in English and Chinese.
5. Regenerate the relevant comparative stakeholder-summary bundle(s) using the new NVIDIA detailed bundle plus existing compatible detailed bundles.
6. Update Wan2.1 reporting docs so users understand the difference between single-device sweeps, detailed comparison bundles, and comparative summaries.

Rollback:
- Revert the new device selectors and generic detailed generator while leaving existing DV detailed reporting intact.
- Remove the generated `h20`/`b200` artifacts and any comparative summary bundle that depends on them.

## Open Questions

- None for the first-pass proposal. If stakeholders later need `H20 fp4` or `B200 fp4`, that should be handled as a follow-on hardware-metadata change rather than assumed here.
