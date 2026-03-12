## Context

The current Wan2.1 DV analysis path can only be run and reported as `fp16`, `fp8`, or `fp8_mixed`, even though the DV device registry already includes `fp4_tflops` for `DV100`, `DV200`, and `DV300`. The published DV stakeholder bundle is therefore precision-specific in practice and currently represents only the `fp8` case.

The sweep and reporting layers also have two important constraints today:
- The hardware sweep CLI and tensor-peak helpers only recognize `fp8` and `fp16` as compute-precision values.
- The DV stakeholder report generator validates that all three DV input runs share exactly one precision context before it will emit a report.

That means “add fp4 stats” is not just a markdown change. The analytic precision contract, sweep metadata, and report artifact layout all need to become explicit about how fp4 is represented and published.

Stakeholders for this change are:
- Readers using DV stakeholder reports for early hardware sizing decisions
- Authors regenerating analytic DV sweeps and comparison bundles
- Future report consumers who need to compare fp8 and fp4 without guessing which precision the numbers came from

## Goals / Non-Goals

**Goals:**
- Add first-class `fp4` support to the Wan2.1 device-sweep contract and emitted metadata.
- Produce fp4 DV100, DV200, and DV300 sweep artifacts using the same workload grid and DP semantics as the current fp8 runs.
- Keep the detailed DV stakeholder bundle precision-specific so each report remains readable and internally consistent.
- Add a compact cross-precision DV stakeholder summary that compares fp8 and fp4 side by side.
- Keep English and Chinese stakeholder outputs aligned for the new precision-aware artifacts.

**Non-Goals:**
- Add measured runtime benchmarking or validate the fp4 assumptions against serving traces.
- Change the existing DP concurrency model or workload grid.
- Introduce Chinese-localized figure assets.
- Introduce `fp4_mixed` or other extra precision profiles in the same change.

## Decisions

### 1. Treat `fp4` as a first-class analytic precision profile

The change will add an explicit Wan2.1 precision profile named `fp4`, with `compute_precision=fp4` and `storage_bits=4`.

Rationale:
- The user request is for fp4 stats, not just a marketing label on fp8 results.
- The existing Wan2.1 precision contract already treats precision profiles as the source of truth for both compute peak selection and IO/VRAM scaling, so fp4 should follow the same pattern.
- This produces self-describing artifacts and avoids hidden special cases in the sweep code.

Alternative considered:
- Model fp4 as “fp4 compute with fp8 storage” inside the same `fp4` label.
Rejected because it overloads one label with two assumptions and makes the resulting artifacts hard to interpret. If mixed fp4 storage assumptions are needed later, that should be a separate profile such as `fp4_mixed`.

### 2. Keep detailed DV stakeholder reports single-precision

The existing detailed DV report shape will remain one precision context per bundle. The system will generate one fp8 bundle and one fp4 bundle, each with the same English/Chinese report pair, figure set, and comparison table shape.

Rationale:
- The current detailed report contains many precision-dependent sections: precision assumptions, single-request breakdown, live tensor consumption, device peaks, sizing gaps, and appendices.
- Folding fp8 and fp4 into one detailed document would roughly double the report size and make the appendix sections much harder to scan.
- Reusing the current single-precision report structure minimizes implementation churn and keeps existing report-reading habits intact.

Alternative considered:
- Replace the existing report with one combined multi-precision report.
Rejected because it optimizes for artifact count at the expense of readability.

### 3. Add a separate compact cross-precision summary bundle

In addition to the detailed fp8 and fp4 bundles, the system will generate a smaller cross-precision summary artifact that compares `(device, precision)` pairs side by side for the common workload slice.

Rationale:
- Stakeholders will otherwise need to open two long detailed reports to answer simple questions such as “how much does DV300 improve from fp8 to fp4?”
- A compact summary can focus on the handful of metrics that decision-makers care about most without duplicating the full appendix tables.
- This keeps the detailed bundles useful for deep reading while creating a quick-entry comparison layer.

Alternative considered:
- Skip the cross-precision summary and rely only on separate fp8/fp4 bundles.
Rejected because it satisfies correctness but leaves the main comparison workflow clumsy.

### 4. Preserve compatibility checks within a precision group, relax them across grouped precisions

The detailed report generator will still require all three DV inputs in a given bundle to share one workload slice and one precision profile. The new cross-precision summary will require shared workload/device grid semantics across fp8 and fp4 groups, but it will allow the precision value itself to differ by group.

Rationale:
- Precision-specific detailed reports should remain internally coherent.
- The current strict “all inputs must share one precision” rule is correct for one detailed bundle but too strict for a higher-level fp8-vs-fp4 comparison view.

Alternative considered:
- Remove precision compatibility checks altogether.
Rejected because it would make it too easy to publish mixed-context reports with inconsistent workload assumptions.

### 5. Keep new artifacts additive

The change will add new fp4 device-run artifacts and new precision-aware comparison outputs without deleting or renaming the existing fp8 bundle shape.

Rationale:
- Existing links and references remain valid while the new reports are validated.
- Rollback is straightforward because the new work is additive rather than in-place destructive.

Alternative considered:
- Reuse the existing comparison directory and overwrite it with multi-precision content.
Rejected because it makes validation, diffing, and rollback harder.

## Risks / Trade-offs

- [Risk] `fp4` may be interpreted as a deployment-ready claim rather than a first-pass analytic assumption. → Mitigation: keep explicit precision-profile disclosure in every generated report and summary, and preserve the existing scope note that these are analytic sizing artifacts.
- [Risk] The change expands the precision contract beyond what some helper functions currently support. → Mitigation: update the shared Wan2.1 precision helpers and add regression tests at the helper and reporting layers.
- [Risk] Maintaining English and Chinese variants across fp8, fp4, and summary outputs increases wording surface area. → Mitigation: keep shared numeric summaries and shared figure generation paths, and localize only the markdown text layer.
- [Risk] A compact cross-precision summary could drift from the detailed per-precision reports if it recomputes logic separately. → Mitigation: build the summary from the same per-device/per-precision summary objects that feed the detailed reports.

## Migration Plan

1. Extend Wan2.1 precision configs and helper functions to accept `fp4`.
2. Regenerate DV100, DV200, and DV300 sweeps under `fp4` using the same workload grid as the current fp8 runs.
3. Generate precision-specific DV comparison bundles for `fp8` and `fp4`.
4. Generate the compact cross-precision summary bundle from the validated fp8 and fp4 comparison inputs.
5. Update Wan2.1 docs to describe the precision-aware artifact layout.

Rollback:
- Leave existing fp8 artifacts untouched until the new fp4 and summary bundles are validated.
- If needed, remove the new fp4 and precision-summary artifacts while retaining the current fp8 bundle and prior reporting entrypoints.

## Open Questions

- None for this change. If a mixed storage assumption is needed later, it should be proposed explicitly as a follow-on profile such as `fp4_mixed` rather than folded into `fp4`.
