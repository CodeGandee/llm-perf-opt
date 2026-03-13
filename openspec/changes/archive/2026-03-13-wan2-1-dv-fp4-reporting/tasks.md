## 1. Extend the Wan2.1 precision and sweep contract

- [x] 1.1 Add the Wan2.1 `fp4` precision profile and update the precision-selection helpers so device tensor-peak selection and metadata support `compute_precision=fp4`
- [x] 1.2 Extend the Wan2.1 hardware sweep CLI and metadata/reporting surfaces to accept `precision=fp4` and to fail fast when a selected device profile does not expose fp4 support
- [x] 1.3 Add regression coverage for fp4 precision selection, emitted precision metadata, and unsupported fp4/device combinations

## 2. Generate precision-specific DV detailed reporting

- [x] 2.1 Update the DV detailed reporting flow so per-device figures and stakeholder bundles remain single-precision but label their precision context explicitly
- [x] 2.2 Generate fp4 sweep artifacts for `DV100`, `DV200`, and `DV300` using the same workload grid and DP semantics as the current fp8 runs
- [x] 2.3 Generate precision-specific English and Chinese DV stakeholder bundles for fp8 and fp4 without output-path collisions
- [x] 2.4 Add regression coverage for precision-specific DV compatibility checks and bilingual detailed bundle generation

## 3. Add the compact fp8-vs-fp4 summary layer

- [x] 3.1 Implement a compact DV cross-precision summary flow that consumes compatible fp8 and fp4 comparison groups and compares `(device, precision)` pairs side by side
- [x] 3.2 Add English and Chinese cross-precision summary outputs that reuse the same computed comparison data while localizing only the markdown text
- [x] 3.3 Add regression coverage for grouped fp8/fp4 compatibility validation and cross-precision summary artifact generation

## 4. Regenerate artifacts and document the new layout

- [x] 4.1 Update Wan2.1 reporting documentation to describe the new fp8, fp4, and cross-precision DV artifact layout and interpretation
- [x] 4.2 Regenerate the DV fp4 device runs and the precision-aware comparison bundles, and verify the resulting artifacts match the documented precision contexts
