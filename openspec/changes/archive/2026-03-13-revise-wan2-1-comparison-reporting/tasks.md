## 1. Build scenario-first reporting inputs

- [x] 1.1 Add shared reporting helpers that load raw Wan2.1 sweep runs as scenario records with device, precision, source run, and summary-metric fields.
- [x] 1.2 Add shared compatibility validation for per-device detailed bundles and selected summaries using raw scenario metadata instead of subgroup comparison bundles.
- [x] 1.3 Update bundle metadata writers to record selected scenario ids, source run ids, and final scenario ordering directly.

## 2. Implement per-device all-precision detailed bundles

- [x] 2.1 Add a per-device detailed reporting entrypoint that accepts two or more compatible scenarios for one device and writes bundles under the new `comparisons/by-device/` layout.
- [x] 2.2 Generate English per-device all-precision detailed artifacts, comparison tables, and figures from the selected precision scenarios.
- [x] 2.3 Generate Chinese per-device all-precision detailed artifacts from the same data and enforce rejection of singleton or mixed-device inputs.

## 3. Rework selected-scenario comparative summaries

- [x] 3.1 Refactor comparative summary reporting to accept explicit scenario selections from raw sweep runs instead of subgroup comparison bundles.
- [x] 3.2 Apply deterministic DV-first ordering and render English summary scaffold markdown with deterministic factual sections, explicit analysis placeholder blocks, draft warnings, and figures from the selected scenarios.
- [x] 3.3 Generate Chinese summary scaffold markdown from the same selected scenario data, preserve the same editable boundaries, and support the current curated summary preset of all available scenarios except `NGU800P`.

## 4. Migrate outputs, docs, and validation

- [x] 4.1 Update comparison output layout, legacy-bundle handling, and Wan2.1 reporting docs to reflect `by-device`, `summaries`, archived subgroup bundles, and the manual LLM-editing workflow for summary placeholder blocks.
- [x] 4.2 Add regression coverage for same-device validation, summary scenario ordering, bilingual output alignment, scenario-level metadata, and generated-vs-editable placeholder boundaries.
- [x] 4.3 Regenerate the current target outputs for DV per-device all-precision bundles and the selected summary scaffold excluding `NGU800P`, then verify README/examples point to the new workflow.
