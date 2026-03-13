## Why

Wan2.1 comparison reporting currently multiplies outputs by device family and precision grouping, which leaves `reports/hardware_sweeps/comparisons/` with too many parallel bundles for stakeholders to navigate. The reporting need is simpler: one detailed report per device across its available precision variants, plus one curated cross-device summary built from selected `(device, precision)` scenarios.

## What Changes

- Add per-device all-precision detailed stakeholder reporting that compares every available precision run for one device in a single bundle.
- Rework comparative stakeholder summary reporting to select scenarios directly from raw `(device, precision)` sweep runs instead of stitching together subgroup comparison bundles.
- Generate selected summary markdown as an editable scaffold with deterministic factual content, figures, and explicit placeholder blocks for analysis text that users will later fill manually with an LLM.
- Support curated scenario selection for the summary flow, with the current default selection including all available scenarios except `NGU800P`.
- Reorganize comparison outputs and metadata around report role and source scenarios so the comparison tree is easier to understand and legacy subgroup bundles can be archived or de-emphasized.

## Capabilities

### New Capabilities
- `wan2-1-device-all-precision-stakeholder-reporting`: Generate one detailed stakeholder bundle per device that compares that device's available precision scenarios in one shared analytic context.
- `wan2-1-device-all-precision-cn-stakeholder-reporting`: Generate a Chinese detailed stakeholder bundle alongside the English per-device all-precision detailed report from the same source scenarios and figures.

### Modified Capabilities
- `wan2-1-comparative-stakeholder-summary-reporting`: Change comparative summary reporting to consume an explicit selection of `(device, precision)` scenarios directly from sweep runs, preserve DV-first ordering, emit editable scaffold markdown with placeholder analysis blocks, and record scenario-level source metadata instead of source comparison bundles.
- `wan2-1-comparative-cn-stakeholder-summary-reporting`: Change the Chinese comparative summary to mirror the revised scenario-selection scope, metadata, figures, and placeholder-scaffold workflow used by the English summary.

## Impact

- Affected code: Wan2.1 reporting scripts, comparison-bundle metadata writers, report directory layout, and README/docs that describe the reporting workflow.
- Affected artifacts: `comparison-table.csv`, bundle metadata, detailed stakeholder markdown, comparative summary markdown, and comparison figures.
- User workflow impact: selected summary markdown becomes a generated draft scaffold that users may edit manually with an LLM after generation.
- Affected systems: `extern/modelmeter/models/wan2_1/scripts/reporting/`, `extern/modelmeter/models/wan2_1/reports/hardware_sweeps/comparisons/`, and the related OpenSpec capability specs for Wan2.1 reporting.
