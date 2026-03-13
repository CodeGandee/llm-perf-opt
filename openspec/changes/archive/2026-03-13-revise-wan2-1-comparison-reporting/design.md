## Context

Wan2.1 reporting currently has three practical layers:

- single-device sweep runs under `reports/hardware_sweeps/<device>/<run_id>/`
- detailed comparison bundles that combine multiple devices under one shared precision profile
- comparative summary bundles that combine multiple detailed bundles

That structure worked for the first DV fp8, DV fp4, and NVIDIA fp8 outputs, but it scales poorly because every new grouping dimension creates another sibling bundle under `reports/hardware_sweeps/comparisons/`. The stakeholder ask is now clearer and simpler than the existing bundle graph:

- one detailed report per device across that device's available precision scenarios
- one curated cross-device summary across selected `(device, precision)` scenarios

The current selected summary set is “everything except `NGU800P`,” which means the summary needs to support explicit scenario selection rather than deriving its scope from intermediate subgroup bundles.

This change therefore needs a new source model for reporting. Instead of treating subgroup comparison bundles as the source of truth, the reporting layer should treat each completed sweep run as a scenario and build both report products directly from selected scenarios under one validated analytic context.

The selected cross-device summary also has a second complication: some stakeholder-facing analysis text is not something we want to hard-code in Python, but it is also not something we want to solve with an in-code LLM orchestration pipeline.
The better boundary for this repo is to generate the final summary markdown as a deterministic scaffold that already contains the factual content, tables, figures, and machine-derived fact bullets, while leaving explicit placeholder blocks for the user to fill manually with a reasoning LLM after generation.

Constraints:

- Generated artifacts stay filesystem-only under `extern/modelmeter/models/wan2_1/reports/hardware_sweeps/`.
- Existing single-device sweep outputs and their `summary.md` files remain valid.
- English and Chinese artifacts must stay numerically aligned and share one data path.
- The current multi-device detailed and comparative-summary commands may need to remain temporarily for rollback and migration safety, but they should stop being the primary documented workflow.

## Goals / Non-Goals

**Goals:**
- Make scenario selection the primary input model for Wan2.1 comparison reporting.
- Add one detailed bundle per device that compares all selected precision scenarios for that device.
- Rework the comparative summary so it consumes selected scenarios directly rather than subgroup comparison bundles.
- Preserve deterministic ordering with the DV series first in cross-device summaries.
- Generate selected summary markdown as an LLM-ready scaffold with deterministic factual sections and explicit placeholder regions for analysis text.
- Separate comparison outputs by report role so the directory tree is easier to scan.
- Keep detailed reports long-form and summary reports concise.

**Non-Goals:**
- Redesign the Wan2.1 analytic sweep model, workload grid, or device metadata model.
- Remove or rewrite single-device `summary.md` generation.
- Create dashboard-style multi-workload comparisons.
- Force single-precision devices to emit degenerate “all-precision” bundles.
- Localize figure assets separately for Chinese.
- Add an automated LLM fill pipeline, slot JSON protocol, or final compose step inside the codebase.

## Decisions

### 1. Reporting will use a scenario-first source model built from raw sweep runs

Both new report products will load source scenarios directly from `reports/hardware_sweeps/<device>/<run_id>/`, where one scenario is one completed sweep run with one device selector and one precision profile.

Each scenario record will carry:

- `device_selector`
- `device_name`
- `device_family`
- `run_id`
- `precision`
- `compute_precision`
- `storage_bits`
- workload/context fields needed for compatibility validation
- computed summary metrics used by reporting

Rationale:

- It removes the current dependency on subgroup bundles as intermediate source artifacts.
- The same scenario abstraction can feed both per-device detailed bundles and selected summary bundles.
- Metadata can record source runs directly, which makes report provenance clearer.

Alternative considered:

- Keep comparative summaries sourced from detailed subgroup bundles.
Rejected because it preserves the current group explosion and keeps scenario selection indirect.

### 2. The reporting surface will have exactly two primary comparison products

The new comparison workflow will produce:

- `by-device` detailed bundles: one device, multiple precision scenarios, rich stakeholder narrative
- `summaries` bundles: selected scenarios across devices, short stakeholder summary

The detailed per-device bundle will require at least two compatible scenarios for the same device. Devices with only one precision run today, such as `H20`, `B200`, and `NGU800P`, will continue to rely on their existing single-run `summary.md` outputs until additional precision runs exist.

Rationale:

- These two products match the stakeholder questions directly.
- They replace the current family-based detailed groupings without losing the rich report surface.
- They avoid one universal report that would either be too long or too shallow.

Alternative considered:

- Replace everything with one global grand comparison report.
Rejected because it cannot serve the need for device-specific detailed analysis across precisions.

### 3. Cross-device summaries will use explicit scenario selection and freeze that selection in metadata

The summary generator will accept an explicit list of selected scenarios. The current recommended preset is “all currently available scenarios except `NGU800P`,” but the report contract will not hard-code that exclusion as the only supported selection.

Summary bundle metadata will record:

- selected scenario ids
- source run ids
- any applied include/exclude filters or preset name
- final scenario order used for tables and figures

Rationale:

- Stakeholders want a curated summary, not every possible scenario by default forever.
- Freezing selection in metadata preserves reproducibility even when discovery helpers or presets are used.

Alternative considered:

- Auto-include every compatible scenario unconditionally.
Rejected because it turns curation into an implicit side effect of whatever runs happen to exist on disk.

### 4. Cross-device scenario ordering will reuse the existing DV-first rule and keep non-DV ordering simple

Summary ordering will keep the existing rule that `DV` scenarios appear before non-DV scenarios. Within the DV series, devices remain ordered `dv100`, `dv200`, `dv300`. Within one device, scenarios remain grouped by device before precision. For non-DV scenarios, ordering will stay simple and deterministic by normalized device name.

Rationale:

- DV-first is an explicit stakeholder preference.
- Reusing the current non-DV alphabetical behavior avoids introducing extra policy without a clear requirement.

Alternative considered:

- Introduce a multi-family ranking such as `dv`, then `nvidia`, then `ngu`.
Rejected because only DV-first is currently required and extra family ordering would be arbitrary.

### 5. Selected summary markdown will be a deterministic scaffold with inline analysis placeholders

The summary generator will write `stakeholder-summary.en.md` and `stakeholder-summary.cn.md` as editable markdown scaffolds.
Each summary scaffold will contain:

- fully rendered factual sections such as scope, workload slice, scenario list, comparison table, and generated figures
- machine-derived fact bullets such as rankings, ratios, and bottleneck facts when they are directly computed from the selected scenarios
- explicit placeholder blocks for analysis sections that the user will later rewrite in place with a reasoning LLM
- visible boundary markers separating generated factual regions from LLM-editable regions

The code will not invoke an LLM, fill JSON slots, or compose a second markdown artifact.
The human workflow is: generate scaffold, manually invoke LLM against the markdown, replace placeholder blocks in place, and stop rerunning the generator afterward unless the user is willing to overwrite those edits.

Rationale:

- It keeps the code path deterministic, testable, and fully local.
- It avoids overcommitting Python logic to narrative analysis that is better handled by a reasoning model.
- It avoids adding extra intermediate artifacts such as slot-fill JSON that the user does not want to manage.
- It gives the user one markdown file per locale to hand directly to an LLM.

Alternative considered:

- Introduce a structured JSON fill-and-compose pipeline.
Rejected because the user will perform the reasoning step manually and the extra artifact surface would add more complexity than value.

Alternative considered:

- Continue generating final stakeholder prose entirely in Python.
Rejected because richer combined-summary analysis is not always easily or safely derivable from scenario selection alone.

### 6. Detailed and summary bundles will keep different artifact depth, but share one data path

Per-device detailed bundles will keep:

- `stakeholder-report.en.md`
- `stakeholder-report.cn.md`
- `comparison-table.csv`
- `bundle-metadata.json`
- richer comparison figures
- appendix-style tables

Selected summary bundles will keep:

- `stakeholder-summary.en.md`
- `stakeholder-summary.cn.md`
- `comparison-table.csv`
- `bundle-metadata.json`
- a compact figure set

The selected summary markdown files are draft scaffolds rather than fully authored final prose, but they are still the primary summary artifact path produced by the code.
Both products will be generated from the same scenario selection and validation helpers so compatibility logic, scenario ids, and core metrics stay aligned.

Rationale:

- Stakeholders still need both long-form and short-form reporting.
- Sharing the data path reduces divergence and makes bilingual outputs easier to keep aligned.

Alternative considered:

- Collapse both products into one artifact format.
Rejected because the detailed appendix content and the concise executive summary serve different audiences.

### 7. Comparison outputs will be reorganized by role, and existing subgroup bundles become legacy

The new output layout under `reports/hardware_sweeps/comparisons/` will separate current products from historical subgroup bundles:

- `by-device/<run_id>-<device>-all-precision/`
- `summaries/<run_id>-selected-scenarios/`
- `legacy/` for older family-based subgroup bundles when archived

Rationale:

- This removes the current flat pile of unrelated bundle names.
- The path itself communicates whether a bundle is detailed or summary.

Alternative considered:

- Keep the current flat directory and only change naming conventions.
Rejected because the root directory would remain noisy even with better names.

## Risks / Trade-offs

- [Risk] Direct raw-run sourcing introduces a second compatibility path unless shared helpers are factored cleanly. → Mitigation: implement shared scenario loading and compatibility validation used by both report roles.
- [Risk] Two primary comparison products still exist, which could be seen as complexity. → Mitigation: the split is by stakeholder need and is reflected explicitly in the directory layout and docs.
- [Risk] Single-precision devices will not immediately get a new per-device detailed bundle. → Mitigation: keep the existing per-run `summary.md` as the detailed artifact for those devices until more precision runs exist.
- [Risk] Historical flat comparison bundles may continue to confuse readers during migration. → Mitigation: stop documenting them as current outputs and move them under `legacy/` when practical.
- [Risk] Existing automation may assume summary metadata references subgroup bundles. → Mitigation: document the new scenario-level metadata contract and preserve source run ids in every bundle.
- [Risk] Users may rerun the generator after manually filling placeholder sections and accidentally overwrite LLM-authored prose. → Mitigation: place a prominent draft warning at the top of generated summaries and document a regenerate-before-edit workflow.
- [Risk] A manual LLM may rewrite factual sections or alter numeric claims when editing the markdown. → Mitigation: mark generated factual regions and LLM-editable placeholder regions explicitly in the markdown template and instruct users to keep edits inside the placeholder blocks.

## Migration Plan

1. Add shared scenario-loading and compatibility-validation helpers that operate directly on raw sweep runs.
2. Implement the per-device all-precision detailed reporting path on top of those shared helpers.
3. Rework comparative summary reporting to consume selected scenarios directly and emit summary scaffolds with deterministic facts, figures, and inline placeholder blocks for analysis.
4. Regenerate the current desired outputs:
   - per-device detailed bundles for `DV100`, `DV200`, and `DV300`
   - one selected summary scaffold covering the current default selection of all scenarios except `NGU800P`
5. Update Wan2.1 reporting documentation and README examples to describe the manual LLM-editing workflow for summary placeholder blocks.
6. Archive or de-emphasize the old family-based subgroup bundles under `comparisons/legacy/`.

Rollback:

- Keep the current subgroup-bundle generators available until the new flows are validated.
- If the new scenario-first reporting path misbehaves, restore the old comparison docs and continue generating the previous subgroup bundles while preserving the raw sweep runs.

## Open Questions

- Should the new summary flow expose only explicit scenario lists, or also a named preset such as `all-except-ngu800p` in the initial CLI surface?
- Should both `stakeholder-summary.en.md` and `stakeholder-summary.cn.md` ship as scaffold drafts with placeholders in the first implementation, or should the placeholder workflow be limited to one locale initially?
