## Context

Wan2.1 currently has two report layers for DV hardware sweeps.

- Detailed comparison bundles are precision-specific and produce `stakeholder-report.en.md`, `stakeholder-report.cn.md`, `comparison-table.csv`, and figures for one validated comparison context.
- A newer compact summary path exists, but it is implemented as a specific `fp8`-vs-`fp4` flow in [run_make_dv_precision_summary_report.py](extern/modelmeter/models/wan2_1/scripts/reporting/run_make_dv_precision_summary_report.py), with fp8/fp4-specific CLI arguments, fp8/fp4-specific wide CSV columns, and report wording centered on that one pairwise comparison.

That current summary path is already narrower than the expected next use cases.
We want one short stakeholder-facing comparative summary that can answer questions such as “how does Wan2.1 compare across DV300 fp4, DV300 fp8, B200 fp8, and H200 fp8 under the same workload slice?” without reintroducing long appendix tables or rewriting the summary format every time a new hardware family is added.

The summary should therefore treat each compared case as a reusable scenario under one shared analytic context, while continuing to reuse the already generated detailed comparison bundles as the source of truth.

## Goals / Non-Goals

**Goals:**
- Generalize the compact summary from a hard-coded `fp8`-vs-`fp4` report into a reusable comparative stakeholder summary.
- Reuse existing detailed comparison bundle outputs rather than reloading raw sweep runs or recomputing summary logic separately.
- Keep the comparative report short, conclusion-first, and suitable for stakeholder reading.
- Standardize the comparative artifact naming on `stakeholder-summary.en.md` and `stakeholder-summary.cn.md`.
- Make the summary data model and validation logic work for future hardware and precision additions such as `DV`, `B200`, `H200`, `fp8`, and `fp4`.

**Non-Goals:**
- Replace the detailed per-bundle stakeholder reports.
- Redesign the underlying Wan2.1 analytic sweep model or its workload grid semantics.
- Add localized Chinese figure assets.
- Solve every future grouping problem such as heterogeneous multi-workload dashboards in this change.

## Decisions

### 1. The comparative summary will consume detailed comparison bundles, not raw sweep runs

The new summary layer will accept multiple compatible detailed comparison bundles as inputs and derive comparative scenario rows from their `comparison-table.csv` and `bundle-metadata.json` files.

Rationale:
- The detailed bundle is already the validated, stable summary product for one comparison context.
- Reusing it avoids duplicating raw-sweep compatibility logic and reduces drift between detailed and summary outputs.
- This keeps the comparative summary additive and cheap to regenerate after new hardware bundles are produced.

Alternative considered:
- Read raw per-device sweep directories directly.
Rejected because it would duplicate detailed-report logic, create a second compatibility surface, and make summary regeneration more fragile.

### 2. The summary data model will be scenario-based and long-format

Each row in the comparative `comparison-table.csv` will represent one scenario under a shared context, where a scenario is one hardware and precision combination such as `DV300 fp4` or `B200 fp8`.

The row schema will be generic and include fields such as:
- `scenario_id`
- `scenario_label`
- `device_name`
- `precision`
- `compute_precision`
- `storage_bits`
- `primary_bottleneck`
- `latency_batch1_s`
- `peak_throughput_8gpu_videos_s`
- `required_dominant_vs_peak_8gpu`
- utilization-at-peak fields
- source bundle identifiers

Rationale:
- Long-format rows scale naturally to more than two scenarios.
- The report layer can still derive pairwise insights when two scenarios are related, but storage and validation no longer assume exactly two precision columns.
- The CSV becomes reusable for future ranking views, dashboards, or downstream analysis.

Alternative considered:
- Keep the current wide-format fp8/fp4 columns and extend them with more prefixed fields later.
Rejected because it bakes pairwise assumptions into the data model and becomes unmanageable once more hardware families or precisions are added.

### 3. Comparative summaries get their own artifact identity and file names

Comparative bundles will use a generic bundle identity such as `comparative-summary` and will emit:
- `stakeholder-summary.en.md`
- `stakeholder-summary.cn.md`
- `comparison-table.csv`
- `bundle-metadata.json`
- a small figure set

Detailed bundles will continue to emit `stakeholder-report.en.md` and `stakeholder-report.cn.md`.

Rationale:
- The naming cleanly separates “detailed report” from “comparative summary.”
- Future users can infer the intended depth from the file name alone.
- This avoids overloading the word “report” for two different artifact styles.

Alternative considered:
- Keep reusing `stakeholder-report.*.md` for the summary bundle.
Rejected because it blurs the distinction between the long detailed bundles and the short comparative artifact.

### 4. The comparative markdown will stay conclusion-focused and table-light

The summary markdown will contain:
- a short context block
- executive takeaways
- one compact scenario comparison table
- a small figure set for headline metrics
- brief grouped observations when useful

It will not contain appendix sweep tables or section-by-section reproductions of the detailed reports.

Rationale:
- Stakeholders asked for a shorter conclusive comparison artifact.
- The detailed bundles already preserve the deep tables for traceability.
- Keeping the summary short makes it more robust as the number of scenarios grows.

Alternative considered:
- Merge large parts of the detailed fp8/fp4 tables into one combined summary.
Rejected because it recreates the same readability problem under a different file name.

### 5. Comparative compatibility will be defined by shared context, not shared hardware or precision

The comparative summary will require all source detailed bundles to share:
- `input_struct`
- `model_mode`
- `util_profile`
- `device_num`
- compatible comparison semantics

It will allow variation in:
- hardware/device family
- precision profile
- vendor-specific device names

Rationale:
- This preserves apples-to-apples comparisons on workload semantics while allowing the summary to compare different hardware and precision scenarios.
- It matches the future need to place `DV`, `B200`, `H200`, and other scenarios in one report when they are analytically comparable.

Alternative considered:
- Require same device family and vary only precision.
Rejected because that would hard-code the current `fp8`-vs-`fp4` use case back into the contract.

### 6. The Chinese summary will share numeric content but not literal phrasing

The Chinese summary will be generated from the same scenario rows, metadata, and figure references as the English summary, while the markdown prose will be written as native Chinese and keep critical technical terminology in English where that is clearer.

Rationale:
- This matches the existing Wan2.1 Chinese stakeholder-report approach.
- It keeps the bilingual surface consistent without requiring a second data pipeline.

Alternative considered:
- Make the Chinese summary a direct sentence-by-sentence translation of the English markdown.
Rejected because the existing reporting direction is to prefer native stakeholder prose over literal translation.

## Risks / Trade-offs

- [Risk] Replacing the current fp8-vs-fp4-specific summary schema may break assumptions in any downstream tooling that expects prefixed fp8/fp4 columns. → Mitigation: keep the detailed per-precision bundles unchanged and document the new comparative summary CSV schema explicitly.
- [Risk] A generic comparative summary may become verbose as more scenarios are added. → Mitigation: constrain the markdown to one compact table and a small set of ranking-style figures, with grouped takeaways instead of per-scenario mini-sections.
- [Risk] Bundle compatibility may be underspecified for future hardware families. → Mitigation: validate only the shared analytic context and keep source bundle identifiers in metadata and CSV for traceability.
- [Risk] The current implementation and docs use fp8-vs-fp4 wording in several places. → Mitigation: update naming consistently in generator CLI text, metadata, docs, and generated artifacts as part of the same change.

## Migration Plan

1. Introduce the new comparative-summary capability and generic stakeholder-summary artifact names.
2. Refactor or replace the current fp8-vs-fp4 summary generator so it accepts multiple detailed comparison bundles and emits generic scenario rows.
3. Preserve detailed bundle generation unchanged.
4. Regenerate the current DV comparative summary from the existing fp8 and fp4 detailed bundles using the new generic comparative path.
5. Update Wan2.1 reporting docs to describe the distinction between detailed reports and comparative summaries.

Rollback:
- Leave the detailed bundle generators untouched so the summary layer can be rolled back independently.
- If the generic summary path causes problems, remove the new comparative artifacts and restore the prior pairwise summary entrypoint while keeping the underlying detailed fp8/fp4 bundles intact.

## Open Questions

- None for this proposal. The grouping and narrative heuristics can remain simple in the first implementation as long as the data model and artifact contract are generic.
