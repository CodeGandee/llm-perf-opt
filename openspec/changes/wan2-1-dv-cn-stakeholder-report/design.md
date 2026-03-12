## Context

The current DV stakeholder-report generator at [run_make_dv_stakeholder_report.py](/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/reporting/run_make_dv_stakeholder_report.py) computes shared summaries, writes shared figures/CSV, and renders one English markdown artifact, `stakeholder-report.en.md`. The repo already has a Wan2.1 legacy Chinese stakeholder report at [stakeholder-report.cn.md](/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/reports/ngu800p_sweeps/stakeholder-report.cn.md), which establishes both the file suffix convention and the expectation that a stakeholder-facing Chinese markdown report can coexist with shared figures.

This follow-on change is localized reporting work, not a new analytics pipeline. The sweep math, compatibility checks, comparison CSV, and generated figures should remain single-source. The design problem is how to add Chinese output without duplicating the current DV generator or coupling report wording changes too tightly to one language.

## Goals / Non-Goals

**Goals:**
- Generate `stakeholder-report.cn.md` alongside `stakeholder-report.en.md` from the same DV comparison run.
- Reuse the same computed summaries, figures, copied architecture image, and `comparison-table.csv`.
- Localize report headings, paragraphs, table column labels, Q&A copy, appendix framing, and explanatory notes into Chinese.
- Make the Chinese report read naturally to a native Chinese reader instead of sounding like a literal translation, while preserving critical technical terms in English where precision matters.
- Preserve the current English report behavior so the existing artifact and downstream references remain stable.
- Match the existing Wan2.1 naming pattern by using `.cn.md` rather than introducing a new suffix.

**Non-Goals:**
- Generating a second set of Chinese-localized figures.
- Changing sweep math, compatibility validation, or the comparison CSV schema.
- Translating legacy NGU reports or retrofitting multilingual support across unrelated Wan2.1 reporting scripts.
- Introducing external i18n dependencies or a general localization framework for the whole repository.

## Decisions

### 1. Keep one reporting entrypoint and emit both markdown variants from shared data

Chosen approach:
- Keep [run_make_dv_stakeholder_report.py](/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/reporting/run_make_dv_stakeholder_report.py) as the single entrypoint for the DV comparison bundle.
- Compute summaries, figures, and CSV once.
- Render two markdown documents from the same in-memory comparison context:
  - `stakeholder-report.en.md`
  - `stakeholder-report.cn.md`

Rationale:
- The English and Chinese outputs should remain numerically identical because they come from the same comparison data.
- A single entrypoint avoids divergence in CLI arguments, validation logic, and emitted artifact layout.

Alternatives considered:
- Add a second `run_make_dv_stakeholder_report_cn.py` script.
  Rejected because it would duplicate the entire report-construction path and make future wording changes error-prone.

### 2. Introduce locale-aware report rendering, not a post-generation translation step

Chosen approach:
- Refactor the report builder so locale-sensitive strings are isolated from shared numeric/table construction.
- Use either a small locale text bundle or two renderer functions with shared helpers for tables, lists, and figures.

Rationale:
- The current generator contains substantial inline English copy, so implementation needs an explicit place for localized headings and prose.
- A structured renderer keeps table values and formulas shared while allowing Chinese-specific wording that reads naturally instead of mechanically translated prose.
- This also gives the Chinese path room to preserve critical English technical terminology such as `batch size`, `MemIO`, `tensor-core compute`, and `Data-parallel replication` where those terms are already the clearest labels in stakeholder discussions.

Alternatives considered:
- Generate the English report first and then mechanically transform it into Chinese.
  Rejected because the report includes nuanced Q&A explanations and stakeholder-facing phrasing that should be authored directly in Chinese.

### 3. Reuse the same figures and image references in the Chinese report

Chosen approach:
- The Chinese report references the exact same `figures/` assets as the English report.
- The Chinese markdown explains the figures in Chinese, but the figure titles and axis labels remain the existing generated English assets for this phase.

Rationale:
- This delivers the requested Chinese stakeholder report without opening a separate font/localized-matplotlib problem.
- The existing Wan2.1 Chinese report precedent already mixes Chinese markdown with shared figure assets.

Alternatives considered:
- Localize the plot titles and axis labels as part of this change.
  Rejected for now because it would require a second layer of figure localization, font handling, and broader changes in [hardware_sweep_reporting.py](/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/hardware_sweep_reporting.py).

### 4. Treat `.cn.md` as the repository contract for Chinese stakeholder markdown

Chosen approach:
- Output the Chinese artifact as `stakeholder-report.cn.md`.
- Update Wan2.1 docs/readmes that enumerate generated DV comparison outputs so both language variants are discoverable.

Rationale:
- The only existing Wan2.1 Chinese stakeholder artifact uses `.cn.md`.
- Matching the current repository convention is lower risk than introducing `.zh.md` or locale suffix flags inconsistently.

Alternatives considered:
- Use `stakeholder-report.zh.md` or `stakeholder-report.zh-CN.md`.
  Rejected because there is no existing Wan2.1 precedent for those names and they would fragment report discovery.

## Risks / Trade-offs

- [Bilingual content drift] → Mitigation: keep English and Chinese renderers fed by the same summary/table helpers so only prose differs.
- [Chinese wording becomes stale when English sections change later] → Mitigation: centralize locale-specific section templates in one file and update docs/tests to check for both outputs.
- [Readers may expect Chinese-localized figures too] → Mitigation: state in the design and spec that this change localizes markdown output only and reuses shared figures.
- [Markdown structure regressions across two language paths] → Mitigation: preserve the existing block-boundary helpers and add focused tests for both `.en.md` and `.cn.md` generation.

## Migration Plan

- No data migration is required.
- Regenerating a comparison bundle will add `stakeholder-report.cn.md` next to the existing English report.
- Existing consumers of `stakeholder-report.en.md` remain unaffected.

## Open Questions

- None for artifact creation. The implementation path is straightforward as long as Chinese figure localization stays out of scope for this change.
