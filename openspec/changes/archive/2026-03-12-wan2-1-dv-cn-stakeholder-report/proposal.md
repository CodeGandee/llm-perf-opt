## Why

The current DV stakeholder reporting flow only generates [stakeholder-report.en.md](extern/modelmeter/models/wan2_1/reports/hardware_sweeps/comparisons/20260311-dv-comparison/stakeholder-report.en.md), but the immediate stakeholder need now includes a Chinese version for the same DV comparison bundle. Wan2.1 already has a legacy Chinese stakeholder-report precedent at [stakeholder-report.cn.md](extern/modelmeter/models/wan2_1/reports/ngu800p_sweeps/stakeholder-report.cn.md), so the DV reporting path should provide the same deliverable without requiring manual translation after every regeneration.

## What Changes

- Extend the consolidated DV stakeholder reporting flow so one run can emit a Chinese markdown report alongside the existing English report.
- Add localized Chinese report text for the DV stakeholder sections, headings, explanatory notes, table labels, and appendix framing while reusing the same computed data, figures, and comparison CSV.
- Make the Chinese report read like native Chinese stakeholder writing rather than a literal sentence-by-sentence translation, while keeping critical technical terminology in English where that is clearer and more precise.
- Standardize the output naming on `stakeholder-report.cn.md` to match the existing Wan2.1 report pattern.
- Update report/documentation references so the generated DV comparison bundle clearly includes both English and Chinese stakeholder markdown outputs.
- Generate the first Chinese stakeholder report artifact for the current DV comparison bundle.

## Capabilities

### New Capabilities
- `wan2-1-dv-cn-stakeholder-reporting`: Generate a Chinese DV stakeholder markdown report from the same comparison inputs and figures used by the English DV report.

### Modified Capabilities
- None.

## Impact

- Affected code: [run_make_dv_stakeholder_report.py](extern/modelmeter/models/wan2_1/scripts/reporting/run_make_dv_stakeholder_report.py) and adjacent Wan2.1 reporting docs/readmes.
- Affected outputs: `extern/modelmeter/models/wan2_1/reports/hardware_sweeps/comparisons/<comparison_run_id>/stakeholder-report.cn.md` in addition to the existing English report and shared figures/CSV.
- Scope boundary: this change localizes markdown reporting only; it does not require Chinese-localized figure titles or a second figure-generation pipeline.
