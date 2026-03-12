## 1. Localized DV Report Generation

- [x] 1.1 Refactor the DV stakeholder-report builder so shared comparison data can render both English and Chinese markdown outputs from one reporting run.
- [x] 1.2 Add Chinese-localized headings, explanatory prose, table labels, Q&A notes, and appendix framing for the DV stakeholder report while preserving the current English output.
- [x] 1.3 Make the Chinese wording read like native Chinese stakeholder writing instead of literal translation, while keeping critical technical terminology in English where clarity depends on it.
- [x] 1.4 Update the reporting entrypoint to write `stakeholder-report.cn.md` alongside `stakeholder-report.en.md` and keep the shared figures and `comparison-table.csv` unchanged.

## 2. Documentation And Validation

- [x] 2.1 Update Wan2.1 reporting docs/readmes to list both `stakeholder-report.en.md` and `stakeholder-report.cn.md` in the DV comparison bundle.
- [x] 2.2 Add or extend focused tests so the reporting flow verifies both language outputs are generated and continue to use clean markdown block formatting.
- [x] 2.3 Review the generated Chinese report text for native phrasing and consistent retention of critical English technical terms.

## 3. Artifact Regeneration

- [x] 3.1 Regenerate the current DV comparison bundle so it includes the first `stakeholder-report.cn.md` artifact for the existing comparison run.
- [x] 3.2 Validate the regenerated bundle and confirm the English and Chinese stakeholder reports coexist with the same shared figures and comparison CSV.
