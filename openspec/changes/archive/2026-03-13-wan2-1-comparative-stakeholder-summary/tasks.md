## 1. Generalize the comparative summary pipeline

- [x] 1.1 Refactor the current fp8-vs-fp4 summary entrypoint so it accepts multiple detailed comparison bundles under one shared Wan2.1 comparison context instead of hard-coded `fp8` and `fp4` inputs.
- [x] 1.2 Replace the current wide fp8/fp4 summary row builder with a generic scenario-row data model and write the new long-format comparative `comparison-table.csv`.
- [x] 1.3 Update comparative bundle metadata and output-guard logic to use a generic comparative-summary identity and to record all included source detailed bundles and scenarios.

## 2. Render concise stakeholder-summary artifacts

- [x] 2.1 Generate `stakeholder-summary.en.md` with a short context block, executive takeaways, one compact scenario comparison table, and a small figure set without appendix-style sweep tables.
- [x] 2.2 Generate `stakeholder-summary.cn.md` from the same comparative scenario data while localizing the markdown into native Chinese and keeping critical technical terms in English where clearer.
- [x] 2.3 Generalize the comparison figures and labels so they work for arbitrary hardware and precision scenarios rather than only the current fp8-vs-fp4 pair.

## 3. Align detailed-report contracts and documentation

- [x] 3.1 Update the detailed DV reporting contract and any related helper or CLI wording so `stakeholder-report.{en,cn}.md` remains the detailed per-bundle artifact and the new comparative bundle clearly owns `stakeholder-summary.{en,cn}.md`.
- [x] 3.2 Update Wan2.1 reporting documentation and README examples to describe the distinction between detailed reports and comparative summaries, including the generic multi-bundle input model.

## 4. Validate and regenerate

- [x] 4.1 Add regression coverage for comparative input compatibility validation, generic scenario-row CSV output, and bilingual `stakeholder-summary.{en,cn}.md` generation.
- [x] 4.2 Regenerate the current DV comparative summary from the existing fp8 and fp4 detailed bundles using the new generic comparative-summary flow and verify that the output stays concise and conclusive.
