## Why

The current combined fp8/fp4 stakeholder summary is modeled as a pairwise `DV fp8 vs fp4` report, which is already too specific for the next comparison needs. We now need a shorter stakeholder-facing comparative summary that can reuse existing detailed comparison bundles while scaling to future hardware and precision scenarios such as `DV`, `B200`, `H200`, `fp8`, and `fp4` under one shared Wan2.1 workload context.

## What Changes

- Add a generic comparative stakeholder-summary flow that consumes multiple compatible detailed comparison bundles and produces concise English and Chinese summary markdown artifacts named `stakeholder-summary.en.md` and `stakeholder-summary.cn.md`.
- Define the summary around reusable scenario rows such as `(device, precision)` rather than hard-coded `fp8_*` and `fp4_*` pairwise columns.
- Keep the comparative summary short and conclusion-focused: shared context, executive takeaways, compact comparison tables, and a small figure set, without repeating appendix-style sweep tables from the detailed reports.
- Generalize bundle metadata and `comparison-table.csv` for the summary so future hardware families such as NVIDIA devices can be added without redesigning the format.
- Rename the summary concept from fp8-vs-fp4-specific wording to a generic comparative-summary artifact model while preserving reuse of the already generated detailed bundle results.

## Capabilities

### New Capabilities
- `wan2-1-comparative-stakeholder-summary-reporting`: Generate a short English comparative Wan2.1 stakeholder summary from multiple compatible detailed comparison bundles.
- `wan2-1-comparative-cn-stakeholder-summary-reporting`: Generate a short Chinese comparative Wan2.1 stakeholder summary aligned with the English comparative summary while keeping critical technical terms in English where clearer.

### Modified Capabilities
- `wan2-1-dv-stakeholder-reporting`: Remove the generic summary responsibility from the DV detailed-report capability so it remains focused on detailed, precision-specific DV stakeholder reports.
- `wan2-1-dv-cn-stakeholder-reporting`: Remove the generic summary responsibility from the Chinese DV detailed-report capability so it remains aligned with the English detailed-report scope.

## Impact

- Affected code: `extern/modelmeter/models/wan2_1/scripts/reporting/run_make_dv_precision_summary_report.py` or its replacement, related report helpers, validation logic, and Wan2.1 reporting docs.
- Affected artifacts: comparative summary bundle metadata, comparative `comparison-table.csv`, `stakeholder-summary.en.md`, `stakeholder-summary.cn.md`, and a small shared figure set.
- Affected specs: new comparative-summary reporting capabilities plus updated DV reporting specs so detailed bundles and comparative summaries have cleanly separated contracts.
