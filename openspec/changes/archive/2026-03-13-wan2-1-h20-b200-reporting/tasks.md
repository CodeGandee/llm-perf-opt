## 1. Extend Wan2.1 device sweep support

- [x] 1.1 Add `h20` and `b200` to the Wan2.1-local supported device selector map and ensure normalized sweep metadata records their compute, MemIO, and interconnect assumptions.
- [x] 1.2 Confirm the existing precision-selection path behaves correctly for the new selectors, including fast failure for unsupported `fp4` requests on `H20` and `B200`.
- [x] 1.3 Add regression coverage for `h20`/`b200` selector support, emitted metadata, and unsupported precision/device combinations.

## 2. Generalize the detailed stakeholder-report pipeline

- [x] 2.1 Implement a generic detailed multi-device stakeholder-report generator that accepts multiple compatible device runs under one shared Wan2.1 context and writes standard detailed bundle artifacts.
- [x] 2.2 Generate bilingual detailed bundle outputs `stakeholder-report.en.md` and `stakeholder-report.cn.md` from the same comparison rows, metadata, and figures, with native Chinese prose and critical technical terms preserved in English where clearer.
- [x] 2.3 Keep the existing DV detailed-report entrypoint as a compatibility wrapper over the new generic detailed generator so current DV workflows and bundle names remain stable.
- [x] 2.4 Add regression coverage for generic detailed-bundle compatibility checks, bilingual output generation, and continued DV wrapper behavior.

## 3. Publish H20 and B200 Wan2.1 metrics

- [x] 3.1 Generate first-pass Wan2.1 `fp8` sweep artifacts for `H20` and `B200` using the same workload grid and DP semantics as the current comparison bundles.
- [x] 3.2 Generate an English and Chinese detailed comparison bundle for the new NVIDIA runs and include explicit header disclosure that the hardware values come from the shared registry and are first-pass analytic assumptions.
- [x] 3.3 Regenerate a comparative stakeholder-summary bundle that reuses the new NVIDIA detailed bundle together with an existing compatible DV detailed bundle under the same shared workload context.

## 4. Document and verify the new flow

- [x] 4.1 Update Wan2.1 reporting documentation and README examples to describe `h20`/`b200` sweep usage, the generic detailed comparison generator, and the relationship between detailed bundles and comparative summaries.
- [x] 4.2 Verify the generated `H20`/`B200` detailed bundle and regenerated comparative summary artifacts, and confirm the existing DV detailed-report workflow still produces the expected outputs through the wrapper path.
