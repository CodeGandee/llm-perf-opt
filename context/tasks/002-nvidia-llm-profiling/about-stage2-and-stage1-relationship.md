# Stage 2 ↔ Stage 1 Relationship

## Overview

Stage 2 builds on Stage 1 by running the same LLM workload while orchestrating NVIDIA vendor profilers around it. Concretely, Stage 2 launches Nsight Systems (nsys) for end‑to‑end timelines and Nsight Compute (ncu) for kernel‑level counters, then generates additional artifacts (e.g., kernels.md). It reuses Stage 1 domain models and exporters where possible and keeps artifacts separate.

## Does Stage 2 read Stage 1 artifacts?

- Default: No. Stage 2 does not import prior `tmp/stage1/...` outputs. It performs a fresh run and writes a self‑contained bundle under `tmp/stage2/<run_id>/`.
- Why: Vendor profiling changes runtime characteristics (overhead), so Stage 2 collects metrics in the same execution that produced the profiler traces, ensuring alignment.
- Optional future: The codebase already includes helpers like `build_summary(metrics_path, out_md)` that can ingest a Stage 1 `metrics.json`. Stage 2 could optionally accept a pointer to a Stage 1 run for comparison or fallback when vendor tools are unavailable, but this is not part of the current tasks.

## What Stage 2 reuses from Stage 1

- Domain models and exporters
  - OperatorRecord (TypedDict) and top‑K operators export
  - Aggregation and MFU estimation paths used by stakeholder reporting
  - Stakeholder summary writer (extended to include Top Kernels)
- Configuration approach
  - Hydra overrides drive dataset/model/runtime similarly to Stage 1
  - NVTX stage ranges (LLM@prefill, LLM@decode_all) delimit stages for vendor tools; these do not depend on Stage 1 artifacts

## What Stage 2 adds

- Vendor tooling orchestration
  - nsys: `--trace=cuda,nvtx,osrt`, NVTX‑gated capture, stats/SQLite export
  - ncu: `--target-processes all`, `--nvtx --nvtx-include`, Roofline/SOL metrics, raw CSV export
- Kernel attribution
  - New KernelRecord type and `kernels.md` export (sorted by total time, calls, mean ms)
- Provenance
  - Per‑run bundle under `tmp/stage2/<run_id>/` with `env.json`, `config.yaml`, profiler outputs, operators/kernels tables, and stakeholder summary

## Artifacts separation

- Stage 1: `tmp/stage1/<run_id>/` (operators.md, report.md, metrics.json, etc.)
- Stage 2: `tmp/stage2/<run_id>/` (nsys `.qdrep`/`.nsys-rep`, `nsys` CSV/SQLite, `ncu` `.ncu-rep`/raw CSV, kernels.md, stakeholder_summary.md, provenance files)
- No cross‑reading by default; integration and comparisons are left as optional future work.

## Why this split

- Reproducibility: Keep profiler‑affected runs separate from non‑profiled baseline runs
- Independence: Each user story (Stage 2) is independently testable without Stage 1 run history
- Clarity: Distinct artifact trees make it easy to diff and share results

## Minimizing Stage 1 overhead during Stage 2 runs

When Stage 2 wraps the Stage 1 runner, keep the Stage 1 PyTorch profiler as light as possible to avoid compounding overhead with nsys/ncu.

Recommended Hydra overrides for Stage 2 launches:

```bash
# Prefer minimal or disabled PyTorch profiler and no warmup
profiling=@profiling/torch/torch-profiler.min \
profiling.activities=[cpu] \
+profiling.record_shapes=false +profiling.profile_memory=false +profiling.with_stack=false \
+profiling.warmup_rounds=0 +profiling.rep_max_new_tokens=16 \
outputs.save_predictions=false outputs.visualization.enable=false
```

Notes:
- Prefer the alias `+torch_profiler.enabled=false` to fully skip the representative PyTorch profiling block (code supports both this alias and `+profiling.enabled=false`). If not skipping, use the minimal preset plus `warmup_rounds=0` and a small `rep_max_new_tokens` to limit cost.
- Keep NVTX ranges enabled (LLM@prefill, LLM@decode_all) so nsys/ncu can still isolate stages.
- Consider `repeats=1` for Stage 2, and ensure `hydra.run.dir` points to `tmp/stage2/<run_id>` to collocate vendor outputs.

## Future extensions (non‑blocking)

- Merge view: Single report that overlays Stage 1 aggregates with Stage 2 kernel attribution
- Pointer input: `+stage1.run_dir=/abs/tmp/stage1/<rid>` to import metrics.json when skipping recomputation
- Fallback: If nsys/ncu missing, fall back to Stage 1 flow and note partial artifacts (aligns with FR‑008)

## References
- Spec: `specs/002-nvidia-llm-profiling/spec.md`
- Data model: `specs/002-nvidia-llm-profiling/data-model.md`
- Tasks: `specs/002-nvidia-llm-profiling/tasks.md`
- Hint: `context/hints/nv-profile-kb/howto-manage-nsys-ncu-processes-for-llm.md`
