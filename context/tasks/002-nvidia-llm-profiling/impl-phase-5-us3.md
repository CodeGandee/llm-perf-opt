# Implementation Guide: US3 — Stakeholder Report

Phase: 5 | Feature: Stage 2 — NVIDIA-Backed Deep LLM Profiling | Tasks: T020–T023

## Files

### Created
- tests/unit/test_stakeholder_summary.py

### Modified
- src/llm_perf_opt/profiling/export.py
- src/llm_perf_opt/runners/deep_profile_runner.py

## Public APIs

### T021: Extend stakeholder summary to include Top Kernels

```python
from llm_perf_opt.data.models import KernelRecord

def write_stakeholder_summary(path: str,
    top_ops: list[dict],
    stage_takeaways: dict[str, str],
    stats: dict | None = None,
    top_kernels: list[KernelRecord] | list[dict] | None = None,
) -> None: ...
```

Behavior
- Adds a "Top Kernels" section if `top_kernels` is provided. Accepts either `KernelRecord` or dicts.
- Keeps existing sections: Environment, Aggregates, Per-Stage Timings, MFU, Stage Takeaways, Top Operators, Recommendations.

## Integration

The Stage 2 runner generates `stakeholder_summary.md` after profiler runs:

```python
# After NCU CSV is written and kernels.md generated
metrics_json = MAIN/"stage1/metrics.json"
stats = map_stage1_metrics_to_stats(metrics_json)
write_stakeholder_summary(MAIN/"stakeholder_summary.md",
    top_ops=[],              # Stage 2 does not collect operators; see Stage 1
    stage_takeaways=heuristic_msgs(stats),
    stats=stats,
    top_kernels=topkern_list  # from NCU CSV parsing
)

# Also write a lightweight Stage 2 report with links and quick tables
write_stage2_report(MAIN/"report.md", stats)
```

Outputs
- `tmp/stage2/<run_id>/stakeholder_summary.md` (Stage 2)
- `tmp/stage2/<run_id>/report.md` (links + Per-Stage Timings + MFU)
- Reuses Stage 1 artifacts under `tmp/stage2/<run_id>/stage1/`

## Testing

Optional unit test checks the summary layout:

```bash
pixi run pytest tests/unit/test_stakeholder_summary.py -q
```

## Summary
- Added `top_kernels` support to `write_stakeholder_summary` (accepts KernelRecord or dicts).
- Stage 2 runner now writes `stakeholder_summary.md` and a lightweight `report.md` in the MAIN directory.
- Summary uses Stage 1 `metrics.json` for Aggregates/MFU and NCU CSV for top kernels.
- Unified outputs preserved: all files under `tmp/stage2/<run_id>/`.

## References
- Spec: specs/002-nvidia-llm-profiling/spec.md (US3)
