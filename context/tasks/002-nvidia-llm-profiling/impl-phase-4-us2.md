# Implementation Guide: US2 — Kernels Export

Phase: 4 | Feature: Stage 2 — NVIDIA-Backed Deep LLM Profiling | Tasks: T017–T019

## Files

### Created
- tests/unit/test_kernels_export.py

### Modified
- src/llm_perf_opt/profiling/export.py
- src/llm_perf_opt/runners/deep_profile_runner.py

## Public APIs

### T018: Kernels export helpers

Leverage the existing domain model and parsers introduced in earlier phases:
- Data model: `KernelRecord` in `src/llm_perf_opt/data/models.py`
- Parsers: `parse_ncu_csv`, `parse_ncu_json`, `kernels_from_ncu_rows` in `src/llm_perf_opt/profiling/kernels.py`

Add the following helpers to `src/llm_perf_opt/profiling/export.py`:

```python
from typing import Iterable
from llm_perf_opt.data.models import KernelRecord

def top_n_kernels(records: Iterable[KernelRecord], n: int = 20) -> list[KernelRecord]: ...
def write_kernel_markdown(records: Iterable[KernelRecord], path: str, top_k: int = 20) -> None: ...
```

Behavior
- Sort by `total_ms` desc, compute mean if not provided (fallback `total_ms / max(calls,1)`).
- Write a Markdown table with columns: `kernel_name`, `total_ms`, `calls`, `mean_ms`, `device`.
- Accept `path` ending with `.md`; strip suffix for mdutils compatibility (like operator export).

## Integration Flow

Outputs are unified under a single MAIN run directory (Hydra `hydra.run.dir`), typically `tmp/stage2/<timestamp>`.

```mermaid
graph LR
    A[Stage 2 MAIN dir] --> R[ncu/raw.csv]
    R --> P[parse_ncu_csv + kernels_from_ncu_rows]
    P --> E[export.top_n_kernels + write_kernel_markdown]
    E --> M[MAIN/kernels.md]
```

Runner integration (append after NCU run) in `src/llm_perf_opt/runners/deep_profile_runner.py`:

```python
from llm_perf_opt.profiling.kernels import parse_ncu_csv, kernels_from_ncu_rows
from llm_perf_opt.profiling.export import top_n_kernels, write_kernel_markdown

ncu_csv = artifacts.path("ncu/raw.csv")
if Path(ncu_csv).exists():
    rows = parse_ncu_csv(ncu_csv)
    krecs = kernels_from_ncu_rows(rows, device=str(getattr(getattr(cfg, "stage1_runner", {}), "device", "cuda:0")))
    topk = top_n_kernels(krecs, n=int(getattr(getattr(cfg, "run", {}), "top_n_kernels", 20)))
    write_kernel_markdown(topk, str(artifacts.path("kernels.md")), top_k=len(topk))
```

Notes
- If Nsight Systems report is missing (known NVTX gating limitation), `kernel_regex` seeding is skipped automatically; NCU raw capture still produces `ncu/raw.csv` for export.
- NCU `--list-sections` is written to `MAIN/ncu/sections.txt` for version‑aware tuning; US2 export does not depend on sections being present.

## Testing

- Unit tests: headers, sorting, mean calculation

```bash
pixi run pytest tests/unit/test_kernels_export.py -q
```

- Manual end‑to‑end

```bash
pixi run stage2-profile 'hydra.run.dir=tmp/stage2/${now:%Y%m%d-%H%M%S}'
# or target GPU 1
pixi run stage2-profile-gpu1 'hydra.run.dir=tmp/stage2/${now:%Y%m%d-%H%M%S}'
```

Expect `tmp/stage2/<run_id>/kernels.md` and `tmp/stage2/<run_id>/ncu/raw.csv`.

## Summary
- Unified outputs: MAIN at `tmp/stage2/<run_id>`; kernels export writes `MAIN/kernels.md`.
- Reuse-first: `KernelRecord` model and NCU parsers from prior phases.
- Robust to missing NSYS report: top‑N seed optional; export proceeds from `ncu/raw.csv`.
- Config toggles: NVTX gating via `nsys.gating_nvtx`/`ncu.gating_nvtx`; they do not change export semantics.

## References
- Spec: specs/002-nvidia-llm-profiling/spec.md (US2)
- Tasks: specs/002-nvidia-llm-profiling/tasks.md (T017–T019)
