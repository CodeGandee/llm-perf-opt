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

```python
from typing import Iterable, TypedDict

class KernelRecord(TypedDict, total=False):
    kernel_name: str
    device: str
    total_ms: float
    calls: int
    mean_ms: float

def top_n_kernels(records: Iterable[KernelRecord], n: int = 20) -> list[KernelRecord]: ...
def write_kernel_markdown(records: Iterable[KernelRecord], path: str, top_k: int = 20) -> None: ...
```

## Phase Integration

```mermaid
graph LR
    N[ncu raw.csv] --> P[parser]
    P --> K[kernels.md]
    K --> S[stakeholder summary]
```

## Testing

```bash
pixi run pytest tests/unit/test_kernels_export.py -q
```

## References
- Spec: specs/002-nvidia-llm-profiling/spec.md (US2)
