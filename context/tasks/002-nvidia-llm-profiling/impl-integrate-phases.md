# Phase Integration Guide: Stage 2 — NVIDIA-Backed Deep LLM Profiling

**Feature**: `002-nvidia-llm-profiling` | **Phases**: 7

## Overview

Stage 2 layers NVIDIA Nsight tools on top of the Stage 1 runner to capture end-to-end timelines (nsys) and per-kernel metrics (ncu). NVTX stage ranges gate collection; Hydra controls configuration and ensures artifacts land under `tmp/stage2/<run_id>/`. Kernel metrics feed the new `kernels.md` while existing operator/aggregate logic powers stakeholder reports.

## Phase Flow

```mermaid
sequenceDiagram
    participant User
    participant Pixi
    participant Runner as deep_profile_runner
    participant Hydra
    participant NVTX
    participant Nsys
    participant NsysStats as nsys stats/export
    participant Ncu
    participant Parser as parsers (nsys_stats, ncu raw)
    participant Exporter as exporters (operators/kernels/summary)
    participant Storage as tmp/stage2/<run_id>

    User->>Pixi: pixi run stage2-profile
    Pixi->>Runner: python -m ...deep_profile_runner +overrides
    Runner->>Hydra: resolve runner config (conf/runner/stage2.yaml)
    Runner->>Storage: create run_dir (nsys/, ncu/, env.json, config.yaml)
    Runner->>NVTX: ensure LLM@prefill / LLM@decode_all
    Runner->>Nsys: build_nsys_cmd(argv, NVTX capture)
    Nsys-->>Storage: run.qdrep/.nsys-rep
    Runner->>NsysStats: export CSV + SQLite
    NsysStats-->>Storage: summary.csv, run.sqlite
    Runner->>Parser: select top‑N kernels by total time
    Runner->>Ncu: build_ncu_cmd(argv, NVTX include=LLM@decode_all/)
    Ncu-->>Storage: decode.ncu-rep, decode_raw.csv
    Runner->>Parser: parse_ncu_csv → KernelRecord[]
    Runner->>Exporter: write kernels.md + stakeholder_summary.md
    Exporter-->>Storage: kernels.md, stakeholder_summary.md
```

## Artifact Flow Between Phases

```mermaid
graph TD
    subgraph P1["Phase 1: Setup"]
        P1T1[T001: nsys wrapper] --> P1A1[vendor/nsys.py]
        P1T2[T002: ncu wrapper] --> P1A2[vendor/ncu.py]
        P1T3[T003: pixi task] --> P1A3[pyproject.toml]
        P1T4[T004: Hydra cfg] --> P1A4[conf/runner/stage2.yaml]
    end

    subgraph P2["Phase 2: Foundational"]
        P2T1[T007: KernelRecord] --> P2A1[data/models.py]
        P2T2[T008: Artifacts] --> P2A2[run_dir layout]
        P2T3[T009: ncu parser] --> P2A3[kernels list]
    end

    subgraph P3["Phase 3: US1"]
        P3T1[T012: runner] --> P3A1[run_id]
        P3T2[T033: nsys stats/sqlite] --> P3A2[summary.csv, run.sqlite]
        P3T3[T034: ncu raw csv] --> P3A3[decode_raw.csv]
    end

    subgraph P4["Phase 4: US2"]
        P4T1[T019: kernels.md] --> P4A1[kernels.md]
    end

    subgraph P5["Phase 5: US3"]
        P5T1[T021: stakeholder] --> P5A1[stakeholder_summary.md]
    end

    P1A4 -.->|read| P3T1
    P2A2 -.->|write| P3T1
    P3A2 -.->|drive top‑N| P3T3
    P3A3 -.->|source| P4T1
    P4A1 -.->|input| P5T1
```

## System Architecture

```mermaid
classDiagram
    class NsysWrapper {
        +build_nsys_cmd(out_base: Path, work_argv: list[str]) list[str]
    }
    class NcuWrapper {
        +build_ncu_cmd(out_base: Path, work_argv: list[str], nvtx_expr: str) list[str]
    }
    class Launch {
        +build_work_argv(module: str, overrides: list[str]) list[str]
    }
    class Artifacts {
        +root: Path
        +env_json() Path
        +config_yaml() Path
        +inputs_yaml() Path
    }
    class NsysStats {
        +select_top_kernels(summary_csv: Path) list[str]
    }
    class KernelsParser {
        +parse_ncu_csv(path: str) list[KernelRecord]
    }
    class Exporters {
        +write_kernel_markdown(records, path)
        +write_stakeholder_summary(...)
    }

    Launch --> NsysWrapper
    Launch --> NcuWrapper
    Artifacts --> NsysWrapper
    Artifacts --> NcuWrapper
    NsysWrapper --> NsysStats
    NcuWrapper --> KernelsParser
    KernelsParser --> Exporters
```

## Use Cases

```mermaid
graph LR
    Actor((Perf Engineer))

    UC1["Run Deep Profile - US1"]
    UC2["Export Top Kernels - US2"]
    UC3["Generate Stakeholder Report - US3"]
    UC4["Profile Other LLM - US4"]

    Actor --> UC1
    Actor --> UC2
    Actor --> UC3
    Actor --> UC4

    UC1 -.prerequisite.-> UC2
    UC2 -.prerequisite.-> UC3
```

## Activity Flow

```mermaid
stateDiagram-v2
    [*] --> Setup
    Setup --> Foundation: wrappers & artifacts ready
    Foundation --> US1: runner + nsys/ncu
    US1 --> US2: ncu raw parsed
    US2 --> US3: kernels included in summary
    US3 --> Polish
    Polish --> [*]
```

## Inter-Phase Dependencies

### Phase 1 → Phase 2

Artifacts
- `conf/runner/stage2.yaml` — runner configuration (mkdir `conf/runner/` if missing)
- `conf/profiling/*` — external profiler presets (torch/nsys/ncu)
- `pyproject.toml` Pixi task — entrypoint for US1

Code Dependencies
```python
# Phase 2 uses Phase 1 wrappers
from llm_perf_opt.profiling.vendor.nsys import build_nsys_cmd
from llm_perf_opt.profiling.vendor.ncu import build_ncu_cmd
```

### Phase 2 → Phase 3

Artifacts
- Run dir layout (`nsys/`, `ncu/`) created by Artifacts

Code Dependencies
```python
from llm_perf_opt.profiling.artifacts import Artifacts
from llm_perf_opt.data.models import KernelRecord
```

### Phase 3 → Phase 4/5

Artifacts
- `nsys` CSV, `ncu` raw CSV → sources for kernels.md
- `kernels.md` → input for stakeholder summary

Code Dependencies
```python
from llm_perf_opt.profiling.nsys_stats import select_top_kernels
from llm_perf_opt.profiling.kernels import parse_ncu_csv
from llm_perf_opt.profiling.export import write_kernel_markdown, write_stakeholder_summary
```

## Data Flow Timeline

```mermaid
gantt
    title Stage 2 Implementation Timeline
    dateFormat YYYY-MM-DD

    section Phase 1
    Wrappers & Config (T001–T004)    :p1, 2025-10-30, 2d

    section Phase 2
    Artifacts & Parser (T007–T011)   :p2, after p1, 2d

    section Phase 3
    Runner & Profilers (T012–T016,T030–T037) :p3, after p2, 3d

    section Phase 4
    Kernels Export (T017–T019)       :p4, after p3, 1d

    section Phase 5
    Stakeholder Report (T020–T023)   :p5, after p4, 1d

    section Phase 6
    Cross-Model (T024–T026)          :p6, after p5, 1d

    section Phase 7
    Polish (T027–T029)               :p7, after p6, 1d
```

## Integration Testing

```bash
# Unit tests (exporters)
pixi run pytest tests/unit/test_kernels_export.py -q
pixi run pytest tests/unit/test_stakeholder_summary.py -q

# Manual end-to-end
pixi run stage2-profile -- +run.mode=deep +inputs.manifest=/abs/path/inputs.yaml

# Validate artifacts exist
python - <<'PY'
from pathlib import Path
root = Path('tmp/stage2')
run = sorted([p for p in root.iterdir() if p.is_dir()])[-1]
for rel in ['nsys', 'ncu', 'kernels.md', 'stakeholder_summary.md', 'env.json', 'config.yaml']:
    p = run / rel
    print(rel, 'OK' if p.exists() else 'MISSING')
PY
```

## Critical Integration Points

1. NVTX range labels must match hint (`LLM@prefill`, `LLM@decode_all`) for correct ncu filtering.
2. Hydra must set `hydra.run.dir` to `tmp/stage2/<run_id>` so nsys/ncu outputs land under the same run.
3. nsys export/statistics must occur before selecting top kernels to feed ncu.
4. ncu must run with `--target-processes all` and NVTX include to bound collection and capture child processes.
5. Stakeholder summary must include both operators and kernels tables for complete attribution.

## References
- Individual phase guides: `context/tasks/002-nvidia-llm-profiling/impl-phase-*.md`
- Spec: `specs/002-nvidia-llm-profiling/spec.md`
- Tasks breakdown: `specs/002-nvidia-llm-profiling/tasks.md`
- Data model: `specs/002-nvidia-llm-profiling/data-model.md`
- Hint: `context/hints/nv-profile-kb/howto-manage-nsys-ncu-processes-for-llm.md`
