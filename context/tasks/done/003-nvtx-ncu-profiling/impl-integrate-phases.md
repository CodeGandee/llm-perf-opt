# Phase Integration Guide: NVTX-based NCU Regional Profiling

**Feature**: `003-nvtx-ncu-profiling` | **Phases**: 5 + Polish

## Overview

This feature adds NVTX-range–aware Nsight Compute profiling to Stage 2. We introduce a dummy model for deterministic kernels, extend the NCU command builder with replay mode, and implement per-region aggregation/export. The flow remains Hydra-driven and reuses existing Stage 2 infrastructure and artifacts layout.

## Phase Flow

```mermaid
sequenceDiagram
    participant User
    participant Setup as Phase1: Dummy Model & Test
    participant Found as Phase2: NCU Builder + Models
    participant US1 as Phase3: Region Replay + Export
    participant US2 as Phase4: Kernel Filters
    participant US3 as Phase5: Sections/Metrics
    participant Storage

    Note over User,Setup: Phase 1: Setup & Manual Test
    User->>Setup: run manual NVTX script (A, A::A1, B)

    Note over Setup,Found: Phase 2: Foundations
    Setup->>Found: ensure configs & builder support replay_mode
    Found->>Storage: presets updated (replay_mode=kernel)

    Note over Found,US1: Phase 3: Region Replay
    User->>US1: deep_profile_runner (replay_mode=range)
    US1->>Storage: ncu/raw.csv, ncu/.ncu-rep
    US1->>Storage: ncu/regions/{per-region}, ncu/regions/report.{md,json}

    Note over US1,US2: Phase 4: Kernel Filters
    User->>US2: add kernel include/exclude rules
    US2->>Storage: filtered reports

    Note over US2,US3: Phase 5: Sections/Metrics
    User->>US3: select sections/metrics
    US3->>Storage: sections_report.txt, updated report.md
```

## Artifact Flow Between Phases

```mermaid
graph TD
    subgraph Phase1["Phase 1: Setup"]
        P1T1[T001: Dummy Configs]
        P1T2[T002/T003: Dummy Model]
        P1T3[T004: Manual Ranges]
    end

    subgraph Phase2["Phase 2: Foundational"]
        P2T1[T007: Models]
        P2T2[T008: NCU Builder]
        P2T3[T009-T011: Presets]
    end

    subgraph Phase3["Phase 3: US1"]
        P3T1[T012: Runner Replay]
        P3T2[T013: Assembler]
        P3T3[T014: Exporter]
        P3T4[T015: Region Paths]
    end

    subgraph Phase4["Phase 4: US2"]
        P4T1[T018: Kernel Filters]
        P4T2[T020: Selection]
    end

    subgraph Phase5["Phase 5: US3"]
        P5T1[T023: Sections/Metrics]
        P5T2[T025: Export Summaries]
    end

    P1T1 --> P3T1
    P1T2 --> P3T1
    P1T3 --> P3T1
    P2T1 --> P3T2
    P2T2 --> P3T1
    P2T3 --> P3T1
    P3T1 -->|raw.csv| P3T2
    P3T2 -->|reports| P3T3
    P3T3 -->|"report.{md,json}"| P4T1
    P4T1 --> P4T2
    P4T2 --> P5T1
    P5T1 --> P5T2
```

## System Architecture

```mermaid
classDiagram
    class Artifacts {
        +out_dir(stage) Path
        +path(name) Path
        +tmp_dir(stage) Path
    }
    class NCUProfileRegion {
        +name: str
        +parent: Optional[str]
        +depth: int
        +process: Optional[str]
        +device: Optional[str]
    }
    class NCUProfileRegionReport {
        +region: NCUProfileRegion
        +total_ms: float
        +kernel_count: int
        +json_path: Optional[str]
        +markdown_path: Optional[str]
    }
    class NCUBuilder {
        +build_ncu_cmd(..., replay_mode)
    }
    class DeepProfileRunner {
        +main(cfg) -> None
    }
    class RegionAssembler {
        +assemble_region_reports(rows) List[NCUProfileRegionReport]
    }
    class RegionExporter {
        +export_region_reports(artifacts, reports)
    }

    Artifacts <.. DeepProfileRunner
    NCUBuilder <.. DeepProfileRunner
    RegionAssembler <.. DeepProfileRunner
    RegionExporter <.. DeepProfileRunner
    NCUProfileRegion <.. NCUProfileRegionReport
```

## Use Cases

```mermaid
graph LR
    U((User))
    UC1[Replay by Range]
    UC2[Filter Kernels]
    UC3[Select Sections]
    U --> UC1 --> UC2 --> UC3
```

## Activity Flow

```mermaid
stateDiagram-v2
    [*] --> Setup
    Setup --> Foundations: dummy ready
    Foundations --> Replay: replay_mode=range
    Replay --> Export: regions built
    Export --> Filter: kernel include/exclude
    Filter --> Select: sections/metrics
    Select --> [*]
```

## Inter-Phase Dependencies

### Phase 1 → Phase 3

Artifacts:
- `conf/model/dummy_shallow_resnet/*` — dummy model configs used in all tests
- `tests/manual/ncu/manual_nvtx_regions.py` — deterministic NVTX ranges

### Phase 2 → Phase 3

Code Dependencies:
```python
from llm_perf_opt.profiling.vendor.ncu import build_ncu_cmd  # replay_mode support
from llm_perf_opt.data.ncu_regions import NCUProfileRegion, NCUProfileRegionReport
```

## Data Flow Timeline

```mermaid
gantt
    title NVTX Region Profiling Timeline
    dateFormat YYYY-MM-DD

    section Phase 1
    Dummy Model (T001–T004) :p1, 2025-11-06, 1d

    section Phase 2
    Data Models (T007)      :p2a, after p1, 0.5d
    NCU Builder (T008)      :p2b, after p2a, 0.5d

    section Phase 3
    Runner + Export (T012–T014):p3, after p2b, 1d

    section Phase 4
    Kernel Filters (T018–T020):p4, after p3, 0.5d

    section Phase 5
    Sections/Metrics (T023–T025):p5, after p4, 0.5d
```

## Integration Testing

```bash
# End-to-end with dummy model, range replay, sections
pixi run -e rtx5090 python -m llm_perf_opt.runners.deep_profile_runner \
  model/dummy_shallow_resnet/arch@model=dummy_shallow_resnet.default \
  model/dummy_shallow_resnet/infer@infer=dummy_shallow_resnet.default \
  pipeline.ncu.enable=true \
  pipeline.ncu.ncu_cli.replay_mode=range \
  pipeline.ncu.ncu_cli.sections='[SpeedOfLight,Occupancy]'

# Validate region artifacts exist
rg -n "# NCU Region Reports" tmp/profile-output/*/ncu/regions/report.md
```

## Critical Integration Points

1. Phase 2 replay_mode must be honored by runner and builder
2. CSV must include range-identifying columns for grouping; assembler falls back to "(unlabeled)"
3. Region exporter must not break if sections import is skipped (no .ncu-rep)

## References
- Phase guides: `context/tasks/003-nvtx-ncu-profiling/impl-phase-*.md`
- Spec: `specs/003-nvtx-ncu-profiling/spec.md`
- Tasks: `specs/003-nvtx-ncu-profiling/tasks.md`
