# Phase Integration Guide: Wan2.1 Analytic FLOP Model

**Feature**: `004-wan2-1-analytic-model` | **Phases**: 6

## Overview

This feature delivers a ModelMeter-style analytic model for Wan2.1-T2V-14B and a static-analysis workflow that emits a structured `report.json`, optional `summary.md`, and verification outputs under `tmp/profile-output/<run_id>/static_analysis/wan2_1/`.
The phases build incrementally: Phase 1 prepares the local model metadata reference, Phase 2 standardizes report schema and compatibility, Phase 3 implements report generation (MVP), Phase 4 enforces FLOP accuracy gates, Phase 5 adds hotspot tooling, and Phase 6 hardens documentation and repo-wide quality gates.

## Phase Flow

**MUST HAVE: End-to-End Sequence Diagram**

```mermaid
sequenceDiagram
    participant Dev as Developer<br/>CLI
    participant BS as models/wan2.1-t2v-14b<br/>bootstrap.sh
    participant HY as Hydra<br/>compose+instantiate
    participant AN as Wan2_1StaticAnalyzer<br/>run
    participant AM as Wan2_1DiTModel<br/>BaseLayer tree
    participant VR as verify scripts<br/>run_verify_*
    participant RT as report tools<br/>compare
    participant FS as tmp/profile-output<br/>static_analysis

    Note over Dev,FS: Phase 1: Setup
    Dev->>BS: bash bootstrap.sh
    BS-->>Dev: source-data symlink ready

    Note over Dev,FS: Phase 2: Foundational schema
    Dev->>Dev: run unit tests<br/>(schema + paths)

    Note over Dev,FS: Phase 3: US1 report generation
    Dev->>AN: run analyzer<br/>(profile_id, run_id)
    AN->>HY: compose wan2_1_t2v_14b
    HY-->>AN: instantiated layers
    AN->>AM: evaluate costs<br/>(per layer + total)
    AN->>FS: write report.json<br/>+ summary.md
    FS-->>Dev: artifacts available

    Note over Dev,FS: Phase 4: US2 verification
    Dev->>VR: run_verify_layers<br/>(workload)
    VR->>HY: read tolerances<br/>from config
    VR->>AM: analytic FLOPs
    VR-->>Dev: pass/fail + diffs

    Note over Dev,FS: Phase 5: US3 hotspots
    Dev->>RT: load report.json
    RT->>RT: compute top-k
    RT-->>Dev: hotspot tables

    Note over Dev,FS: Phase 6: Polish
    Dev->>Dev: ruff + mypy + pytest
```

## Artifact Flow Between Phases

```mermaid
graph TD
    subgraph P1["Phase 1: Setup"]
        P1T1[T001: Bootstrap model ref] --> P1A1[models/wan2.1-t2v-14b/source-data];
    end

    subgraph P2["Phase 2: Foundational"]
        P2T1[T005: analytic_common] --> P2A1[src/llm_perf_opt/data/analytic_common.py];
        P2T2[T010: wan2_1_analytic] --> P2A2[src/llm_perf_opt/data/wan2_1_analytic.py];
        P2T3[T012: paths] --> P2A3[src/llm_perf_opt/utils/paths.py];
    end

    subgraph P3["Phase 3: US1 Report"]
        P3T1[T024: Hydra configs] --> P3A1[extern/modelmeter/models/wan2_1/configs];
        P3T2[T025: Analyzer] --> P3A2[tmp/profile-output/<run_id>/static_analysis/wan2_1/report.json];
        P3T3[T037: Summary] --> P3A3[tmp/profile-output/<run_id>/static_analysis/wan2_1/summary.md];
    end

    subgraph P4["Phase 4: US2 Verify"]
        P4T1[T031: verify layers] --> P4A1[tmp/profile-output/<run_id>/static_analysis/wan2_1/verify/layers.json];
        P4T2[T032: verify end2end] --> P4A2[tmp/profile-output/<run_id>/static_analysis/wan2_1/verify/end2end.json];
    end

    subgraph P5["Phase 5: US3 Hotspots"]
        P5T1[T038: report tools] --> P5A1[console output];
    end

    P1A1 -.->|reads config.json| P3A1;
    P2A1 -.->|schema used by| P3T2;
    P2A2 -.->|report wrapper used by| P3T2;
    P3A2 -.->|consumed by| P4T1;
    P3A2 -.->|consumed by| P5T1;
    P3A2 -.->|rendered to| P3A3;
```

## System Architecture

```mermaid
classDiagram
    class Wan2_1StaticAnalyzer {
        +run(cfg) Wan2_1AnalyticModelReport
    }

    class Wan2_1DiTModel {
        +set_num_inference_steps(steps)
        +forward_tensor_core_flops() float
        +forward_cuda_core_flops() float
    }

    class Wan2_1TransformerBlock {
        +forward_tensor_core_flops() float
        +forward_cuda_core_flops() float
    }

    class Wan2_1AnalyticModelReport {
        +report_id: str
        +model: Wan2_1ModelSpec
        +workload: Wan2_1WorkloadProfile
        +modules: list
        +module_metrics: list
    }

    class VerifyLayers {
        +run(workload) pass_fail
    }

    class SummaryRenderer {
        +render(report) str
    }

    class ReportTools {
        +load(path) report
        +top_k(report) table
        +compare(a,b) diff
    }

    Wan2_1StaticAnalyzer --> Wan2_1DiTModel: instantiates
    Wan2_1DiTModel --> Wan2_1TransformerBlock: contains
    Wan2_1StaticAnalyzer --> Wan2_1AnalyticModelReport: emits
    VerifyLayers --> Wan2_1DiTModel: evaluates
    SummaryRenderer --> Wan2_1AnalyticModelReport: renders
    ReportTools --> Wan2_1AnalyticModelReport: loads
```

## Use Cases

```mermaid
graph LR
    Actor((Developer))

    UC1[Bootstrap local model reference]
    UC2[Generate Wan2.1 report.json]
    UC3[Verify FLOPs within 5 percent]
    UC4[Compare hotspots across workloads]

    Actor --> UC1;
    Actor --> UC2;
    Actor --> UC3;
    Actor --> UC4;

    UC1 -.->|prerequisite| UC2;
    UC2 -.->|prerequisite| UC3;
    UC2 -.->|prerequisite| UC4;
```

## Activity Flow

```mermaid
stateDiagram-v2
    [*] --> Setup
    Setup --> FoundationReady: symlink + scaffolds
    FoundationReady --> ReportGen
    ReportGen --> Verify: report.json created
    Verify --> Hotspots: verification passed
    Verify --> ReportGen: verification failed
    Hotspots --> Polish
    Polish --> [*]: gates pass
```

## Inter-Phase Dependencies

### Phase 1 → Phase 2

**Artifacts**:
- `models/wan2.1-t2v-14b/source-data/` provides local model metadata for later configs and report “model spec” fields.

**Code Dependencies**:
- None (Phase 2 is pure-Python schema and path helpers).

### Phase 2 → Phase 3

**Artifacts**:
- Shared schema types from `src/llm_perf_opt/data/analytic_common.py` are required for the Wan analyzer to build `report.json`.
- Path helpers from `src/llm_perf_opt/utils/paths.py` are required for consistent output layout.

**Code Dependencies**:

```python
from llm_perf_opt.data.wan2_1_analytic import Wan2_1AnalyticModelReport
from llm_perf_opt.utils.paths import wan2_1_report_path


def write_report(report: Wan2_1AnalyticModelReport) -> None:
    out_path = wan2_1_report_path(report.report_id)
    ...
```

### Phase 3 → Phase 4

**Artifacts**:
- `tmp/profile-output/<run_id>/static_analysis/wan2_1/report.json` is the shared input to verification and hotspot tooling.

**Code Dependencies**:
- Verification scripts import the analytic model package (`modelmeter.models.wan2_1`) and must use the same geometry and FLOP conventions as the analyzer.

### Phase 3 → Phase 5

**Artifacts**:
- `report.json` is the input to summary rendering and report comparison tools.

**Code Dependencies**:
- Hotspot tooling depends on the shared schema invariants (stable module ids and consistent metric semantics).

## Integration Testing

```bash
# Phase 2: schema and path helper tests
pixi run pytest tests/unit/data/test_analytic_common.py
pixi run pytest tests/unit/utils/test_paths_wan2_1.py

# Phase 3: report generation (may skip if local model reference unavailable)
pixi run pytest tests/integration/wan2_1/test_wan2_1_analyzer_report.py

# Phase 4: verification (may skip if local reference unavailable)
pixi run pytest tests/integration/wan2_1/test_verify_layers_ci_tiny.py
```

## Critical Integration Points

1. Phase 1 bootstrap correctness: `models/wan2.1-t2v-14b/source-data/config.json` must be readable or analyzer/verification must fail with an actionable message.
2. Stable module ids: transformer blocks and key subcomponents must have deterministic ids so layer-by-layer verification and hotspot comparisons are meaningful across runs.
3. FLOP counting convention alignment: verification comparisons must use the same convention as the chosen reference measurement (`torch.utils.flop_counter.FlopCounterMode`), and any intentional omissions must be explicit and controlled (for example via a “torch-visible” mode flag).
4. Step scaling semantics: total FLOPs scale with `num_inference_steps`, while reported weight/activation memory must not scale with steps.
5. Output layout invariants: all artifacts must live under `tmp/profile-output/<run_id>/static_analysis/wan2_1/` so manual inspection and automation can locate them predictably.

## References

- Individual phase guides:
  - `context/tasks/working/004-wan2-1-analytic-model/impl-phase-1-setup.md`
  - `context/tasks/working/004-wan2-1-analytic-model/impl-phase-2-foundational.md`
  - `context/tasks/working/004-wan2-1-analytic-model/impl-phase-3-us1-report.md`
  - `context/tasks/working/004-wan2-1-analytic-model/impl-phase-4-us2-verify.md`
  - `context/tasks/working/004-wan2-1-analytic-model/impl-phase-5-us3-hotspots.md`
  - `context/tasks/working/004-wan2-1-analytic-model/impl-phase-6-polish.md`
- Spec: `specs/004-wan2-1-analytic-model/spec.md`
- Tasks breakdown: `specs/004-wan2-1-analytic-model/tasks.md`
- Data model: `specs/004-wan2-1-analytic-model/data-model.md`
- Contracts: `specs/004-wan2-1-analytic-model/contracts/`
