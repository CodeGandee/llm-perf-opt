# Phase Integration Guide: Basic Profiling for DeepSeek‑OCR (Stage 1)

**Feature**: `001-profile-deepseek-ocr` | **Phases**: 6

## Overview

Phase 1 establishes the packaging, contracts, and profiling scaffolding. Phase 2 wires foundational data models and NVTX helpers into the runner. Phase 3 completes US1 by producing stage segmentation, operator summary, and MFU with aggregated metrics. Phase 4 produces a stakeholder-friendly summary. Phase 5 captures reproducibility metadata, and Phase 6 polishes APIs, typing, and docs to meet constitution gates.

## Phase Flow

```mermaid
sequenceDiagram
    participant User
    participant Runner as LLMProfileRunner
    participant Session as OCR Session
    participant Utils as MFU/Aggregate/Export
    participant Storage as tmp/

    Note over User,Storage: Phase 1-2: Setup & Foundation
    User->>Runner: install + run --help
    Runner->>Session: load_once()
    Runner->>Utils: helpers available

    Note over User,Storage: Phase 3: Core Profiling
    User->>Runner: run(images, repeats)
    Runner->>Session: run_inference(image)
    Runner->>Utils: aggregate+export
    Utils->>Storage: report.md, metrics.json

    Note over User,Storage: Phase 4: Summary
    Runner->>Utils: top_n_operators()
    Utils->>Storage: stakeholder_summary.md

    Note over User,Storage: Phase 5: Reproducibility
    Runner->>Storage: inputs.yaml, env.json, assumptions.md

    Note over User,Storage: Phase 6: Polish
    User->>Runner: review docs & typing
```

## Artifact Flow Between Phases

```mermaid
graph TD
    subgraph P1["Phase 1: Setup"]
        P1T1[T001: Packaging]
        P1T2[T002: Contracts]
        P1T3[T003-7: Harness+Utils]
        P1T4[T008: Runner CLI]
    end

    subgraph P2["Phase 2: Foundation"]
        P2T1[T010: Data Models]
        P2T2[T011: Converters]
        P2T3[T012: NVTX Helpers]
        P2T4[T013: Wiring]
    end

    subgraph P3["Phase 3: US1"]
        P3T1[T020–T027]
        P3A1[report.md]
        P3A2[metrics.json]
    end

    subgraph P4["Phase 4: US2"]
        P4T1[T030–T032]
        P4A1[stakeholder_summary.md]
    end

    subgraph P5["Phase 5: US3"]
        P5A1[inputs.yaml]
        P5A2[env.json]
        P5A3[assumptions.md]
    end

    subgraph P6["Phase 6: Polish"]
        P6T1[Docstrings]
        P6T2[Typing/Lint]
        P6T3[README/Quickstart]
    end

    P1T4 --> P2T4
    P2T4 --> P3T1
    P3A2 --> P4T1
    P1T4 --> P5A1
```

## System Architecture

```mermaid
classDiagram
    class Contracts {
        +LLMProfileRequest
        +LLMProfileAccepted
    }

    class Harness {
        +nvtx_range(name)
    }

    class Utils {
        +mfu(...)
        +mean_std(values)
        +write_operator_markdown(records, path)
    }

    class Runner {
        +main()
        +run_stage1_once(image) dict
        +summarize_runs(results) dict
        +write_outputs(dir, summary, ops, raw)
    }

    Contracts --> Runner : build request
    Harness --> Runner : NVTX ranges
    Utils --> Runner : helpers
```

## Use Cases

```mermaid
graph LR
    Actor((Engineer))

    UC1[Setup Environment]
    UC2[Profile DeepSeek‑OCR]
    UC3[Generate Output]
    UC4[Reproduce Run]

    Actor --> UC1
    Actor --> UC2
    Actor --> UC3
    Actor --> UC4

    UC1 -.prerequisite.-> UC2
    UC2 -.produces.-> UC3
    UC2 -.produces.-> UC4
```
