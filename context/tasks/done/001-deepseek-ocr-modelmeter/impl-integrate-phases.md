# Phase Integration Guide: DeepSeek-OCR Analytic Modeling in ModelMeter

**Feature**: `001-deepseek-ocr-modelmeter` | **Phases**: 9

## Overview

This feature adds a theoretical analytic modeling pipeline for DeepSeek-OCR on top of existing profiling workflows. Phases 1–2 establish environment, artifacts, and core data models; Phases 3–6 implement analytic layers and a runner by module (vision, LLaMA, decoder, core) and emit JSON/YAML plus Markdown docs; Phase 7 exposes machine-readable exports and contracts for capacity planning; Phase 8 aligns abstractions with existing profiling pipelines; and Phase 9 hardens quality via documentation, tests, and tooling.

## Phase Flow

```mermaid
sequenceDiagram
    participant Engineer
    participant Phase1 as Setup
    participant Phase2 as Foundation
    participant Phase3 as VisionLayers
    participant Phase4 as LLaMALayers
    participant Phase5 as DecoderLayers
    participant Phase6 as CorePipeline
    participant Phase7 as ExportContracts
    participant Phase8 as Reuse
    participant Phase9 as Polish
    participant Storage

    Note over Engineer,Storage: Phase 1 – Setup
    Engineer->>Phase1: verify Pixi env & TorchInfo artifacts
    Phase1->>Storage: ensure reports/torchinfo-unique-layers.{json,md}

    Note over Engineer,Storage: Phase 2 – Foundational
    Engineer->>Phase2: implement data models & helpers
    Phase2->>Storage: ready-to-use layer package skeletons and TargetOperatorList loader

    Note over Engineer,Storage: Phase 3–6 – Analytic Layers & Runner
    Engineer->>Phase3: implement vision BaseLayer subclasses
    Engineer->>Phase4: implement LLaMA primitives
    Engineer->>Phase5: implement decoder + MoE layers
    Engineer->>Phase6: implement core aggregator & analytic mode
    Phase6->>Storage: write AnalyticModelReport JSON/YAML and Markdown layer docs

    Note over Engineer,Storage: Phase 7 – Exports & Contracts
    Engineer->>Phase7: wire contracts & what-if tooling
    Phase7->>Storage: machine-readable artifacts consumed by external tools

    Note over Engineer,Storage: Phase 8 – Reuse in Existing Workflows
    Engineer->>Phase8: align naming & integrate with profiling exports
    Phase8->>Storage: unified reports that include DeepSeek-OCR analytic data

    Note over Engineer,Storage: Phase 9 – Polish
    Engineer->>Phase9: docs, tests, lint, type-check
    Phase9-->>Engineer: ready-to-ship analytic modeling pipeline
```

## Artifact Flow Between Phases

```mermaid
graph TD
    subgraph Phase1["Phase 1: Setup"]
        P1T1[T001: Pixi env] --> P1A1[rtx5090 Python 3.11]
        P1T2[T002: TorchInfo artifacts] --> P1A2[torchinfo-unique-layers-json-and-md]
    end

    subgraph Phase2["Phase 2: Foundation"]
        P2T1[T004: Layer skeletons] --> P2A1[DeepSeek-OCR BaseLayer stubs]
        P2T2[T005/T006: Data models] --> P2A2[AnalyticModelReport, TargetOperatorList]
        P2T3[T007/T008: Helpers] --> P2A3[Path helpers, loader]
    end

    subgraph Phase3["Phase 3: Vision Layers"]
        P3T1[T009–T021: Vision BaseLayer impls]
        P3A1[AnalyticModelReport JSON/YAML]
        P3A2[Markdown layer docs]
    end

    subgraph Phase4["Phase 4: LLaMA Layers"]
        P4T1[T022–T023: LLaMA primitives]
    end

    subgraph Phase5["Phase 5: Decoder Layers"]
        P5T1[T024–T028: Decoder/MoE layers]
    end

    subgraph Phase6["Phase 6: Core & Pipeline"]
        P6T1[T029–T034: Aggregator, runner, Markdown]
    end

    subgraph Phase7["Phase 7: Planning Exports"]
        P7T1[T037–T038: Metrics aggregation] --> P7A1[Detailed module/operator metrics]
        P7T2[T039/T040: Contracts & CLI]
        P7T3[T041: What-if script]
    end

    subgraph Phase8["Phase 8: Reuse"]
        P8T1[T044/T045: Naming & exports] --> P8A1[Unified reporting views]
        P8T2[T043/T046: Manual script & docs]
    end

    subgraph Phase9["Phase 9: Polish"]
        P9T1[T047/T048: Docs & tests]
        P9T2[T049/T050: Lint/type & docs]
    end

    P1A2 -.->|reads| P2T3
    P2A2 -.->|constructs| P6T1
    P2A3 -.->|locates output| P6T1
    P3A1 -.->|consumed by| P7T1
    P3A1 -.->|consumed by| P7T3
    P3A2 -.->|linked from| P8T2
    P7A1 -.->|displayed by| P8T1
```

## References

- Phase guides:
  - `context/tasks/001-deepseek-ocr-modelmeter/impl-phase-1-setup.md`
  - `context/tasks/001-deepseek-ocr-modelmeter/impl-phase-2-foundational.md`
  - `context/tasks/001-deepseek-ocr-modelmeter/impl-phase-3-vision.md`
  - `context/tasks/001-deepseek-ocr-modelmeter/impl-phase-4-llama.md`
  - `context/tasks/001-deepseek-ocr-modelmeter/impl-phase-5-decoder.md`
  - `context/tasks/001-deepseek-ocr-modelmeter/impl-phase-6-core.md`
  - `context/tasks/001-deepseek-ocr-modelmeter/impl-phase-7-us2.md`
  - `context/tasks/001-deepseek-ocr-modelmeter/impl-phase-8-us3.md`
  - `context/tasks/001-deepseek-ocr-modelmeter/impl-phase-9-polish.md`
- Spec: `specs/001-deepseek-ocr-modelmeter/spec.md`
- Data model: `specs/001-deepseek-ocr-modelmeter/data-model.md`
- Contracts: `specs/001-deepseek-ocr-modelmeter/contracts/`
