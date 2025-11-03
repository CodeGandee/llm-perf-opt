# Implementation Plan: Stage 2 — NVIDIA-Backed Deep LLM Profiling

**Branch**: `002-nvidia-llm-profiling` | **Date**: 2025-10-29 | **Spec**: specs/002-nvidia-llm-profiling/spec.md
**Input**: Feature specification from `/specs/002-nvidia-llm-profiling/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Stage 2 extends Stage 1 (NVTX-based) profiling by capturing kernel-level GPU metrics for DeepSeek-OCR and producing actionable stakeholder reports. We will use NVIDIA tools to obtain kernel durations, achieved occupancy, memory throughput, and utilization, export top‑operators and top‑kernels tables (with mean ms), and integrate per‑stage timing into the Aggregates section. Vision processing (sam+clip+projector) remains a note, not a separate stage row. Cross‑model applicability is planned as a lower‑priority extension.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11 (Pixi env)  
**Primary Dependencies**: PyTorch 2.5.x, Transformers 4.46.x, Tokenizers 0.20.x, NVTX, NVIDIA Nsight Systems (nsys), NVIDIA Nsight Compute (ncu)  
**Storage**: Local filesystem artifacts under `tmp/` and `context/` (no DB)  
**Testing**: Manual verification via generated artifacts; optional pytest for exporters [NEEDS CLARIFICATION]  
**Target Platform**: Linux workstation with NVIDIA GPU (CUDA 12.4+)  
**Project Type**: Single project (CLI/run scripts + report exporters)  
**Performance Goals**: Default deep profiling overhead ≤ 25% over non‑profiled baseline [NEEDS CLARIFICATION]  
**Constraints**: Trace size bounded (< 2 GB per run) and completion < 30 minutes baseline [NEEDS CLARIFICATION]  
**Scale/Scope**: Single‑GPU runs (MIG/multi‑GPU optional), dataset subset for reproducibility

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Pythonic clarity & docstrings: All new/changed public APIs/classes include
  NumPy‑style docstrings and illustrative examples.
- Typed, linted, formatted: All new code is fully type‑annotated; `mypy` passes
  with zero errors; `ruff` lint/format passes with zero errors.
- OO discipline for functional classes: Member vars prefixed with `m_`; use
  properties for read‑only access and `set_xxx()` for mutations; constructors
  argument‑free with `from_xxx()` factories where needed.
- Data models: Use `attrs` (default) with `@define(kw_only=True)` and `field`
  metadata, or `pydantic` for web schemas; no business logic in models.
- Runtime environment declared: Execution context specified (Pixi preferred;
  otherwise venv). Commands in this plan use that environment explicitly.
- Testing plan: Manual tests provided for major functionality with file paths
  under `tests/manual/…`. Automated tests only if requested or critical; specify
  locations under `tests/unit/…` and `tests/integration/…` if included.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
src/
├── llm_perf_opt/
│   ├── runners/
│   ├── profiling/
│   └── exporters/
tests/
├── manual/
│   └── stage2_profile/
├── integration/
└── unit/
```

**Structure Decision**: Single‑project repository. We extend `src/llm_perf_opt/` runners and exporters, add planning docs under `specs/002-nvidia-llm-profiling/`, and place manual tests under `tests/manual/stage2_profile/`.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
 
## Phase 0: Outline & Research

Unknowns (from Technical Context):
- Profiling overhead target and acceptable bound [NEEDS CLARIFICATION]
- Max trace size and representative input set for Stage 2 [NEEDS CLARIFICATION]
- Degree of automated testing for exporters [NEEDS CLARIFICATION]

Research Tasks:
- Investigate best practices for `nsys` + `ncu` on LLM inference workloads (single‑GPU)
- Determine manageable input sizes that still trigger representative kernels
- Establish artifact retention strategy and size limits

Output: specs/002-nvidia-llm-profiling/research.md (decisions with rationale and alternatives)

## Phase 1: Design & Contracts

Deliverables:
- specs/002-nvidia-llm-profiling/data-model.md (Reuse‑first: reuse Stage 1 models; add KernelRecord only; optional LLMProfileReport.kernels_topk; ProfilingSession is an on‑disk provenance bundle; ModelTarget embedded in provenance)
- specs/002-nvidia-llm-profiling/contracts/profile.yaml (OpenAPI endpoints for run and artifact retrieval)
- specs/002-nvidia-llm-profiling/quickstart.md (Pixi/NVIDIA tools usage; manual test flow)
- Update agent context via `.specify/scripts/bash/update-agent-context.sh codex`

## Phase 2: Readiness Check (stop here)

Re‑evaluate Constitution Gates after Phase 1. If any remain violated, justify in Complexity Tracking or adjust design.
