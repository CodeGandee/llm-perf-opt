# Implementation Plan: Basic Profiling for DeepSeek‑OCR (Stage 1)

**Branch**: `001-profile-deepseek-ocr` | **Date**: 2025-10-28 | **Spec**: /data2/huangzhe/code/llm-perf-opt/specs/001-profile-deepseek-ocr/spec.md
**Input**: Feature specification from /data2/huangzhe/code/llm-perf-opt/specs/001-profile-deepseek-ocr/spec.md

**Note**: This plan follows the .specify workflow and satisfies constitution gates. Artifacts generated under: /data2/huangzhe/code/llm-perf-opt/specs/001-profile-deepseek-ocr

## Summary

Goal: Produce a Stage 1 profiling report for DeepSeek‑OCR that (a) segments runtime into prefill and decode via NVTX ranges, (b) emits an operator‑level time/memory summary using PyTorch Profiler, and (c) provides early MFU estimates at model‑level and per‑stage using measured throughput and an analytical FLOPs‑per‑token approximation derived from model config.

Approach: Use Pixi environment (CUDA 12) with PyTorch 2.5+, Transformers 4.46.3, NVTX, and Hydra. Instrument manual HF run with NVTX ranges for prefill/decode, wrap with PyTorch Profiler to collect operator stats, and compute MFU using tokens/sec and approximated FLOPs/token from model configuration. Support repeated passes over a 10–20 image input set and aggregate metrics (mean/variance). Output a human‑readable report plus a concise stakeholder summary.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11 (Pixi env; pyproject requires >=3.11)  
**Primary Dependencies**: PyTorch 2.5.x (CUDA 12.4 wheels), Transformers 4.46.3, Tokenizers 0.20.x, NVTX, Hydra/OmegaConf, attrs/pydantic, NVML (nvidia-ml-py), ruff, mypy  
**Storage**: N/A (local files for artifacts under `/data2/huangzhe/code/llm-perf-opt/tmp` and `/data2/huangzhe/code/llm-perf-opt/specs/001-profile-deepseek-ocr`)  
**Testing**: pytest (manual tests prioritized under `/data2/huangzhe/code/llm-perf-opt/tests/manual/`; optional unit/integration later)  
**Target Platform**: Linux server with NVIDIA GPU; CUDA 12.0 system requirement (Pixi)  
**Project Type**: single  
**Performance Goals**: Stable Stage 1 timings (prefill/decode), operator ranking, early MFU; numeric thresholds for MFU accuracy NEEDS CLARIFICATION (addressed in research)  
**Constraints**: Repeated runs yield ±10% variance for timings and MFU per SC‑002; GPU memory sufficiency is required; exact batch size and decoding settings NEEDS CLARIFICATION (addressed in research)  
**Scale/Scope**: Single‑GPU profiling of 10–20 images; repeatable workflow with aggregation; no distributed profiling in Stage 1

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

Gate Evaluation (pre‑design): PASS. No violations expected. Runtime = Pixi; typing/lint via mypy/ruff; data models via attrs/pydantic for schemas and report serialization. Re‑evaluate after Phase 1 design.

## Project Structure

### Documentation (this feature)

```text
/data2/huangzhe/code/llm-perf-opt/specs/001-profile-deepseek-ocr/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 (not created here)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
/data2/huangzhe/code/llm-perf-opt/
├── pyproject.toml             # Pixi env + tasks + project metadata
├── conf/                      # Hydra config tree (present; README stubs, YAML planned)
├── models/                    # model weights (symlinked: deepseek-ocr -> external repo)
├── datasets/                  # dataset organization (present, to be expanded per guide)
├── extern/                    # reference sources (present)
├── src/
│   └── llm_perf_opt/
│       ├── profiling/         # profiling harness (nvtx, torch profiler, nvml)
│       ├── runners/           # entry points (e.g., stage1 runner)
│       └── data/              # schema + serialization helpers
├── tests/
│   ├── manual/                # manual scripts (DeepSeek‑OCR HF driver)
│   ├── unit/                  # optional unit tests
│   └── integration/           # optional integration tests
├── tmp/                       # local artifacts (Stage 1 traces, reports)
└── context/                   # knowledge base; project structure guide lives here
```

**Structure Decision**: Conform to current repo layout and adopt ideas from `context/hints/nv-profile-kb/about-profile-project-structure.md`. All code lives under the unified package `llm_perf_opt` with subpackages `profiling/`, `runners/`, and `data/`. Stage 1 artifacts live under `tmp/` (not `runs/` yet). Hydra config tree exists under `conf/` and will be populated in a later phase.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |

No complexity exceptions anticipated for Stage 1; repository remains single‑project.
