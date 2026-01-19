# Implementation Plan: Wan2.1 Analytic FLOP Model

**Branch**: `004-wan2-1-analytic-model` | **Date**: 2026-01-16 | **Spec**: /data1/huangzhe/code/llm-perf-opt/specs/004-wan2-1-analytic-model/spec.md
**Input**: Feature specification from `/specs/004-wan2-1-analytic-model/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command.

## Summary

Implement a ModelMeter-style analytic model for Wan2.1-T2V-14B (starting with the diffusion transformer core) and a matching static-analysis runner that produces machine-readable and human-readable artifacts under `/data1/huangzhe/code/llm-perf-opt/tmp/profile-output/<run_id>/static_analysis/`.

High-level approach:
- Extend `extern/modelmeter/models/` with a new `modelmeter.models.wan2_1` package that composes typed `BaseLayer` components and exposes Hydra configs mirroring the DeepSeek-OCR analytic model pattern.
- Introduce shared analytic report data models in `src/llm_perf_opt/data/analytic_common.py` and reuse them for both DeepSeek-OCR and Wan2.1 reports, keeping model-specific spec/workload types separate.
- Provide verification scripts that compare analytic FLOP counts to PyTorch `torch.utils.flop_counter.FlopCounterMode` references (layer-by-layer and end-to-end within the analytic scope) and enforce the ≤5% error budget for the standard workload set.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: Hydra (omegaconf), attrs, PyTorch (reference flop counter), ModelMeter analytic layers (`modelmeter.layers.*`)  
**Storage**: Filesystem artifacts under `/data1/huangzhe/code/llm-perf-opt/tmp/profile-output/<run_id>/` (static_analysis/, nsys/, ncu/, torch_profiler/)  
**Testing**: pytest (unit/integration), manual scripts under `tests/manual/`  
**Target Platform**: Linux; GPU optional for analytic evaluation, but verification against reference FLOPs may require CUDA depending on the reference model implementation  
**Project Type**: Single Python package with Hydra-based runners and ModelMeter extension under `extern/modelmeter/`  
**Performance Goals**: End-to-end and per-layer FLOP estimates within ≤5% of the reference for the standard workload set; report generation completes in ≤60 seconds for representative workloads on a developer workstation  
**Constraints**: No weights/checkpoints committed; all run artifacts under `tmp/`; keep module naming stable for “layer-by-layer” comparisons; avoid breaking DeepSeek-OCR report consumers while extracting shared report types  
**Scale/Scope**: Target model is Wan2.1-T2V-14B; primary workload knobs are batch size, frames, height, width, inference steps, and text length; v1 focuses on the diffusion transformer core

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Gate status: PASS (no expected constitution violations).
- Post-design re-check: PASS (design artifacts generated under `/data1/huangzhe/code/llm-perf-opt/specs/004-wan2-1-analytic-model/` and no additional tools beyond existing repo dependencies are introduced).
- Runtime environment: use the Pixi `rtx5090` environment for all reference FLOP verification scripts and (optionally) for the analyzer runner to match the CUDA toolchain used by other modelmeter verification scripts.
  - `pixi run -e rtx5090 python -m llm_perf_opt.runners.wan2_1_analyzer ...`
  - `pixi run -e rtx5090 python -m modelmeter.models.wan2_1.scripts.verify.run_verify_* ...`
- Lint/type gates:
  - `pixi run ruff check .`
  - `pixi run mypy src`
- Test plan:
  - Unit tests under `tests/unit/` for token-geometry helpers, scaling invariants, and report serialization invariants.
  - Integration tests under `tests/integration/` for end-to-end report generation and (when a local Wan2.1 reference implementation is available) reference FLOP comparisons for the standard workload set.
  - Manual test under `tests/manual/wan2_1/manual_wan2_1_static_analysis.py` that generates artifacts under a user-specified `tmp/profile-output/<run_id>/` directory for inspection.
- Artifacts: all outputs go under `/data1/huangzhe/code/llm-perf-opt/tmp/profile-output/<run_id>/` with a dedicated subtree for Wan2.1 static analysis (e.g., `static_analysis/wan2_1/`).

## Project Structure

### Documentation (this feature)

```text
/data1/huangzhe/code/llm-perf-opt/specs/004-wan2-1-analytic-model/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
    ├── openapi.yaml
    ├── python-contracts.md
    └── MAPPING.md
```

### Source Code (repository root)

```text
/data1/huangzhe/code/llm-perf-opt/
├── conf/                                      # Hydra config groups (project runners)
├── extern/modelmeter/models/
│   ├── common/                                # shared ModelMeter mixins/utilities
│   ├── deepseek_ocr/                          # reference pattern to mirror
│   └── wan2_1/                                # (new) Wan2.1 analytic model package
├── models/wan2.1-t2v-14b/                      # external model reference symlink setup
├── src/llm_perf_opt/
│   ├── data/                                  # analytic report data models
│   └── runners/                               # Hydra runners (static analysis entrypoints)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── manual/
└── tmp/                                       # run outputs and scratch space (no committed artifacts)
```

**Structure Decision**: Follow the DeepSeek-OCR ModelMeter pattern: add `extern/modelmeter/models/wan2_1/` containing typed `BaseLayer` implementations + Hydra configs and verification scripts, and add a lightweight runner under `src/llm_perf_opt/runners/` that emits a shared-schema report under `tmp/profile-output/<run_id>/static_analysis/wan2_1/`.

## Complexity Tracking

No constitution violations anticipated.
