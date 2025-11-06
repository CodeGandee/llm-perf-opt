# Implementation Plan: NVTX-based NCU Regional Profiling

**Branch**: `003-nvtx-ncu-profiling` | **Date**: 2025-11-06 | **Spec**: /workspace/code/llm-perf-opt/specs/003-nvtx-ncu-profiling/spec.md
**Input**: Feature specification from `/specs/003-nvtx-ncu-profiling/spec.md`

## Summary

Extend the existing Nsight Compute profiling facility to support NVTX range–scoped profiling and reporting. Only kernels occurring inside NVTX-marked regions are profiled, the output aggregates metrics by region (including nested regions), and users can filter kernels within each region by include/exclude patterns. Metrics and sections remain fully configurable via Hydra, consistent with current NCU presets.

High-level approach:
- Discover or accept a configured set of NVTX region labels and run one NCU capture per region using `--nvtx --nvtx-include <expr>`; write per‑region artifacts under `ncu/regions/<region>/` plus a consolidated machine‑readable summary.
- Preserve current deep profiling flows and presets; add config to enable region mode and kernel selection patterns.
- Generate a per‑region report (human + JSON) and an aggregate summary across processes/devices, with inclusive parent totals for nested regions.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: Hydra (omegaconf), mdutils, attrs, nvtx (runtime), Nsight Systems/Compute CLIs (nsys/ncu)  
**Storage**: Filesystem artifacts under `/workspace/code/llm-perf-opt/tmp/profile-output/<run_id>/` (nsys/, ncu/, torch_profiler/, static_analysis/)  
**Testing**: pytest (unit/integration), manual scripts under `tests/manual/`  
**Target Platform**: Linux with NVIDIA GPUs and CUDA toolchain on PATH  
**Project Type**: Single Python package with Hydra‑based runners  
**Performance Goals**: Region‑mode overhead ≤ 15% over baseline deep profiling for same workload; per‑region report generation ≤ 10 seconds for ≤ 20 regions  
**Constraints**: No breaking changes to current CLI and config usage; defaults remain off until enabled via config; report must be reproducible from artifacts only  
**Scale/Scope**: Typical runs produce ≤ 30 regions; each region may include up to hundreds of kernel launches; multi‑process (DDP) supported via per‑scope and aggregate outputs

## Constitution Check

Gate pre‑design status: PASS (commitment; to be enforced during implementation)

- Pythonic clarity & docstrings: All new public APIs/classes will include NumPy‑style docstrings and examples.
- Typed, linted, formatted: New code fully type‑annotated; `mypy`/`ruff` passing required pre‑merge.
- OO discipline for functional classes: Service classes will use `m_` member vars, properties, and `set_xxx()`/`from_xxx()` per constitution.
- Data models: Use `attrs` for JSON artifacts (NCUProfileRegion, NCUProfileRegionReport) with `@define(kw_only=True)`; reuse existing `KernelRecord`; no business logic in models.
- Runtime environment declared: Pixi‑based commands provided; avoid system Python.
- Testing plan: Provide manual scenario `tests/manual/ncu/manual_nvtx_regions.py`; add unit tests for parsers/aggregators.

## Project Structure

### Documentation (this feature)

```text
/workspace/code/llm-perf-opt/specs/003-nvtx-ncu-profiling/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
    └── openapi.yaml
```

### Relevant Source Code (repository root)

```text
/workspace/code/llm-perf-opt/
├── conf/
│   ├── config.yaml                         # mounts profiling presets
│   └── profiling/
│       ├── nsys/                           # Nsight Systems presets
│       └── ncu/                            # Nsight Compute presets (metrics/sections, optional nvtx.include)
├── docs/
│   ├── configuration.md                    # how presets/config work
│   ├── running.md                          # NCU workflow & scripts
│   └── internals.md                        # NVTX ranges, runners
├── scripts/ncu/release/                    # production NCU scripts
│   ├── ncu-profile-kernels.sh              # current kernel-level tool
│   ├── ncu-profile-kernels.py
│   └── README.md
├── src/llm_perf_opt/
│   ├── runners/
│   │   └── deep_profile_runner.py          # Stage‑2 orchestrator (nsys + ncu)
│   ├── profiling/
│   │   ├── nvtx_utils.py                   # NVTX helper ranges
│   │   ├── vendor/ncu.py                   # ncu command builders (supports --nvtx)
│   │   ├── nsys_stats.py                   # extract top kernels from nsys
│   │   ├── kernels.py                      # NCU CSV/JSON parsing
│   │   └── export.py                       # Markdown exports
│   └── data/models.py                      # KernelRecord, etc.
└── tests/
    ├── unit/
    ├── integration/
    └── manual/
```

**Structure Decision**: Single Python package (Hydra runners). Extend Stage‑2 runner and profiling helpers; retain script compatibility. New artifacts live under `ncu/regions/<region_name>/` plus consolidated JSON and Markdown at `ncu/regions/`.

## Complexity Tracking

No constitution violations anticipated.
