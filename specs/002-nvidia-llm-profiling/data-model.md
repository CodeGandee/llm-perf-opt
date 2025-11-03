# Data Model — Stage 2 NVIDIA Deep Profiling (Reuse-First)

This feature reuses existing domain and contract data models wherever possible and introduces only the minimum new structures needed for deep GPU profiling. Prefer extension over duplication.

## Reuse Strategy

- Stage timings and aggregates: Reuse Stage 1 internal models in `src/llm_perf_opt/data/models.py` and the runner’s existing `aggregates.stage_ms` mapping.
  - StageTiming (per-run): `StageTiming` attrs class already exists and is used by `LLMProfileReport`.
  - Aggregates (mean/std): `Stats` and the `aggregates` dict in `LLMProfileReport` are retained. Per‑stage stats remain under `aggregates.stage_ms` (map: stage → {mean, std}).
- Operators: Reuse existing
  - Domain: `OperatorSummary` (attrs)
  - Export/IO: `OperatorRecord` (TypedDict) in `src/llm_perf_opt/profiling/export.py`
- Provenance/environment: Reuse `profiling.hw.capture_env()`/`write_env_json()` and Hydra config files as the canonical source of environment/config metadata.

## Extensions Introduced in Stage 2

Only the following are added for deep GPU kernel attribution. Everything else above is reused.

1) KernelRecord (new)

- Purpose: Represent aggregated per‑kernel metrics from Nsight Compute for export (`kernels.md`) and internal use.
- Suggested shape (domain):
  - `kernel_name: str`
  - `device: str` (GPU index/name)
  - `total_ms: float`
  - `calls: int`
  - `mean_ms: float` (derived = total_ms / max(calls, 1))

2) LLMProfileReport (extend, optional)

- Keep existing fields; optionally add:
  - `kernels_topk: list[KernelRecord]` — top‑N kernels by total time for stakeholder reports.
- Note: Per‑stage timing aggregates remain in `aggregates.stage_ms` (no new StageTiming aggregate type required).

3) ProfilingSession (no new runtime model)

- Definition: A provenance bundle persisted on disk (not a new class), consisting of `env.json`, `config.yaml`, `inputs.yaml`, `report.md`, `operators.md`, `kernels.md`, profiler binaries (`.qdrep`, `.ncu-rep`), and any CSV/JSON exports under `tmp/stage2/<run_id>/`.
- If a programmatic model is desired later, prefer a thin attrs wrapper that references existing domain types rather than redefining them.

4) ModelTarget (embed, not a new top‑level model)

- Model identity (name/variant/params/dtype) is recorded in provenance (env/config) and may be included as metadata in `LLMProfileReport` notes or a `model` sub‑object. Avoid a separate duplicated model unless future contracts require it.

## Relationships (conceptual)

- One profiling session (provenance bundle) references a single model target, contains aggregate stage timings and top‑K operators/kernels, and persists all artifacts on disk.

## Validation Rules

- Non‑negative timing fields; `mean_ms` computed, not user‑supplied.
- Stages: top‑level `{prefill, decode}`; “vision (sam+clip+projector)” remains a note within prefill, not a separate top‑level stage row.
- Artifact paths written under `tmp/stage2/<run_id>/`; only existing paths are listed.

## Implementation Notes (for engineers)

- Internal domain models: `src/llm_perf_opt/data/models.py`
- Operator record type (export): `src/llm_perf_opt/profiling/export.py` → `OperatorRecord`
- Proposed kernel parsing/export: `src/llm_perf_opt/profiling/kernels.py` (parser) and `export.write_kernel_markdown()`
- Provenance writers: `src/llm_perf_opt/profiling/hw.py` → `capture_env`, `write_env_json`

