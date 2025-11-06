# Research: NVTX-based NCU Regional Profiling

## Decisions

- Decision: Region discovery and capture via NVTX include expressions + range replay
  Rationale: Using `ncu --nvtx --nvtx-include <expr> --replay-mode range` (or `app-range`) aggregates results per NVTX range with deterministic attribution and no kernel‑name heuristics.
  Alternatives considered: Single pass + post‑hoc NVTX correlation (NCU lacks reliable region export); time‑window filtering (not supported directly); rely only on top‑K kernel regex from NSYS (insufficient for region attribution).

- Decision: Nested region semantics are inclusive
  Rationale: Aligns with timeline/wall‑time expectations and simplifies reconciliation; exclusive values may be derived if needed.
  Alternatives: Exclusive parent totals; dual reporting by default (adds verbosity).

- Decision: Kernel selection within each region uses include/exclude name patterns
  Rationale: Flexible, reproducible, and aligns with existing config patterns. Exclude takes precedence; default is include all.
  Alternatives: Top‑N by time; hybrid (adds precedence complexity without clear added value for this feature).

- Decision: Multi‑process/device scope provides both per‑scope and aggregate outputs
  Rationale: Per‑rank debugging and single‑pane summaries are both valuable; aggregate is toggleable.
  Alternatives: Only aggregate; only per‑scope (either reduces utility).

- Decision: Metrics/sections remain configurable via existing `conf/profiling/ncu/ncu.*.yaml` presets
  Rationale: Matches current facility and avoids duplication.
  Alternatives: New parallel config tree (adds maintenance burden).

- Decision: Artifact layout under Hydra run dir
  Rationale: Co‑locate with Stage‑2 artifacts: `/tmp/profile-output/<run_id>/ncu/regions/<region>/...` plus consolidated `ncu/regions/{report.md,report.json}`.
  Alternatives: Separate top‑level output (fragments run context).

## Constraints & Practices

- Preserve existing CLI contract and defaults; expose native NCU CLI flags under `pipeline.ncu.ncu_cli.*` (1:1 mapping). Examples: `replay_mode`, `nvtx.include`, `sections`, `metrics`, `target_processes`, `kernel_name`, `kernel_name_base`. No standalone `region_mode` config.
- Use `attrs` data models for NCUProfileRegion and NCUProfileRegionReport in the Python package; reuse `KernelRecord`; emit JSON via `.asdict()`.
- Keep manual test under `tests/manual/ncu/manual_nvtx_regions.py`.
- Document configuration and examples in `docs/running.md` and `docs/configuration.md`.

## Open Items resolved here

- NVTX label conventions: default include patterns `LLM@*`, `prefill*`, `decode*`; user may override via config list.
- Aggregation across processes/devices: per‑scope JSON array + optional aggregate object; aggregate enabled by default.
- Error handling: If no NVTX matches → generate baseline section noting "no matching regions" and exit 0.
