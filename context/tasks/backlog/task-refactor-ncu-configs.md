# Refactor Plan: Nsight Compute Configs (Hydra Runners)

## What to Refactor
- YAML presets under `conf/profiling/ncu/`:
  - `ncu.default.yaml`, `ncu.rtx3090.yaml`, `ncu.rtx3090.compute.yaml`, `ncu.rtx3090.memory.yaml`.
- No changes to `scripts/ncu/release/ncu-profile-kernels.py` (per direction). These presets remain for Hydra runners only.

## Why Refactor
- Improve usability: run the standalone NCU kernel profiler with consistent presets without duplicating CLI flags.
- Enhance readability: use clear YAML maps/lists with descriptions, while preserving existing Hydra compatibility for Stage-2 runners.
- Reduce config drift: a single source (YAML) drives both orchestrated runs and the standalone script.

## How to Refactor
1. Introduce a single normalized schema under `ncu_cli` for all Nsight Compute presets in `conf/profiling/ncu/*.yaml` (remove duplicated top-level keys):
   - `target_processes` → `--target-processes`
   - `nvtx.include` → `--nvtx --nvtx-include`
   - `set` → `--set`
   - `metrics` (YAML list or null) → `--metrics`
   - `sections` (YAML list) → `--section` repeated
   - `export.csv` (bool) → signals CSV exports in tooling
2. Keep `conf/config.yaml` defaults composing these presets into `pipeline.ncu`; runner code will be refactored to read `pipeline.ncu.ncu_cli.*` instead of the old top-level fields.
3. Update README under `conf/profiling/` documenting the new `ncu_cli` schema (lists/maps for clarity).

## Impact Analysis
- Backward compatible:
  - Script: unchanged; still uses its own CLI flags and defaults.
- Risks: Stage‑2 runner code reading old keys will need updates (planned in follow-up).
- Mitigation: Provide clear mapping in README; initial fallbacks in runner may still function (e.g., default `set=roofline`).

## Expected Outcome
- Single set of presets drives both orchestrated and ad‑hoc kernel profiling.
- Clear, documented YAML structure improves discoverability and reduces mistakes.
- Easier to switch devices/presets via one CLI flag.

## Before / After Examples

Before (excerpt) `conf/profiling/ncu/ncu.rtx3090.memory.yaml`:
```yaml
target_processes: all
nvtx_include: decode*
set: roofline
sections:
  - SpeedOfLight
  - MemoryWorkloadAnalysis
metrics: null
csv: true
```

After (uses only `ncu_cli`):
```yaml
description: Nsight Compute preset: RTX 3090 (GA102) — Memory focus
ncu_cli:
  target_processes: all
  nvtx:
    include: decode*
  set: roofline
  sections:
    - SpeedOfLight
    - MemoryWorkloadAnalysis
  metrics: null
  export:
    csv: true
```

Example: Default preset aligned with script defaults (hard-coded metrics + sections)
```yaml
description: Default Nsight Compute preset aligned with script defaults
ncu_cli:
  target_processes: all
  nvtx:
    include: decode*
  set: roofline
  # Sections and concise metrics approximating the script behavior
  sections:
    - SpeedOfLight
    - MemoryWorkloadAnalysis
    - Occupancy
    - SchedulerStats
  metrics:
    - flop_count_hp
    - flop_count_sp
    - gpu__time_duration.sum
    - sm__throughput.avg.pct_of_peak_sustained_elapsed
    - dram__throughput.avg.pct_of_peak_sustained_elapsed
  export:
    csv: true
```

Script usage before:
```bash
python3 scripts/ncu/release/ncu-profile-kernels.py \
  --kernel-regex 'flash_fwd.*' -- ./bench --size 2048
```

Script usage remains unchanged (script does not read these presets).

## References
- Code: `src/llm_perf_opt/profiling/vendor/ncu.py` (for future alignment), `scripts/ncu/release/ncu-profile-kernels.py` (unchanged)
- Configs: `conf/profiling/ncu/*.yaml`
- Nsight Compute CLI User Guide (NVIDIA)
- Context7 library id (for reference): /nvidia/nsight-compute

## Code Using Outdated NCU Configs (to update)

- src/llm_perf_opt/runners/deep_profile_runner.py:293
  - Calls `build_ncu_cmd(...)` with values read from `pipeline.ncu.*` (old keys).
  - Current reads: `nvtx_include`, `set`, `metrics` (CSV string), `sections` (list), `gating_nvtx`.
  - Change: read from `pipeline.ncu.ncu_cli.*` and map as:
    - `nvtx_expr` ← `cfg.pipeline.ncu.ncu_cli.nvtx.include`
    - `set_name` ← `cfg.pipeline.ncu.ncu_cli.set`
    - `metrics` ← `cfg.pipeline.ncu.ncu_cli.metrics` (list or null)
    - `sections` ← `cfg.pipeline.ncu.ncu_cli.sections` (list)
    - `use_nvtx` remains from `cfg.pipeline.ncu.gating_nvtx` (boolean)
    - Optional: `target_processes` ← `cfg.pipeline.ncu.ncu_cli.target_processes`

- src/llm_perf_opt/runners/deep_profile_runner.py:296,300,301,339,343
  - Lines explicitly referencing `nvtx_include`, `set`, `metrics` should switch to nested `ncu_cli` reads.

- src/llm_perf_opt/profiling/vendor/ncu.py:59
  - `build_ncu_cmd(...)` currently hardcodes `--target-processes all` and expects `metrics` as CSV string.
  - Change: extend signature to accept `target_processes: Optional[str] = 'all'` and `metrics: Union[str, Sequence[str], None]`.
  - Convert metrics list to CSV internally; keep `_filter_metrics(...)` compatible with both string and list.
  - If `sections` is provided, allowing `metrics=None` is fine (section defaults only).

- conf/config.yaml:72
  - Comment references old fields: “Keys (nvtx_include, set, metrics, sections) inherited from the preset.”
  - Change to: “Keys are read from `pipeline.ncu.ncu_cli.*` (target_processes, nvtx.include, set, metrics, sections, export.*).”

- docs/configuration.md:15
  - Mentions Nsight Compute presets but not fields; add a brief `ncu_cli` field list and example.

- conf/profiling/README.md:1
  - Already updated to `ncu_cli`; review for accuracy and examples with list-based `metrics` and nested NVTX include.

### Script Integrations (no CLI changes)

- scripts/ncu/release/ncu-profile-kernels.py
  - Keep CLI interface unchanged (no `--ncu-config`).
  - Internal defaults must align with `ncu.default` preset:
    - `DEFAULT_SECTIONS = ["SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy", "SchedulerStats"]`
    - Do not pass `--metrics` by default; rely on section defaults (equivalent to `metrics: null`).
  - Provenance remains, but update description/comments to reflect alignment with `ncu_cli`.

- scripts/ncu/release/ncu-profile-kernels.sh
  - Keep CLI interface unchanged.
  - Internal defaults must align with `ncu.default` preset:
    - `DEFAULT_SECTIONS="SpeedOfLight MemoryWorkloadAnalysis Occupancy SchedulerStats"`
    - Do not pass `--metrics` by default; rely on section defaults.
  - Comments/help text should mention that defaults mirror the runner presets.

### Before (deep_profile_runner extract)
```python
ncu_cfg = getattr(getattr(cfg, "pipeline", {}), "ncu", {})
gating_nvtx_ncu = bool(getattr(ncu_cfg, "gating_nvtx", True))
ncu_set = str(getattr(ncu_cfg, "set", "roofline")) or "roofline"
ncu_metrics = str(getattr(ncu_cfg, "metrics", "")) or None
secs = getattr(ncu_cfg, "sections", None)
ncu_sections = [str(s) for s in secs] if isinstance(secs, (list, tuple)) else None
ncu_cmd = build_ncu_cmd(
    ncu_out,
    work,
    nvtx_expr=str(getattr(ncu_cfg, "nvtx_include", "decode*")),
    kernel_regex=kernel_regex,
    csv_log=csv_log,
    use_nvtx=gating_nvtx_ncu,
    set_name=ncu_set,
    metrics=(None if ncu_sections else ncu_metrics),
    sections=ncu_sections,
)
```

### After (deep_profile_runner extract)
```python
ncu_cfg = getattr(getattr(cfg, "pipeline", {}), "ncu", {})
ncu_cli = getattr(ncu_cfg, "ncu_cli", {})
gating_nvtx_ncu = bool(getattr(ncu_cfg, "gating_nvtx", True))

nvtx_expr = str(getattr(getattr(ncu_cli, "nvtx", {}), "include", "decode*"))
set_name = str(getattr(ncu_cli, "set", "roofline"))
metrics_val = getattr(ncu_cli, "metrics", None)   # list[str] or None
sections_val = getattr(ncu_cli, "sections", None)  # list[str] or None
target_procs = str(getattr(ncu_cli, "target_processes", "all"))

ncu_cmd = build_ncu_cmd(
    ncu_out,
    work,
    nvtx_expr=nvtx_expr,
    kernel_regex=kernel_regex,
    csv_log=csv_log,
    use_nvtx=gating_nvtx_ncu,
    set_name=set_name,
    metrics=metrics_val,     # builder accepts list or str
    sections=sections_val,
    target_processes=target_procs,
)
```
