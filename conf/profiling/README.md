Profiling Config Group

Toggles and options for profilers used during runs.

Example keys:
- `nsys`: { enable, args }
- `ncu`:  { enable, gating_nvtx, ncu_cli }

Nsight Compute (`profiling/ncu/*`)
- New schema groups CLI arguments under `ncu_cli` for clarity and portability:
  - `ncu_cli.target_processes`: string (e.g., `all`) → `--target-processes`.
  - `ncu_cli.nvtx.include`: optional string (e.g., `decode*`) → when set and NVTX gating is enabled, passes `--nvtx --nvtx-include`. If omitted/null, no NVTX args are passed.
  - `ncu_cli.set`: string (e.g., `roofline`) → `--set`.
  - `ncu_cli.metrics`: list of metric names, or `null` to omit `--metrics`.
  - `ncu_cli.sections`: YAML list of NCU sections (e.g., SpeedOfLight, ComputeWorkloadAnalysis, ...).
  - `ncu_cli.export.csv`: boolean, request CSV-friendly exports (tooling-defined).
  - `ncu_cli.force_overwrite`: boolean, when true pass `--force-overwrite` to overwrite existing outputs.
  - `ncu_cli.kernel_name_base`: string, one of {`demangled`, `mangled`} → `--kernel-name-base`.
  - Defaults: the `ncu.default` preset mirrors the scripts’ behavior but explicitly hard-codes both sections and a concise metrics list:
    - sections: [SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats]
    - metrics: [flop_count_hp, flop_count_sp, gpu__time_duration.sum, sm__throughput.avg.pct_of_peak_sustained_elapsed, dram__throughput.avg.pct_of_peak_sustained_elapsed]

Notes
- Avoid hard-coding device-specific metrics unless necessary; prefer sections or set presets for portability.
- For GA102 (RTX 3090) configs, `ncu_cli.metrics: null` is used to rely on section defaults due to metric stability across NCU versions.
- `nvml`: { enable }
