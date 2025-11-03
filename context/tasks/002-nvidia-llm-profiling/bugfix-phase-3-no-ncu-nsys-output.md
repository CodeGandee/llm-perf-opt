# Bugfix Plan: Phase 3 — Missing Nsight Outputs (nsys/ncu)

Scope: specs/002-nvidia-llm-profiling Phase 3 (US1) runner and vendor shims

## Issues Observed
- Nsight Systems
  - No `.qdrep`/`.nsys-rep` produced in `nsys/`; subsequent `nsys stats` fails with “Specified input file … does not exist”.
  - `nsys export` error: “unrecognised option '--sqlite'”.
  - “Importer binary … not found” messages during stats/export on this host.
- Nsight Compute
  - `ncu` CSV contains an error: `--section ".*SpeedOfLight.*" did not match any section`.
  - Sections naming differs by version; `--section` does not accept regex patterns.
- Hydra run dir
  - Stage 2 runs landed under `outputs/YYYY-MM-DD/HH-MM` instead of `tmp/stage2/<run_id>/` as per `conf/runner/stage2.yaml`.
- CUPTI conflicts
  - `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED` when PyTorch profiler is enabled alongside `nsys`.
- Device mismatch (vendor model)
  - Initial `cuda:1` usage triggered “Expected all tensors to be on the same device, cuda:1 and cuda:0”. Resolved by `CUDA_VISIBLE_DEVICES=1` and targeting `cuda:0` inside the workload.

## Root Cause Hypotheses
- Nsight Systems CLI syntax/version mismatch:
  - Older releases use `nsys export --type sqlite <report>` (not `--sqlite`).
  - Report file extension may be `.nsys-rep` instead of `.qdrep` depending on version.
  - Environment constraints (missing importer, insufficient privileges) may prevent capture/export.
- Nsight Compute section mismatch:
  - `--section` expects exact identifiers (e.g., `SpeedOfLight`, `SpeedOfLight_RooflineChart`); regex is not supported.
  - The selected `--set roofline` already covers roofline sections for many versions; extra `--section` can break on some installs.
- Hydra config resolution:
  - Using `config_path="conf"` and `config_name="runner/stage2"` loads the config, but hydra.run.dir still defaulted to `outputs/…` on this host. Needs explicit override/verification.

## Proposed Fixes

### A) Nsight Systems (.qdrep/.nsys-rep) and Export
1. Make export compatible with multiple versions:
   - Update vendor helper to prefer `nsys export --type sqlite <report>`.
   - Fallback: `nsys stats <report>` generates SQLite if missing (per docs).
2. Detect the produced report filename and extension after `nsys profile`:
   - Check for `<out_base>.nsys-rep` and `<out_base>.qdrep`; use whichever exists.
   - If neither exists, log a clear warning and skip stats/export gracefully.
3. Optionally enable SQLite creation at profile time:
   - Use `nsys profile --stats=true …` to auto-generate SQLite alongside the report when supported.
4. Add an environment readiness check and guidance:
   - Run `nsys status --environment`; if critical capabilities are missing, log actionable hints.

### B) Nsight Compute (sections/CSV)
1. Remove the `--section ".*SpeedOfLight.*"` filter from the default command.
   - Rely on `--set roofline` only.
   - Optionally probe `ncu --list-sections` once and include `--section SpeedOfLight` only if present.
2. Keep `--target-processes all`, NVTX include, metrics set, and `--csv --log-file` intact.
3. Preserve top‑K kernel regex support (demangled base) when names are available; otherwise, profile all decode kernels.

### C) Unified main run directory (Hydra + both stages)
1. Use `conf/runner/stage2.yaml` to define a Stage 2 main dir and set Hydra `run.dir` to it.
2. In the Stage 2 runner, use `HydraConfig.get().run.dir` as MAIN and create artifacts under MAIN (`nsys/`, `ncu/`, provenance files).
3. Launch Stage 1 with `hydra.run.dir=<MAIN>/stage1` via `build_work_argv(..., hydra_run_dir=...)` so Stage 1 writes into the same MAIN directory under a subfolder.
4. All temporary and Hydra outputs from both stages land under MAIN.

### D) CUPTI and device hygiene
1. Keep `torch_profiler.enabled=false` in workload overrides when running Nsight Systems to avoid CUPTI subscriber conflicts.
2. Provide a Pixi task pattern for multi‑GPU hosts:
   - `CUDA_VISIBLE_DEVICES=<idx> … +stage1_runner.device=cuda:0` to avoid vendor hard‑coded `.cuda()` mismatches, and select no‑static via `runner@stage1_runner=stage1.no-static` if needed.

## Step‑by‑Step Changes
1. vendor/nsys.py
   - Replace `build_nsys_export_sqlite_cmd` with `--type sqlite` form; accept input as base or full filename.
   - Add helper `resolve_nsys_report_path(out_base)` that returns existing `<base>.nsys-rep` or `<base>.qdrep`.
2. profiling/nsys_stats.py
   - Keep `top_kernels_from_nsys_summary` unchanged.
   - Add a wrapper to call `nsys stats <report>` without prior export; handle failures.
3. profiling/vendor/ncu.py
   - Remove `--section` injection from default; make it optional/conditional.
4. conf/profiling/ncu/ncu.default.yaml
   - Drop `section:`; keep `set: roofline` and `metrics: …`.
5. runners/deep_profile_runner.py
   - After `nsys profile`, call `resolve_nsys_report_path` and gate stats/sqlite export on file existence.
   - If `nsys` version is older, prefer `nsys stats` and `nsys export --type sqlite`.
   - Add a startup log of `HydraConfig.get().run.dir` and a warning if it doesn’t start with `tmp/stage2`.
   - Keep `torch_profiler.enabled=false` in overrides; retain NVTX gating and `decode*` include.

## Validation Plan
- Tool detection
  - Log `nsys --version` and, if necessary, adjust export flags accordingly.
  - Probe `ncu --list-sections` once and record whether `SpeedOfLight` is available.
- Functional smoke
  - Run `pixi run stage2-profile-gpu1 repeats=1 '+infer.max_new_tokens=4'`.
  - Verify presence of Nsight Systems report `<run>.nsys-rep` or `<run>.qdrep` (if the environment supports capture) and that `nsys stats` no longer errors.
  - Verify `ncu/raw.csv` contains CSV headers instead of an error line.
- Artifacts
  - Confirm `tmp/stage2/<run_id>/` contains `nsys/`, `ncu/`, and provenance files.

## Impact & Risks
- Low risk to Stage 1: changes are isolated to Stage 2 runner and vendor wrappers.
- Nsight Systems behavior varies by version; we mitigate by probing version and checking for report files.
- On locked‑down systems, capture may still be prevented; plan logs clear guidance and fails open (skips stats/export).

## References
- Nsight Systems export docs (2022.4): `nsys export --type sqlite`
  - https://docs.nvidia.com/nsight-systems/2022.4/nsys-exporter/overview.html
- Generating stats/SQLite via `nsys profile --stats=true` and `nsys stats`
  - https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-applications-with-nsight-systems_701.html
- Nsight Compute roofline/sections naming and `--set roofline`
  - https://indico.cern.ch/event/962112/contributions/4110591/attachments/2159863/3643851/CERN_Nsight_Compute.pdf
  - https://forums.developer.nvidia.com/t/how-to-profile-roofline/279352
- Hydra config groups and search path (Context7)
  - /facebookresearch/hydra
