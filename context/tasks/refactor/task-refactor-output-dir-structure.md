# Refactor Plan: Output Directory Structure Unification

## What to Refactor
- The current deep profiling run produces nested, confusing paths such as:
  - `<run>/tmp/stage2/<run>/nsys/...` and `<run>/tmp/stage2/<run>/stage1/...`
- Paths passed to vendor tools (nsys/ncu) and the workload are sometimes relative to the Hydra run dir, causing duplicated prefixes when `hydra.job.chdir=true`.
- Ephemeral vs. final artifacts are intermingled; users expect a clean split:
  - Final artifacts at `<main-output-dir>/<pipeline-stage-dir>/...` and top-level consolidated reports.
  - Temporary files at `<main-output-dir>/tmp/<pipeline-stage-dir>/...`.

## Why Refactor
- Predictability: Users should immediately find profiler outputs under `<run>/nsys/` and `<run>/ncu/`, and workload artifacts under `<run>/stage1/`.
- Robustness: Avoid path duplication when Hydra changes the working directory.
- Cleanliness: Separate ephemeral scratch files under `<run>/tmp/<stage>/` while keeping published reports in stable locations.

## How to Refactor

1) Define directory contract (single source of truth)
- Main run dir: `${hydra.run.dir}` (absolute). Contains:
  - `static_analysis/` — static analyzer outputs and logs
  - `torch_profiler/` — representative PyTorch profiler outputs (operators.md, trace files if any)
  - `nsys/` — Nsight Systems report(s), summaries, SQLite, commands
  - `ncu/` — Nsight Compute CSV, .ncu-rep, sections reports, commands
  - `tmp/` — transient scratch for each pipeline stage and internal workload
    - `tmp/static_analysis/`
    - `tmp/torch_profiler/`
    - `tmp/nsys/`
    - `tmp/ncu/`
    - `tmp/workload/` — scratch for the internal workload used during NSYS/NCU capture (not user-facing)
  - Top-level consolidated artifacts produced by orchestrator
    - `kernels.md`, `stakeholder_summary.md`, `report.md`, `env.json`, `config.yaml`, `inputs.yaml`

2) Make Hydra run dir absolute
- In `conf/config.yaml`, ensure `artifacts.runs_dir` is absolute via `${hydra:runtime.cwd}` and use it for `hydra.run.dir`.
- Update CLI tasks to prefer absolute `hydra.run.dir=${hydra:runtime.cwd}/...` if overridden.

3) Hard‑resolve artifact root to absolute path
- Update `Artifacts.set_root()` to call `.resolve()` on the provided path:
  - Prevents vendor tools from duplicating relative prefixes when `-o` receives a non‑absolute path and the process CWD is already the run dir.

4) Provide intentful helpers in Artifacts
- Add methods:
  - `out_dir(stage: Literal['static_analysis','torch_profiler','nsys','ncu']) -> Path`
  - `tmp_dir(stage: Literal['static_analysis','torch_profiler','nsys','ncu','workload']) -> Path`
  - Keep `path(name)` for simple cases.
- Ensure these create directories on demand and always return absolute paths.

5) Refactor deep_profile_runner to the new helpers
- Internal workload during NSYS/NCU runs (not a user-facing pipeline stage):
  - Set `hydra.run.dir` for the workload to `Artifacts.tmp_dir('workload')` so any incidental outputs don’t pollute user-facing directories.
  - Route any explicit scratch files (e.g., warmup images) to `Artifacts.tmp_dir('workload')` or stage-specific tmp dirs.
- When running the pipeline.torch_profiler stage by itself (outside NSYS/NCU), set its run dir to `Artifacts.out_dir('torch_profiler')`.
- Nsight Systems:
  - Use `Artifacts.out_dir('nsys') / 'run'` as the output base for `--output`.
  - Keep CLI command logs under `<run>/nsys/cmd.txt`.
  - Optionally use `Artifacts.tmp_dir('nsys')` for any staging (rare; nsys uses system tmp).
- Nsight Compute:
  - Use `Artifacts.out_dir('ncu')` for CSV and `.ncu-rep` outputs; keep `cmd.txt`, `cmd-rerun.txt` there.
  - Use `Artifacts.tmp_dir('ncu')` for any scratch intermediates if needed.

6) Consolidate top‑level reports
- Continue writing `kernels.md`, `stakeholder_summary.md`, and overall `report.md` at the run root.
- Optional: copy/link the NSYS summary CSV filename(s) to predictable names (e.g., `nsys/summary.csv`).

7) Backwards compatibility and migration
- Existing runs remain untouched; new runs follow the contract.
- Optionally add a `scripts/migrate-run-layout.py` to flatten the most common nested pattern by moving `<run>/tmp/<run>/*` back into `<run>/*` when detected.

8) Validation
- Add a small test (or manual check) to ensure that for a run with `pipeline.nsys.enable=true` and `pipeline.ncu.enable=false`:
  - `<run>/nsys/run.nsys-rep` exists; no nested `<run>/tmp/stage2/<run>/nsys/...`.
  - `<run>/ncu` doesn’t exist (or is empty) when disabled.
  - `<run>/stage1` contains stage‑1 artifacts when enabled.

## Expected Outcome
- Clear, stable layout:
  - Final artifacts: `<run>/(stage1|nsys|ncu)/...`
  - Ephemeral: `<run>/tmp/(stage1|nsys|ncu)/...`
  - Aggregated: `<run>/*.md`, `<run>/env.json`, `<run>/config.yaml`, `<run>/inputs.yaml`
- No more duplicated `tmp/stage2/<run>` segments in paths emitted by vendor tools.

## Example Code Snippets

Before: relative out paths cause duplicated prefixes
```python
# deep_profile_runner.py (current)
main_dir = Path(HydraConfig.get().run.dir)           # could be relative
artifacts = Artifacts.from_root(main_dir)            # stores relative root
nsys_out = artifacts.path("nsys/run")                # 'tmp/stage2/<ts>/nsys/run'
nsys_cmd = build_nsys_cmd(nsys_out, work, ...)       # process CWD is 'tmp/stage2/<ts>'
# nsys writes to 'tmp/stage2/<ts>/tmp/stage2/<ts>/nsys/run.nsys-rep'
```

After: absolute root + pipeline/output helpers
```python
# artifacts.py
def set_root(self, root: Path | str) -> None:
    rp = Path(root).resolve()  # ensure absolute
    (rp / "nsys").mkdir(parents=True, exist_ok=True)
    (rp / "ncu").mkdir(parents=True, exist_ok=True)
    (rp / "stage1").mkdir(parents=True, exist_ok=True)
    (rp / "tmp").mkdir(parents=True, exist_ok=True)
    self.m_root = rp

def out_dir(self, stage: str) -> Path: return self.root / stage

def tmp_dir(self, stage: str) -> Path:
    p = self.root / "tmp" / stage
    p.mkdir(parents=True, exist_ok=True)
    return p

# deep_profile_runner.py
artifacts = Artifacts.from_root(HydraConfig.get().run.dir)
# For NSYS: run the internal workload in a temp area, not a user-facing pipeline dir
work = build_work_argv(
    "llm_perf_opt.runners.llm_profile_runner",
    overrides=["pipeline.torch_profiler.enable=false", "pipeline.static_analysis.enable=false"],
    hydra_run_dir=str(artifacts.tmp_dir("workload")),
    chdir=True,
)
nsys_base = artifacts.out_dir("nsys") / "run"
nsys_cmd = build_nsys_cmd(nsys_base, work, ...)
```

## Impact Analysis
- Behavior changes
  - Paths become absolute; scripts manipulating relative paths might need updates.
  - Warmup scratch files should live under `<run>/tmp/torch_profiler/_warmup.png` (or `<run>/tmp/workload/` when profiling under NSYS/NCU).
- Risks
  - Any external tooling relying on the previous nested layout will need minor adjustments.
  - CI or docs that referenced the old layout must be updated.
- Mitigations
  - Provide a simple migration script or docs note to flatten existing runs.
  - Keep legacy path lookups for one release cycle where inexpensive (e.g., search both new and old NSYS summary locations).

## References
- Code
  - src/llm_perf_opt/profiling/artifacts.py
  - src/llm_perf_opt/runners/deep_profile_runner.py
  - src/llm_perf_opt/profiling/vendor/nsys.py, vendor/ncu.py
- Config
  - conf/config.yaml (hydra.run.dir, artifacts.runs_dir, pipeline.*)
- Third‑party libraries
  - Hydra: /facebookresearch/hydra
