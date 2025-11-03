# Refactor Plan: Unify Runner Configuration (remove stage1/stage2 split)

## What to Refactor
- Consolidate runner configuration currently split across:
  - `conf/config.yaml` (Stage 1 defaults and composition)
  - `conf/runner/stage1.default.yaml`, `conf/runner/stage1.no-static.yaml`
  - `conf/runner/stage2.yaml` (deep profiler orchestrator)
- Unify the execution entrypoints:
  - `src/llm_perf_opt/runners/llm_profile_runner.py` (Stage 1)
  - `src/llm_perf_opt/runners/deep_profile_runner.py` (Stage 2)
- Normalize artifacts and pipeline toggles across static analysis, PyTorch profiler, Nsight Systems (nsys), and Nsight Compute (ncu).

## Why Refactor
- Clarity: “stage1/stage2” are internal development phases, not user-facing run modes. The split confuses users and spreads config across multiple files.
- Single source of truth: One runner config should declaratively enable/disable static analysis, PyTorch profiler, nsys, and ncu. No cross-runner coupling.
- Extensibility: Future profilers or checks (e.g., NVML sampling, memory tracers) should be plug-in toggles under one config namespace.
- Consistency: Artifacts layout and gating semantics (e.g., NVTX gating for nsys/ncu) should be consistent and configured in one place.

## How to Refactor

Phase 1 — Config design and new orchestrator (backward compatible)
1) Make `conf/config.yaml` the single Hydra entrypoint for ALL runs (orchestrator and workload) with a single pipeline:
   - General run knobs: `run.device`, `run.repeats`, `run.mode`, `run.dataset_subset_filelist`, `run.top_n_kernels`.
   - Group stage settings under `pipeline.<stage>` with an explicit `enable` flag per stage:
     - `pipeline.static_analysis.enable`, `pipeline.torch_profiler.enable`, `pipeline.nsys.enable`, `pipeline.ncu.enable`.
     - Each stage’s detailed knobs live under its own `pipeline.<stage>` object.
   - Mount existing profiling presets directly under the nested keys:
     - `/profiling/torch@pipeline.torch_profiler: torch-profiler.default`
     - `/profiling/nsys@pipeline.nsys: nsys.default`
     - `/profiling/ncu@pipeline.ncu: ncu.default`
   - Artifacts root under a unified tree, e.g., `tmp/runs/<ts>/`.
   - Keep compatibility aliases at root level for Stage‑1 code paths (see schema):
     - `analysis.static` mirrors `pipeline.static_analysis`
     - `torch_profiler` mirrors `pipeline.torch_profiler`

2) Create a new orchestrator entrypoint `src/llm_perf_opt/runners/profile_orchestrator.py`:
   - Hydra decorator: `config_path='../../../conf', config_name='config'` (uses the unified `conf/config.yaml`).
   - Executes pipeline steps based on toggles in order:
     a. Static analysis (if enabled)
     b. PyTorch profiler representative run (if enabled)
     c. Nsight Systems capture (if enabled)
     d. Nsight Compute capture (if enabled)
   - Uses current helpers where possible:
     - Reuse `build_work_argv(...)` to call the Stage 1 workload with overrides.
     - Reuse vendor builders for nsys/ncu (already encapsulated in `vendor/nsys.py` and `vendor/ncu.py`).
   - Writes provenance and reports under unified artifacts root.

3) Refactor Stage‑1 runner to read the new schema (no aliases):
   - Update `llm_profile_runner.py` to consume `pipeline.static_analysis.*` and `pipeline.torch_profiler.*` directly.
   - Remove legacy fallbacks to `analysis.static` and `runners.analysis.static`.

4) Preserve existing entrypoints as thin wrappers with warnings:
   - `llm_profile_runner.py` and `deep_profile_runner.py` become wrappers that load `runner/default.yaml` with appropriate toggles and call the orchestrator, printing a deprecation notice.
   - Keep existing Pixi tasks working by switching them to the orchestrator with equivalent flags.

Phase 2 — Config consolidation and cleanup
5) Consolidate into `conf/config.yaml` as the only entrypoint:
   - Remove reliance on `conf/runner/stage2.yaml`. Compose profiler presets under `pipeline.*` in `conf/config.yaml`.
   - Deprecate `conf/runner/` configs; keep them temporarily as thin includes pointing to `conf/config.yaml` patterns (optional).

5) Normalize artifacts API
   - Generalize `src/llm_perf_opt/profiling/artifacts.py` from Stage-2-specific to a neutral manager that initializes `nsys/`, `ncu/`, and `workload/` subdirs under `tmp/runs/<ts>/`.
   - Provide helpers `Artifacts.for_run(root)` and deprecate `create_stage2_root()`.

Phase 3 — Internal code alignment (no user-facing changes)
6) Factor “workload” settings into a small struct or helper consumed by both profiler paths:
   - E.g., `WorkloadConfig(device, repeats, dataset_subset_filelist, infer.max_new_tokens, ...)`.
   - The orchestrator passes these to either the in-process Stage 1 function or to `build_work_argv` for subprocess execution.

7) Align NVTX gating semantics
   - Keep the fixed behavior already implemented: if `nsys.capture_range=nvtx` and `nsys.nvtx_capture` omitted/empty → hard error.
   - Respect `nsys.capture_range_end` when set; omit flag when null/empty.
   - Mirror the same gating toggle for NCU (`ncu.gating_nvtx`, `ncu.nvtx_include`).

8) Tests and docs
   - Unit tests for orchestrator: toggles wiring, artifacts layout, and command-line emission for nsys/ncu.
   - Update quickstart/specs to show unified config and examples.

## Proposed Unified Config Schema

New: `conf/config.yaml` (single entrypoint)
```yaml
# Unified runner defaults

defaults:
  - dataset: omnidocbench
  - model/deepseek_ocr/arch@model: deepseek_ocr.default
  - model/deepseek_ocr/infer@infer: deepseek_ocr.default
  - profiling/torch@pipeline.torch_profiler: torch-profiler.default
  - profiling/nsys@pipeline.nsys: nsys.default
  - profiling/ncu@pipeline.ncu: ncu.default
  - _self_

run:
  mode: deep                 # deep | light
  device: cuda:0
  repeats: 3
  dataset_subset_filelist: null
  top_n_kernels: 30          # used by NCU focus selection

artifacts:
  runs_dir: ${hydra:runtime.cwd}/tmp/runs/${now:%Y%m%d-%H%M%S}

pipeline:
  static_analysis:
    enable: true
    # Detailed static-analysis settings live here
    write_reports: true
    use_analytic_fallback: true
    use_synthetic_inputs: true

  torch_profiler:
    enable: true
    # Inherits detailed knobs from profiling/torch preset mounted above
    # You may still override here, e.g. activities: [cpu, cuda]

  nsys:
    enable: false
    gating_nvtx: true
    # Inherits detailed knobs from profiling/nsys preset (mounted under this key)
    # capture_range mirrors Nsight Systems CLI:
    #   nvtx | cudaProfilerApi | hotkey | none
    # If capture_range=nvtx, nvtx_capture MUST be provided and non-empty; omitted/empty is an error.
    # capture_range_end: omit/empty/null → do not pass flag
    # Other inherited keys: trace, sample, nvtx_capture, capture_range_end

  ncu:
    enable: false
    gating_nvtx: true
    # Inherits detailed knobs from profiling/ncu preset
    # Typical keys: nvtx_include, set, metrics, sections

hydra:
  run:
    dir: ${artifacts.runs_dir}
  output_subdir: null
  job:
    chdir: true
```

Mapping from old configs:
- `conf/runner/stage1.default.yaml` → `pipeline.static_analysis.*` (Stage‑1 runner updated to read these keys directly).
- `conf/runner/stage1.no-static.yaml` → override `pipeline.static_analysis.enable=false`.
- `conf/runner/stage2.yaml` → `run.*`, and `pipeline.nsys.*` / `pipeline.ncu.*` with `enable=true` as appropriate.

## Before/After Code Snippets

Before: deep profiler ties to Stage 2 config and injects Stage 1 overrides
```python
# src/llm_perf_opt/runners/deep_profile_runner.py
@hydra.main(version_base=None, config_path="../../../conf", config_name="runner/stage2")
def main(cfg: DictConfig) -> None:
    # Build workload argv for Stage 1 runner
    overrides = [
        "runner@stage1_runner=stage1.no-static",  # disable static under NSYS
        f"device={device_sel}",
        f"repeats={stage1_repeats}",
        "torch_profiler.enabled=false",
    ]
    work = build_work_argv("llm_perf_opt.runners.llm_profile_runner", overrides, ...)
    # NSYS, NCU orchestration ...
```

After: unified orchestrator reads one config and runs steps by toggles
```python
# src/llm_perf_opt/runners/profile_orchestrator.py
@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    artifacts = Artifacts.from_root(HydraConfig.get().run.dir)

    # 1) Static analysis
    if bool(cfg.pipeline.static_analysis.enable):
        run_static_analysis(cfg, artifacts)

    # 2) PyTorch profiler
    if bool(cfg.pipeline.torch_profiler.enable):
        run_stage1_workload(cfg, artifacts, torch_profiler_enabled=True)

    # Prepare workload argv once for vendor tools
    work = build_work_argv(
        "llm_perf_opt.runners.llm_profile_runner",
        overrides=[
          f"device={cfg.run.device}",
          f"repeats={cfg.run.repeats}",
          "torch_profiler.enabled=false",
          "analysis.static.enabled=false",
        ],
        hydra_run_dir=str(artifacts.path("workload")),
        chdir=True,
        run_mode=str(cfg.run.mode),
    )

    # 3) Nsight Systems
    if bool(cfg.pipeline.nsys.enable):
        nsys_cmd = build_nsys_cmd(
            artifacts.path("nsys/run"), work,
            nvtx_capture=cfg.pipeline.nsys.nvtx_capture if (cfg.pipeline.nsys.capture_range == "nvtx" and cfg.pipeline.nsys.gating_nvtx) else "none",
            trace=cfg.pipeline.nsys.trace, sample=cfg.pipeline.nsys.sample,
            capture=cfg.pipeline.nsys.capture_range,
            capture_end=(cfg.pipeline.nsys.capture_range_end or None),
        )
        subprocess.run(nsys_cmd, check=False)

    # 4) Nsight Compute
    if bool(cfg.pipeline.ncu.enable):
        ncu_cmd = build_ncu_cmd(
            artifacts.path("ncu/decode"), work,
            nvtx_expr=cfg.pipeline.ncu.nvtx_include, use_nvtx=cfg.pipeline.ncu.gating_nvtx,
            set_name=cfg.pipeline.ncu.set, metrics=(cfg.pipeline.ncu.metrics or None), sections=(cfg.pipeline.ncu.sections or None),
        )
        subprocess.run(ncu_cmd, check=False)
```

After (Stage‑1 code): switch to pipeline.* keys
```python
# src/llm_perf_opt/runners/llm_profile_runner.py (extract)

# Read PyTorch profiler config from unified schema
tp_cfg = getattr(getattr(cfg, 'pipeline', {}), 'torch_profiler', {})
prof_enabled = bool(getattr(tp_cfg, 'enable', True))
activities = [str(x).lower() for x in list(getattr(tp_cfg, 'activities', ['cpu', 'cuda']))]

# Read static analysis config from unified schema
sa_cfg = getattr(getattr(cfg, 'pipeline', {}), 'static_analysis', {})
if bool(getattr(sa_cfg, 'enable', True)):
    use_analytic = bool(getattr(sa_cfg, 'use_analytic_fallback', True))
    use_synth = bool(getattr(sa_cfg, 'use_synthetic_inputs', True))
    # ... run static analyzer with these flags ...
```

## Impact Analysis
- Backward compatibility:
  - Existing `stage1-run` and `stage2-profile` Pixi tasks will be redirected to the orchestrator with equivalent toggles; initial release keeps old entrypoints printing a deprecation warning.
  - `conf/config.yaml` remains for Stage‑1‑only workflows; documentation guides users to the unified `conf/runner/default.yaml`.
- Risks:
  - Hydra composition changes can break overrides; mitigate with explicit key paths and tests.
  - Artifacts path changes may affect downstream scripts; provide symlink or compatibility note for `tmp/stage1`/`tmp/stage2` to `tmp/runs/<ts>/workload`.
  - NVTX gating strictness (already fixed) now applies in unified pipeline; clearly documented in config comments.
- Testing:
  - Unit tests for orchestrator toggle matrix and vendor command emission (incl. `--capture-range-end`).
  - Smoke tests for each pipeline toggle independently and in combination.

## Expected Outcome
- One cohesive configuration controls static analysis, PyTorch profiling, Nsight Systems, and Nsight Compute.
- Cleaner mental model: “enable what you need” instead of picking an arbitrary stage.
- Easier extension for new profilers or analyzers via additional toggles and presets.
- Consistent artifacts layout and provenance across all modes.

Note on multiple NVTX stages
- Nsight Systems `--nvtx-capture` accepts a single range expression (optionally with domain). Use `--capture-range-end=repeat[:N]` to capture multiple occurrences of that same range.
- Capturing multiple different NVTX range names in one run is not supported by a single expression; run separate profiles or converge on a shared label/pattern where feasible.

## References
- Code
  - src/llm_perf_opt/runners/llm_profile_runner.py
  - src/llm_perf_opt/runners/deep_profile_runner.py
  - src/llm_perf_opt/profiling/artifacts.py
  - conf/config.yaml
  - conf/runner/stage1.default.yaml, conf/runner/stage1.no-static.yaml, conf/runner/stage2.yaml
- Profiler presets
  - conf/profiling/nsys/nsys.default.yaml (capture_range, nvtx_capture, capture_range_end behavior)
  - conf/profiling/ncu/ncu.default.yaml
  - conf/profiling/torch/torch-profiler.*.yaml
- Third‑party libraries
  - Hydra: /facebookresearch/hydra
  - PyTorch: /pytorch/pytorch
- External docs
  - Nsight Systems User Guide — capture range and NVTX gating
    https://docs.nvidia.com/nsight-systems/UserGuide/index.html
  - NVIDIA Developer Forums — NVTX capture behavior
    https://forums.developer.nvidia.com/t/using-capture-range-nvtx/254091
