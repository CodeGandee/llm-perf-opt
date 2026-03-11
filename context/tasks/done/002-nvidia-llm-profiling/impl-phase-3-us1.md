# Implementation Guide: US1 — Deep Profiling Session

Phase: 3 | Feature: Stage 2 — NVIDIA-Backed Deep LLM Profiling | Tasks: T012–T016, T030–T037

## Files

### Created
- src/llm_perf_opt/runners/deep_profile_runner.py
- tests/manual/stage2_profile/manual_stage2_profile.py
- src/llm_perf_opt/profiling/vendor/launch.py
- src/llm_perf_opt/profiling/nsys_stats.py

### Modified
- src/llm_perf_opt/profiling/vendor/nsys.py
- src/llm_perf_opt/profiling/vendor/ncu.py
- src/llm_perf_opt/profiling/nvtx_utils.py
- conf/runner/stage2.yaml
- src/llm_perf_opt/profiling/artifacts.py

## Public APIs

### T012: Hydra entrypoint

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='../../../conf', config_name='runner/stage2')
def main(cfg: DictConfig) -> None:
    # 1) Create run dir + provenance
    # 2) Build work argv with Hydra overrides
    # 3) Run nsys and ncu subprocesses
    # 4) Export operators/kernels tables, stakeholder summary
    pass
```

### T030/T031: NVTX ranges — use existing session tags

NVTX segmentation already exists in the model session (`dsocr_session.py`) via
`prefill`/`decode` ranges and sub-stage hooks (`sam`, `clip`, `projector`). Do
not add new NVTX helpers; instead, align profiler filters to these labels.

- Nsight Systems: use NVTX range gating with a concrete range name (e.g., `nvtx_capture=decode` or `prefill`).
- Nsight Compute: include `decode*` to focus on decode kernels.

### T032: Hydra‑aware argv builder

```python
from typing import Sequence

def build_work_argv(module: str, overrides: Sequence[str]) -> list[str]:
    return ["python", "-m", module, *overrides]
```

## Usage Flow

```mermaid
sequenceDiagram
    participant Runner
    participant Nsys
    participant Ncu
    participant Artifacts

    Runner->>Artifacts: create MAIN run_dir (shared)
    Runner->>Runner: set Hydra run.dir = MAIN
    Runner->>Nsys: build_nsys_cmd(MAIN/nsys/run, work_argv)
    Nsys-->>Runner: .qdrep/.nsys-rep
    Runner->>Ncu: build_ncu_cmd(MAIN/ncu/decode, work_argv, nvtx=decode*)
    Ncu-->>Runner: .ncu-rep + raw.csv
    Runner-->>Artifacts: env.json, config.yaml, inputs.yaml (under MAIN)
    Runner->>Stage1: launch with hydra.run.dir = MAIN/stage1
```

## Pseudocode

```python
art = Artifacts(MAIN)
work = build_work_argv('llm_perf_opt.runners.llm_profile_runner', hydra_overrides, hydra_run_dir=str(art/"stage1"))
subprocess.run(build_nsys_cmd(art.root/"nsys"/"run", work), check=True)
subprocess.run(build_ncu_cmd(art.root/"ncu"/"decode", work, nvtx_expr='decode*'), check=True)
```

## Testing

```bash
pixi run python tests/manual/stage2_profile/manual_stage2_profile.py
```

## Summary
- Unified config and outputs
  - Runner config path consolidated under `conf/runner/` (previous `conf/runners/` merged).
  - One MAIN run directory per Stage 2 run (Hydra `hydra.run.dir`), typically under `tmp/stage2/<timestamp>`.
  - Stage 1 artifacts live in `MAIN/stage1/`; all temporary and Hydra outputs for both stages stay under MAIN.
  - Legacy `outputs/` is deprecated; artifacts now land under `tmp/` and are ignored by VCS.
- Nsight Systems (nsys)
  - Uses NVTX range gating by default. Set a specific capture expression that matches emitted ranges, e.g. `nsys.nvtx_capture=prefill` (default) or `decode`.
  - Enables non‑registered NVTX strings via `--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0` for Python NVTX labels.
  - Resolves `.nsys-rep`/`.qdrep` automatically and exports summary/SQLite when a report exists (`nsys export --type sqlite`).
- Nsight Compute (ncu)
  - NVTX gating default is on; toggle with `ncu.gating_nvtx` (true|false). When off, no NVTX include or kernel regex is applied.
  - Drops brittle `--section` filters; prefers stable presets/metrics and `--set roofline` when available.
  - Supports `--list-sections`; writes `MAIN/ncu/sections.txt` for inspection and version‑dependent tuning.
  - When available, top‑N kernels are derived from NSYS stats to form a `kernel_regex` seed; falls back gracefully if stats are missing.
- CLI/argv and Hydra integration
  - `build_work_argv` injects Hydra overrides for Stage 1, including `hydra.run.dir=MAIN/stage1` and `hydra.job.chdir=false`.
  - Device mapping works with `CUDA_VISIBLE_DEVICES` and `+stage1_runner.device=cuda:0` so GPU index `1` can be targeted reliably.
- Stage 1 coordination
  - Static analyzer disabled during deep profiling via `runner@stage1_runner=stage1.no-static`.
  - Torch profiler disabled to avoid CUPTI conflicts: `torch_profiler.enabled=false`.
- Presets and provenance
  - Defaults live in `conf/profiling/nsys/nsys.default.yaml` and `conf/profiling/ncu/ncu.default.yaml`.
  - Provenance files (`env.json`, `config.yaml`, `inputs.yaml`) are written under MAIN for each run.

### Troubleshooting
- If no NSYS report appears under `MAIN/nsys/` with gating enabled:
  - Ensure the capture expression matches emitted NVTX range names (`prefill`, `decode`, `sam`, `clip`, `projector`). Try `nsys.nvtx_capture=decode`.
  - Confirm `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` is in effect (we inject via `--env-var`).
  - As a temporary fallback, disable gating (`+nsys.gating_nvtx=false`) to force capture, then re‑enable after confirming ranges.

### Open Issue (under review)
- On this setup, `nsys.nvtx_capture=decode` produced "No reports were generated", while `nsys.nvtx_capture=prefill` produced a valid `run.nsys-rep` and `summary_cuda_gpu_kern_sum.csv`. Default is now `prefill` to ensure reliable capture. This variance is left for further code review and validation. See: `context/logs/code-review/20251030-061211-phase3-nvtx-gating-issue.md`.


## References
- Quickstart: specs/002-nvidia-llm-profiling/quickstart.md
- Hint: context/hints/nv-profile-kb/howto-manage-nsys-ncu-processes-for-llm.md
