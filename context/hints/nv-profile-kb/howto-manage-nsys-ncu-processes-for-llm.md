# Howto — Manage nsys/ncu Processes for Python LLM Profiling

> Scope: Python-based LLM profiling that captures end-to-end timelines with Nsight Systems (nsys) and kernel-level metrics with Nsight Compute (ncu). Optimized for low overhead, reproducible runs, and actionable artifacts (top kernels, MFU, stage timings).

## TL;DR (Best Practices)
- Use nsys for full-run timelines; use ncu only on selected ranges/kernels (top-N) to keep overhead bounded.
- Gate collection with NVTX: nsys `--capture-range=nvtx`, ncu `--nvtx --nvtx-include`.
- Spawn profilers as external processes via `subprocess.run([...])`; avoid shell-string quoting and recursion.
- Capture child processes: ncu `--target-processes all`; nsys traces children by default when `--trace=osrt` is enabled.
- Limit ncu to Roofline/SOL sections first; expand metrics only when needed.
- Export to CSV/SQLite and aggregate offline; keep trace size < 2 GB per run.
- For MFU, compute from ncu FLOPs/time or cross-check via tokens/s; see `about-mfu-analysis.md`.

## When to Use Which Tool
- nsys (Nsight Systems): end-to-end CPU↔GPU timeline, NVTX range validation, kernel launch/overlap, memcpys.
- ncu (Nsight Compute): per-kernel counters (achieved FLOPs, occupancy, memory throughput) for selected ranges/kernels.

## Instrumentation (NVTX)
Annotate stages so profilers can filter exactly those regions (low overhead, accurate attribution):

```python
import nvtx, torch

with torch.inference_mode():
    with nvtx.annotate("LLM@prefill", domain="LLM"):
        out = model(input_ids=prompt_ids, use_cache=True)
        kv = out.past_key_values
    with nvtx.annotate("LLM@decode_all", domain="LLM"):
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_token, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
```

Sources: NVTX SDK; Nsight Systems User Guide.

## Launch Pattern (Python → external nsys/ncu)
Prefer process-level wrapping with explicit argv lists (no shell) and absolute paths. Combine with Hydra by passing overrides directly in argv so the runner configures datasets, modes, and output dirs consistently.

```python
import os, subprocess, sys
from pathlib import Path

root = Path.cwd()
run_id = "2025-10-29T15-20-01Z"
art = root / "tmp" / "stage2" / run_id
art.mkdir(parents=True, exist_ok=True)

# Hydra overrides: prefer absolute paths and an explicit hydra.run.dir matching art
hydra_overrides = [
    f"hydra.run.dir={art}",
    "hydra.job.chdir=true",
    "device=cuda:0",
    "repeats=1",
    "profiling.activities=[cpu,cuda]",
    "+run.mode=deep",
    "+inputs.manifest=/abs/path/to/inputs.yaml",
]

work = [sys.executable, "-m", "llm_perf_opt.runners.llm_profile_runner", *hydra_overrides]

# 1) Nsight Systems (timeline)
nsys = [
    "nsys", "profile",
    "--trace=cuda,nvtx,osrt", "--sample=none",
    "--capture-range=nvtx", "--nvtx-capture=range@LLM",
    "-o", str(art / "nsys" / "run"),
]
subprocess.run(nsys + work, check=True, env=os.environ.copy())

# 2) Nsight Compute (top-N kernel metrics) — decode as example
ncu = [
    "ncu", "--target-processes", "all",
    "--nvtx", "--nvtx-include", "LLM@decode_all/",
    "--set", "roofline", "--section", ".*SpeedOfLight.*",
    "--metrics", "flop_count_hp,flop_count_sp,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "-o", str(art / "ncu" / "decode"),
]
subprocess.run(ncu + work, check=True, env=os.environ.copy())

# Export CSV for parsers
subprocess.run(["ncu", "--import", str(art/"ncu"/"decode.ncu-rep"),
                "--page", "raw", "--csv", "--log-file", str(art/"ncu"/"decode_raw.csv")], check=True)
```

Notes:
- Pass argv as a list. Avoid shell quoting and environment side-effects.
- Keep separate output roots: `tmp/stage2/<run_id>/{nsys,ncu}/`.
- For long runs, profile subsets using NVTX capture only around stages of interest.

Sources: Nsight Systems User Guide; Nsight Compute Profiling Guide; RCAC guide; PyTorch discuss.

### Hydra config example (Stage 2)
Use a dedicated Hydra config to drive Stage 2 and keep all knobs versioned:

```yaml
# conf/profiling/stage2.yaml
defaults:
  - /profiling/torch@profiling: torch-profiler.min
  - _self_

run:
  mode: deep          # deep|light
  top_n_kernels: 30   # used by ncu selection

artifacts:
  stage2_dir: ${hydra:runtime.cwd}/tmp/stage2/${now:%Y%m%d-%H%M%S}

hydra:
  run:
    dir: ${artifacts.stage2_dir}
  output_subdir: null
  job:
    chdir: true
```

Then launch via:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --capture-range=nvtx --nvtx-capture=range@LLM \
  -o tmp/stage2/nsys/run \
  python -m llm_perf_opt.runners.llm_profile_runner \
  profiling=@profiling/torch/torch-profiler.min \
  +run.mode=deep +inputs.manifest=/abs/path/inputs.yaml \
  hydra.run.dir=$(pwd)/tmp/stage2/$(date +%Y%m%d-%H%M%S) hydra.job.chdir=true
```

Notes:
- Override `hydra.run.dir` per run to keep artifacts collocated with nsys/ncu outputs.
- If your runner itself spawns child processes, prefer `ncu --target-processes all` and set `CUDA_VISIBLE_DEVICES` explicitly.

### Hydra multirun (sweeps) with profilers
- Avoid a single `nsys` around a Hydra `-m` sweep; instead, iterate runs and invoke `nsys`/`ncu` separately per config to keep one report per run.
- Programmatic pattern:

```python
for mode in ["deep", "light"]:
    run_dir = art.parent / f"{run_id}-{mode}"
    ovs = [f"hydra.run.dir={run_dir}", f"run.mode={mode}", "hydra.job.chdir=true"]
    subprocess.run(nsys + work + ovs, check=True)
    subprocess.run(ncu + work + ovs, check=True)
```

If you still want Hydra `-m`, pass it between the module and overrides:

```python
work_mr = [sys.executable, "-m", "llm_perf_opt.runners.llm_profile_runner", "-m", *hydra_overrides]
subprocess.run(nsys + work_mr, check=True)
```

## Process Tree & Overhead Controls
- Child processes: `ncu --target-processes all`. nsys captures children when `--trace=osrt` is set; they appear as separate processes in the timeline.
- Start/Stop control:
  - Prefer NVTX gating (`--capture-range=nvtx`, `--nvtx-include`).
  - Fallback: `--capture-range=cudaProfilerApi` + `cudaProfilerStart/Stop` (legacy; works if NVTX not available).
- Reduce overhead:
  - nsys: `--sample=none`; limit traces (`cuda,nvtx,osrt`), gate with NVTX ranges.
  - ncu: start with `--set roofline` or selected `--section` sets; avoid `--set full` for full app runs.
  - Profile top-N kernels only: identify from nsys timeline or a first ncu scan, then filter via `--kernel-id`, `--kernel-name`, `--launch-skip/--launch-count`.

Sources: Nsight Systems User Guide; Nsight Compute Profiling Guide; NERSC/NASA docs; community gists.

## Artifact Workflow
1) Run nsys full timeline → `.qdrep`/`.nsys-rep` under `tmp/stage2/<run_id>/nsys/`.
2) Select stages/kernels:
   - From NVTX-labeled spans; from nsys “CUDA GPU Kernels” table (by total time).
3) Run ncu on selected stages/kernels only → `.ncu-rep` under `tmp/stage2/<run_id>/ncu/`.
4) Export to CSV → parse into `KernelRecord[]`, aggregate top kernels, compute mean ms.
5) Compute MFU per stage (see below) and include in stakeholder report.

CLI helpers:

```bash
# nsys → CSV / SQLite
nsys stats --report summary --format csv -o out tmp/stage2/<run>/nsys/run.nsys-rep
nsys export --sqlite out.sqlite tmp/stage2/<run>/nsys/run.qdrep

# ncu → CSV
ncu --import tmp/stage2/<run>/ncu/decode.ncu-rep --page raw --csv --log-file decode_raw.csv
```

Sources: Nsight Systems docs (`nsys stats`, `nsys export`); Nsight Compute CLI.

## Reproducibility Controls
- Device selection: `CUDA_VISIBLE_DEVICES=0` (or Hydra config), log device name.
- Clock & cache control (ncu): `--clock-control=base`, `--cache-control=all` for apples-to-apples micro-metrics.
- Tensor Core boost state: set to stable in ncu if available for consistency.
- Record env/config/inputs manifests alongside profiler outputs for each run_id.

Sources: Nsight Compute CLI; NASA/Modular guides.

## MFU Integration (per stage)
See `context/hints/nv-profile-kb/about-mfu-analysis.md` for equations and concrete ncu commands. In short:

```
Achieved_GFLOPs/s_stage = FLOPs_stage / Time_stage
MFU_stage = Achieved_GFLOPs/s_stage / Peak_stage
```

Use `flop_count_*` + `gpu__time_duration.sum` from ncu raw CSV; Peak from Speed-of-Light/Roofline sections.

## Common Pitfalls
- Profiling entire app with ncu `--set full`: excessive overhead. Filter by NVTX and sample a few launches.
- Missing NVTX ranges: no gating; add `domain="LLM"` and consistent labels (`LLM@prefill`, `LLM@decode_all`).
- Shell-quoted subprocess invocations: brittle; pass argv lists.
- Trace bloat: cap artifacts, compress older runs, keep last N.

## References
- Nsight Systems User Guide — https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Nsight Compute Profiling Guide — https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- Nsight Compute CLI (User Manual) — https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- RCAC: Nsight Systems quick guide — https://www.rcac.purdue.edu/knowledge/profilers/nvidia_nsight_systems
- PyTorch discuss: Using Nsight Systems — https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59
- “Favorite nsys commands” (community gist) — https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223
- NASA HECC Nsight Compute CLI — https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-cuda-kernels-with-nsight-compute-cli_706.html
- Modular: NVIDIA profiling basics — https://puzzles.modular.com/puzzle_30/nvidia_profiling_basics.html
