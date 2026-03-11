# Wan2.1 Profiling Requirements (Draft)

This document captures requirements for profiling the Wan2.1-T2V-14B model in this repo, and lists the existing entrypoints/resources that can be reused.

## Profiling Modalities (What We Mean By "Profiling")
- Analytic/static modeling: closed-form FLOP/IO/weights/activation estimates from the ModelMeter analytic model (fast, deterministic, no GPU execution required).
- Runtime profiling: measure real GPU execution (latency, kernel timeline, utilization, memory) using Torch profiler / Nsight Systems / Nsight Compute (requires a runnable inference workload).

## Prerequisites (Local Assets and Tooling)
- Model files are expected to be available via `models/wan2.1-t2v-14b/source-data` (a symlink to the local Wan2.1-T2V-14B directory).
- GPU pinning can be enforced via `.env` (e.g., `CUDA_VISIBLE_DEVICES=6,7` so the process sees only two GPUs as logical `cuda:0` and `cuda:1`).
- Pixi environment: use the default environment via `pixi run ...` unless a specific environment is required by the hardware/tooling context or by user instruction.
- Nsight tools: `nsys` and `ncu` must be available on `PATH` (or invoked via absolute paths) for runtime capture workflows.

## Repo Entrypoints and Resources

### Model Assets (Local Mounts / Symlinks)
- `models/wan2.1-t2v-14b/README.md`: describes the external reference layout and env vars (`LLM_MODELS_ROOT`, `WAN21_T2V_14B_PATH`).
- `models/wan2.1-t2v-14b/bootstrap.sh`: creates/repairs the `models/wan2.1-t2v-14b/source-data` symlink.

## Requirements (Fill In)

### Experiment Working Directory (Must Follow Pattern)
Use an experiment workspace under `tmp/` following the `explore-dnn-model` skill layout (ignore the tutorial subtree).

Default experiment directory:
- `tmp/wan21-profiling-<time>-fp16/`

Standard directory layout:
```
tmp/wan21-profiling-<time>-fp16/
  scripts/
  inputs/
  outputs/
  reports/
```

### Target Workload (Current Decision)
- [ ] Resolution: 1280x720 (720P; width x height) for T2V-14B (max supported generation resolution per `models/wan2.1-t2v-14b/source-data/README.md`).
- [ ] Frames: 81.
- [ ] Text input length: prompt must be >= 200 words.
- [ ] Diffusion steps (profiling run): 10 (diffusion-only; no VAE decode).
- [ ] Extrapolation target: estimate 50-step diffusion time from the 10-step per-step timing data.
- [ ] Warmup: before profiling, run a warmup diffusion-only run (steps = 1; not included in metrics aggregation).
- [ ] Timing repetitions: collect timings for 3 diffusion-only runs (exclude warmup from timing stats).

### Precision (Required)
- [ ] Use FP16 only.

### Metrics to Record (Profiling Output)
- [ ] Precision configuration: effective autocast dtype(s) and any relevant upstream mixed-precision overrides (if any).
- [ ] Stage timings (seconds): measure text encoder, diffusion, and VAE decode as separate stages (even if diffusion-only is the primary focus).
- [ ] Diffusion step timing (10 steps): time each diffusion step (ms) and record the 10-step total.
- [ ] Extrapolated diffusion time (50 steps): estimate 50-step diffusion time derived from the 10-step data (document the exact formula).
- [ ] Timing must exclude all disk IO time (ensure timers stop before any file write and do not include any output serialization/saving steps).
- [ ] Do not include text encoder / VAE decode time inside diffusion step timings (time them as separate stages).
- [ ] Peak VRAM used during generation (report peak GPU memory usage for the process/run).
- [ ] FLOP-derived throughput (TFLOP/s): collect FLOP estimates for text encoder, 1 diffusion step (cond+uncond forward only), and VAE decode; report TFLOP/s per stage and end-to-end (do not display raw FLOP counts in the report).

### Pre-Run Validation (Idle GPU Requirement)
- [ ] Before starting warmup/profiling, verify the target GPUs are idle: utilization <= 1% and VRAM usage <= 1%.

### Placeholder
- [ ] Add additional requirements here (capture tools, NVTX ranges, measurement method, acceptance criteria, outputs, etc.).
