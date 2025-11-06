# Quickstart: NVTX Range–Scoped Nsight Compute (native args)

## Prerequisites

- NVIDIA driver + CUDA toolkit installed, `ncu` on PATH.
- Pixi environment set up in repo root: `/workspace/code/llm-perf-opt`.

## Understanding Replay Modes

NCU replay modes control how kernels are re-executed to collect all requested profiling data:

- **`kernel`** (default): Replays individual kernel launches transparently during execution. Use for per-kernel profiling results. Works with most applications without requiring determinism.

- **`range`**: Replays NVTX ranges (defined by `nvtxRangePush/Pop` or NVTX expressions) transparently during execution. Produces **aggregated results per range** instead of per-kernel. Requires NVTX annotations in your code.

- **`app-range`**: Relaunches the entire application multiple times to profile NVTX ranges. Produces aggregated results per range. **Requires deterministic execution** (same kernels/order every run). Use when transparent range replay isn't feasible.

- **`application`**: Relaunches the entire application multiple times (not range-scoped). **Requires deterministic execution**. Rarely needed; prefer `kernel` mode for most cases.

**Choosing a mode:**
- Need per-kernel details? → Use `kernel`
- Need range-level aggregates with transparent replay? → Use `range`
- Need range-level aggregates but deterministic relaunch is easier? → Use `app-range`

## Range Replay (aggregate per NVTX range)

Hydra mirrors native `ncu` flags under `pipeline.ncu.ncu_cli.*` (1:1 mapping):
- `ncu_cli.replay_mode` → `--replay-mode <kernel|range|app-range|application>`
- `ncu_cli.nvtx.include` → `--nvtx --nvtx-include <expr>` (NVTX gating is on by default)
- `ncu_cli.sections` → repeated `--section <Name>`
- `ncu_cli.metrics` → `--metrics <csv>`
- `ncu_cli.target_processes` → `--target-processes <...>`

Example: profile an NVTX range as a single aggregated result

```bash
pixi run python -m llm_perf_opt.runners.deep_profile_runner \
  pipeline.ncu.enable=true \
  pipeline.ncu.ncu_cli.replay_mode=range \
  pipeline.ncu.ncu_cli.nvtx.include='LLM@*' \
  'pipeline.ncu.ncu_cli.sections=[SpeedOfLight,MemoryWorkloadAnalysis,Occupancy,SchedulerStats]'
```

Deterministic relaunch (application replay):

```bash
pixi run python -m llm_perf_opt.runners.deep_profile_runner \
  pipeline.ncu.enable=true \
  pipeline.ncu.ncu_cli.replay_mode=app-range \
  pipeline.ncu.ncu_cli.nvtx.include='prefill*'
```

## NVTX filter without range replay (per‑kernel inside range)

If you only need to limit to kernels inside a range (keep per‑kernel results):

```bash
pixi run python -m llm_perf_opt.runners.deep_profile_runner \
  pipeline.ncu.enable=true \
  pipeline.ncu.ncu_cli.replay_mode=kernel \
  pipeline.ncu.ncu_cli.nvtx.include='decode*'
```

## Combining NVTX range and kernel name filters

To profile only specific kernels (by name pattern) within an NVTX range, combine both filters.
The filters work in AND logic: kernels must match both the NVTX range **and** the kernel name pattern.

Exact kernel name match within NVTX range:

```bash
pixi run python -m llm_perf_opt.runners.deep_profile_runner \
  pipeline.ncu.enable=true \
  pipeline.ncu.ncu_cli.replay_mode=kernel \
  pipeline.ncu.ncu_cli.nvtx.include='LLM@*' \
  pipeline.ncu.ncu_cli.kernel_name='my_exact_kernel_name' \
  pipeline.ncu.ncu_cli.kernel_name_base=demangled
```

Regex kernel name match within NVTX range (e.g., all matmul kernels in decode phase):

```bash
pixi run python -m llm_perf_opt.runners.deep_profile_runner \
  pipeline.ncu.enable=true \
  pipeline.ncu.ncu_cli.replay_mode=kernel \
  pipeline.ncu.ncu_cli.nvtx.include='decode*' \
  pipeline.ncu.ncu_cli.kernel_name='regex:.*matmul.*' \
  pipeline.ncu.ncu_cli.kernel_name_base=demangled \
  'pipeline.ncu.ncu_cli.sections=[SpeedOfLight,MemoryWorkloadAnalysis,Occupancy]'
```

**Note:** Since NCU 2021.1+, exact match is the default behavior. Use the `regex:` prefix explicitly for pattern matching. Only kernels launched inside the specified NVTX range(s) that also match the kernel name filter will be profiled.

## Outputs

Artifacts are written under `tmp/profile-output/<run_id>/ncu/`.
- Aggregated range runs produce a single range result (`.ncu-rep`) per matched range and optional section exports.
- If `ncu_cli.sections` are set, a text export is included (e.g., `sections_report.txt`).
