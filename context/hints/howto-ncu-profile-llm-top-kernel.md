# How to use Nsight Compute (ncu) to profile a specific hot kernel in an LLM (e.g., DeepSeek‑OCR 3B)

This guide shows a reliable workflow to profile a single CUDA kernel from a PyTorch LLM pipeline using Nsight Compute. It assumes you’ve first used Nsight Systems (nsys) to identify top kernels and now want deep metrics for one kernel.

## 1) Identify the kernel name from Nsight Systems

Generate an NSYS report and extract the kernel summary (includes full demangled names):

```bash
# Example capture (adjust your launch)
nsys profile --trace=cuda,osrt -o tmp/nsys/run \
  pixi run -e rtx5090 python -m llm_perf_opt.runners.llm_profile_runner \
    'hydra.run.dir=tmp/profile-output/${now:%Y%m%d-%H%M%S}' \
    device=cuda:0 pipeline.torch_profiler.enable=false pipeline.static_analysis.enable=false \
    dataset.subset_filelist=datasets/omnidocbench/subsets/dev-3.txt \
    dataset.sampling.num_epochs=1 dataset.sampling.num_samples_per_epoch=1 infer.max_new_tokens=64

# Export per‑kernel sums CSV with full names
nsys stats --report gpukernsum --format csv \
  --output tmp/nsys/summary tmp/nsys/run.qdrep

# Inspect top kernels by time
column -s, -t < tmp/nsys/summary__gpukernsum.csv | head -n 20
```

Pick a short, stable substring from the top kernel name (e.g., `gemvx`, `flash_fwd_splitkv_kernel`, `sdpa`, `cutlass`).

References:
- Nsight Systems User Guide (stats): https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Forum tip for full kernel names: https://forums.developer.nvidia.com/t/getting-full-kernel-name-from-nsys/173011

## 2) Profile that kernel with Nsight Compute (no NVTX required)

Use `--kernel-name` as a regex filter, demangled names, and limit collection to a practical set.

Minimal command (paste‑ready):

```bash
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name '.*gemvx.*' \
  --set roofline \
  --replay-mode kernel \
  -o tmp/ncu/top1_gemv \
  pixi run -e rtx5090 python -m llm_perf_opt.runners.llm_profile_runner \
    'hydra.run.dir=tmp/ncu-work/${now:%Y%m%d-%H%M%S}' \
    dataset.subset_filelist=datasets/omnidocbench/subsets/dev-3.txt \
    device=cuda:0 \
    pipeline.torch_profiler.enable=false pipeline.static_analysis.enable=false \
    dataset.sampling.num_epochs=1 dataset.sampling.num_samples_per_epoch=1 dataset.sampling.randomize=false \
    infer.max_new_tokens=64
```

Target a specific occurrence when launches repeat a lot:

```bash
# Profile only the 3rd matching launch
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name '.*flash_fwd_splitkv_kernel.*' \
  --launch-skip 2 --launch-count 1 \
  --set roofline --replay-mode kernel \
  -o tmp/ncu/top1_flash \
  pixi run -e rtx5090 python -m llm_perf_opt.runners.llm_profile_runner ...
```

Collect an even smaller set (faster):

```bash
# Focused sections (examples: SpeedOfLight + Occupancy)
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name '.*sdpa.*' \
  --section '.*SpeedOfLight.*' \
  --section '.*Occupancy.*' \
  -o tmp/ncu/sdpa_sol_occ \
  pixi run -e rtx5090 python -m llm_perf_opt.runners.llm_profile_runner ...
```

References:
- Nsight Compute CLI options (kernel filters, sets, sections): https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- StackOverflow (partial name regex & launch filters):
  - https://stackoverflow.com/questions/68713720/filter-on-partial-kernel-name-with-nsight-compute
  - https://stackoverflow.com/questions/72249180/can-i-skip-ahead-to-profile-a-specific-invocation-of-a-specific-kernel

## 3) Why NCU can be slow, and how to speed it up

NCU often performs multi‑pass “replay” to collect metrics. This can trigger device memory save/restore and large overhead, especially for memory‑writing kernels typical in LLMs.

- Replay overhead basics:
  - More requested metrics → more replay passes → more save/restore → slower. Use a lighter set like `--set roofline` or a few sections/metrics. 
  - Kernel replay saves device memory; if this is too slow, consider application replay: `--replay-mode application` (relaunches the app per pass). Depending on your startup cost vs kernel size, this can be faster.
- Keep the workload tiny:
  - Use a small dataset subset and `infer.max_new_tokens` kept low (e.g., 8–64) to reach your kernel quickly with minimal work.
  - Disable unrelated profiling in the workload: `pipeline.torch_profiler.enable=false`, `pipeline.static_analysis.enable=false`.
- Warm‑up: You don’t need to add special warm‑up for NCU; it handles it internally. If your app JITs extensions at first run, do one quick run without NCU to populate caches before profiling.

References:
- Nsight Compute Profiling Guide (replay overhead, kernel vs application replay): https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- Forum notes on replay slowness and guidance: https://forums.developer.nvidia.com/t/qusetion-about-kernel-warmup-and-replay-control/327599

## 4) Troubleshooting

- “Taking minutes before model loads”: NCU may be preparing replay, listing metrics, and instrumenting processes; reduce collection scope (use `--set roofline` or a few `--section` flags) and limit the workload as shown above.
- “Backing up device memory in system memory. Kernel replay might be slow”: try `--replay-mode application`, or reduce metrics.
- No report created: ensure your kernel regex matches; try a more generic substring, or use `--launch-count 1` with `--launch-skip` to hit the correct occurrence.
- Python/child processes: always use `--target-processes all` for PyTorch.
- Open the report: `ncu-ui tmp/ncu/top1_gemv.ncu-rep` (or `nv-nsight-cu`).

---

This workflow balances accuracy and speed for large LLMs by: (1) scoping the workload, (2) filtering by kernel name, and (3) reducing replay passes via lighter metric sets.

