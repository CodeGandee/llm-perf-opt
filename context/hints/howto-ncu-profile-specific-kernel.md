# How to profile a specific CUDA kernel with Nsight Compute (ncu) from a PyTorch workload

This hint shows a fast workflow to:
- Identify hot kernels with Nsight Systems (nsys)
- Focus Nsight Compute (ncu) on just one kernel by name (no NVTX tagging)

Works with Python/PyTorch programs. Replace example script/commands with yours.

## 1) Find hot kernels with Nsight Systems

Capture a representative run:

```bash
# Record CUDA + OS runtime; produces tmp/nsys/run.qdrep
nsys profile --trace=cuda,osrt -o tmp/nsys/run \
  python your_script.py --your-args
```

Extract kernel time summary and full kernel names:

```bash
# CSV with full kernel names and aggregated times
nsys stats --report gpukernsum --format csv \
  --output tmp/nsys/kernel-sum tmp/nsys/run.qdrep

# Inspect top kernels by total time
column -s, -t < tmp/nsys/kernel-sum__gpukernsum.csv | head -n 30
```

Pick one kernel name (or a distinctive substring/regex). For very long names, keep a short stable fragment (e.g., `flash_attn`, `sdpa`, `gemm`, `layer_norm`, etc.).

References:
- Nsight Systems CLI docs (stats): https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Forum tip for full kernel names: https://forums.developer.nvidia.com/t/getting-full-kernel-name-from-nsys/173011

## 2) Profile just that kernel with Nsight Compute

Use `--kernel-name` (regex) to filter the kernel(s) that ncu profiles. No NVTX tagging required.

Minimal example:

```bash
# Target any Python child processes and profile only kernels matching the regex
ncu \
  --target-processes all \
  --kernel-name '.*flash_attn.*' \
  --set full \
  --replay-mode kernel \
  --output tmp/ncu/flash_attn \
  python your_script.py --your-args
```

Notes:
- `--kernel-name` accepts a regex; quote it for the shell.
- `--target-processes all` ensures kernels from Python workers/child procs are captured.
- `--replay-mode kernel` reduces overhead for multi-pass collection.
- Change `--set` for faster runs: e.g., `--set roofline` or select sections/metrics explicitly.

Pick occurrences when the kernel launches multiple times:

```bash
# Profile the 3rd occurrence only
ncu \
  --target-processes all \
  --kernel-name '.*sdpa.*' \
  --launch-skip 2 --launch-count 1 \
  --set roofline \
  --output tmp/ncu/sdpa_run \
  python your_script.py --your-args
```

Collect a focused subset of sections/metrics (faster):

```bash
# Speed-of-Light + Occupancy only (examples; adjust to your needs)
ncu \
  --target-processes all \
  --kernel-name '.*gemm.*' \
  --section '.*SpeedOfLight.*' \
  --section '.*Occupancy.*' \
  --output tmp/ncu/gemm_sol_occ \
  python your_script.py --your-args
```

Open reports in GUI later (optional):

```bash
# Open in Nsight Compute UI (or use nv-nsight-cu)
ncu-ui tmp/ncu/flash_attn.ncu-rep
```

References:
- Nsight Compute CLI options: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- Regex kernel selection discussion: https://forums.developer.nvidia.com/t/profile-multiple-kernels-once-each/289032
- General Nsight profiling notes: https://gpuhackshef.readthedocs.io/en/latest/tools/nvidia-profiling-tools.html

## 3) Tips & troubleshooting

- Kernel names can be long/mangled. Prefer short, stable substrings in your regex.
- If multiple different kernels match, refine the regex or use `--launch-skip/--launch-count`.
- For multi-process PyTorch workloads (e.g., DataLoader workers), keep `--target-processes all`.
- If your GPU is newer (e.g., RTX 50 / sm_120), ensure your PyTorch build supports it; otherwise CUDA kernels won’t run.
- If names change across versions, re-run `nsys stats` to refresh the kernel name you filter on.
