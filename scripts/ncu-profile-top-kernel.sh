#!/usr/bin/env bash
set -euo pipefail

# Hardcoded Nsight Compute run to profile the top kernel seen in NSYS summary.
# Light profiling: collect only the "Speed Of Light" section for quick, high‑level
# utilization, minimizing replay overhead. The script focuses ncu on a single
# kernel (by name regex) and runs a minimal Stage‑1 workload under the rtx5090
# Pixi environment.

# Top-kernel observed in: tmp/profile-output/20251103-164000/nsys/summary_cuda_gpu_kern_sum.csv
# Example entry includes: internal::gemvx::kernel<...>
KERNEL_REGEX='.*gemvx.*'

# Output base for the .ncu-rep report
OUT_BASE="tmp/ncu_top1_gemvx"

# Ensure tmp exists
mkdir -p tmp

echo "[ncu] Report base: ${OUT_BASE} (final report: ${OUT_BASE}.ncu-rep)"

set -x
ncu \
  --target-processes all \
  --section SpeedOfLight \
  --replay-mode application \
  -o "${OUT_BASE}" \
  pixi run -e rtx5090 python -m llm_perf_opt.runners.llm_profile_runner \
    'hydra.run.dir=tmp/ncu-work/${now:%Y%m%d-%H%M%S}' \
    dataset.subset_filelist=datasets/omnidocbench/subsets/dev-3.txt \
    device=cuda:0 \
    pipeline.torch_profiler.enable=false \
    pipeline.static_analysis.enable=false \
    dataset.sampling.num_epochs=1 \
    dataset.sampling.num_samples_per_epoch=1 \
    dataset.sampling.randomize=false \
    infer.max_new_tokens=64
set +x

echo "[ncu] Done. Report at: ${OUT_BASE}.ncu-rep"
