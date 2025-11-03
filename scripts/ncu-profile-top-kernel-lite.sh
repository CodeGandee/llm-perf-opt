#!/usr/bin/env bash
set -euo pipefail

# LITE/FAST version of Nsight Compute kernel profiling.
# Optimized for speed by:
# 1. --kernel-id :::1    → Profile only FIRST instance of each kernel (avoids excessive replays)
# 2. --section minimal   → Collect only SpeedOfLight (basic utilization, no roofline)
# 3. Small workload      → 1 image, 64 tokens
#
# Expected runtime: ~30 seconds to 2 minutes (vs. 5-30+ minutes for roofline)
# Use this for quick iteration, then use the full script for deep roofline analysis.

# Top-kernel observed in: tmp/profile-output/20251103-164000/nsys/summary_cuda_gpu_kern_sum.csv
KERNEL_REGEX='.*gemvx.*'

# Output base for the .ncu-rep report
OUT_BASE="tmp/ncu_lite_gemvx"

# Ensure tmp exists
mkdir -p tmp

echo "[ncu-lite] Target kernel regex: ${KERNEL_REGEX}"
echo "[ncu-lite] Optimization: --kernel-id :::1 (profile first instance only)"
echo "[ncu-lite] Metrics: SpeedOfLight only (minimal set for fast profiling)"
echo "[ncu-lite] Report base: ${OUT_BASE} (final: ${OUT_BASE}.ncu-rep)"
echo ""

set -x
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name "${KERNEL_REGEX}" \
  --kernel-id :::1 \
  --section SpeedOfLight \
  --replay-mode kernel \
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

echo ""
echo "[ncu-lite] Done! Report at: ${OUT_BASE}.ncu-rep"
echo "[ncu-lite] Open with: ncu-ui ${OUT_BASE}.ncu-rep"
echo ""
echo "Next steps:"
echo "  - For roofline analysis, use: scripts/ncu-profile-top-kernel.sh"
echo "  - For custom sections, add: --section SpeedOfLight --section Occupancy"
echo "  - Profile Nth instance: add --launch-skip N-1 --launch-count 1"
