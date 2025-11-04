#!/usr/bin/env bash
set -euo pipefail

# Test script for ncu-profile-kernels.sh
#
# This script runs NCU kernel profiling on the top 3 kernels from the
# top-10-kernels.yaml config using the RTX 5090 environment.
#
# Usage:
#   ./test-ncu-profile.sh
#
# Or with custom topk:
#   TOPK=5 ./test-ncu-profile.sh

# --- Configuration ---
TOPK=${TOPK:-3}
KERNEL_CONFIG="scripts/ncu/examples/top-10-kernels.yaml"
LAUNCH_SKIP=${LAUNCH_SKIP:-50}
LAUNCH_COUNT=${LAUNCH_COUNT:-1}
PIXI_ENV=${PIXI_ENV:-rtx5090}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
NUM_SAMPLES=${NUM_SAMPLES:-1}

# Generate timestamp for output directory
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="tmp/ncu-profile-py/${TIMESTAMP}"

echo "=========================================="
echo "NCU Kernel Profiling Test"
echo "=========================================="
echo "Config: $KERNEL_CONFIG"
echo "Top K: $TOPK"
echo "Output: $OUTPUT_DIR"
echo "Pixi env: $PIXI_ENV"
echo "Launch skip/count: $LAUNCH_SKIP/$LAUNCH_COUNT"
echo "=========================================="
echo ""

# Run the profiling
scripts/ncu/release/ncu-profile-kernels.sh \
  --kernel-config "$KERNEL_CONFIG" \
  --topk "$TOPK" \
  --output-dir "$OUTPUT_DIR" \
  --num-kernel-call-skip "$LAUNCH_SKIP" \
  --num-kernel-call-profile "$LAUNCH_COUNT" \
  -- \
  python -m llm_perf_opt.runners.llm_profile_runner \
    hydra.run.dir=tmp/ncu-work/\${now:%Y%m%d-%H%M%S} \
    dataset.subset_filelist=datasets/omnidocbench/subsets/dev-3.txt \
    device=cuda:0 \
    pipeline.torch_profiler.enable=false \
    pipeline.static_analysis.enable=false \
    dataset.sampling.num_epochs=1 \
    dataset.sampling.num_samples_per_epoch="$NUM_SAMPLES" \
    dataset.sampling.randomize=false \
    infer.max_new_tokens="$MAX_NEW_TOKENS"

echo ""
echo "=========================================="
echo "Profiling complete!"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
