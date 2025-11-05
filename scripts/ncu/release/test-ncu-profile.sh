#!/usr/bin/env bash
set -euo pipefail

# NOTE: This script is intended to be run inside a Pixi environment.
# Use:   pixi run -e <pixi-env> ./scripts/ncu/release/test-ncu-profile.sh [--bash|--python]
# Running it directly will use the system Python/Nsight tools and may miss
# required dependencies (PyTorch, ruamel.yaml/pyyaml, etc.).

# Test script for ncu-profile-kernels (bash/python variants)
#
# This script runs NCU kernel profiling on the top 3 kernels from the
# top-10-kernels.yaml config using the RTX 5090 environment.
#
# Usage:
#   ./test-ncu-profile.sh [--bash|--python]
#
# Flags:
#   --bash     Use Bash profiler (ncu-profile-kernels.sh). Default.
#   --python   Use Python profiler (ncu-profile-kernels.py).
#
# Examples:
#   ./test-ncu-profile.sh                       # bash branch (default)
#   ./test-ncu-profile.sh --python              # python branch
#   TOPK=5 ./test-ncu-profile.sh --bash         # override topk

# --- Configuration ---
TOPK=${TOPK:-3}
KERNEL_CONFIG="scripts/ncu/examples/top-10-kernels.yaml"
LAUNCH_SKIP=${LAUNCH_SKIP:-50}
LAUNCH_COUNT=${LAUNCH_COUNT:-1}
PIXI_ENV=${PIXI_ENV:-rtx5090}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
NUM_SAMPLES=${NUM_SAMPLES:-1}

# Parse branch flag (default: bash)
BRANCH="bash"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bash) BRANCH="bash"; shift ;;
    --python) BRANCH="python"; shift ;;
    -h|--help)
      grep -E "^# (Usage|Flags|Examples):|^#\s{2,}--|^#\s{2,}\.|^#\s{2,}[A-Z]" -n "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) break ;;
  esac
done

# Generate timestamp for output directory
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="tmp/ncu-profile/${TIMESTAMP}"

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

if [[ "$BRANCH" == "python" ]]; then
  echo "Running Python profiler branch (ncu-profile-kernels.py)"
  python scripts/ncu/release/ncu-profile-kernels.py \
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
else
  echo "Running Bash profiler branch (ncu-profile-kernels.sh)"
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
fi

echo ""
echo "=========================================="
echo "Profiling complete!"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
