#!/usr/bin/env bash
set -euo pipefail

# Run Stage 2 profiling with full capture (NVTX gating disabled)
# Optionally enable Stage 1 static model analysis output.
#
# Usage:
#   tests/manual/stage2_profile/run-stage2-full.sh [--gpu=N] [--with-static-analysis] [additional hydra overrides]
# Examples:
#   tests/manual/stage2_profile/run-stage2-full.sh
#   tests/manual/stage2_profile/run-stage2-full.sh --gpu=1
#   tests/manual/stage2_profile/run-stage2-full.sh --gpu=1 --with-static-analysis infer.max_new_tokens=32

EXTRA_OVERRIDES=()
WITH_STATIC=false
SUBSET_DEFAULT="datasets/omnidocbench/subsets/dev-20.txt"
SUBSET_PATH="$SUBSET_DEFAULT"

for arg in "$@"; do
  case "$arg" in
    --gpu=*)
      GPU_IDX="${arg#--gpu=}"
      export CUDA_VISIBLE_DEVICES="$GPU_IDX"
      # Map to cuda:0 within the remapped visible set
      EXTRA_OVERRIDES+=("+stage1_runner.device=cuda:0")
      ;;
    --with-static-analysis)
      WITH_STATIC=true
      ;;
    --subset=*)
      SUBSET_PATH="${arg#--subset=}"
      ;;
    *)
      EXTRA_OVERRIDES+=("$arg")
      ;;
  esac
done

# Use Hydra time-stamped run dir under tmp/stage2/
RUN_OVERRIDE='hydra.run.dir=tmp/stage2/${now:%Y%m%d-%H%M%S}'

# Ensure subset path is absolute for Stage 1 runner
if [[ "$SUBSET_PATH" != /* ]]; then
  SUBSET_ABS="$(pwd)/$SUBSET_PATH"
else
  SUBSET_ABS="$SUBSET_PATH"
fi

CMD=(pixi run stage2-profile "$RUN_OVERRIDE" +nsys.gating_nvtx=false +ncu.gating_nvtx=false \
  ncu=ncu.rtx3090.compute \
  "+run.dataset_subset_filelist=$SUBSET_ABS" \
  "+run.stage1_repeats=20")

# If requested, enable Stage 1 static model analysis (undo the default disable in Stage 2)
if [[ "$WITH_STATIC" == "true" ]]; then
  CMD+=("+stage1_runner.disable_static=false")
fi

if ((${#EXTRA_OVERRIDES[@]})); then
  CMD+=("${EXTRA_OVERRIDES[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

# Resolve the latest run directory and print reports
RUN_DIR=$(ls -1td tmp/stage2/* | head -n1)
echo "\nStage 2 run directory: $RUN_DIR"
echo "\nArtifacts (depth 2):"
find "$RUN_DIR" -maxdepth 2 -type f -printf "%p\n" | sort

echo -e "\nLatest report (report.md):\n"
if [[ -f "$RUN_DIR/report.md" ]]; then
  sed -n '1,200p' "$RUN_DIR/report.md"
else
  echo "report.md not found"
fi

echo -e "\nStakeholder summary (stakeholder_summary.md):\n"
if [[ -f "$RUN_DIR/stakeholder_summary.md" ]]; then
  sed -n '1,200p' "$RUN_DIR/stakeholder_summary.md"
else
  echo "stakeholder_summary.md not found"
fi

# Print Stage 1 static analysis report if present
if [[ -f "$RUN_DIR/stage1/static_compute.md" ]]; then
  echo -e "\nStage 1 static analysis (stage1/static_compute.md):\n"
  sed -n '1,200p' "$RUN_DIR/stage1/static_compute.md"
fi

exit 0
