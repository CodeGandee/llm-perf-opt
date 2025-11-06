#!/usr/bin/env bash
set -euo pipefail

# Run Stage 2 profiling on a single image (convenience wrapper).
# - Disables NVTX gating (nsys/ncu) for full capture.
# - Defaults to compute-focused preset for RTX3090.
# - Lets you pick GPU via --gpu=N (maps Stage 1 to cuda:0 within that mapping).
# - Optionally enable Stage 1 static model analysis via --with-static-analysis.
# - Optionally pick an image via --image=/abs/path/to/image.(png|jpg)
#
# Examples:
#   tests/manual/stage2_profile/run-stage2-simple.sh
#   tests/manual/stage2_profile/run-stage2-simple.sh --gpu=1
#   tests/manual/stage2_profile/run-stage2-simple.sh --image=/workspace/datasets/OpenDataLab___OmniDocBench/images/foo.png
#   tests/manual/stage2_profile/run-stage2-simple.sh --with-static-analysis

WITH_STATIC=false
EXTRA_OVERRIDES=()
IMG_PATH=""
GPU_SET="false"

for arg in "$@"; do
  case "$arg" in
    --gpu=*)
      GPU_IDX="${arg#--gpu=}"
      export CUDA_VISIBLE_DEVICES="$GPU_IDX"
      EXTRA_OVERRIDES+=("+stage1_runner.device=cuda:0")
      GPU_SET="true"
      ;;
    --with-static-analysis)
      WITH_STATIC=true
      ;;
    --image=*)
      IMG_PATH="${arg#--image=}"
      ;;
    *)
      EXTRA_OVERRIDES+=("$arg")
      ;;
  esac
done

# Default image if not provided explicitly (commonly used sample in logs)
if [[ -z "$IMG_PATH" ]]; then
  IMG_PATH="$(pwd)/datasets/omnidocbench/source-data/images/PPT_1001115_eng_page_003.png"
fi

if [[ ! -f "$IMG_PATH" ]]; then
  echo "Image not found: $IMG_PATH" >&2
  exit 1
fi

# Prepare a one-line subset file (absolute path)
mkdir -p tmp
SUBSET_FILE="$(pwd)/tmp/stage2_single_image.txt"
printf "%s\n" "$IMG_PATH" > "$SUBSET_FILE"

# Default to GPU 1 if user did not specify --gpu (maps Stage 1 to cuda:0)
if [[ "$GPU_SET" != "true" ]]; then
  export CUDA_VISIBLE_DEVICES=1
  EXTRA_OVERRIDES+=("+stage1_runner.device=cuda:0")
fi

RUN_OVERRIDE='hydra.run.dir=tmp/stage2/${now:%Y%m%d-%H%M%S}'

CMD=(pixi run stage2-profile "$RUN_OVERRIDE" \
  +nsys.gating_nvtx=false +ncu.gating_nvtx=true \
  ncu=ncu.rtx3090.compute \
  dataset.subset_filelist=$SUBSET_FILE \
  dataset.sampling.num_epochs=1 \
  dataset.sampling.num_samples_per_epoch=1)

if [[ "$WITH_STATIC" == "true" ]]; then
  CMD+=("+stage1_runner.disable_static=false")
fi

if ((${#EXTRA_OVERRIDES[@]})); then
  CMD+=("${EXTRA_OVERRIDES[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

RUN_DIR=$(ls -1td tmp/stage2/* | head -n1)
echo "\nStage 2 run directory: $RUN_DIR"
echo "\nArtifacts (depth 2):"
find "$RUN_DIR" -maxdepth 2 -type f -printf "%p\n" | sort

echo -e "\nStage 2 report (report.md):\n"
if [[ -f "$RUN_DIR/report.md" ]]; then
  sed -n '1,200p' "$RUN_DIR/report.md"
fi

echo -e "\nStakeholder summary (stakeholder_summary.md):\n"
if [[ -f "$RUN_DIR/stakeholder_summary.md" ]]; then
  sed -n '1,200p' "$RUN_DIR/stakeholder_summary.md"
fi

echo -e "\nNCU raw CSV (head):\n"
if [[ -f "$RUN_DIR/ncu/raw.csv" ]]; then
  sed -n '1,60p' "$RUN_DIR/ncu/raw.csv"
else
  echo "ncu/raw.csv not found"
fi

echo -e "\nNCU sections report (head):\n"
if [[ -f "$RUN_DIR/ncu/sections_report.txt" ]]; then
  sed -n '1,120p' "$RUN_DIR/ncu/sections_report.txt"
else
  echo "ncu/sections_report.txt not found"
fi

exit 0
