#!/usr/bin/env bash
set -euo pipefail

# Nsight Systems profiling for DeepSeek-OCR (per NVTX stage + combined).
#
# Runs Stage-2 runner (deep_profile_runner) multiple times:
#   - Per NVTX stage: prefill, decode, sam, clip, projector (default)
#   - Combined (no gating): full application capture ignoring NVTX
#
# Outputs are organized under:
#   <output-dir>/per-stage/{prefill,decode,sam,clip,projector}/
#   <output-dir>/all-stage/
#
# Usage examples (recommended via Pixi RTX5090 env):
#   pixi run -e rtx5090 scripts/nsys/nsys-profile-dsocr-stages.sh
#   pixi run -e rtx5090 scripts/nsys/nsys-profile-dsocr-stages.sh \
#     --gpu=0 --samples=10 --subset=datasets/omnidocbench/subsets/dev-20.txt
#   pixi run -e rtx5090 scripts/nsys/nsys-profile-dsocr-stages.sh \
#     --stages=prefill --max-new-tokens=64 infer.context_len_mode=auto
#
# Notes
# - deep_profile_runner ensures 'nsys' exists; we also check early for a clearer error.
# - The Stage-1 workload is automatically launched with torch_profiler and
#   static_analysis disabled to avoid overhead during Nsight capture.

RED=""; GREEN=""; YELLOW=""; BOLD=""; RESET=""
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
fi

print_help() {
  cat <<'USAGE'
Nsight Systems profiling per NVTX range (DeepSeek-OCR)

Options:
  --gpu=N                 Set CUDA_VISIBLE_DEVICES to N (maps to device=cuda:0)
  --subset=PATH           Dataset filelist (default: datasets/omnidocbench/subsets/dev-20.txt)
  --samples=N             Samples per epoch for Stage-1 workload (default: 20)
  --max-new-tokens=N      Inference max_new_tokens (default: 64)
  --stages=list           Comma-separated NVTX ranges to capture (default: prefill,decode,sam,clip,projector)
  --no-combined           Skip the combined (no NVTX gating) capture
  --run-dir=PATH          Override base output dir (default: tmp/nsys-profile/<ts>)
  -h, --help              Show this help

Any additional arguments are passed as Hydra overrides, e.g.:
  infer.context_len_mode=auto model.dtype=bf16

Examples:
  pixi run -e rtx5090 scripts/nsys/nsys-profile-dsocr-stages.sh --gpu=0 --samples=10 \
    --subset=datasets/omnidocbench/subsets/dev-20.txt infer.max_new_tokens=64
USAGE
}

require_cmd() { for c in "$@"; do command -v "$c" >/dev/null || { echo "missing: $c" >&2; exit 127; }; done; }

require_cmd date
if ! command -v nsys >/dev/null 2>&1; then
  echo "${RED}nsys not found in PATH${RESET}. Install Nsight Systems and retry." >&2
  exit 127
fi

# Defaults
GPU_IDX=""
SUBSET_PATH="datasets/omnidocbench/subsets/dev-20.txt"
SAMPLES="20"
MAX_NEW_TOKENS="64"
STAGES="prefill,decode,sam,clip,projector"
RUN_DIR_BASE=""
DO_COMBINED=true
EXTRA_OVERRIDES=()

for arg in "$@"; do
  case "$arg" in
    --gpu=*) GPU_IDX="${arg#--gpu=}" ;;
    --subset=*) SUBSET_PATH="${arg#--subset=}" ;;
    --samples=*) SAMPLES="${arg#--samples=}" ;;
    --max-new-tokens=*) MAX_NEW_TOKENS="${arg#--max-new-tokens=}" ;;
    --stages=*) STAGES="${arg#--stages=}" ;;
    --run-dir=*) RUN_DIR_BASE="${arg#--run-dir=}" ;;
    --no-combined) DO_COMBINED=false ;;
    -h|--help) print_help; exit 0 ;;
    *) EXTRA_OVERRIDES+=("$arg") ;;
  esac
done

# Timestamped base output
TS="$(date +%Y%m%d-%H%M%S)"
if [[ -z "$RUN_DIR_BASE" ]]; then
  RUN_DIR_BASE="tmp/nsys-profile/$TS"
fi

# Prep environment mapping and echo configuration
if [[ -n "$GPU_IDX" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDX"
  echo "${BOLD}GPU${RESET}: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (device=cuda:0)"
else
  echo "${BOLD}GPU${RESET}: using default visibility (device=cuda:0)"
fi

# Ensure subset path is absolute for robust Hydra file resolution in workload
if [[ "$SUBSET_PATH" != /* ]]; then
  SUBSET_ABS="$(pwd)/$SUBSET_PATH"
else
  SUBSET_ABS="$SUBSET_PATH"
fi

echo "${BOLD}Output Base${RESET}: $RUN_DIR_BASE"
echo "${BOLD}Subset${RESET}: $SUBSET_ABS"
echo "${BOLD}Samples/epoch${RESET}: $SAMPLES"
echo "${BOLD}Max new tokens${RESET}: $MAX_NEW_TOKENS"
echo "${BOLD}Stages${RESET}: $STAGES"
echo "${BOLD}Combined capture${RESET}: $([[ "$DO_COMBINED" == true ]] && echo enabled || echo disabled)"
if ((${#EXTRA_OVERRIDES[@]})); then
  echo "${BOLD}Extra overrides${RESET}: ${EXTRA_OVERRIDES[*]}"
fi

IFS=',' read -r -a STAGE_LIST <<< "$STAGES"

for stage in "${STAGE_LIST[@]}"; do
  s_trim="${stage//[[:space:]]/}"
  [[ -n "$s_trim" ]] || continue
  OUT_DIR="$RUN_DIR_BASE/per-stage/$s_trim"
  mkdir -p "$OUT_DIR"

  echo "\n${YELLOW}==> Profiling NVTX range: ${s_trim}${RESET}"
  CMD=(python -m llm_perf_opt.runners.deep_profile_runner \
    "hydra.run.dir=$OUT_DIR" \
    pipeline.nsys.enable=true \
    pipeline.ncu.enable=false \
    pipeline.nsys.capture_range=nvtx \
    "pipeline.nsys.nvtx_capture=$s_trim" \
    pipeline.nsys.capture_range_end=stop \
    device=cuda:0 \
    "dataset.subset_filelist=$SUBSET_ABS" \
    "dataset.sampling.num_epochs=1" \
    "dataset.sampling.num_samples_per_epoch=$SAMPLES" \
    dataset.sampling.randomize=false \
    "infer.max_new_tokens=$MAX_NEW_TOKENS")

  if ((${#EXTRA_OVERRIDES[@]})); then
    CMD+=("${EXTRA_OVERRIDES[@]}")
  fi

  echo "Running: ${CMD[*]}"
  "${CMD[@]}"

  echo "\nArtifacts for stage '$s_trim':"
  if [[ -d "$OUT_DIR/nsys" ]]; then
    find "$OUT_DIR/nsys" -maxdepth 1 -type f -printf "%p\n" | sort || true
  else
    echo "(nsys dir not found)"
  fi
done

if [[ "$DO_COMBINED" == true ]]; then
  # Full application capture (ignore NVTX gating)
  OUT_DIR="$RUN_DIR_BASE/all-stage"
  mkdir -p "$OUT_DIR"
  echo "\n${YELLOW}==> Profiling combined (no NVTX gating)${RESET}"
  CMD=(python -m llm_perf_opt.runners.deep_profile_runner \
    "hydra.run.dir=$OUT_DIR" \
    pipeline.nsys.enable=true \
    pipeline.ncu.enable=false \
    pipeline.nsys.capture_range=none \
    pipeline.nsys.gating_nvtx=false \
    device=cuda:0 \
    "dataset.subset_filelist=$SUBSET_ABS" \
    "dataset.sampling.num_epochs=1" \
    "dataset.sampling.num_samples_per_epoch=$SAMPLES" \
    dataset.sampling.randomize=false \
    "infer.max_new_tokens=$MAX_NEW_TOKENS")

  if ((${#EXTRA_OVERRIDES[@]})); then
    CMD+=("${EXTRA_OVERRIDES[@]}")
  fi

  echo "Running: ${CMD[*]}"
  "${CMD[@]}"

  echo "\nArtifacts for combined:"
  if [[ -d "$OUT_DIR/nsys" ]]; then
    find "$OUT_DIR/nsys" -maxdepth 1 -type f -printf "%p\n" | sort || true
  else
    echo "(nsys dir not found)"
  fi
fi

echo "\n${GREEN}Done${RESET}. Base run directory: $RUN_DIR_BASE"
