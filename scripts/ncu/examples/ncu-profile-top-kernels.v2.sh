#!/usr/bin/env bash
set -euo pipefail

# Batch profile multiple kernels listed in a YAML file using yq.
# Each kernel's regex pattern is passed to ncu-profile-kernel.v2.sh
# with a dedicated output subdirectory.
#
# Usage:
#   ncu-profile-top-kernels.v2.sh [--yaml <file>] [--output-root <dir>] [options]

# --- Check for yq ---
if ! command -v yq >/dev/null 2>&1; then
  echo "[batch-v2] ERROR: 'yq' is required to parse YAML. Please install yq (mikefarah/yq)." >&2
  exit 2
fi

# --- Color logger ---
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  B=$'\033[1m'; BL=$'\033[34m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; C=$'\033[36m'; X=$'\033[0m'
else
  B=""; BL=""; G=""; Y=""; R=""; C=""; X=""
fi
log()   { printf "%b[batch-v2]%b %s\n" "$B$BL" "$X" "$*"; }
ok()    { printf "%b[batch-v2]%b %b%s%b\n" "$B$G" "$X" "$G" "$*" "$X"; }
warn()  { printf "%b[batch-v2]%b %b%s%b\n" "$B$Y" "$X" "$Y" "$*" "$X"; }
error() { printf "%b[batch-v2]%b %b%s%b\n" "$B$R" "$X" "$R" "$*" "$X" 1>&2; }

# --- Defaults ---
YAML=""
OUT_ROOT=""
LAUNCH_SKIP=200
LAUNCH_COUNT=1
FORCE_OVERWRITE="no"

show_help() {
  cat <<USAGE
Usage: $(basename "$0") [--yaml <file>] [--output-root <dir>] [options]

Options:
  --yaml <file>          YAML file with kernels (default: scripts/ncu/examples/top-kernel-example.yaml)
  --output-root <dir>    Root output directory (default: <repo>/tmp/ncu-profile-batch-v2/<timestamp>)
  --launch-skip <n>      Skip first N invocations per kernel (default: 200)
  --launch-count <n>     Profile N invocations per kernel (default: 1)
  --force-overwrite      Overwrite existing report files

YAML format:
  kernels:
    - name: 'Full kernel name from nsys'
      regex: 'internal::gemvx::kernel<.*\\\\(int\\\\)7.*>'
      description: 'Optional description'

Examples:
  # Profile all kernels from default YAML
  $(basename "$0")

  # Custom YAML and output location
  $(basename "$0") --yaml my-kernels.yaml --output-root /tmp/profiles

  # Profile with different sampling
  $(basename "$0") --launch-skip 500 --launch-count 3
USAGE
}

# --- Arg parsing ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yaml)
      [[ $# -lt 2 ]] && { error "--yaml requires a value"; exit 2; }
      YAML="$2"; shift 2;;
    --output-root)
      [[ $# -lt 2 ]] && { error "--output-root requires a value"; exit 2; }
      OUT_ROOT="$2"; shift 2;;
    --launch-skip)
      [[ $# -lt 2 ]] && { error "--launch-skip requires a value"; exit 2; }
      LAUNCH_SKIP="$2"; shift 2;;
    --launch-count)
      [[ $# -lt 2 ]] && { error "--launch-count requires a value"; exit 2; }
      LAUNCH_COUNT="$2"; shift 2;;
    --force-overwrite)
      FORCE_OVERWRITE="yes"; shift;;
    -h|--help) show_help; exit 0;;
    --) shift; break;;
    *) warn "Unknown arg: $1 (ignored)"; shift;;
  esac
done

# --- Setup paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ -z "$YAML" ]] && YAML="${SCRIPT_DIR}/top-kernel-example.yaml"

if [[ ! -f "$YAML" ]]; then
  error "YAML file not found: $YAML"
  exit 3
fi

if [[ -z "$OUT_ROOT" ]]; then
  TS="$(date +%Y%m%d-%H%M%S)"
  OUT_ROOT="${REPO_ROOT}/tmp/ncu-profile-batch-v2/${TS}"
fi
mkdir -p "$OUT_ROOT"

# --- Log configuration ---
log "yaml: $YAML"
log "output_root: $OUT_ROOT"
log "launch_skip: $LAUNCH_SKIP, launch_count: $LAUNCH_COUNT"

# --- Read kernels from YAML ---
# Extract both regex and description for each kernel
mapfile -t REGEX_LIST < <(yq -r '.kernels[].regex' "$YAML")
mapfile -t DESC_LIST < <(yq -r '.kernels[].description' "$YAML")

if [[ ${#REGEX_LIST[@]} -eq 0 ]]; then
  error "No kernels found in YAML: $YAML"
  exit 4
fi

log "Found ${#REGEX_LIST[@]} kernel(s) to profile"

# --- Write batch provenance ---
BATCH_YAML="${OUT_ROOT}/batch-info.yaml"
cat > "$BATCH_YAML" <<EOF
timestamp: '$(date -Is)'
script: '$(basename "$0")'
version: 'v2-simplified'
yaml_source: '${YAML}'
output_root: '${OUT_ROOT}'
launch_skip: ${LAUNCH_SKIP}
launch_count: ${LAUNCH_COUNT}
force_overwrite: ${FORCE_OVERWRITE}
kernel_count: ${#REGEX_LIST[@]}
kernels:
EOF

for idx in "${!REGEX_LIST[@]}"; do
  echo "  - regex: '${REGEX_LIST[$idx]}'" >> "$BATCH_YAML"
  echo "    description: '${DESC_LIST[$idx]}'" >> "$BATCH_YAML"
done

# --- Profile each kernel ---
idx=0
SUCCESS_COUNT=0
FAIL_COUNT=0

for idx in "${!REGEX_LIST[@]}"; do
  kidx=$((idx+1))
  regex="${REGEX_LIST[$idx]}"
  desc="${DESC_LIST[$idx]}"

  outdir="${OUT_ROOT}/k${kidx}"
  mkdir -p "$outdir"

  printf "\n%b[batch-v2] [#%d/%d]%b Profiling kernel...\n" "$B$C" "$kidx" "${#REGEX_LIST[@]}" "$X"
  printf "%b[batch-v2]%b   Regex: %b%s%b\n" "$B$BL" "$X" "$C" "$regex" "$X"
  printf "%b[batch-v2]%b   Desc:  %s\n" "$B$BL" "$X" "$desc"
  printf "%b[batch-v2]%b   Output: %s\n" "$B$BL" "$X" "$outdir"

  # Build arguments for v2 script
  V2_ARGS=(
    --kernel-regex "$regex"
    --output-dir "$outdir"
    --launch-skip "$LAUNCH_SKIP"
    --launch-count "$LAUNCH_COUNT"
  )

  [[ "$FORCE_OVERWRITE" == "yes" ]] && V2_ARGS+=(--force-overwrite)

  # Run profiler
  if "${SCRIPT_DIR}/ncu-profile-kernel.v2.sh" "${V2_ARGS[@]}"; then
    ok "[#${kidx}] SUCCESS"
    SUCCESS_COUNT=$((SUCCESS_COUNT+1))
  else
    error "[#${kidx}] FAILED (see logs above)"
    FAIL_COUNT=$((FAIL_COUNT+1))
  fi
done

# --- Summary ---
echo ""
log "===== Batch Profiling Complete ====="
ok "Successful: ${SUCCESS_COUNT}/${#REGEX_LIST[@]}"
[[ $FAIL_COUNT -gt 0 ]] && warn "Failed: ${FAIL_COUNT}/${#REGEX_LIST[@]}"
log "Reports under: ${OUT_ROOT}"

# Update batch provenance
cat >> "$BATCH_YAML" <<EOF
completed: true
success_count: ${SUCCESS_COUNT}
fail_count: ${FAIL_COUNT}
EOF

[[ $FAIL_COUNT -gt 0 ]] && exit 1
exit 0
