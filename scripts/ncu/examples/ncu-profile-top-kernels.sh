#!/usr/bin/env bash
set -euo pipefail

# Batch profile multiple kernels listed in a YAML file using yq.
# Each kernel name is converted to an anchored regex and passed to
# ncu-profile-kernel.sh with a dedicated output subdirectory.

if ! command -v yq >/dev/null 2>&1; then
  echo "[batch] ERROR: 'yq' is required to parse YAML. Please install yq (mikefarah/yq)." >&2
  exit 2
fi

# Minimal color logger
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  B=$'\033[1m'; BL=$'\033[34m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; X=$'\033[0m'
else
  B=""; BL=""; G=""; Y=""; R=""; X=""
fi
log()   { printf "%b[batch]%b %s\n" "$B$BL" "$X" "$*"; }
warn()  { printf "%b[batch]%b %b%s%b\n" "$B$Y" "$X" "$Y" "$*" "$X"; }
error() { printf "%b[batch]%b %b%s%b\n" "$B$R" "$X" "$R" "$*" "$X" 1>&2; }

YAML=""
OUT_ROOT=""
MODE="detailed"
REPLAY="kernel"
KID=":::1"
SKIP_DISCOVERY="no"

show_help() {
  cat <<USAGE
Usage: $(basename "$0") [--yaml <file>] [--output-root <dir>] \\
                         [--ncu-mode fast|detailed] [--replay-mode kernel|application] \\
                         [--kernel-id <id>] [--skip-discovery]

Defaults:
  --yaml            scripts/ncu/examples/top-kernel-example.yaml
  --output-root     <repo>/tmp/ncu-profile-batch/<timestamp>
  --ncu-mode        detailed
  --replay-mode     kernel
  --kernel-id       :::1
  --skip-discovery  Speed up by skipping regex verification (off by default)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yaml)
      [[ $# -lt 2 ]] && { error "--yaml requires a value"; exit 2; }
      YAML="$2"; shift 2;;
    --output-root)
      [[ $# -lt 2 ]] && { error "--output-root requires a value"; exit 2; }
      OUT_ROOT="$2"; shift 2;;
    --ncu-mode)
      [[ $# -lt 2 ]] && { error "--ncu-mode requires a value (fast|detailed)"; exit 2; }
      case "$2" in fast|detailed) MODE="$2";; *) error "invalid --ncu-mode: $2"; exit 2;; esac
      shift 2;;
    --replay-mode)
      [[ $# -lt 2 ]] && { error "--replay-mode requires a value (kernel|application)"; exit 2; }
      case "$2" in kernel|application) REPLAY="$2";; *) error "invalid --replay-mode: $2"; exit 2;; esac
      shift 2;;
    --kernel-id)
      [[ $# -lt 2 ]] && { error "--kernel-id requires a value (e.g., :::1)"; exit 2; }
      KID="$2"; shift 2;;
    --skip-discovery)
      SKIP_DISCOVERY="yes"; shift;;
    -h|--help) show_help; exit 0;;
    --) shift; break;;
    *) warn "Unknown arg: $1 (ignored)"; shift;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ -z "$YAML" ]] && YAML="${SCRIPT_DIR}/top-kernel-example.yaml"

if [[ -z "$OUT_ROOT" ]]; then
  TS="$(date +%Y%m%d-%H%M%S)"
  OUT_ROOT="${REPO_ROOT}/tmp/ncu-profile-batch/${TS}"
fi
mkdir -p "$OUT_ROOT"

log "yaml: $YAML"
log "out_root: $OUT_ROOT"
log "mode=$MODE replay=$REPLAY kid=$KID skip_discovery=$SKIP_DISCOVERY"

# Read kernels from YAML
mapfile -t KLIST < <(yq -r '.kernels[]' "$YAML")
if [[ ${#KLIST[@]} -eq 0 ]]; then
  error "No kernels found in YAML: $YAML"; exit 3
fi

idx=0
for k in "${KLIST[@]}"; do
  idx=$((idx+1))
  # Escape regex metacharacters for exact match, then anchor
  # Character class escapes: ] [ . \ * ^ $ ( ) + ? { | }
  # Put ] first and [ second to make them literal in the character class
  esc=$(printf '%s' "$k" | sed -e 's/\\/\\\\/g' -e 's/[][\.*^$()+?{|}]/\\&/g')
  regex="^${esc}$"
  outdir="${OUT_ROOT}/k${idx}"
  mkdir -p "$outdir"
  log "[#${idx}] profiling kernel → $k"

  # Build optional discovery flag
  DISC_ARGS=()
  [[ "$SKIP_DISCOVERY" == "yes" ]] && DISC_ARGS=(--skip-discovery)

  "${SCRIPT_DIR}/ncu-profile-kernel.sh" \
    --kernel-name-regex "$regex" \
    --output-dir "$outdir" \
    --ncu-mode "$MODE" \
    --replay-mode "$REPLAY" \
    --kernel-id "$KID" \
    "${DISC_ARGS[@]}"
done

log "Done. Reports under: $OUT_ROOT"

