#!/usr/bin/env bash
set -euo pipefail

# Nsight Compute Kernel Profiler (via runner facility)
#
# This script mirrors scripts/ncu/release/ncu-profile-kernels.sh CLI and behavior,
# but builds and executes the Nsight Compute command using the project's
# Python builder (src/llm_perf_opt/profiling/vendor/ncu.py).
#
# Synopsis:
#   scripts/ncu/release/ncu-profile-kernels-via-runner.sh [options] -- <launch-command> [launch-args]
#
# Options are identical to ncu-profile-kernels.sh
#   --kernel-config <yaml>         YAML listing kernels with 'name' and 'regex'
#   --kernel-regex <regex>         Single demangled-name regex to match kernels
#   --output-dir <dir>             Defaults to tmp/ncu-profile/<timestamp>
#   --topk <K>                     First K kernels from YAML (with --kernel-config)
#   --extra-sections <s...>        Extra NCU sections beyond defaults
#   --num-kernel-call-skip <N>     Skip first N kernel invocations (default: 200)
#   --num-kernel-call-profile <M>  Profile M invocations after skipping (default: 1)
#   --force-overwrite              Overwrite existing reports
#   --                             Separator before target launch command
#
# Default sections:
#   SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats

# --- Color logging ---
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  CLR_RESET=$'\033[0m'; CLR_BOLD=$'\033[1m'; CLR_RED=$'\033[31m'
  CLR_GREEN=$'\033[32m'; CLR_YELLOW=$'\033[33m'; CLR_BLUE=$'\033[34m'
  CLR_CYAN=$'\033[36m'
else
  CLR_RESET=""; CLR_BOLD=""; CLR_RED=""; CLR_GREEN=""; CLR_YELLOW=""; CLR_BLUE=""; CLR_CYAN=""
fi
log_info()  { printf "%b[ncu-profile]%b %s\n"  "$CLR_BOLD$CLR_BLUE" "$CLR_RESET" "$*"; }
log_ok()    { printf "%b[ncu-profile]%b %b%s%b\n" "$CLR_BOLD$CLR_GREEN" "$CLR_RESET" "$CLR_GREEN" "$*" "$CLR_RESET"; }
log_warn()  { printf "%b[ncu-profile]%b %b%s%b\n" "$CLR_BOLD$CLR_YELLOW" "$CLR_RESET" "$CLR_YELLOW" "$*" "$CLR_RESET" >&2; }
log_err()   { printf "%b[ncu-profile]%b %b%s%b\n" "$CLR_BOLD$CLR_RED" "$CLR_RESET" "$CLR_RED" "$*" "$CLR_RESET" >&2; }
log_highlight() { printf "%b%s%b" "$CLR_CYAN" "$1" "$CLR_RESET"; }

# --- Defaults ---
KERNEL_CONFIG=""
KERNEL_REGEX=""
OUT_DIR=""
TOPK=""
EXTRA_SECTIONS=()
LAUNCH_SKIP=200
LAUNCH_COUNT=1
FORCE_OVERWRITE="no"
LAUNCH_CMD=()

# Default sections (aligned with Python version)
DEFAULT_SECTIONS="SpeedOfLight MemoryWorkloadAnalysis Occupancy SchedulerStats"

# --- Help ---
show_help() {
  cat <<'USAGE'
Usage: ncu-profile-kernels-via-runner.sh [options] -- <launch-command> [launch-args]

Required (one of):
  --kernel-config <yaml-path>  Path to YAML file with kernel names/regex patterns
  --kernel-regex <regex>       Single regex pattern for kernel matching

Options:
  --output-dir <dir>           Directory for profiling results (default: tmp/ncu-profile/<timestamp>)
  --topk <num>                 Profile only top K kernels from YAML (requires --kernel-config)
  --extra-sections <s1> <s2>   Additional ncu sections beyond defaults
  --num-kernel-call-skip <N>   Skip first N kernel invocations (default: 200)
  --num-kernel-call-profile <M> Profile M invocations after skipping (default: 1)
  --force-overwrite            Overwrite existing reports

Launch Command:
  --                           Separator before target application command
  <launch-command>             Command to launch target application
  [launch-args]                Arguments for the target application

Examples:
  # Profile single kernel
  ncu-profile-kernels-via-runner.sh --kernel-regex 'gemvx::kernel<.*\(int\)7.*>' \
    --output-dir tmp/gemvx \
    -- python -m llm_perf_opt.runners.llm_profile_runner device=cuda:0

  # Profile multiple kernels from YAML
  ncu-profile-kernels-via-runner.sh --kernel-config top-kernels.yaml \
    --extra-sections SourceCounters \
    -- python inference.py --model deepseek

  # Profile only top 3 kernels from YAML
  ncu-profile-kernels-via-runner.sh --kernel-config top-kernels.yaml --topk 3 \
    -- python inference.py

Default sections: SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats
USAGE
}

# --- Parse arguments ---
while [[ $# > 0 ]]; do
  case "$1" in
    --kernel-config)
      [[ $# -lt 2 ]] && { log_err "--kernel-config requires a value"; exit 2; }
      KERNEL_CONFIG="$2"; shift 2;;
    --kernel-regex)
      [[ $# -lt 2 ]] && { log_err "--kernel-regex requires a value"; exit 2; }
      KERNEL_REGEX="$2"; shift 2;;
    --output-dir)
      [[ $# -lt 2 ]] && { log_err "--output-dir requires a value"; exit 2; }
      OUT_DIR="$2"; shift 2;;
    --topk)
      [[ $# -lt 2 ]] && { log_err "--topk requires a value"; exit 2; }
      TOPK="$2"; shift 2;;
    --extra-sections)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        EXTRA_SECTIONS+=("$1")
        shift
      done
      ;;
    --num-kernel-call-skip)
      [[ $# -lt 2 ]] && { log_err "--num-kernel-call-skip requires a value"; exit 2; }
      LAUNCH_SKIP="$2"; shift 2;;
    --num-kernel-call-profile)
      [[ $# -lt 2 ]] && { log_err "--num-kernel-call-profile requires a value"; exit 2; }
      LAUNCH_COUNT="$2"; shift 2;;
    --force-overwrite)
      FORCE_OVERWRITE="yes"; shift;;
    -h|--help)
      show_help; exit 0;;
    --)
      shift
      LAUNCH_CMD=("$@")
      break;;
    *)
      log_err "Unknown argument: $1"
      show_help
      exit 2;;
  esac
done

# --- Validation ---
if [[ -z "$KERNEL_CONFIG" && -z "$KERNEL_REGEX" ]]; then
  log_err "ERROR: Either --kernel-config or --kernel-regex is required"
  show_help
  exit 2
fi

if [[ -n "$KERNEL_CONFIG" && -n "$KERNEL_REGEX" ]]; then
  log_err "ERROR: --kernel-config and --kernel-regex are mutually exclusive"
  exit 2
fi

if [[ ${#LAUNCH_CMD[@]} -eq 0 ]]; then
  log_err "ERROR: Launch command is required after -- separator"
  show_help
  exit 2
fi

if [[ -n "$TOPK" ]]; then
  if [[ -n "$KERNEL_REGEX" ]]; then
    log_err "ERROR: --topk can only be used with --kernel-config, not with --kernel-regex"
    exit 2
  fi
  if [[ ! "$TOPK" =~ ^[0-9]+$ ]] || [[ "$TOPK" -le 0 ]]; then
    log_err "ERROR: --topk must be a positive integer"
    exit 2
  fi
fi

# --- Setup paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ -z "$OUT_DIR" ]] && OUT_DIR="${REPO_ROOT}/tmp/ncu-profile/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

# Python availability (for builder + optional YAML parse)
if command -v python3 &>/dev/null; then PY_CMD=$(command -v python3); elif command -v python &>/dev/null; then PY_CMD=$(command -v python); else log_err "Python 3 is required"; exit 2; fi

# Build sections list
ALL_SECTIONS="$DEFAULT_SECTIONS"
if [[ ${#EXTRA_SECTIONS[@]} -gt 0 ]]; then
  ALL_SECTIONS="$ALL_SECTIONS ${EXTRA_SECTIONS[*]}"
fi

log_info "Output directory: $(log_highlight "$OUT_DIR")"
log_info "Sections: $ALL_SECTIONS"
log_info "Launch skip/count: $LAUNCH_SKIP/$LAUNCH_COUNT"

# --- Check tools ---
if ! command -v ncu &>/dev/null; then
  log_err "ncu command not found. Please ensure Nsight Compute is installed and in PATH."
  exit 2
fi

# --- Load kernels ---
KERNELS_JSON=$(mktemp)
trap "rm -f $KERNELS_JSON" EXIT

if [[ -n "$KERNEL_CONFIG" ]]; then
  # Batch mode from YAML
  if [[ ! -f "$KERNEL_CONFIG" ]]; then
    log_err "Kernel config file not found: $KERNEL_CONFIG"
    exit 2
  fi

  log_info "Kernel config: $(log_highlight "$KERNEL_CONFIG")"

  # Prefer yq-go if present, else Python
  if command -v yq &>/dev/null && yq --version 2>&1 | grep -qiE "(mikefarah|version.*v[0-9]+\.[0-9]+\.[0-9]+)"; then
    if [[ -n "$TOPK" ]]; then
      ORIGINAL_COUNT=$(yq eval '.kernels | length' "$KERNEL_CONFIG")
      yq eval ".kernels | .[0:${TOPK}]" -o=json "$KERNEL_CONFIG" > "$KERNELS_JSON"
      log_info "Limiting to top $TOPK kernel(s) from $ORIGINAL_COUNT total"
    else
      yq eval '.kernels' -o=json "$KERNEL_CONFIG" > "$KERNELS_JSON"
    fi
  else
    "$PY_CMD" - "$KERNEL_CONFIG" "$TOPK" > "$KERNELS_JSON" <<'PYTHON_SCRIPT'
import sys, json
try:
    import ruamel.yaml
    yaml = ruamel.yaml.YAML(typ='safe')
except ImportError:
    try:
        import yaml as _pyyaml
        yaml = _pyyaml
    except ImportError:
        print("ERROR: ruamel.yaml or pyyaml is required", file=sys.stderr)
        sys.exit(1)

yaml_path = sys.argv[1]
topk = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

with open(yaml_path, 'r', encoding='utf-8') as f:
    data = yaml.load(f) if hasattr(yaml, 'load') else yaml.safe_load(f)

kernels = data.get('kernels', [])
if topk:
    try:
        topk_int = int(topk)
        kernels = kernels[:topk_int]
        print(f"# Limiting to top {topk_int} kernels", file=sys.stderr)
    except Exception:
        pass

json.dump(kernels, sys.stdout)
PYTHON_SCRIPT
  fi
else
  # Single kernel mode
  printf '[{"name": "user_provided", "regex": "%s", "description": "Single kernel profiling"}]' "$KERNEL_REGEX" > "$KERNELS_JSON"
fi

# Count kernels
if command -v jq &>/dev/null; then
  KERNEL_COUNT=$(jq length < "$KERNELS_JSON")
else
  # Fallback count via Python
  KERNEL_COUNT=$("$PY_CMD" -c 'import sys,json; print(len(json.load(open(sys.argv[1]))))' "$KERNELS_JSON")
fi

# Determine rank width (min 4)
RANK_WIDTH=${#KERNEL_COUNT}
(( RANK_WIDTH < 4 )) && RANK_WIDTH=4

SUCCESS_COUNT=0
FAILED_COUNT=0

# Helper: compute MD5 hex of a string
compute_md5() {
  local s="$1"
  if command -v md5sum >/dev/null 2>&1; then
    printf "%s" "$s" | md5sum | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    printf "%s" "$s" | md5 | awk '{print $NF}'
  else
    printf "%s" "$s" | "$PY_CMD" -c 'import sys,hashlib; print(hashlib.md5(sys.stdin.buffer.read()).hexdigest())'
  fi
}

for i in $(seq 0 $((KERNEL_COUNT - 1))); do
  if command -v jq &>/dev/null; then
    KERNEL_NAME=$(jq -r ".[$i].name" < "$KERNELS_JSON")
    KERNEL_REGEX_CURRENT=$(jq -r ".[$i].regex" < "$KERNELS_JSON")
    KERNEL_DESC=$(jq -r ".[$i].description // empty" < "$KERNELS_JSON")
  else
    KERNEL_INFO=$("$PY_CMD" -c "import sys, json; k=json.load(sys.stdin)[$i]; print(k['name']); print(k['regex']); print(k.get('description',''))" < "$KERNELS_JSON")
    KERNEL_NAME=$(echo "$KERNEL_INFO" | sed -n '1p')
    KERNEL_REGEX_CURRENT=$(echo "$KERNEL_INFO" | sed -n '2p')
    KERNEL_DESC=$(echo "$KERNEL_INFO" | sed -n '3p')
  fi

  KERNEL_NUM=$((i + 1))

  log_info ""
  log_info "================================================================================"
  log_info "Profiling kernel $KERNEL_NUM/$KERNEL_COUNT"
  log_info "Name: $KERNEL_NAME"
  [[ -n "$KERNEL_DESC" ]] && log_info "Description: $KERNEL_DESC"
  log_info "Regex: $(log_highlight "$KERNEL_REGEX_CURRENT")"
  log_info "================================================================================"

  RANK_PADDED=$(printf "%0${RANK_WIDTH}d" "$KERNEL_NUM")
  NAME_MD5=$(compute_md5 "$KERNEL_NAME")
  KERNEL_DIR="$OUT_DIR/kernel_${RANK_PADDED}_${NAME_MD5}"
  mkdir -p "$KERNEL_DIR"

  OUTPUT_BASE="${KERNEL_DIR}/ncu"
  REPORT_PATH="${OUTPUT_BASE}.ncu-rep"

  # Build + run NCU using Python builder; inject launch skip/count before work argv
  log_info "Running via builder: ncu (sections: $ALL_SECTIONS) ${LAUNCH_CMD[*]}"

  # Prepare JSON payload for Python
  JSON_PAYLOAD=$(mktemp)
  trap 'rm -f "$JSON_PAYLOAD"' EXIT
  {
    printf '{'
    printf '"out_base":"%s",' "$OUTPUT_BASE"
    printf '"sections":["%s"],' "${ALL_SECTIONS// /","}"
    printf '"kernel_regex":"%s",' "regex:${KERNEL_REGEX_CURRENT}"
    printf '"work_argv":['
    for ((j=0; j<${#LAUNCH_CMD[@]}; j++)); do
      arg=${LAUNCH_CMD[$j]}
      printf '"%s"' "${arg//"/\"}"
      if (( j < ${#LAUNCH_CMD[@]} - 1 )); then printf ','; fi
    done
    printf '],'
    printf '"target_processes":"all",'
    printf '"force_overwrite":%s,' "$([[ "$FORCE_OVERWRITE" == "yes" ]] && echo true || echo false)"
    printf '"kernel_name_base":"demangled",'
    printf '"launch_skip":%d,' "$LAUNCH_SKIP"
    printf '"launch_count":%d' "$LAUNCH_COUNT"
    printf '}'
  } > "$JSON_PAYLOAD"

  set +e
  "$PY_CMD" - "$JSON_PAYLOAD" <<'PYBUILDER'
import json, sys, subprocess
from pathlib import Path
from llm_perf_opt.profiling.vendor.ncu import build_ncu_cmd

payload_path = sys.argv[1]
with open(payload_path, 'r', encoding='utf-8') as f:
    P = json.load(f)

cmd = build_ncu_cmd(
    out_base=Path(P['out_base']),
    work_argv=P['work_argv'],
    nvtx_expr=None,
    kernel_regex=P['kernel_regex'],
    csv_log=None,
    use_nvtx=False,
    set_name='roofline',
    metrics=None,
    sections=P['sections'],
    target_processes=P.get('target_processes','all'),
    force_overwrite=bool(P.get('force_overwrite', False)),
    kernel_name_base=P.get('kernel_name_base','demangled'),
)

# Insert launch skip/count flags before the work argv
work_len = len(P['work_argv'])
insert_at = len(cmd) - work_len if work_len <= len(cmd) else len(cmd)
cmd[insert_at:insert_at] = ["--launch-skip", str(int(P.get('launch_skip', 200))), "--launch-count", str(int(P.get('launch_count', 1)))]

print('[ncu-profile] executing:', ' '.join(cmd))
ret = subprocess.run(cmd, check=False).returncode
sys.exit(ret)
PYBUILDER
  NCU_EXIT_CODE=$?
  set -e

  if [[ $NCU_EXIT_CODE -eq 0 ]]; then
    log_ok "Profiling complete: $REPORT_PATH"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

    # Export CSVs
    if [[ -f "$REPORT_PATH" ]]; then
      log_info "Exporting section CSVs..."

      for section in $ALL_SECTIONS; do
        SECTION_CSV="${OUTPUT_BASE}.section_${section}.csv"
        if ncu --csv --section "$section" --import "$REPORT_PATH" > "$SECTION_CSV" 2>/dev/null; then
          log_ok "Section CSV: $SECTION_CSV"
        else
          log_warn "Could not export section: $section"
        fi
      done

      # Try SpeedOfLight_RooflineChart
      ROOFLINE_CSV="${OUTPUT_BASE}.section_SpeedOfLight_RooflineChart.csv"
      if ncu --csv --section SpeedOfLight_RooflineChart --import "$REPORT_PATH" > "$ROOFLINE_CSV" 2>/dev/null; then
        log_ok "Section CSV: $ROOFLINE_CSV"
      fi

      # Export details page
      DETAILS_CSV="${OUTPUT_BASE}.details.csv"
      if ncu --csv --page details --import "$REPORT_PATH" > "$DETAILS_CSV" 2>/dev/null; then
        log_ok "Details CSV: $DETAILS_CSV"
      else
        log_warn "Could not export details CSV"
      fi
    else
      log_warn "Report not found, skipping CSV export: $REPORT_PATH"
    fi
  else
    log_err "Profiling failed for kernel: $KERNEL_NAME (exit code: $NCU_EXIT_CODE)"
    FAILED_COUNT=$((FAILED_COUNT + 1))
  fi
done

log_info ""
log_info "================================================================================"
log_ok "Profiling complete: $SUCCESS_COUNT succeeded, $FAILED_COUNT failed"
log_ok "Results saved to: $OUT_DIR"
log_info "================================================================================"

(( FAILED_COUNT > 0 )) && exit 1 || exit 0

