#!/usr/bin/env bash
set -euo pipefail

# Nsight Compute Kernel Profiler (Bash)
#
# Synopsis:
#   scripts/ncu/release/ncu-profile-kernels.sh [options] -- <launch-command> [launch-args]
#
# Description:
#   Profiles CUDA kernels with NVIDIA Nsight Compute (ncu). Works with any console
#   command (Python, compiled binary, or shell). Supports single-regex and YAML batch
#   modes. Outputs are written per kernel under:
#     <output-dir>/kernel_<rank>_<md5(kernel-name)>/ncu.*
#   where <rank> is zero‑padded to max(4, len(total_kernels)).
#
# Requirements:
#   - ncu on PATH (Nsight Compute CLI)
#   - For --kernel-config mode: 'yq' (mikefarah, Go) or Python 3 with ruamel.yaml/pyyaml
#     and jq (recommended) or Python for JSON parsing
#   - Recommended: run inside Pixi env for CUDA/PyTorch toolchain
#       pixi run -e <env> ./scripts/ncu/release/ncu-profile-kernels.sh ...
#
# Options (common):
#   --kernel-config <yaml>         YAML listing kernels with 'name' and 'regex'
#   --kernel-regex <regex>         Single demangled-name regex to match kernels
#   --output-dir <dir>             Defaults to tmp/ncu-profile/<timestamp>
#   --topk <K>                     First K kernels from YAML (with --kernel-config)
#   --ncu-config <yaml>            Optional NCU CLI config YAML (plain YAML; e.g., conf/profiling/ncu/ncu.default.yaml)
#   --extra-sections <s...>        Extra NCU sections beyond defaults
#   --num-kernel-call-skip <N>     Skip first N kernel invocations (default: 200)
#   --num-kernel-call-profile <M>  Profile M invocations after skipping (default: 1)
#   --force-overwrite              Overwrite existing reports
#   --                             Separator before target launch command
#
# Default sections:
#   SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats, SpeedOfLight_RooflineChart
#
# Examples:
#   # Profile single kernel (Python module)
#   pixi run -e rtx5090 ./scripts/ncu/release/ncu-profile-kernels.sh \
#     --kernel-regex 'flash_fwd.*' -- \
#     python -m llm_perf_opt.runners.llm_profile_runner device=cuda:0
#
#   # Batch from YAML (limit to top 3) and add SourceCounters
#   pixi run -e rtx5090 ./scripts/ncu/release/ncu-profile-kernels.sh \
#     --kernel-config scripts/ncu/examples/top-10-kernels.yaml \
#     --topk 3 --extra-sections SourceCounters -- \
#     ./bin/infer --device cuda:0 --model deepseek-ocr
#
#   # C++ binary with custom sampling
#   ./scripts/ncu/release/ncu-profile-kernels.sh \
#     --kernel-regex 'my_kernel_.*' --num-kernel-call-skip 500 --num-kernel-call-profile 5 -- \
#     /path/to/app --arg1
#
# Output layout:
#   <output-dir>/
#     command.yaml
#     kernel_0001_<md5>/
#       ncu.ncu-rep
#       ncu.section_<Section>.csv
#       ncu.details.csv
#
# Tips:
#   - Quote regex to avoid shell globbing
#   - Set NO_COLOR=1 to disable colored logs
#   - Use -- to terminate profiler options; everything after is passed as-is to target
#
# Run with -h for on-demand help.

# Based on NVIDIA best practices from Nsight Compute CLI docs

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
NCU_CONFIG_FILE=""
EXTRA_SECTIONS=()
LAUNCH_SKIP=200
LAUNCH_COUNT=1
FORCE_OVERWRITE="no"
LAUNCH_CMD=()

# Default sections (aligned with Python version), include RooflineChart for physical roofline support
DEFAULT_SECTIONS="SpeedOfLight MemoryWorkloadAnalysis Occupancy SchedulerStats SpeedOfLight_RooflineChart"
# Default metrics to ensure roofline-computable data even without external config
# These keep overhead modest while enabling FLOPs/s and AI computation in analysis.
DEFAULT_METRICS="flop_count_hp,flop_count_sp,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"

# --- Help ---
show_help() {
  cat <<'USAGE'
Usage: ncu-profile-kernels.sh [options] -- <launch-command> [launch-args]

Required (one of):
  --kernel-config <yaml-path>  Path to YAML file with kernel names/regex patterns
  --kernel-regex <regex>       Single regex pattern for kernel matching

Options:
  --output-dir <dir>           Directory for profiling results (default: tmp/ncu-profile/<timestamp>)
  --topk <num>                 Profile only top K kernels from YAML (requires --kernel-config)
  --ncu-config <yaml>          Optional NCU CLI config YAML to derive sections/preset
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
  ncu-profile-kernels.sh --kernel-regex 'gemvx::kernel<.*\(int\)7.*>' \
    --output-dir tmp/gemvx \
    -- python -m llm_perf_opt.runners.llm_profile_runner device=cuda:0

  # Profile multiple kernels from YAML
  ncu-profile-kernels.sh --kernel-config top-kernels.yaml \
    --extra-sections SourceCounters \
    -- python inference.py --model deepseek

  # Profile only top 3 kernels from YAML
  ncu-profile-kernels.sh --kernel-config top-kernels.yaml --topk 3 \
    -- python inference.py

Default sections: SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats, SpeedOfLight_RooflineChart
USAGE
}

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
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
    --ncu-config)
      [[ $# -lt 2 ]] && { log_err "--ncu-config requires a value"; exit 2; }
      NCU_CONFIG_FILE="$2"; shift 2;;
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

# Build sections list and optional preset from config (fallback to defaults)
ALL_SECTIONS="$DEFAULT_SECTIONS"
ALL_METRICS="$DEFAULT_METRICS"
NCU_PRESET=""

if [[ -n "$NCU_CONFIG_FILE" ]]; then
  if [[ -f "$NCU_CONFIG_FILE" ]]; then
    # Try to load 'ncu_cli.set' and 'ncu_cli.sections' using python (ruamel or pyyaml)
    if command -v python3 &>/dev/null || command -v python &>/dev/null; then
      PY_CMD=$(command -v python3 2>/dev/null || command -v python)
      CFG_OUT=$($PY_CMD - "$NCU_CONFIG_FILE" <<'PY'
import sys, json
try:
    from ruamel.yaml import YAML
    y = YAML(typ='safe')
    data = y.load(open(sys.argv[1]))
except Exception:
    import yaml
    data = yaml.safe_load(open(sys.argv[1]))
cfg = data.get('ncu_cli', data)
preset = cfg.get('set')
sections = cfg.get('sections') or []
metrics = cfg.get('metrics') or []
print(preset or "")
print(json.dumps(sections))
print(json.dumps(metrics))
PY
      )
      PRESET_LINE=$(printf "%s" "$CFG_OUT" | sed -n '1p')
      SECTIONS_JSON=$(printf "%s" "$CFG_OUT" | sed -n '2p')
      METRICS_JSON=$(printf "%s" "$CFG_OUT" | sed -n '3p')
      if [[ -n "$SECTIONS_JSON" && "$SECTIONS_JSON" != "null" ]]; then
        if command -v jq &>/dev/null; then
          CONFIG_SECTIONS=$(printf "%s" "$SECTIONS_JSON" | jq -r '.[]' | xargs)
        else
          CONFIG_SECTIONS=$(printf "%s" "$SECTIONS_JSON" | $PY_CMD -c 'import sys,json; print(" ".join(json.load(sys.stdin)))')
        fi
      fi
      if [[ -n "$METRICS_JSON" && "$METRICS_JSON" != "null" ]]; then
        if command -v jq &>/dev/null; then
          CONFIG_METRICS=$(printf "%s" "$METRICS_JSON" | jq -r '.[]' | paste -sd, -)
        else
          CONFIG_METRICS=$(printf "%s" "$METRICS_JSON" | $PY_CMD - <<'PY2'
import sys, json
arr = json.load(sys.stdin)
print(",".join(arr))
PY2
          )
        fi
      fi
      if [[ -n "${CONFIG_SECTIONS:-}" ]]; then
        ALL_SECTIONS="$CONFIG_SECTIONS"
      fi
      if [[ -n "${CONFIG_METRICS:-}" ]]; then
        ALL_METRICS="$CONFIG_METRICS"
      fi
      if [[ -n "$PRESET_LINE" ]]; then
        NCU_PRESET="$PRESET_LINE"
      fi
      log_info "Loaded NCU config: $(log_highlight "$NCU_CONFIG_FILE")"
    else
      log_warn "Python not found to parse --ncu-config; using hardcoded defaults"
    fi
  else
    log_warn "NCU config file not found: $NCU_CONFIG_FILE (using defaults)"
  fi
else
  log_info "No --ncu-config specified; using hardcoded defaults"
fi

# Always ensure RooflineChart section is present to support physical roofline exports
case " $ALL_SECTIONS " in
  *" SpeedOfLight_RooflineChart "*) : ;;  # present
  *) ALL_SECTIONS="$ALL_SECTIONS SpeedOfLight_RooflineChart" ;;
esac

if [[ ${#EXTRA_SECTIONS[@]} -gt 0 ]]; then
  ALL_SECTIONS="$ALL_SECTIONS ${EXTRA_SECTIONS[*]}"
fi

log_info "Output directory: $(log_highlight "$OUT_DIR")"
[[ -n "$NCU_PRESET" ]] && log_info "Preset: $NCU_PRESET"
log_info "Sections: $ALL_SECTIONS"
log_info "Metrics: ${ALL_METRICS:-<none>}"
log_info "Launch skip/count: $LAUNCH_SKIP/$LAUNCH_COUNT"

# --- Check tools ---
if ! command -v ncu &>/dev/null; then
  log_err "ncu command not found. Please ensure Nsight Compute is installed and in PATH."
  exit 2
fi

# Check for YAML parsing tools only if --kernel-config is used
if [[ -n "$KERNEL_CONFIG" ]]; then
  # Check for yq and determine which version
  if command -v yq &>/dev/null; then
    YQ_VERSION=$(yq --version 2>&1 || echo "")
    # Check if it's mikefarah's yq (Go-based) - look for "mikefarah" or version pattern like "v4.x.x"
    if echo "$YQ_VERSION" | grep -qiE "(mikefarah|version.*v[0-9]+\.[0-9]+\.[0-9]+)"; then
      YAML_PARSER="yq-go"
    else
      # It's the Python-based yq (jq wrapper) - not compatible with our syntax
      YAML_PARSER=""
    fi
  fi

  # Fall back to Python if yq is not available or wrong version
  if [[ -z "$YAML_PARSER" ]]; then
    if command -v python3 &>/dev/null || command -v python &>/dev/null; then
      YAML_PARSER="python"
      PYTHON_CMD=$(command -v python3 2>/dev/null || command -v python)
    else
      log_err "YAML parsing requires either 'yq' (Go version by mikefarah) or 'python3/python' to be installed"
      log_err "Install yq: https://github.com/mikefarah/yq"
      log_err "Or install Python 3 with ruamel.yaml or pyyaml"
      exit 2
    fi
  fi
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

  # Parse YAML and extract kernels to JSON
  if [[ "$YAML_PARSER" == "yq-go" ]]; then
    # Use yq (Go version by mikefarah) for parsing
    if [[ -n "$TOPK" ]]; then
      ORIGINAL_COUNT=$(yq eval '.kernels | length' "$KERNEL_CONFIG")
      yq eval ".kernels | .[0:${TOPK}]" -o=json "$KERNEL_CONFIG" > "$KERNELS_JSON"
      log_info "Limiting to top $TOPK kernel(s) from $ORIGINAL_COUNT total"
    else
      yq eval '.kernels' -o=json "$KERNEL_CONFIG" > "$KERNELS_JSON"
    fi
  else
    # Use Python for parsing
    $PYTHON_CMD - "$KERNEL_CONFIG" "$TOPK" > "$KERNELS_JSON" <<'PYTHON_SCRIPT'
import sys
import json
try:
    import ruamel.yaml
    yaml = ruamel.yaml.YAML(typ='safe')
except ImportError:
    try:
        import yaml
    except ImportError:
        print("ERROR: ruamel.yaml or pyyaml is required", file=sys.stderr)
        sys.exit(1)

yaml_path = sys.argv[1]
topk = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

with open(yaml_path) as f:
    if 'ruamel.yaml' in sys.modules:
        data = yaml.load(f)
    else:
        data = yaml.safe_load(f)

kernels = data.get('kernels', [])

if topk:
    topk_int = int(topk)
    original_count = len(kernels)
    kernels = kernels[:topk_int]
    print(f"# Limiting to top {topk_int} kernel(s) from {original_count} total", file=sys.stderr)

json.dump(kernels, sys.stdout)
PYTHON_SCRIPT
  fi

  if [[ $? -ne 0 ]]; then
    log_err "Failed to parse kernel config YAML"
    exit 2
  fi

  # Count kernels based on available tools
  if [[ "$YAML_PARSER" == "yq-go" ]]; then
    KERNEL_COUNT=$(echo "$(<"$KERNELS_JSON")" | jq 'length')
  else
    KERNEL_COUNT=$(echo "$(<"$KERNELS_JSON")" | $PYTHON_CMD -c "import sys, json; print(len(json.load(sys.stdin)))")
  fi
  log_info "Loaded $KERNEL_COUNT kernel(s) from config"

  KERNEL_SOURCE="yaml_config"
else
  # Single kernel mode
  log_info "Kernel regex: $(log_highlight "$KERNEL_REGEX")"
  echo '[{"name": "user_provided", "regex": "'"$KERNEL_REGEX"'", "description": "Single kernel profiling"}]' > "$KERNELS_JSON"
  KERNEL_COUNT=1
  KERNEL_SOURCE="single_regex"
fi

# --- Write provenance ---
PROV_YAML="$OUT_DIR/command.yaml"
cat > "$PROV_YAML" <<EOF
timestamp: '$(date -Iseconds)'
script: '$(basename "$0")'
version: 'v1-bash'
output_dir: '$OUT_DIR'
kernel_source: '$KERNEL_SOURCE'
launch_skip: $LAUNCH_SKIP
launch_count: $LAUNCH_COUNT
sections:
$(echo "$ALL_SECTIONS" | tr ' ' '\n' | sed 's/^/- /')
launch_command:
$(printf '%s\n' "${LAUNCH_CMD[@]}" | sed 's/^/- /')
EOF

if [[ -n "$KERNEL_REGEX" ]]; then
  echo "kernel_regex: '$KERNEL_REGEX'" >> "$PROV_YAML"
fi
if [[ -n "$KERNEL_CONFIG" ]]; then
  echo "kernel_config: '$KERNEL_CONFIG'" >> "$PROV_YAML"
fi
if [[ -n "$TOPK" ]]; then
  echo "topk: $TOPK" >> "$PROV_YAML"
fi

# Add tool versions
NCU_VERSION=$(ncu --version 2>&1 | head -1 || echo "unknown")
echo "ncu_version: '$NCU_VERSION'" >> "$PROV_YAML"
BASH_VERSION_INFO="$BASH_VERSION"
echo "bash_version: '$BASH_VERSION_INFO'" >> "$PROV_YAML"

log_ok "Provenance: $PROV_YAML"

# --- Profile each kernel ---
SUCCESS_COUNT=0
FAILED_COUNT=0

# Determine rank width for directory naming (min 4 digits)
RANK_WIDTH=${#KERNEL_COUNT}
if (( RANK_WIDTH < 4 )); then RANK_WIDTH=4; fi

# Helper: compute MD5 hex of a string (kernel name)
compute_md5() {
  local s="$1"
  if command -v md5sum >/dev/null 2>&1; then
    printf "%s" "$s" | md5sum | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    # macOS fallback
    printf "%s" "$s" | md5 | awk '{print $NF}'
  elif command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
    local PY_CMD
    PY_CMD=$(command -v python3 2>/dev/null || command -v python)
    printf "%s" "$s" | "$PY_CMD" -c 'import sys,hashlib; print(hashlib.md5(sys.stdin.buffer.read()).hexdigest())'
  else
    log_err "md5sum or python is required to compute MD5"
    exit 2
  fi
}

for i in $(seq 0 $((KERNEL_COUNT - 1))); do
  # Extract kernel info using available tools
  if command -v jq &>/dev/null; then
    # Use jq for JSON parsing (standard tool)
    KERNEL_NAME=$(jq -r ".[$i].name" < "$KERNELS_JSON")
    KERNEL_REGEX_CURRENT=$(jq -r ".[$i].regex" < "$KERNELS_JSON")
    KERNEL_DESC=$(jq -r ".[$i].description // empty" < "$KERNELS_JSON")
  elif [[ "$YAML_PARSER" == "python" ]]; then
    # Fallback to Python
    KERNEL_INFO=$($PYTHON_CMD -c "import sys, json; k=json.load(sys.stdin)[$i]; print(k['name']); print(k['regex']); print(k.get('description', ''))" < "$KERNELS_JSON")
    KERNEL_NAME=$(echo "$KERNEL_INFO" | sed -n '1p')
    KERNEL_REGEX_CURRENT=$(echo "$KERNEL_INFO" | sed -n '2p')
    KERNEL_DESC=$(echo "$KERNEL_INFO" | sed -n '3p')
  else
    log_err "JSON parsing requires 'jq' or 'python'. Please install jq."
    exit 2
  fi

  KERNEL_NUM=$((i + 1))

  log_info ""
  log_info "================================================================================"
  log_info "Profiling kernel $KERNEL_NUM/$KERNEL_COUNT"
  log_info "Name: $KERNEL_NAME"
  [[ -n "$KERNEL_DESC" ]] && log_info "Description: $KERNEL_DESC"
  log_info "Regex: $(log_highlight "$KERNEL_REGEX_CURRENT")"
  log_info "================================================================================"

  # Create output directory for this kernel (match Python naming):
  # kernel_<rank>_<md5-of-kernel-name>, rank padded to RANK_WIDTH (min 4)
  RANK_PADDED=$(printf "%0${RANK_WIDTH}d" "$KERNEL_NUM")
  NAME_MD5=$(compute_md5 "$KERNEL_NAME")
  KERNEL_DIR="$OUT_DIR/kernel_${RANK_PADDED}_${NAME_MD5}"
  mkdir -p "$KERNEL_DIR"

  OUTPUT_BASE="${KERNEL_DIR}/ncu"
  REPORT_PATH="${OUTPUT_BASE}.ncu-rep"

  # Build NCU command
  NCU_ARGS=(
    --kernel-name-base demangled
    --kernel-name "regex:$KERNEL_REGEX_CURRENT"
    --launch-skip "$LAUNCH_SKIP"
    --launch-count "$LAUNCH_COUNT"
    -o "$OUTPUT_BASE"
  )

  # Add preset if provided
  if [[ -n "$NCU_PRESET" ]]; then
    NCU_ARGS+=(--set "$NCU_PRESET")
  fi

  # Add sections
  for section in $ALL_SECTIONS; do
    NCU_ARGS+=(--section "$section")
  done

  # Add metrics (comma-separated) to guarantee availability for analysis (FLOPs/s & AI)
  if [[ -n "${ALL_METRICS:-}" ]]; then
    NCU_ARGS+=(--metrics "$ALL_METRICS")
  fi

  [[ "$FORCE_OVERWRITE" == "yes" ]] && NCU_ARGS+=(--force-overwrite)

  # Run NCU profiling
  log_info "Running: ncu ${NCU_ARGS[*]} ${LAUNCH_CMD[*]}"

  # Run ncu, capture exit code but don't exit script on failure (set -e is active)
  set +e
  ncu "${NCU_ARGS[@]}" "${LAUNCH_CMD[@]}"
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
        # For RooflineChart, include body details to expose numeric roofline data
        if [[ "$section" == "SpeedOfLight_RooflineChart" ]]; then
          if ncu --csv --section "$section" --print-details all --import "$REPORT_PATH" > "$SECTION_CSV" 2>/dev/null; then
            [[ -s "$SECTION_CSV" ]] && log_ok "Section CSV: $SECTION_CSV" || log_warn "RooflineChart has no numeric details (empty)"
          else
            log_warn "Could not export section with details: $section"
          fi
        else
          if ncu --csv --section "$section" --import "$REPORT_PATH" > "$SECTION_CSV" 2>/dev/null; then
            log_ok "Section CSV: $SECTION_CSV"
          else
            log_warn "Could not export section: $section"
          fi
        fi
      done

      # Export collected metrics, if any
      if [[ -n "${ALL_METRICS:-}" ]]; then
        METRICS_CSV="${OUTPUT_BASE}.metrics.csv"
        if ncu --csv --metrics "$ALL_METRICS" --import "$REPORT_PATH" > "$METRICS_CSV" 2>/dev/null; then
          [[ -s "$METRICS_CSV" ]] && log_ok "Metrics CSV: $METRICS_CSV" || log_warn "Metrics CSV empty"
        else
          log_warn "Could not export metrics CSV"
        fi
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

# --- Summary ---
log_info ""
log_info "================================================================================"
log_ok "Profiling complete: $SUCCESS_COUNT succeeded, $FAILED_COUNT failed"
log_ok "Results saved to: $OUT_DIR"
log_info "================================================================================"

if [[ $FAILED_COUNT -gt 0 ]]; then
  exit 1
fi
