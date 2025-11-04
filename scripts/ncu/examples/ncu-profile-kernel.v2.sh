#!/usr/bin/env bash
set -euo pipefail

# Simplified Nsight Compute kernel profiler (v2)
#
# Based on NVIDIA best practices from nsight-compute CLI docs
# - Uses --launch-skip/--launch-count for sampling (no replay mode needed)
# - Uses --kernel-name-base demangled with regex patterns directly
# - Collects comprehensive sections (SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats)
#
# Usage:
#   ncu-profile-kernel.v2.sh --kernel-regex 'pattern' [options]

# --- Color logging ---
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  CLR_RESET=$'\033[0m'; CLR_BOLD=$'\033[1m'; CLR_RED=$'\033[31m'
  CLR_GREEN=$'\033[32m'; CLR_YELLOW=$'\033[33m'; CLR_BLUE=$'\033[34m'
  CLR_CYAN=$'\033[36m'
else
  CLR_RESET=""; CLR_BOLD=""; CLR_RED=""; CLR_GREEN=""; CLR_YELLOW=""; CLR_BLUE=""; CLR_CYAN=""
fi
log_info()  { printf "%b[ncu-v2]%b %s\n"  "$CLR_BOLD$CLR_BLUE" "$CLR_RESET" "$*"; }
log_ok()    { printf "%b[ncu-v2]%b %b%s%b\n" "$CLR_BOLD$CLR_GREEN" "$CLR_RESET" "$CLR_GREEN" "$*" "$CLR_RESET"; }
log_warn()  { printf "%b[ncu-v2]%b %b%s%b\n" "$CLR_BOLD$CLR_YELLOW" "$CLR_RESET" "$CLR_YELLOW" "$*" "$CLR_RESET" >&2; }
log_err()   { printf "%b[ncu-v2]%b %b%s%b\n" "$CLR_BOLD$CLR_RED" "$CLR_RESET" "$CLR_RED" "$*" "$CLR_RESET" >&2; }

# --- Defaults ---
KERNEL_REGEX=""
OUT_DIR=""
LAUNCH_SKIP=200
LAUNCH_COUNT=1
FORCE_OVERWRITE="no"

# --- Sections to collect (aligned with original script) ---
# Collect comprehensive section-based analysis:
# - SpeedOfLight: high-level compute/memory/L1/L2 utilization
# - MemoryWorkloadAnalysis: detailed memory subsystem breakdown
# - Occupancy: theoretical vs achieved occupancy
# - SchedulerStats: warp scheduling and stall analysis
SECTIONS="SpeedOfLight MemoryWorkloadAnalysis Occupancy SchedulerStats"

# --- Arg parsing ---
show_help() {
  cat <<USAGE
Usage: $(basename "$0") --kernel-regex <pattern> [options]

Required:
  --kernel-regex <pattern>   Regex pattern for kernel name (no 'regex:' prefix needed)

Options:
  --output-dir <dir>         Output directory (default: <repo>/tmp/ncu-profile-v2/<timestamp>)
  --launch-skip <n>          Skip first N kernel invocations (default: 200)
  --launch-count <n>         Profile N invocations (default: 1)
  --force-overwrite          Overwrite existing report files

Examples:
  # Profile gemvx (int)7 variant
  $(basename "$0") --kernel-regex 'internal::gemvx::kernel<.*\\(int\\)7.*>'

  # Profile with custom skip
  $(basename "$0") --kernel-regex 'unrolled_elementwise_kernel<.*direct_copy' --launch-skip 500
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-regex)
      [[ $# -lt 2 ]] && { log_err "--kernel-regex requires a value"; exit 2; }
      KERNEL_REGEX="$2"; shift 2;;
    --output-dir)
      [[ $# -lt 2 ]] && { log_err "--output-dir requires a value"; exit 2; }
      OUT_DIR="$2"; shift 2;;
    --launch-skip)
      [[ $# -lt 2 ]] && { log_err "--launch-skip requires a value"; exit 2; }
      LAUNCH_SKIP="$2"; shift 2;;
    --launch-count)
      [[ $# -lt 2 ]] && { log_err "--launch-count requires a value"; exit 2; }
      LAUNCH_COUNT="$2"; shift 2;;
    --force-overwrite)
      FORCE_OVERWRITE="yes"; shift;;
    -h|--help) show_help; exit 0;;
    --) shift; break;;
    *) log_warn "Unknown arg: $1 (ignored)"; shift;;
  esac
done

if [[ -z "$KERNEL_REGEX" ]]; then
  log_err "ERROR: --kernel-regex is required"
  show_help
  exit 2
fi

# --- Setup paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ -z "$OUT_DIR" ]] && OUT_DIR="${REPO_ROOT}/tmp/ncu-profile-v2/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

OUT_BASE="${OUT_DIR}/ncu"
REPORT="${OUT_BASE}.ncu-rep"

# --- Check tools ---
CUR_NCU="$(command -v ncu || true)"
CUR_PY="$(command -v python || true)"
[[ -z "$CUR_PY" ]] && { log_err "No 'python' found on PATH"; exit 2; }
[[ -z "$CUR_NCU" ]] && { log_err "No 'ncu' found on PATH"; exit 2; }

# --- Log config ---
log_info "Kernel regex: ${CLR_CYAN}${KERNEL_REGEX}${CLR_RESET}"
log_info "Output dir: ${OUT_DIR}"
log_info "Launch skip/count: ${LAUNCH_SKIP}/${LAUNCH_COUNT}"
log_info "Sections: ${SECTIONS}"
log_info "ncu: ${CUR_NCU}"
log_info "python: ${CUR_PY}"

# --- Write provenance ---
PROV_YAML="${OUT_DIR}/command.yaml"
cat > "$PROV_YAML" <<EOF
timestamp: '$(date -Is)'
script: '$(basename "$0")'
version: 'v2-simplified'
output_dir: '${OUT_DIR}'
report: '${REPORT}'
kernel_regex: '${KERNEL_REGEX}'
launch_skip: ${LAUNCH_SKIP}
launch_count: ${LAUNCH_COUNT}
sections: '${SECTIONS}'
python: '${CUR_PY}'
ncu: '${CUR_NCU}'
target_command: |
  python -m llm_perf_opt.runners.llm_profile_runner \\
    'hydra.run.dir=tmp/ncu-work/\${now:%Y%m%d-%H%M%S}' \\
    dataset.subset_filelist=datasets/omnidocbench/subsets/dev-3.txt \\
    device=cuda:0 \\
    pipeline.torch_profiler.enable=false \\
    pipeline.static_analysis.enable=false \\
    dataset.sampling.num_epochs=1 \\
    dataset.sampling.num_samples_per_epoch=1 \\
    dataset.sampling.randomize=false \\
    infer.max_new_tokens=64
EOF

# --- Build ncu arguments ---
NCU_ARGS=(
  --kernel-name-base demangled
  --kernel-name "regex:${KERNEL_REGEX}"
  --launch-skip "$LAUNCH_SKIP"
  --launch-count "$LAUNCH_COUNT"
  -o "$OUT_BASE"
)

# Add each section
for section in $SECTIONS; do
  NCU_ARGS+=(--section "$section")
done

[[ "$FORCE_OVERWRITE" == "yes" ]] && NCU_ARGS+=(--force-overwrite)

# --- Run ncu profiling ---
log_info "Starting profiling..."
set -x
ncu \
  "${NCU_ARGS[@]}" \
  python -m llm_perf_opt.runners.llm_profile_runner \
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

# --- Export CSV ---
if [[ -f "$REPORT" ]]; then
  log_ok "Profiling complete!"
  log_info "Exporting section CSVs..."

  # Export each section collected
  for section in $SECTIONS; do
    section_csv="${OUT_BASE}.section_${section}.csv"
    if ncu --csv --section "$section" --import "$REPORT" > "$section_csv" 2>/dev/null; then
      log_ok "Section CSV: $section_csv"
    else
      log_warn "Could not export section: $section"
    fi
  done

  # Also try SpeedOfLight_RooflineChart (useful visualization data)
  if ncu --csv --section SpeedOfLight_RooflineChart --import "$REPORT" > "${OUT_BASE}.section_SpeedOfLight_RooflineChart.csv" 2>/dev/null; then
    log_ok "Section CSV: ${OUT_BASE}.section_SpeedOfLight_RooflineChart.csv"
  fi

  # Export details page
  if ncu --csv --page details --import "$REPORT" > "${OUT_BASE}.details.csv" 2>/dev/null; then
    log_ok "Details CSV: ${OUT_BASE}.details.csv"
  else
    log_warn "Could not export details CSV"
  fi

  log_ok "Report: ${REPORT}"
  echo "completed: true" >> "$PROV_YAML"
else
  log_err "Profiling failed - no report generated"
  exit 1
fi
