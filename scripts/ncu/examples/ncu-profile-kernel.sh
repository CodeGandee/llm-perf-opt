#!/usr/bin/env bash
set -euo pipefail

# Nsight Compute kernel profiler helper
#
# Summary
# - Profiles a targeted CUDA kernel (by demangled name regex) from a Python workload.
# - Detailed mode (default) collects section pages (SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats)
#   for rich, section-based exports.
# - Fast mode collects a curated metric set for bound classification and cache/stall signals.
# - After profiling, the script exports CSVs (sections or details page) for downstream analysis.
#
# Requirements
# - `ncu` available on PATH (system Nsight Compute preferred)
# - Python with PyTorch installed (used for GPU SM arch preflight)
#
# Configurable CLI options
# - `--kernel-name-regex <regex>` : Demangled regex to match kernel names. If omitted, profiles all kernels.
# - `--output-dir <dir>`          : Directory to write outputs. Default: <workspace>/tmp/ncu-profile/<timestamp>
# - `--ncu-mode {fast|detailed}`  : Collection mode; default detailed
# - `--replay-mode {kernel|application}` : Replay strategy; default kernel (faster for Python/LLM)
# - `--kernel-id <id>`            : Kernel instance selector; default ':::1'
#
# Note: Coloring can be disabled via NO_COLOR env; all other config is via CLI only.

# --- simple color logger ----------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  CLR_RESET=$'\033[0m'
  CLR_BOLD=$'\033[1m'
  CLR_RED=$'\033[31m'
  CLR_GREEN=$'\033[32m'
  CLR_YELLOW=$'\033[33m'
  CLR_BLUE=$'\033[34m'
  CLR_MAGENTA=$'\033[35m'
  CLR_CYAN=$'\033[36m'
else
  CLR_RESET=""; CLR_BOLD=""; CLR_RED=""; CLR_GREEN=""; CLR_YELLOW=""; CLR_BLUE=""; CLR_MAGENTA=""; CLR_CYAN="";
fi
log_info()  { printf "%b[ncu]%b %s\n"  "$CLR_BOLD$CLR_BLUE" "$CLR_RESET" "$*"; }
log_note()  { printf "%b[ncu]%b %b%s%b\n" "$CLR_BOLD$CLR_BLUE" "$CLR_RESET" "$CLR_CYAN" "$*" "$CLR_RESET"; }
log_warn()  { printf "%b[ncu]%b %b%s%b\n" "$CLR_BOLD$CLR_YELLOW" "$CLR_RESET" "$CLR_YELLOW" "$*" "$CLR_RESET" 1>&2; }
log_ok()    { printf "%b[ncu]%b %b%s%b\n" "$CLR_BOLD$CLR_GREEN" "$CLR_RESET" "$CLR_GREEN" "$*" "$CLR_RESET"; }
log_err()   { printf "%b[ncu]%b %b%s%b\n" "$CLR_BOLD$CLR_RED" "$CLR_RESET" "$CLR_RED" "$*" "$CLR_RESET" 1>&2; }

# Defaults (CLI only)
KERNEL_REGEX=""
OUT_DIR=""
OUT_BASE=""
MODE="detailed"
REPLAY_MODE="kernel"
KERNEL_ID=":::1"
SKIP_DISCOVERY="no"

# Args parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-name-regex)
      [[ $# -lt 2 ]] && { log_err "--kernel-name-regex requires a value"; exit 2; }
      KERNEL_REGEX="$2"; shift 2;;
    --output-dir)
      [[ $# -lt 2 ]] && { log_err "--output-dir requires a value"; exit 2; }
      OUT_DIR="$2"; shift 2;;
    --ncu-mode)
      [[ $# -lt 2 ]] && { log_err "--ncu-mode requires a value (fast|detailed)"; exit 2; }
      case "$2" in fast|detailed) MODE="$2";; *) log_err "invalid --ncu-mode: $2"; exit 2;; esac
      shift 2;;
    --replay-mode)
      [[ $# -lt 2 ]] && { log_err "--replay-mode requires a value (kernel|application)"; exit 2; }
      case "$2" in kernel|application) REPLAY_MODE="$2";; *) log_err "invalid --replay-mode: $2"; exit 2;; esac
      shift 2;;
    --kernel-id)
      [[ $# -lt 2 ]] && { log_err "--kernel-id requires a value (e.g., :::1)"; exit 2; }
      KERNEL_ID="$2"; shift 2;;
    --skip-discovery)
      SKIP_DISCOVERY="yes"; shift;;
    -h|--help)
      cat <<USAGE
Usage: $(basename "$0") [--kernel-name-regex <regex>] [--output-dir <dir>] \\
                         [--ncu-mode fast|detailed] [--replay-mode kernel|application] \\
                         [--kernel-id <id>] [--skip-discovery]

Options:
  --skip-discovery  Skip the initial regex verification phase (faster but less safe)
USAGE
      exit 0;;
    --) shift; break;;
    *) log_warn "Unknown arg: $1 (ignored)"; shift;;
  esac
done

# Resolve paths and defaults
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
[[ -z "$OUT_DIR" ]] && OUT_DIR="${REPO_ROOT}/tmp/ncu-profile/$(date +%Y%m%d-%H%M%S)"
[[ -z "$OUT_BASE" ]] && OUT_BASE="${OUT_DIR}/ncu"
mkdir -p "$OUT_DIR"

# Context and tools
INSIDE_PIXI="no"
[[ -n "${PIXI_ENVIRONMENT_NAME:-}" || -n "${PIXI_PROJECT_ROOT:-}" ]] && INSIDE_PIXI="yes"
CUR_PY="$(command -v python || true)"
CUR_NCU="$(command -v ncu || true)"
[[ -z "$CUR_PY" ]] && { log_err "No 'python' found on PATH."; exit 2; }

log_info "inside_pixi=${INSIDE_PIXI}"
log_note "python: $CUR_PY"
log_note "ncu: ${CUR_NCU}"
log_note "output dir: ${OUT_DIR}"
if [[ -n "$KERNEL_REGEX" ]]; then
  printf "%b[ncu]%b kernel regex: %b%s%b\n" "$CLR_BOLD$CLR_BLUE" "$CLR_RESET" "$CLR_MAGENTA" "$KERNEL_REGEX" "$CLR_RESET"
else
  printf "%b[ncu]%b kernel regex: %b%s%b\n" "$CLR_BOLD$CLR_BLUE" "$CLR_RESET" "$CLR_MAGENTA" "<ALL>" "$CLR_RESET"
fi
printf "%b[ncu]%b report base: %b%s%b (final: %b%s.ncu-rep%b)\n" \
  "$CLR_BOLD$CLR_BLUE" "$CLR_RESET" "$CLR_GREEN" "$OUT_BASE" "$CLR_RESET" "$CLR_GREEN" "$OUT_BASE" "$CLR_RESET"

# Build argument arrays
NAME_FILTER_ARGS=()
if [[ -n "$KERNEL_REGEX" ]]; then
  # Use kernel-name regex and disable kernel renaming for consistent matching
  NAME_FILTER_ARGS=(--kernel-name-base demangled --kernel-name "regex:${KERNEL_REGEX}" --rename-kernels off)
fi
COLLECT_ARGS=()
if [[ "$MODE" == "detailed" ]]; then
  COLLECT_ARGS=(--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section SchedulerStats)
  log_note "mode: detailed; sections: SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats"
else
  METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
gpu__time_duration.sum"
  COLLECT_ARGS=(--metrics "$METRICS")
  log_note "mode: fast; metrics: ${METRICS}"
fi

# Gather torch/GPU info for provenance
GPU_NAME=""; SM_ARCH_STR=""; SM_CAP_STR=""; TORCH_VER=""; TORCH_CUDA_VER=""; BUILT_ARCHES_RAW=""
TORCH_INFO=$("${CUR_PY}" - <<'PY'
import sys
try:
    import torch
except Exception:
    print("TORCH_IMPORT_ERROR=1"); sys.exit(0)
if not torch.cuda.is_available():
    print("CUDA_AVAILABLE=0"); sys.exit(0)
dev = torch.cuda.current_device()
cap = torch.cuda.get_device_capability(dev)
name = torch.cuda.get_device_name(dev)
arch = f"sm_{cap[0]}{cap[1]}"
built = []
try:
    built = torch.cuda.get_arch_list()
except Exception:
    built = []
print("GPU_NAME=" + name)
print("SM_CAP=%d.%d" % cap)
print("SM_ARCH=" + arch)
print("TORCH_VERSION=" + torch.__version__)
print("TORCH_CUDA=" + str(getattr(torch.version, 'cuda', None)))
print("BUILT_ARCHES=" + ",".join(built))
PY
)
while IFS='=' read -r k v; do
  case "$k" in
    GPU_NAME) GPU_NAME="$v";;
    SM_CAP) SM_CAP_STR="$v";;
    SM_ARCH) SM_ARCH_STR="$v";;
    TORCH_VERSION) TORCH_VER="$v";;
    TORCH_CUDA) TORCH_CUDA_VER="$v";;
    BUILT_ARCHES) BUILT_ARCHES_RAW="$v";;
  esac
done <<< "$TORCH_INFO"

# Emit provenance file (command.yaml)
CMD_YAML="${OUT_DIR}/command.yaml"
{
  echo "timestamp: '$(date -Is)'"
  echo "script: '$(basename "$0")'"
  echo "output_dir: '${OUT_DIR}'"
  echo "report: '${OUT_BASE}.ncu-rep'"
  echo "ncu:"
  echo "  verbose: true"
  echo "  target_processes: 'all'"
  if [[ -n "$KERNEL_REGEX" ]]; then
    echo "  kernel_filter_regex: '$KERNEL_REGEX'"
  else
    echo "  kernel_filter_regex: '<ALL>'"
  fi
  echo "  replay_mode: '${REPLAY_MODE}'"
  echo "  kernel_id: '${KERNEL_ID}'"
  echo "  mode: '${MODE}'"
  if [[ "$MODE" == "detailed" ]]; then
    echo "  sections:"
    echo "    - SpeedOfLight"
    echo "    - MemoryWorkloadAnalysis"
    echo "    - Occupancy"
    echo "    - SchedulerStats"
  else
    echo "  metrics: '${METRICS}'"
  fi
  echo "env:"
  echo "  inside_pixi: '${INSIDE_PIXI}'"
  echo "  python: '${CUR_PY}'"
  echo "  ncu: '${CUR_NCU}'"
  echo "system:"
  echo "  gpu_name: '${GPU_NAME}'"
  echo "  sm_capability: '${SM_CAP_STR}'"
  echo "  sm_arch: '${SM_ARCH_STR}'"
  echo "  torch_version: '${TORCH_VER}'"
  echo "  torch_cuda: '${TORCH_CUDA_VER}'"
  echo "  torch_built_arches:"
  IFS=',' read -ra _arches <<< "$BUILT_ARCHES_RAW"
  for a in "${_arches[@]}"; do
    [[ -n "$a" ]] && echo "    - '$a'"
  done
  echo "target_command: |"
  echo "  python -m llm_perf_opt.runners.llm_profile_runner \\\n+    'hydra.run.dir=tmp/ncu-work/\${now:%Y%m%d-%H%M%S}' \\\n+    dataset.subset_filelist=datasets/omnidocbench/subsets/dev-3.txt \\\n+    device=cuda:0 \\\n+    pipeline.torch_profiler.enable=false \\\n+    pipeline.static_analysis.enable=false \\\n+    dataset.sampling.num_epochs=1 \\\n+    dataset.sampling.num_samples_per_epoch=1 \\\n+    dataset.sampling.randomize=false \\\n+    infer.max_new_tokens=64"
} > "$CMD_YAML"

# Optional discovery: ensure the filter matches only one kernel
if [[ -n "$KERNEL_REGEX" && "$SKIP_DISCOVERY" != "yes" ]]; then
  DISC_BASE="${OUT_BASE}.discover"
  log_note "discovery: verifying regex matches only the intended kernel"
  set -x
  ncu \
    --target-processes all \
    "${NAME_FILTER_ARGS[@]}" \
    --kernel-id "${KERNEL_ID}" \
    --replay-mode kernel \
    --metrics gpu__time_duration.sum \
    -o "${DISC_BASE}" \
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
  if ncu --import "${DISC_BASE}.ncu-rep" --csv --page details --name-base demangled > "${DISC_BASE}.details.csv" 2>/dev/null; then
    mapfile -t KLIST < <(awk -F, 'NR>1{print $1}' "${DISC_BASE}.details.csv" | sort -u)
    log_note "discovery: unique kernels captured: ${#KLIST[@]}"
    BAD=0
    for n in "${KLIST[@]}"; do
      if ! printf '%s' "$n" | grep -E -q "$KERNEL_REGEX"; then BAD=1; fi
    done
    if [[ ${#KLIST[@]} -eq 0 ]]; then
      log_err "discovery: no kernels matched regex; aborting to avoid profiling extras"; exit 3
    fi
    if [[ ${#KLIST[@]} -gt 1 || $BAD -ne 0 ]]; then
      log_err "discovery: regex is too broad or mismatched; refusing to continue"
      printf '%s\n' "${KLIST[@]}" | nl | sed 's/^/[discovery] /'
      exit 4
    fi
  else
    log_err "discovery: failed to export details; aborting"; exit 5
  fi
elif [[ -n "$KERNEL_REGEX" && "$SKIP_DISCOVERY" == "yes" ]]; then
  log_warn "discovery: SKIPPED (--skip-discovery enabled). Regex may match multiple kernels!"
fi

# SM arch preflight via PyTorch
log_info "Checking PyTorch CUDA arch support for this GPU (cuda:0)..."
"${CUR_PY}" - <<'PYTORCH_CHECK'
import sys
try:
    import torch
except Exception as e:
    print("[ncu] ERROR: Failed to import torch: %s" % (e,), file=sys.stderr)
    sys.exit(10)
if not torch.cuda.is_available():
    print("[ncu] ERROR: torch.cuda.is_available() is False.", file=sys.stderr)
    sys.exit(11)
dev = torch.cuda.current_device()
cap = torch.cuda.get_device_capability(dev)
arch = f"sm_{cap[0]}{cap[1]}"
name = torch.cuda.get_device_name(dev)
try:
    built_arches = torch.cuda.get_arch_list()
except Exception:
    built_arches = []
print(f"[ncu] torch.version: {torch.__version__}, cuda={getattr(torch.version,'cuda',None)}")
print(f"[ncu] gpu: {name}")
print(f"[ncu] device capability: {cap} → {arch}")
print(f"[ncu] PyTorch built arches: {built_arches}")
if arch not in built_arches:
    print("[ncu] ERROR: This PyTorch build does not include kernels for %s. "
          "Upgrade to a build that supports your GPU." % arch, file=sys.stderr)
    sys.exit(12)
sys.exit(0)
PYTORCH_CHECK

set -x
ncu \
  --verbose \
  --target-processes all \
  "${NAME_FILTER_ARGS[@]}" \
  --kernel-id "${KERNEL_ID}" \
  --replay-mode "${REPLAY_MODE}" \
  "${COLLECT_ARGS[@]}" \
  -o "${OUT_BASE}" \
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

# Post-processing: export section CSVs if present
REP="${OUT_BASE}.ncu-rep"
export_section() {
  local section="$1"; local out="${OUT_BASE}.section_${section}.csv"
  [[ ! -f "$REP" ]] && { log_warn "report not found: $REP"; return 0; }
  if ncu --import "$REP" --csv --section "$section" > "$out" 2>/dev/null; then
    log_ok "exported section '$section' → $out"
  else
    log_warn "section '$section' not available in report (or unsupported); skipping"
  fi
}
if [[ "$MODE" == "detailed" ]]; then
  export_section SpeedOfLight
  export_section MemoryWorkloadAnalysis
  export_section Occupancy
  export_section SchedulerStats
  export_section SpeedOfLight_RooflineChart
else
  export_section SpeedOfLight
  export_section MemoryWorkloadAnalysis
fi
if ncu --import "$REP" --csv --page details > "${OUT_BASE}.details.csv" 2>/dev/null; then
  log_ok "exported details page → ${OUT_BASE}.details.csv"
else
  log_warn "could not export details page; try ncu-ui or detailed mode"
fi

# Mark completion in provenance
echo "completed: true" >> "$CMD_YAML"
