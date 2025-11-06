#!/usr/bin/env bash
set -euo pipefail

# Minimal Nsight Compute profiling script for the dummy CUDA workload.
# - Profiles explore/ncu-simple/run-dummy-model.py directly (no runners)
# - Hard-coded settings based on conf/profiling/ncu/ncu.default.yaml
# - Captures kernels within the NVTX range "residual" via range replay
# - Outputs artifacts to tmp/explore/ncu-simple/<ts>

usage() {
  cat <<EOF
Usage: $0 [-i ITERS] [-b BATCH] [-l LOG_INTERVAL]

Options:
  -i ITERS         Number of forward iterations (default: 100)
  -b BATCH         Batch size (default: 1)  [1024x1024 inputs; adjust to avoid OOM]
  -l LOG_INTERVAL  Progress log interval (default: 10)

Examples:
  $0 -i 100 -b 1
  $0 -i 50 -b 2 -l 5

Then inspect: tmp/explore/ncu-simple/<ts>/{run.ncu-rep, raw.csv}
EOF
}

ITERS=100
BATCH=1
LOG_INTERVAL=10
# Limit launch set (skip first 2, then profile 5) per best practices
LAUNCH_SKIP=2
LAUNCH_COUNT=5
while getopts ":i:b:l:h" opt; do
  case "$opt" in
    i) ITERS="$OPTARG" ;;
    b) BATCH="$OPTARG" ;;
    l) LOG_INTERVAL="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

# Resolve repo root from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Timestamped output dir
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="tmp/explore/ncu-simple/${TS}"
mkdir -p "${OUT_DIR}"

# Sanity check: ncu presence
if ! command -v ncu >/dev/null 2>&1; then
  echo "Error: 'ncu' not found in PATH. Ensure Nsight Compute CLI is installed." >&2
  exit 1
fi

echo "[ncu] Output dir: ${OUT_DIR}"
echo "[ncu] Iterations: ${ITERS} | Batch size: ${BATCH} | Log interval: ${LOG_INTERVAL}"
echo "[ncu] NVTX ranges: stem/, residual/, head/, block.conv1/, block.conv2/ (app-range)"

# Hard-coded Nsight Compute settings mirroring conf/profiling/ncu/ncu.default.yaml
SET_NAME="roofline"
KERNEL_NAME_BASE="demangled"
# Use a portable subset of metrics to avoid device-specific failures
# (the full preset includes flop_count_hp/sp which may be unavailable).
METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"
SECTIONS=(SpeedOfLight MemoryWorkloadAnalysis Occupancy SchedulerStats)

# Build ncu command
NCU_OUT_BASE="${OUT_DIR}/run"

NCU_CMD=(
  ncu
  --target-processes all
  --set "${SET_NAME}"
  --replay-mode app-range
  --nvtx
  --nvtx-include stem/
  --nvtx-include residual/
  --nvtx-include head/
  --nvtx-include block.conv1/
  --nvtx-include block.conv2/
  --kernel-name-base "${KERNEL_NAME_BASE}"
  --metrics "${METRICS}"
  -o "${NCU_OUT_BASE}"
  --force-overwrite
  --launch-skip "${LAUNCH_SKIP}"
  --launch-count "${LAUNCH_COUNT}"
)

# Add sections
for sec in "${SECTIONS[@]}"; do
  NCU_CMD+=( --section "$sec" )
done

# Workload command (Python script)
PY_CMD=(
  python explore/ncu-simple/run-dummy-model.py
  --iters "${ITERS}"
  --batch-size "${BATCH}"
  --log-interval "${LOG_INTERVAL}"
)

echo "[ncu] Running: ${NCU_CMD[*]} -- ${PY_CMD[*]}"
"${NCU_CMD[@]}" -- "${PY_CMD[@]}" 2>"${OUT_DIR}/ncu-stderr.txt" || true

# Export raw CSV from the generated report
if [[ -f "${NCU_OUT_BASE}.ncu-rep" ]]; then
  echo "[ncu] Exporting raw CSV"
  ncu --import "${NCU_OUT_BASE}.ncu-rep" \
      --page raw \
      --csv \
      > "${OUT_DIR}/raw.csv" 2>>"${OUT_DIR}/ncu-import-stderr.txt" || true
fi

echo "[ncu] Exporting sections to text (human-readable)"
if [[ -f "${NCU_OUT_BASE}.ncu-rep" ]]; then
  ncu --import "${NCU_OUT_BASE}.ncu-rep" \
      --page raw \
      --section SpeedOfLight \
      --section MemoryWorkloadAnalysis \
      --section Occupancy \
      --section SchedulerStats \
      > "${OUT_DIR}/sections.txt" 2>"${OUT_DIR}/ncu-import-stderr.txt" || true
fi

echo "[ncu] Done. Artifacts:"
echo "  - ${NCU_OUT_BASE}.ncu-rep (open with Nsight Compute UI)"
echo "  - ${OUT_DIR}/sections.txt (human-readable sections)"
echo "  - ${OUT_DIR}/ncu-stderr.txt (capture log)"
echo "  - ${OUT_DIR}/ncu-import-stderr.txt (import log)"
ls -lah "${OUT_DIR}" || true
