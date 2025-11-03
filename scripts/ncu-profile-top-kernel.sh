#!/usr/bin/env bash
set -euo pipefail

# Nsight Compute helper. If running inside a Pixi shell, great; otherwise we
# warn and proceed with whatever `python`/`ncu` are on PATH.

# --- simple color logger ----------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  CLR_RESET=$'\033[0m'
  CLR_BOLD=$'\033[1m'
  CLR_DIM=$'\033[2m'
  CLR_RED=$'\033[31m'
  CLR_GREEN=$'\033[32m'
  CLR_YELLOW=$'\033[33m'
  CLR_BLUE=$'\033[34m'
  CLR_MAGENTA=$'\033[35m'
  CLR_CYAN=$'\033[36m'
else
  CLR_RESET=""; CLR_BOLD=""; CLR_DIM=""; CLR_RED=""; CLR_GREEN=""; CLR_YELLOW=""; CLR_BLUE=""; CLR_MAGENTA=""; CLR_CYAN="";
fi

log_info()  { printf "%b[ncu]%b %s\n"  "$CLR_BOLD$CLR_BLUE" "$CLR_RESET" "$*"; }
log_note()  { printf "%b[ncu]%b %b%s%b\n" "$CLR_BOLD$CLR_BLUE" "$CLR_RESET" "$CLR_CYAN" "$*" "$CLR_RESET"; }
log_warn()  { printf "%b[ncu]%b %b%s%b\n" "$CLR_BOLD$CLR_YELLOW" "$CLR_RESET" "$CLR_YELLOW" "$*" "$CLR_RESET" 1>&2; }
log_ok()    { printf "%b[ncu]%b %b%s%b\n" "$CLR_BOLD$CLR_GREEN" "$CLR_RESET" "$CLR_GREEN" "$*" "$CLR_RESET"; }
log_err()   { printf "%b[ncu]%b %b%s%b\n" "$CLR_BOLD$CLR_RED" "$CLR_RESET" "$CLR_RED" "$*" "$CLR_RESET" 1>&2; }

KERNEL_REGEX='.*gemvx.*'      # adjust to your hot kernel substring
OUT_BASE="tmp/ncu_top1_gemvx"

# Detect whether we are inside ANY pixi environment
INSIDE_PIXI="no"
if [[ -n "${PIXI_ENVIRONMENT_NAME:-}" ]] || [[ -n "${PIXI_PROJECT_ROOT:-}" ]]; then
  INSIDE_PIXI="yes"
fi

CUR_PY="$(command -v python || true)"
CUR_NCU="$(command -v ncu || true)"

if [[ -z "${CUR_PY}" ]]; then
  log_err "No 'python' found on PATH. Aborting."
  exit 2
fi

if [[ "$INSIDE_PIXI" != "yes" ]]; then
  log_warn "Not inside a Pixi shell; proceeding anyway."
  log_warn "Tip: enter env with: pixi shell -e rtx5090"
fi

mkdir -p tmp

log_info "inside_pixi=${INSIDE_PIXI}"
log_note "python: $CUR_PY"
log_note "ncu: ${CUR_NCU}"
printf "%b[ncu]%b kernel regex: %b%s%b\n" "$CLR_BOLD$CLR_BLUE" "$CLR_RESET" "$CLR_MAGENTA" "$KERNEL_REGEX" "$CLR_RESET"
printf "%b[ncu]%b report base: %b%s%b (final: %b%s.ncu-rep%b)\n" \
  "$CLR_BOLD$CLR_BLUE" "$CLR_RESET" "$CLR_GREEN" "$OUT_BASE" "$CLR_RESET" "$CLR_GREEN" "$OUT_BASE" "$CLR_RESET"

# Verify PyTorch supports current device SM (e.g., sm_120 for RTX 5090)
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
    print(
        "[ncu] ERROR: This PyTorch build does not include kernels for %s. "
        "Upgrade to a build that supports your GPU (e.g., sm_120 for RTX 5090)." % arch,
        file=sys.stderr,
    )
    sys.exit(12)
sys.exit(0)
PYTORCH_CHECK

set -x
ncu \
  --verbose \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name "${KERNEL_REGEX}" \
  --launch-count 1 \
  --kill 1 \
  --check-exit-code 0 \
  --profile-from-start on \
  --section SpeedOfLight \
  --replay-mode application \
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
