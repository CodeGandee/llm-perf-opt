#!/usr/bin/env bash
set -euo pipefail

# Nsight Compute: profile the top kernel using the Pixi rtx5090 Python directly
# (no `pixi run` wrapper around the target Python process).

# Configure the kernel filter (demangled name regex) and output base.
KERNEL_REGEX='.*gemvx.*'   # adjust if your top kernel differs
OUT_BASE="tmp/ncu_top1_gemvx_nopixi"

mkdir -p tmp

# Resolve repo root and the Python inside Pixi's rtx5090 env
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYBIN_CAND="$REPO_ROOT/.pixi/envs/rtx5090/bin/python"
if [[ -x "$PYBIN_CAND" ]]; then
  PYBIN="$PYBIN_CAND"
else
  # Fallback: discover via Pixi once (still does not wrap the target process)
  PYBIN="$(pixi run -e rtx5090 which python)"
fi

echo "[ncu] Using python: $PYBIN"
echo "[ncu] Kernel regex: $KERNEL_REGEX"
echo "[ncu] Report base: $OUT_BASE (final: ${OUT_BASE}.ncu-rep)"

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
  "$PYBIN" -m llm_perf_opt.runners.llm_profile_runner \
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

echo "[ncu] Done. Report at: ${OUT_BASE}.ncu-rep"

