#!/usr/bin/env bash
set -euo pipefail

# Nsight Compute via Pixi shell (rtx5090): profile a top kernel without using `pixi run`.

KERNEL_REGEX='.*gemvx.*'      # adjust to your hot kernel substring
OUT_BASE="tmp/ncu_top1_gemvx_pixi_shell"

mkdir -p tmp

echo "[ncu] (pixi shell) kernel regex: ${KERNEL_REGEX}"
echo "[ncu] (pixi shell) report base: ${OUT_BASE} (final: ${OUT_BASE}.ncu-rep)"

# Enter the rtx5090 environment shell and run ncu there.
# Note: keep the Hydra now-expression single-quoted so the shell doesn't expand it.
pixi shell -e rtx5090 <<'EOS'
set -euo pipefail
ncu \
  --verbose \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name ".*gemvx.*" \
  --launch-count 1 \
  --kill 1 \
  --check-exit-code 0 \
  --profile-from-start on \
  --section SpeedOfLight \
  --replay-mode application \
  -o tmp/ncu_top1_gemvx_pixi_shell \
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
EOS

echo "[ncu] (pixi shell) Done. Report at: ${OUT_BASE}.ncu-rep"

