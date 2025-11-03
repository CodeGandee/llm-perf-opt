#!/usr/bin/env bash
set -euo pipefail

# Workspace-level bootstrap. Runs dataset + model bootstraps.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) Dataset: OmniDocBench
DATASET_BOOT="$ROOT_DIR/datasets/omnidocbench/bootstrap.sh"
if [[ -x "$DATASET_BOOT" ]]; then
  echo "[bootstrap] OmniDocBench dataset setup..."
  "$DATASET_BOOT" "$@"
else
  echo "[bootstrap] Skip dataset setup (not found): $DATASET_BOOT" >&2
fi

# 2) Models: DeepSeek-OCR
MODEL_BOOT="$ROOT_DIR/models/bootstrap.sh"
if [[ -x "$MODEL_BOOT" ]]; then
  echo "[bootstrap] Models setup..."
  "$MODEL_BOOT" "$@"
else
  echo "[bootstrap] Skip models setup (not found): $MODEL_BOOT" >&2
fi

echo "[bootstrap] Completed"
