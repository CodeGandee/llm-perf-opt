#!/usr/bin/env bash
set -euo pipefail

# Installs vLLM nightly from source against the currently active Pixi env's Torch.
# - Uses vLLM's helper to adopt the existing Torch install
# - Installs build requirements
# - Installs vLLM in editable mode without build isolation
# - Enables precompiled components to avoid local CUDA compilation

workdir="${INIT_CWD:-$(pwd)}"
repo_dir="${workdir}/tmp/vllm-nightly"

echo "[install-vllm-nightly] Cloning vLLM repo into: ${repo_dir}" >&2
rm -rf "${repo_dir}"
git clone --depth 1 https://github.com/vllm-project/vllm.git "${repo_dir}"

cd "${repo_dir}"

echo "[install-vllm-nightly] Adapting to existing torch installation" >&2
python python/use_existing_torch.py

echo "[install-vllm-nightly] Installing build requirements" >&2
uv pip install -r requirements/build.txt

echo "[install-vllm-nightly] Installing vLLM in editable mode (no build isolation)" >&2
export VLLM_USE_PRECOMPILED=1
uv pip install --no-build-isolation -e .

echo "[install-vllm-nightly] Done." >&2

