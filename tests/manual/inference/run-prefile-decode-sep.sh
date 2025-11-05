#!/usr/bin/env bash
# Run DeepSeek-OCR manual prefill+decode (separate) on one image.
#
# Usage:
#   tests/manual/inference/run-prefile-decode-sep.sh [<relative-image-path>] [--max-new-tokens N]
#
# Notes:
# - The Python runner always uses the vendor grounding prompt, so no prompt is required.
# - Paths should be relative to the workspace root (this repository).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Default image (relative to repo root) if not provided by the user
DEFAULT_IMAGE_REL="datasets/omnidocbench/source-data/images/color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点四年级英语上册_课本_人教新起点英语4A电子课本_page_072.png"

# Optional first arg: image path (must not start with --)
IMAGE_REL=""
if [[ $# -ge 1 && "$1" != --* ]]; then
  IMAGE_REL="$1"; shift || true
else
  IMAGE_REL="${DEFAULT_IMAGE_REL}"
fi

# Optional: collect passthrough args for Python
PY_EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-new-tokens)
      shift
      if [[ -n "${1:-}" ]]; then
        PY_EXTRA_ARGS+=("--max_new_tokens" "${1}")
        shift || true
      fi
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

IMAGE_PATH="${REPO_ROOT}/${IMAGE_REL}"
if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Image not found: ${IMAGE_REL} (resolved: ${IMAGE_PATH})" >&2
  exit 2
fi

cd "${REPO_ROOT}"
if [[ ${#PY_EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Running manual prefill+decode on: ${IMAGE_REL} (${PY_EXTRA_ARGS[*]})"
else
  echo "Running manual prefill+decode on: ${IMAGE_REL} (using Python defaults)"
fi

# Detect python in PATH; prefer $PYTHON if provided
PY_BIN="${PYTHON:-python}"
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
  else
    echo "No python found in PATH. Please run via 'pixi run -e <env> ...' or set PYTHON env var." >&2
    exit 3
  fi
fi

# Heuristic: warn if not running inside a pixi environment
PY_EXE_PATH="$(command -v "${PY_BIN}")"
if [[ "${PY_EXE_PATH}" != *"/.pixi/envs/"* ]]; then
  echo "[warn] This does not look like a Pixi environment (python=${PY_EXE_PATH})." >&2
  echo "       Recommended: pixi run -e rtx5090 $0 ${IMAGE_REL} [--max-new-tokens N]" >&2
fi

exec "${PY_BIN}" tests/manual/inference/manual_dsocr_prefill_decode.py \
  --image "${IMAGE_REL}" \
  "${PY_EXTRA_ARGS[@]}"
