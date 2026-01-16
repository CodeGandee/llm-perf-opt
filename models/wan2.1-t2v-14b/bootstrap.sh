#!/usr/bin/env bash
set -euo pipefail

require_cmd() {
  for c in "$@"; do
    command -v "$c" >/dev/null 2>&1 || { echo "missing required command: $c" >&2; exit 127; }
  done
}

require_cmd ln rm mkdir

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_DIR="${SCRIPT_DIR}"
SRC_LINK="${REF_DIR}/source-data"

MODEL_BASENAME="Wan2.1-T2V-14B"

if [[ -n "${WAN21_T2V_14B_PATH:-}" ]]; then
  TARGET="${WAN21_T2V_14B_PATH}"
else
  : "${LLM_MODELS_ROOT:?set LLM_MODELS_ROOT to the directory containing ${MODEL_BASENAME} (e.g., /data1/huangzhe/llm-models)}"
  TARGET="${LLM_MODELS_ROOT}/${MODEL_BASENAME}"
fi

echo "Bootstrapping external reference in ${REF_DIR} ..."
echo "Target: ${TARGET}"
echo "Link:   ${SRC_LINK} -> ${TARGET}"

if [[ ! -e "${TARGET}" ]]; then
  echo "Target does not exist: ${TARGET}" >&2
  exit 1
fi

mkdir -p "${REF_DIR}"
rm -rf "${SRC_LINK}"
ln -s -- "${TARGET}" "${SRC_LINK}"

echo "Done."

