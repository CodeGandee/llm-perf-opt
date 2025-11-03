#!/usr/bin/env bash
set -euo pipefail

# Model bootstrap for DeepSeek-OCR
# - Creates symlink: models/<repo_link_name> -> <resolved target dir>

require_cmd() { for c in "$@"; do command -v "$c" >/dev/null || { echo "missing: $c" >&2; exit 127; }; done; }
require_cmd yq realpath ln mkdir rm

ASSUME_YES=false
CFG_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=true; shift ;;
    -c|--config) CFG_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help)
      cat <<'USAGE'
Usage: models/bootstrap.sh [--yes] [--config <yaml>]

Actions:
  - Create or replace symlink: models/<name> -> <target model directory>

Config:
  - Default: models/bootstrap.yaml
  - Override: --config path/to/bootstrap.yaml
USAGE
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

RED=""; GREEN=""; YELLOW=""; BOLD=""; RESET=""
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_PATH="${CFG_OVERRIDE:-$SCRIPT_DIR/bootstrap.yaml}"

if [[ ! -f "$CFG_PATH" ]]; then
  echo "${RED}Config not found:${RESET} $CFG_PATH" >&2
  exit 1
fi

MODEL_ROOT_ENV=$(yq -r '.env.model_root_env' "$CFG_PATH")
DEFAULT_MODEL_ROOT=$(yq -r '.env.default_model_root' "$CFG_PATH")
SOURCE_SUBDIR=$(yq -r '.model.source_subdir' "$CFG_PATH")
REPO_LINK_NAME=$(yq -r '.model.repo_link_name' "$CFG_PATH")

set +u; ENV_MODEL_ROOT=${!MODEL_ROOT_ENV-}; set -u
BASE_DIR="${ENV_MODEL_ROOT:-$DEFAULT_MODEL_ROOT}"
TARGET_DIR="$BASE_DIR/$SOURCE_SUBDIR"
LINK_PATH="$SCRIPT_DIR/$REPO_LINK_NAME"

echo "${BOLD}Model bootstrap (DeepSeek-OCR)${RESET}"
echo "Base dir:     $BASE_DIR (env $MODEL_ROOT_ENV preferred)"
echo "Target dir:   $TARGET_DIR"
echo "Repo symlink: $LINK_PATH -> $TARGET_DIR"

confirm() {
  local prompt="$1"
  if $ASSUME_YES; then return 0; fi
  read -r -p "$prompt [y/N]: " ans || true
  [[ ${ans,,} == y* ]]
}

mkdir -p "$(dirname "$LINK_PATH")"
if [[ -e "$LINK_PATH" || -L "$LINK_PATH" ]]; then
  if confirm "Replace existing $(basename "$LINK_PATH")?"; then
    case "$LINK_PATH" in
      "$SCRIPT_DIR"/*) rm -rf -- "$LINK_PATH" ;;
      *) echo "Refuse to remove non-repo path: $LINK_PATH" >&2 ;;
    esac
  else
    echo "Skip updating symlink.";
  fi
fi

if [[ ! -e "$LINK_PATH" && ! -L "$LINK_PATH" ]]; then
  ln -s -- "$TARGET_DIR" "$LINK_PATH"
  echo "${GREEN}Linked${RESET}: $LINK_PATH -> $TARGET_DIR"
fi

# Optional: sanity check for required files
if yq -e '.model.required_files' "$CFG_PATH" >/dev/null 2>&1; then
  mapfile -t REQ < <(yq -r '.model.required_files[]' "$CFG_PATH")
  missing=()
  for f in "${REQ[@]}"; do
    [[ -e "$TARGET_DIR/$f" ]] || missing+=("$f")
  done
  if ((${#missing[@]})); then
    echo "${YELLOW}Warning:${RESET} missing required files in target: ${missing[*]}" >&2
  else
    echo "${GREEN}OK${RESET}: required files present"
  fi
fi

echo "${BOLD}Done${RESET}"

