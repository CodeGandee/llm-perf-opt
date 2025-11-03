#!/usr/bin/env bash
set -euo pipefail

# Dataset-specific bootstrap for OmniDocBench.
# - Creates repo symlink: ./source-data -> <resolved target dir>
# - Optionally extracts known zip archives (images.zip, pdfs.zip) in the target dir
#   after user confirmation.

require_cmd() { for c in "$@"; do command -v "$c" >/dev/null || { echo "missing: $c" >&2; exit 127; }; done; }
require_cmd yq realpath ln mkdir rm

# unzip is optional; only required if we choose to extract zips
if command -v unzip >/dev/null; then HAVE_UNZIP=true; else HAVE_UNZIP=false; fi

ASSUME_YES=false
CFG_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=true; shift;;
    -c|--config) CFG_OVERRIDE="${2:-}"; shift 2;;
    -h|--help)
      cat <<'USAGE'
Usage: datasets/omnidocbench/bootstrap.sh [--yes] [--config <yaml>]

Actions:
  - Create symlink: source-data -> <target dataset directory>
  - Detect images.zip / pdfs.zip in target and prompt to extract

Config:
  - Default config: datasets/omnidocbench/bootstrap.yaml
  - Override: --config path/to/bootstrap.yaml
USAGE
      exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
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

DATA_ROOT_ENV=$(yq -r '.env.data_root_env' "$CFG_PATH")
DEFAULT_DATA_ROOT=$(yq -r '.env.default_data_root' "$CFG_PATH")
SOURCE_SUBDIR=$(yq -r '.dataset.source_subdir' "$CFG_PATH")
REPO_LINK_NAME=$(yq -r '.dataset.repo_link_name' "$CFG_PATH")

set +u; ENV_DATA_ROOT=${!DATA_ROOT_ENV-}; set -u
BASE_DIR="${ENV_DATA_ROOT:-$DEFAULT_DATA_ROOT}"
TARGET_DIR="${BASE_DIR%/}/$SOURCE_SUBDIR"
# Construct link path without dereferencing existing links
LINK_PATH="$SCRIPT_DIR/$REPO_LINK_NAME"

echo "${BOLD}OmniDocBench bootstrap${RESET}"
echo "Base dir:     $BASE_DIR (env $DATA_ROOT_ENV preferred)"
echo "Target dir:   $TARGET_DIR"
echo "Repo symlink: $LINK_PATH -> $TARGET_DIR"

confirm() {
  local prompt="$1"
  if $ASSUME_YES; then return 0; fi
  read -r -p "$prompt [y/N]: " ans || true
  [[ ${ans,,} == y* ]]
}

# Create or update the symlink
mkdir -p "$(dirname "$LINK_PATH")"
if [[ -e "$LINK_PATH" || -L "$LINK_PATH" ]]; then
  if confirm "Replace existing $(basename "$LINK_PATH")?"; then
    # Safety guard: only remove if path resides under dataset folder
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

# Zip extraction (optional)
mapfile -t ZIP_LIST < <(yq -r '.dataset.zips[]?' "$CFG_PATH" 2>/dev/null || true)
if ((${#ZIP_LIST[@]} == 0)); then
  ZIP_LIST=()
fi

if ((${#ZIP_LIST[@]})); then
  if ! $HAVE_UNZIP; then
    echo "${YELLOW}unzip not found; skip zip extraction${RESET}"
  else
    for zipname in "${ZIP_LIST[@]}"; do
      # Get target subdir from map; default to basename w/o .zip
      subdir=$(yq -r --arg z "$zipname" '.dataset.unzip_targets[$z] // ""' "$CFG_PATH" 2>/dev/null || true)
      if [[ -z "$subdir" ]]; then
        subdir="${zipname%.zip}"
      fi
      extract_dir="$TARGET_DIR/$subdir"
      zip_path="$TARGET_DIR/$zipname"

      # Decide if extraction is needed (only if target dir is missing or empty)
      need_extract=false
      if [[ ! -d "$extract_dir" ]]; then
        need_extract=true
      else
        if [[ -z "$(find "$extract_dir" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
          need_extract=true
        fi
      fi

      if ! $need_extract; then
        # Data already present; no zip needed and no message about missing zips
        continue
      fi

      if [[ -f "$zip_path" ]]; then
        if confirm "Extract $(basename "$zip_path") in $TARGET_DIR?"; then
          unzip -q -o "$zip_path" -d "$TARGET_DIR"
          echo "${GREEN}Extracted${RESET}: $zip_path in $TARGET_DIR"
        else
          echo "Skip extraction: $zipname"
        fi
      else
        echo "${YELLOW}Missing zip${RESET}: $zip_path (skip)"
      fi
    done
  fi
fi

echo "${BOLD}Done${RESET}"
