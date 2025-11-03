#!/usr/bin/env bash
set -euo pipefail

# Bootstrap symlink creator following magic-context/instructions/snippets/make-boostrap-symlink-script.md

require_cmd() { for c in "$@"; do command -v "$c" >/dev/null || { echo "missing: $c" >&2; exit 127; }; done; }
require_cmd yq realpath ln mkdir

ASSUME_YES=false
STRICT_FAIL=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=true; shift;;
    --strict) STRICT_FAIL=true; shift;;
    -h|--help) echo "usage: $0 [--yes] [--strict]"; exit 0;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

RED=""; GREEN=""; YELLOW=""; BLUE=""; BOLD=""; DIM=""; RESET=""
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK="$SCRIPT_DIR/data-pack.yaml"

DATA_ROOT_ENV=$(yq -r '.env.data_root_env' "$PACK")
DEFAULT_DATA_ROOT=$(yq -r '.env.default_data_root' "$PACK")

set +u; ENV_DATA_ROOT=${!DATA_ROOT_ENV-}; set -u
SUGGEST_ROOT="${ENV_DATA_ROOT:-$DEFAULT_DATA_ROOT}"
echo "Using data root: ${BOLD}${SUGGEST_ROOT}${RESET} (env ${DATA_ROOT_ENV} preferred)"

print_checks() {
  local base="$1" srcdir="$2"; shift 2
  local req=("$@")
  local dir_raw="${base%/}/$srcdir"
  local dir_abs="$(realpath -m -- "$dir_raw")"
  if [[ -d "$dir_raw" ]]; then
    echo "  DIR  $dir_abs  [${GREEN}OK${RESET}]"
    for f in "${req[@]}"; do
      # Depth=1 existence only; ok for files or dirs; ok if symlink
      local p="$dir_raw/$f"; local p_abs="$(realpath -m -- "$p")"
      [[ -e "$p" ]] && echo "  NAME $p_abs  [${GREEN}OK${RESET}]" \
                     || echo "  NAME $p_abs  [${RED}MISSING${RESET}]"
    done
  else
    echo "  DIR  $dir_abs  [${RED}MISSING${RESET}]"
  fi
}

detected_path=""
detect_valid() {
  local base="$1" srcdir="$2"; shift 2
  local req=("$@")
  local dir_raw="${base%/}/$srcdir"
  [[ -d "$dir_raw" ]] || return 1
  # Validate required names exist at depth=1
  for f in "${req[@]}"; do [[ -e "$dir_raw/$f" ]] || return 1; done
  detected_path="$dir_raw"; return 0
}

confirm_overwrite() {
  local path="$1"
  if $ASSUME_YES; then return 0; fi
  read -r -p "Overwrite $path? (y/N): " ans || true
  [[ ${ans,,} == y* ]]
}

read_one_item() {
  local id="$1" srcdir="$2" linkpath="$3" processed="$4"; shift 4
  local req=("$@")

  echo "\n${BOLD}=== $id ===${RESET}"
  echo "Candidates:"
  if [[ -n "${ENV_DATA_ROOT:-}" ]]; then
    print_checks "$ENV_DATA_ROOT" "$srcdir" "${req[@]}"
  else
    echo "  ${DATA_ROOT_ENV} is unset"
  fi
  print_checks "$DEFAULT_DATA_ROOT" "$srcdir" "${req[@]}"

  detected_path=""; if [[ -n "${ENV_DATA_ROOT:-}" ]]; then detect_valid "$ENV_DATA_ROOT" "$srcdir" "${req[@]}" || true; fi
  if [[ -z "$detected_path" ]]; then detect_valid "$DEFAULT_DATA_ROOT" "$srcdir" "${req[@]}" || true; fi

  echo "Expected: $srcdir/ contains required names (depth=1 only)"
  for f in "${req[@]}"; do echo "  - $f (required)"; done

  local chosen=""
  if $ASSUME_YES; then
    if [[ -n "$detected_path" ]]; then
      chosen=$(realpath -m -- "$detected_path"); echo "Using detected: $chosen"
    else
      if $STRICT_FAIL; then
        echo "ERROR: no detected path for $id" >&2; exit 3
      else
        echo "Skip $id"; return
      fi
    fi
  else
    while true; do
      echo "Enter absolute path (Enter=accept detected; blank=skip if none):"; read -r -p "Path: " inp || true
      local trim="$(printf '%s' "$inp" | sed 's/^\s\+//;s/\s\+$//')"
      if [[ -z "$trim" ]]; then
        if [[ -n "$detected_path" ]]; then chosen=$(realpath -m -- "$detected_path"); echo "Using detected: $chosen"; break; else echo "Skip $id"; return; fi
      else
        chosen=$(realpath -m -- "$trim")
      fi
      [[ -d "$chosen" ]] || { echo "Not a directory: $chosen"; continue; }
      # Validate required names exist at depth=1
      local miss=(); for f in "${req[@]}"; do [[ -e "$chosen/$f" ]] || miss+=("$f"); done
      (( ${#miss[@]} == 0 )) || { echo "Missing required at depth=1: ${miss[*]}"; continue; }
      break
    done
  fi

  mkdir -p "$(dirname "$linkpath")"
  if [[ -e "$linkpath" || -L "$linkpath" ]]; then
    if confirm_overwrite "$linkpath"; then rm -rf -- "$linkpath"; else echo "Skip overwrite: $linkpath"; return; fi
  fi
  ln -s -- "$chosen" "$linkpath"; echo "Linked: $linkpath -> $chosen"
  mkdir -p -- "$processed"; echo "Ensured processed dir: $processed"
}

COUNT=$(yq -r '.data_items | length' "$PACK")
for ((i=0; i<COUNT; i++)); do
  id=$(yq -r ".data_items[$i].id" "$PACK")
  src=$(yq -r ".data_items[$i].source_dir" "$PACK")
  link=$(yq -r ".data_items[$i].repo_link_path" "$PACK")
  proc=$(yq -r ".data_items[$i].procssed_dir" "$PACK")
  # Load required names (may be empty)
  if yq -e ".data_items[$i].required_files" "$PACK" >/dev/null 2>&1; then
    mapfile -t req < <(yq -r ".data_items[$i].required_files[]" "$PACK")
  else
    req=()
  fi
  read_one_item "$id" "$src" "$link" "$proc" "${req[@]}"
done
