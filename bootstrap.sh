#!/usr/bin/env bash
set -euo pipefail

# Workspace-level bootstrap that forwards to the config-driven symlink setup.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$ROOT_DIR/scripts/bootstrap-symlinks.sh"

if [[ ! -x "$SCRIPT" ]]; then
  echo "Error: missing or non-executable $SCRIPT" >&2
  echo "Hint: ensure the repo is intact and run: chmod +x scripts/bootstrap-symlinks.sh" >&2
  exit 1
fi

exec "$SCRIPT" "$@"

