Scripts

Utility scripts for creating symlinks, preparing datasets, taking snapshots of
reference code, or quick sanity runs.

Added:
- `bootstrap-symlinks.sh` — config-driven symlink bootstrapper. Creates repo symlinks
  (e.g., `models/deepseek-ocr`) pointing to external data roots defined in `scripts/data-pack.yaml`.
  - Interactive: `bash scripts/bootstrap-symlinks.sh`
  - Non-interactive: `bash scripts/bootstrap-symlinks.sh --yes`
  - Strict fail when missing: `bash scripts/bootstrap-symlinks.sh --yes --strict`
  - Uses env var `HF_SNAPSHOTS_ROOT` if set; otherwise defaults to the `default_data_root` in YAML.
  - Validation checks the top-level target directory exists and that listed required names exist directly under it (depth=1). Symlink vs. regular file/dir does not matter.

Avoid embedding heavy logic here; keep scripts small and focused.
