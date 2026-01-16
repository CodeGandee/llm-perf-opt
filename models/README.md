Models Directory

Stores model weights and tokenizers used by experiments.

## Setup

Model symlinks (e.g., `models/deepseek-ocr`) should be created using the model bootstrap. These symlinks are not tracked in git to avoid environment‑specific paths.

This directory may also contain per-model subdirectories that manage machine-local references via a `source-data` symlink (for models that are not stored as Hugging Face snapshots).

Options
- Run only model bootstrap:
  - `models/bootstrap.sh --yes` (uses `models/bootstrap.yaml`)
- Run all bootstraps from repo root:
  - `./bootstrap.sh --yes`

`models/bootstrap.yaml` defaults to using a Hugging Face snapshot root (`$HF_SNAPSHOTS_ROOT`) and a snapshot subdir. Adjust to your local clone or snapshot as needed.

## Options

- Symlinks (recommended for large assets):
  - `models/qwen2_5_7b -> /data/weights/qwen2_5_7b/`
  - `models/deepseek-ocr -> /path/to/your/deepseek-ocr/`
- Git submodules (only for small repos or LFS pointers):
  - Pin lightweight adapters or configs; avoid committing large binaries.

**Important**: Do not commit model symlinks or large binaries to this repo. Model paths are environment‑specific and should be configured locally.

## Managed references

- `deepseek-ocr` (symlink at `models/deepseek-ocr`): configured via `models/bootstrap.yaml`.
- `wan2.1-t2v-14b/` (per-ref directory): run `bash models/wan2.1-t2v-14b/bootstrap.sh` to create `models/wan2.1-t2v-14b/source-data`.
