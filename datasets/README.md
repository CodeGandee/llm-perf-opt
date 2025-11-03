Datasets Directory

Organizes datasets with a `source-data` symlink and optional variant subfolders
(e.g., `subset-1k/`, `tokenized-gpt2/`). Include per-dataset `metadata.yaml`
and `README.md` to document provenance and transformations.

Large data should live outside the repo and be linked via symlinks.

Bootstrap
- OmniDocBench: `datasets/omnidocbench/bootstrap.sh --yes` (config: `datasets/omnidocbench/bootstrap.yaml`)
- Or run all from repo root: `./bootstrap.sh --yes`
- Set `$DATASETS_ROOT` to override the default base directory.
