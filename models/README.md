Models Directory

Stores model weights and tokenizers used by experiments.

Options:
- Symlinks (recommended for large assets):
  - `models/qwen2_5_7b -> /data/weights/qwen2_5_7b/`
- Git submodules (only for small repos or LFS pointers):
  - Pin lightweight adapters or configs; avoid committing large binaries.

Prefer symlinks for large weights. Do not commit large binaries to this repo.
