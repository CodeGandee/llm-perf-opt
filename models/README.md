Models Directory

Stores model weights and tokenizers used by experiments.

## Setup

Model symlinks (e.g., `models/deepseek-ocr`) should be created by developers on their host using `bootstrap.sh`. These symlinks are not tracked in git to avoid environment-specific paths.

Run the bootstrap script from the repository root:
```bash
./bootstrap.sh
```

This will create the necessary symlinks to your local model storage based on your environment configuration.

## Options

- Symlinks (recommended for large assets):
  - `models/qwen2_5_7b -> /data/weights/qwen2_5_7b/`
  - `models/deepseek-ocr -> /path/to/your/deepseek-ocr/`
- Git submodules (only for small repos or LFS pointers):
  - Pin lightweight adapters or configs; avoid committing large binaries.

**Important**: Do not commit model symlinks or large binaries to this repo. Model paths are environment-specific and should be configured locally.
