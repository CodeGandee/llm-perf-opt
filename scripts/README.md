Scripts

Utility scripts for creating symlinks, preparing datasets, taking snapshots of
reference code, or quick sanity runs.

Avoid embedding heavy logic here; keep scripts small and focused.

## Environment Setup

### RTX 5090 (Blackwell/sm_120)
- **Install dependencies**: `./scripts/install-deps-rtx5090.sh`
  - Installs PyTorch nightly with CUDA 12.8
  - Builds Triton and Flash Attention with sm_120 support
  - Takes 20-40 minutes (mostly building Flash Attention)
- **Verify setup**: `./scripts/verify-rtx5090.sh`
  - Checks PyTorch, CUDA, Triton, and Flash Attention installation
  - Tests GPU tensor operations

See `context/hints/howto-use-rtx5090-environment.md` for detailed documentation.

## Symlinks
- Models (DeepSeek-OCR): `models/bootstrap.sh --yes` or run `./bootstrap.sh`
- Datasets (OmniDocBench): `datasets/omnidocbench/bootstrap.sh --yes` or run `./bootstrap.sh`
