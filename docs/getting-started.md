# Getting Started

Prerequisites
- GPU machine with CUDA 12.x (Pixi system requirement is set to 12.0)
- Model symlink (DeepSeek-OCR): use `models/bootstrap.sh --yes` or `./bootstrap.sh --yes`
  - Default config points to an HF snapshot root (`$HF_SNAPSHOTS_ROOT`). You can edit `models/bootstrap.yaml` to match your environment (local clone or snapshot hash).
- Dataset symlink: `datasets/omnidocbench/source-data -> /workspace/datasets/OpenDataLab___OmniDocBench`
  - Create via: `datasets/omnidocbench/bootstrap.sh --yes` (reads `datasets/omnidocbench/bootstrap.yaml`)
  - Or run all bootstraps: `./bootstrap.sh --yes`
  - The dataset bootstrap can also extract `images.zip` / `pdfs.zip` when present.

Environment
- This project uses Pixi for environment and tasks. Install Pixi and run tasks via `pixi run <task>`.
- Python requirements are declared in `pyproject.toml`.

Common tasks
- `stage1-run`: torch_profiler + static_analysis to `tmp/profile-output/<run_id>/`
- `stage2-profile`: Nsight Systems capture (no NCU) to the same dir
- `stage-all-run`: runs both into one run dir (no NCU)

Quick checks
```
# Lint and type (scoped to src/)
pixi run ruff check .
pixi run mypy

# Verify GPU PyTorch is present
pixi run python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())
PY
```

Model and dataset layout
- `models/deepseek-ocr` should be a local clone or HF snapshot; we call its `infer()` function for parity.
- `datasets/omnidocbench/source-data/images/*.png|*.jpg` contains sample pages.
 - Subset filelist can be relative to the repo root, e.g., `datasets/omnidocbench/subsets/dev-20.txt`.
