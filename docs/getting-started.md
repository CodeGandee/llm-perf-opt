# Getting Started

Prerequisites
- GPU machine with CUDA 12.x (Pixi system requirement is set to 12.0)
- Local model repo symlink: `models/deepseek-ocr -> /data2/huangzhe/code/DeepSeek-OCR`
- Dataset: `datasets/omnidocbench/source-data/` with images under `images/`

Environment
- This project uses Pixi for environment and tasks. Install Pixi and run tasks via `pixi run <task>`.
- Python requirements are declared in `pyproject.toml`.

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

