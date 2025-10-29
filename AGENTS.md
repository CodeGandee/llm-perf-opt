# llm-perf-opt Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-10-28

## Active Technologies
- Python 3.11 (Pixi env; pyproject requires >=3.11) + PyTorch 2.5.x (CUDA 12.4 wheels), Transformers 4.46.3, Tokenizers 0.20.x, NVTX, Hydra/OmegaConf, attrs/pydantic, NVML (nvidia-ml-py), ruff, mypy (001-profile-deepseek-ocr)
- N/A (local files for artifacts under `/data2/huangzhe/code/llm-perf-opt/tmp` and `/data2/huangzhe/code/llm-perf-opt/specs/001-profile-deepseek-ocr`) (001-profile-deepseek-ocr)
- Python 3.11 (Pixi env) + PyTorch 2.5.x, Transformers 4.46.x, Tokenizers 0.20.x, NVTX, NVIDIA Nsight Systems (nsys), NVIDIA Nsight Compute (ncu) (002-nvidia-llm-profiling)
- Local filesystem artifacts under `tmp/` and `context/` (no DB) (002-nvidia-llm-profiling)

- (001-profile-deepseek-ocr)

## Project Structure

```text
backend/
frontend/
tests/
```

## Commands

# Add commands for 

## Code Style

: Follow standard conventions

## Recent Changes
- 002-nvidia-llm-profiling: Added Python 3.11 (Pixi env) + PyTorch 2.5.x, Transformers 4.46.x, Tokenizers 0.20.x, NVTX, NVIDIA Nsight Systems (nsys), NVIDIA Nsight Compute (ncu)
- 001-profile-deepseek-ocr: Added Python 3.11 (Pixi env; pyproject requires >=3.11) + PyTorch 2.5.x (CUDA 12.4 wheels), Transformers 4.46.3, Tokenizers 0.20.x, NVTX, Hydra/OmegaConf, attrs/pydantic, NVML (nvidia-ml-py), ruff, mypy
- 001-profile-deepseek-ocr: Added Python 3.11 (Pixi env; pyproject requires >=3.11) + PyTorch 2.5.x (CUDA 12.4 wheels), Transformers 4.46.3, Tokenizers 0.20.x, NVTX, Hydra/OmegaConf, attrs/pydantic, NVML (nvidia-ml-py), ruff, mypy


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
