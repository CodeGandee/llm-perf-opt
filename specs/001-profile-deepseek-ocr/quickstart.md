# Quickstart: Stage 1 Profiling for DeepSeek‑OCR

This guide shows how to prepare the environment, data, and run a Stage 1 profiling workflow.
It follows the existing repo layout and adapts ideas from `context/hints/nv-profile-kb/about-profile-project-structure.md`.

## Prerequisites

- OS/GPU: Linux with NVIDIA GPU (CUDA 12 runtime)
- Environment: Pixi (configured by repo)
- Repo root: /data2/huangzhe/code/llm-perf-opt

## 1) Create/enter Pixi environment

```bash
cd /data2/huangzhe/code/llm-perf-opt
pixi run python -V
```

## 2) Prepare model and inputs

- Model: Place the DeepSeek‑OCR model under `/data2/huangzhe/code/llm-perf-opt/models/deepseek-ocr` for offline load.
- Inputs: Put 10–20 images in `/data2/huangzhe/code/llm-perf-opt/data/samples` (png/jpg).

Optional env overrides:

```bash
export DSOCR_MODEL="/data2/huangzhe/code/llm-perf-opt/models/deepseek-ocr"
export DSOCR_IMAGE="/data2/huangzhe/code/llm-perf-opt/data/samples"
export DSOCR_USE_FLASH_ATTN=1   # set 0 to disable
export DSOCR_DEVICE=cuda:0
```

## 3) Manual sanity run (HF-only driver)

```bash
pixi run python /data2/huangzhe/code/llm-perf-opt/tests/manual/deepseek_ocr_hf_manual.py
```

Outputs are written under `/data2/huangzhe/code/llm-perf-opt/tmp/dsocr_outputs` to respect the current project convention.

## 4) Stage 1 profiling workflow (design)

The Stage 1 runner will:
- Add NVTX ranges: `prefill` for the first forward pass after tensors are on device, and `decode` for the token-by-token loop.
- Wrap execution with PyTorch Profiler (CPU+CUDA), export operator summaries.
- Compute MFU (model-level and per-stage) using tokens/sec and analytical FLOPs/token approximations.
- Support repeated runs (default 3) and report mean/std.

Planned CLI (to be implemented in Phase 2):

```bash
pixi run python -m llm_perf_opt.runners.llm_profile_runner \
  --model-path /data2/huangzhe/code/llm-perf-opt/models/deepseek-ocr \
  --input-dir  /data2/huangzhe/code/llm-perf-opt/data/samples \
  --repeats 3 --device cuda:0 --use-flash-attn 1
```

Artifacts will be written under `/data2/huangzhe/code/llm-perf-opt/tmp/stage1/<run_id>/` (not `runs/` yet) and summarized in the Stage 1 report. Hydra configs under `conf/` will be introduced in a later phase to align with the structure guide.

## 5) Interpreting results

- Stage segmentation: Verify prefill/decode times are present.
- Operator summary: Review top operators by total time and CUDA time.
- MFU: Check model-level and per-stage MFU; expect stability within ±10% across repeats.
- Stakeholder notes: Confirm top cost centers and stage attribution are clearly stated.

## 6) Reproducibility artifacts (US3)

Each run emits additional files under the run directory to help reproduce and compare results:

- `inputs.yaml`: Absolute input image paths with basic metadata (width/height/size) and the dataset selection used.
- `env.json`: Minimal environment snapshot (GPU, CUDA, torch, transformers).
- `assumptions.md`: Run assumptions (device, repeats, decoding and preprocessing params, profiling settings).

To rerun with the same setup, point the runner at the same dataset subset and device, keeping the decoding/preprocess values equal to those in `assumptions.md`.
