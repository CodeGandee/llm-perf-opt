# Quickstart: DeepSeek‑OCR Analytic Modeling in ModelMeter

This guide shows how to prepare the environment and run an analytic modeling workflow for DeepSeek‑OCR using
ModelMeter and existing static analysis tooling. It builds on Stage 1/Stage 2 profiling and static TorchInfo analysis.

## Prerequisites

- OS/GPU: Linux with an NVIDIA GPU (RTX 5090‑class or similar, CUDA 12 runtime).
- Environment: Pixi with the `rtx5090` environment configured.
- Repo root: `/workspace/code/llm-perf-opt`
- Model: Local DeepSeek‑OCR weights under `/workspace/code/llm-perf-opt/models/deepseek-ocr`
  (matching `deepseek-ai/DeepSeek-OCR`).

## 1) Verify Pixi environment

```bash
cd /workspace/code/llm-perf-opt
pixi run -e rtx5090 python -V
```

## 2) Ensure static TorchInfo artifacts exist

Analytic modeling relies on TorchInfo static analysis artifacts for DeepSeek‑OCR. If you need to regenerate them:

```bash
cd /workspace/code/llm-perf-opt
pixi run -e rtx5090 python scripts/analytical/dsocr_find_static_components.py \
  --model /workspace/code/llm-perf-opt/models/deepseek-ocr \
  --device cuda:0 \
  --base-size 1024 \
  --image-size 640 \
  --seq-len 512 \
  --crop-mode 1 \
  --output /workspace/code/llm-perf-opt/reports/20211117-dsorc-op-analysis/static-20251118-130533
```

This produces `torchinfo-*.{txt,json,md}` files under
`/workspace/code/llm-perf-opt/reports/20211117-dsorc-op-analysis/static-20251118-130533/`, which serve as the canonical
target operator list for this feature.

## 3) Run DeepSeek‑OCR analytic modeling (CLI sketch)

Once the ModelMeter layers for DeepSeek‑OCR are implemented under
`/workspace/code/llm-perf-opt/extern/modelmeter/models/deepseek_ocr/`, an analytic modeling run for the standard OCR
workload (`dsocr-standard-v1`) should look like:

```bash
cd /workspace/code/llm-perf-opt
pixi run -e rtx5090 python -m llm_perf_opt.runners.dsocr_analyzer \
  --mode analytic \
  --model /workspace/code/llm-perf-opt/models/deepseek-ocr \
  --device cuda:0 \
  --workload-profile-id dsocr-standard-v1
```

Expected behavior:

- Load DeepSeek‑OCR via `DeepSeekOCRSession` and `DeepseekOCRStaticAnalyzer`.
- Read TorchInfo artifacts from `reports/20211117-dsorc-op-analysis/static-20251118-130533/`.
- Map DeepSeek‑OCR modules and PyTorch operators into ModelMeter analytic layers implemented as subclasses of
  `extern/modelmeter/layers/base.py`.
- Produce structured analytic model artifacts (JSON and/or YAML) plus human‑readable Markdown documentation under:
  `/workspace/code/llm-perf-opt/tmp/profile-output/<run_id>/static_analysis/analytic_model/`.
  - Machine‑readable outputs: module hierarchy, per‑module/operator metrics, and workload metadata as `.json`/`.yaml`.
  - Markdown outputs: one or more pages that describe each analyzed layer/operator, its definition, the theoretical
    formulas used to compute FLOPs/I/O/memory, and the rationale and assumptions behind those formulas.

## 4) Interpreting analytic reports

Analytic reports and documentation are expected to include:

- A module hierarchy showing major components (vision encoder, projector, decoder blocks, head).
- Per‑module metrics: predicted runtime, FLOPs, I/O, and memory footprint for the standard OCR workload.
- Operator‑level breakdowns per module (e.g., conv2d vs linear vs attention).
- Layer/operation documentation in Markdown that explains, for each analytic layer/operator:
  - how the layer is defined and parameterized in DeepSeek‑OCR,
  - how theoretical FLOPs/I/O/memory estimates are derived from shapes and call counts, and
  - why particular modeling assumptions or simplifications are used.
- Optional fields for future runtime comparison (e.g., measured total runtime) may appear in the schema but are not
  required or populated in this development stage; focus is on theoretical analytic estimates, structural clarity, and
  documentation that makes the analytic model understandable to humans.

For convenience, a summary view should also be exported, matching the
`DeepSeekOCRAnalyticReportSummary` contract defined in
`/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/contracts/python-contracts.md`.

## 5) Manual validation (planned)

Manual validation scripts for this feature are planned under:

- `/workspace/code/llm-perf-opt/tests/manual/deepseek_ocr/manual_deepseek_ocr_performance_report.py`
  – end‑to‑end performance and memory report generation.
- `/workspace/code/llm-perf-opt/tests/manual/deepseek_ocr/manual_deepseek_ocr_analytic_model.py`
  – analytic model construction and inspection for theoretical correctness (structure and cost estimates), without
  automated comparison against measured runtimes.

These scripts should:

- Use the `dsocr-standard-v1` workload profile (synthetic inputs, no pinned dataset).
- Emit artifacts under `/workspace/code/llm-perf-opt/tmp/profile-output/<run_id>/`.
- Enable human reviewers to inspect module hierarchies, operator breakdowns, and theoretical cost estimates
  (FLOPs/time/memory) for plausibility. There is no requirement in this development stage to compare analytic predictions
  against measured runtimes; correctness is assessed via expert review rather than automated runtime thresholds.
