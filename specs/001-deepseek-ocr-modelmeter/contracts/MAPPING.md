# Contract ↔ CLI Mapping (DeepSeek‑OCR Analytic Modeling)

This document maps analytic modeling API endpoints to local CLI/task behavior for the DeepSeek‑OCR analytic modeling
feature.

- POST `/analytic/deepseek-ocr/run` → `pixi run -e rtx5090 python -m llm_perf_opt.runners.dsocr_analyzer --mode analytic`
  - Builds or refreshes the DeepSeek‑OCR analytic model for the requested `model_variant` and `workload_profile_id`.
  - Uses static artifacts under `/workspace/code/llm-perf-opt/reports/20211117-dsorc-op-analysis/static-20251118-130533/`
    plus ModelMeter layers under `extern/modelmeter/models/deepseek_ocr/`.
  - Writes structured analytic artifacts (JSON and/or YAML) and Markdown documentation to
    `/workspace/code/llm-perf-opt/tmp/profile-output/<run_id>/static_analysis/analytic_model/`, including:
    - machine-readable analytic model files (module hierarchy, metrics, workload metadata), and
    - human-readable Markdown pages describing each analyzed layer/operator, its definition in DeepSeek‑OCR, and the
      theoretical FLOPs/I/O/memory formulas and assumptions used.

- GET `/analytic/deepseek-ocr/{report_id}/summary` → read summary JSON/Markdown under the corresponding run directory
  - Returns a summary view (top modules, theoretical runtime estimates, notes) derived from the analytic report and
    associated Markdown documentation. Runtime comparison fields are optional and may remain unset in this development
    stage.

- GET `/analytic/deepseek-ocr/{report_id}/model` → read full analytic model JSON under the corresponding run directory
  - Returns the full module hierarchy, operator categories, and per‑module metrics for the specified report id.

Notes
- No new long‑running service is required; analytic modeling is executed as a local CLI task within the Pixi
  environment (`rtx5090`).
- Artifact layout should match existing conventions for static analysis and profiling under
  `/workspace/code/llm-perf-opt/tmp/profile-output/<run_id>/`.
