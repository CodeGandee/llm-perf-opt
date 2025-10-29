# LLM Perf Opt — DeepSeek‑OCR Stage 1

LLM Perf Opt provides a reproducible profiling and benchmarking workflow for DeepSeek‑OCR. Phase 3 (US1) is implemented with NVTX ranges, PyTorch operator summaries, and vendor‑style prediction outputs and visualizations.

Key links:
- Runner entry: `src/llm_perf_opt/runners/llm_profile_runner.py`
- Session wrapper: `src/llm_perf_opt/runners/dsocr_session.py`
- Vendor style viz: `src/llm_perf_opt/visualize/annotations.py`
- Config root: `conf/`

What you can do:
- Run a profiling benchmark that writes `report.md`, `operators.md`, `metrics.json` under `tmp/stage1/<run_id>/`.
- Save predictions with vendor‑style annotated images and `result.mmd` per image.
- Compare outputs with the official vendor `infer()`.

See “Getting Started” for environment and quick commands.

