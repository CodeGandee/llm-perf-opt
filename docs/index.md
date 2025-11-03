# LLM Perf Opt — DeepSeek‑OCR Profiling

LLM Perf Opt provides a reproducible profiling and benchmarking workflow for DeepSeek‑OCR.
It supports multiple pipelines from a single config: PyTorch operator profiling, Static analysis, Nsight Systems, and Nsight Compute.

Key links:
- PyTorch runner: `src/llm_perf_opt/runners/llm_profile_runner.py`
- Deep profiling orchestrator: `src/llm_perf_opt/runners/deep_profile_runner.py`
- Session wrapper: `src/llm_perf_opt/runners/dsocr_session.py`
- Vendor style viz: `src/llm_perf_opt/visualize/annotations.py`
- Config root: `conf/`

Quickstart:
- `pixi run stage1-run` — writes `torch_profiler/*` and `static_analysis/*` under `tmp/profile-output/<run_id>/`.
- `pixi run stage2-profile` — writes `nsys/*` under the same run directory.
- `pixi run stage-all-run` — runs both into a single run directory.

See “Getting Started” for environment and quick commands.
