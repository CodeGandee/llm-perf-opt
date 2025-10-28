Project Package (`llm_perf_opt`)

Unified Python package for profiling harnesses, runners, and optimization logic
for this repository.

Subpackages:
- `llm_perf_opt.profiling/` — NVTX helpers, profiler wrappers, NVML sampling
- `llm_perf_opt.runners/` — runtime adapters (e.g., PyTorch, vLLM, TRT-LLM)
- `llm_perf_opt.data/` — dataset utilities and simple preprocessors
