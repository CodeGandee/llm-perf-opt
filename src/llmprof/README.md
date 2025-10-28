Profiling Package (`llmprof`)

Python package intended for profiling harnesses, runners, and utilities.
This package complements the project’s performance optimization goals.

Suggested modules:
- `cli.py`: Hydra entrypoint that wires configs to runners and profilers.
- `profiling/`: NVTX helpers, NVML sampler, profiler wrappers.
- `runners/`: Runtime-specific runner implementations (PyTorch, vLLM, TRT-LLM).
- `data/`: Dataset utilities (loading, preprocessing).

