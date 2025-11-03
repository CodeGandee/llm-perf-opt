# Implementation Guide: Polish & Cross‑Cutting

Phase: 7 | Feature: Stage 2 — NVIDIA-Backed Deep LLM Profiling | Tasks: T027–T029

## Files

### Modified
- specs/002-nvidia-llm-profiling/quickstart.md
- conf/runner/stage2.yaml
- src/llm_perf_opt/profiling/vendor/ncu.py

## Public APIs

### T029: Light-mode configuration

```yaml
# conf/runner/stage2.yaml
run:
  mode: light  # deep|light
defaults:
  - /profiling/nsys@nsys: nsys.default
  - /profiling/ncu@ncu: ncu.default
```

## Testing

```bash
pixi run ruff check .
pixi run mypy src/
```

## References
- Tasks: specs/002-nvidia-llm-profiling/tasks.md
