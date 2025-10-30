# Implementation Guide: Polish & Cross‑Cutting

Phase: 7 | Feature: Stage 2 — NVIDIA-Backed Deep LLM Profiling | Tasks: T027–T029

## Files

### Modified
- specs/002-nvidia-llm-profiling/quickstart.md
- conf/profiling/stage2.yaml
- src/llm_perf_opt/profiling/vendor/ncu.py

## Public APIs

### T029: Light-mode configuration

```yaml
# conf/profiling/stage2.yaml
run:
  mode: deep  # deep|light
ncu:
  metrics_set: roofline   # roofline|minimal
  clock_control: base
  cache_control: all
```

## Testing

```bash
pixi run ruff check .
pixi run mypy src/
```

## References
- Tasks: specs/002-nvidia-llm-profiling/tasks.md
