# Implementation Guide: US4 — Cross‑Model

Phase: 6 | Feature: Stage 2 — NVIDIA-Backed Deep LLM Profiling | Tasks: T024–T026

## Files

### Created
- conf/model/llm/alt.yaml
- tests/manual/stage2_profile/manual_stage2_alt_model.py

### Modified
- src/llm_perf_opt/runners/deep_profile_runner.py

## Public APIs

### T025: Model target in provenance

```python
# inside deep_profile_runner
model_id = cfg.get('model', {}).get('target', 'deepseek-ocr')
provenance = { 'model': { 'id': model_id, 'dtype': 'bf16' } }
```

## Phase Integration

```mermaid
graph LR
    C[config: +model.target] --> R[runner]
    R --> P[provenance: env.json]
```

## Testing

```bash
pixi run python tests/manual/stage2_profile/manual_stage2_alt_model.py
```

## References
- Spec: specs/002-nvidia-llm-profiling/spec.md (US4)
