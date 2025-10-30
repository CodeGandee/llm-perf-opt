# Quickstart — Stage 2 NVIDIA Deep Profiling (DeepSeek-OCR)

## Prerequisites

- NVIDIA GPU with compatible driver; CUDA 12.4
- Nsight Systems (`nsys`) and Nsight Compute (`ncu`) installed and on PATH
- Pixi environment set up for this repo
- Never use system Python; always run via `pixi run ...`

## Run a Profiling Session

1) Run Stage 2 profile (Deep mode)

```
# Uses fixed inputs manifest and deep mode
pixi run stage2-profile -- +run.mode=deep +inputs.manifest=/abs/path/to/inputs.yaml
```

2) Inspect artifacts (latest run dir under `tmp/stage2/<run_id>/`):
- `report.md`, `stakeholder_summary.md`
- `operators.md` and `kernels.md` (sorted by total, includes mean ms)
- `env.json`, `inputs.yaml`, `config.yaml`
- `nsys` timeline (`.qdrep`) and `ncu` summaries (CSV/JSON)

## Running Python Scripts (via Pixi)

- One‑off module run (example):

```
pixi run python -m llm_perf_opt.runners.llm_profile_runner profiling.enabled=true 'profiling.activities=[cpu,cuda]'
```

- Quick environment check:

```
pixi run python -c "import torch; print(torch.__version__)"
```

## Notes

- Vision timing (sam+clip+projector) is documented as a note; not a separate stage row.
- If CUDA times are near zero at operator rows, check `kernels.md` — attribution happens at kernel level.
- If overhead is high, switch to light mode:

```
pixi run stage2-profile -- +run.mode=light
```

### Data model reuse (Stage 2)

- Stage 2 reuses Stage 1 domain models (StageTiming, OperatorSummary/OperatorRecord, LLMProfileReport aggregates) and introduces only `KernelRecord` for the kernels table. ProfilingSession is treated as an on‑disk provenance bundle (env/config/artifacts) rather than a new runtime class.
