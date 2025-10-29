# Quickstart — Stage 2 NVIDIA Deep Profiling (DeepSeek-OCR)

## Prerequisites

- NVIDIA GPU with compatible driver; CUDA 12.4
- Nsight Systems (`nsys`) and Nsight Compute (`ncu`) installed and on PATH
- Pixi environment set up for this repo

## Run a Profiling Session

1) Launch environment

```
pixi shell
```

2) Run Stage 2 profile (Deep mode)

```
# Example: uses fixed inputs manifest and deep mode
pixi run stage2-profile -- +run.mode=deep +inputs.manifest=/abs/path/to/inputs.yaml
```

3) Inspect artifacts (latest run dir under `tmp/stage2/<run_id>/`):
- `report.md`, `stakeholder_summary.md`
- `operators.md` and `kernels.md` (sorted by total, includes mean ms)
- `env.json`, `inputs.yaml`, `config.yaml`
- `nsys` timeline (`.qdrep`) and `ncu` summaries (CSV/JSON)

## Notes

- Vision timing (sam+clip+projector) is documented as a note; not a separate stage row.
- If CUDA times are near zero at operator rows, check `kernels.md` — attribution happens at kernel level.
- If overhead is high, switch to light mode:

```
pixi run stage2-profile -- +run.mode=light
```

