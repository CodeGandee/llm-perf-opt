# Quickstart: Wan2.1 Static Analytic Report (ModelMeter-style)

## Prerequisites

- Pixi environment installed and dependencies synced in `/data1/huangzhe/code/llm-perf-opt` (run `pixi install`).
- A machine-local Wan2.1-T2V-14B checkout or directory available on disk (weights and metadata are not committed to Git).

## Link the local model directory (no downloads)

Set one of the following and create the `source-data` symlink:

```bash
export LLM_MODELS_ROOT=/data1/huangzhe/llm-models
bash models/wan2.1-t2v-14b/bootstrap.sh
```

If your model lives elsewhere:

```bash
export WAN21_T2V_14B_PATH=/path/to/Wan2.1-T2V-14B
bash models/wan2.1-t2v-14b/bootstrap.sh
```

## Run the static analyzer (generate report artifacts)

Use a Hydra run directory rooted under `tmp/profile-output/`:

```bash
pixi run -e rtx5090 python -m llm_perf_opt.runners.wan2_1_analyzer \
  hydra.run.dir='tmp/profile-output/${now:%Y%m%d-%H%M%S}' \
  workload.profile_id=wan2-1-512p
```

## Run verification (≤5% FLOP error budget)

Layer-by-layer verification (transformer blocks and key subcomponents):

```bash
pixi run -e rtx5090 python -m modelmeter.models.wan2_1.scripts.verify.run_verify_layers \
  --workload wan2-1-ci-tiny
```

End-to-end verification (diffusion core across steps, within the analytic scope):

```bash
pixi run -e rtx5090 python -m modelmeter.models.wan2_1.scripts.verify.run_verify_end2end \
  --workload wan2-1-512p
```

## Standard workload set

- `wan2-1-ci-tiny`: batch 1, 1 step, 4 frames, 256×256, text_len 64
- `wan2-1-512p`: batch 1, 50 steps, 16 frames, 512×512, text_len 512
- `wan2-1-720p`: batch 1, 50 steps, 16 frames, 720×1280, text_len 512

## Outputs

The analyzer writes artifacts under:

- `/data1/huangzhe/code/llm-perf-opt/tmp/profile-output/<run_id>/static_analysis/wan2_1/`
  - `report.json` (machine-readable; includes per-layer metrics and totals)
  - `summary.md` (human-readable overview)
  - `verify/` (optional; verification outputs)
