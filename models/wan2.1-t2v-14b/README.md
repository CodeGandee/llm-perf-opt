# Wan2.1-T2V-14B (external reference)

This directory tracks a machine-local installation of the `Wan2.1-T2V-14B` model without committing any weights to Git.

## Layout

- `source-data` (symlink, not committed): points to the actual model directory on your machine.
- `bootstrap.sh` (committed): creates or repairs the `source-data` symlink.

## Setup

Set `LLM_MODELS_ROOT` to the parent directory that contains `Wan2.1-T2V-14B`, then run:

```bash
export LLM_MODELS_ROOT=/data1/huangzhe/llm-models
bash models/wan2.1-t2v-14b/bootstrap.sh
```

If your model lives elsewhere, either set `LLM_MODELS_ROOT` accordingly or set `WAN21_T2V_14B_PATH` to the full path of the model directory.

