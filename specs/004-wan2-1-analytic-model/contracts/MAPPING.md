# Mapping: Wan2.1 Analytic “API” → Repository Entry Points

This project is filesystem- and CLI-driven (not a deployed web service). The OpenAPI contract in `openapi.yaml` is a conceptual interface that maps to Python runners and on-disk artifacts.

## POST /analytic/wan2-1/run

**Repository entry point**:

- `pixi run -e rtx5090 python -m llm_perf_opt.runners.wan2_1_analyzer ...`

**Primary outputs** (under the Hydra run directory):

- `/data1/huangzhe/code/llm-perf-opt/tmp/profile-output/<run_id>/static_analysis/wan2_1/report.json`
- `/data1/huangzhe/code/llm-perf-opt/tmp/profile-output/<run_id>/static_analysis/wan2_1/summary.md` (optional)

## GET /analytic/wan2-1/{report_id}/summary

**Repository entry point**:

- Read and summarize: `/data1/huangzhe/code/llm-perf-opt/tmp/profile-output/<report_id>/static_analysis/wan2_1/report.json`

**Notes**:

- In practice, the “summary” is either produced directly during `run` (as `summary.md`) or derived from `report.json` by lightweight tooling.

## GET /analytic/wan2-1/{report_id}/model

**Repository entry point**:

- Read: `/data1/huangzhe/code/llm-perf-opt/tmp/profile-output/<report_id>/static_analysis/wan2_1/report.json`

## Verification (not an OpenAPI endpoint)

**Repository entry points**:

- `pixi run -e rtx5090 python -m modelmeter.models.wan2_1.scripts.verify.run_verify_layers ...`
- `pixi run -e rtx5090 python -m modelmeter.models.wan2_1.scripts.verify.run_verify_end2end ...`

**Outputs**:

- Verification scripts should write their artifacts under the same run directory subtree, for example:
  - `/data1/huangzhe/code/llm-perf-opt/tmp/profile-output/<run_id>/static_analysis/wan2_1/verify/`
