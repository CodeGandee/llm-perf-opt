# Artifacts

Run directory layout
- Default run dir: `tmp/profile-output/<run_id>/`.
- Pipeline outputs live in subdirectories:
  - `torch_profiler/` — `report.md`, `operators.md`, `metrics.json`, `llm_profile_runner.log`
  - `static_analysis/` — `static_compute.{json,md}` (enabled by `pipeline.static_analysis.enable`)
  - `nsys/` — `run.nsys-rep`, `run.sqlite`, `summary_*.csv`, `cmd.txt`
  - `ncu/` — `raw.csv`, `.ncu-rep`, `sections_report.txt`, `cmd*.txt`
- Ephemeral scratch lives under `tmp/<stage>/` (e.g., `tmp/workload/` during NSYS capture).
- Repro at run root: `env.json`, `config.yaml`, `inputs.yaml`.

Static analysis
- Detailed analyzer report under `static_analysis/`:
  - `static_compute.json` — per‑stage FLOPs, params, activations
  - `static_compute.md` — human‑readable summary

Predictions and visualization (optional)
- Enable with `outputs.save_predictions=true`.
- Files (under `torch_profiler/`):
  - `predictions.jsonl`, `predictions.md`
  - `viz/<stem>/result_with_boxes.jpg`, `viz/<stem>/result.mmd`, `viz/_thumbs/<stem>.jpg`

Notes on profiler tables
- CUDA time often aggregates under kernel entries (e.g., `cudaLaunchKernel`, `flash_attn::*`) and may be 0 for many high‑level `aten::*` rows.
- Mean ms is `total_time_ms / calls`.
