# Artifacts

Run directory
- Hydra writes to `tmp/stage1/<run_id>/`.
- Log file: `llm_profile_runner.log` captures preprocessing, prefill/decode timings, token counts.

Core files
- `report.md` — human summary with aggregates and MFU per stage (uses device peak TFLOPs mapping)
- `operators.md` — top‑K operator table from PyTorch profiler (sorted by total; includes Mean ms per call)
- `metrics.json` — machine‑readable summary (includes `aggregates.stage_ms` with prefill/decode and sub‑stages if present)
- `stakeholder_summary.md` — stakeholder tables: Environment, Aggregates, Per‑Stage Timings (ms), MFU, Top Operators, Recommendations

Reproducibility
- `env.json` — GPU/CUDA/torch/transformers versions
- `inputs.yaml` — dataset selection and absolute image paths with width/height/bytes
- `assumptions.md` — device, repeats, decoding/preprocess/profiling/dataset knobs

Predictions and visualization (optional)
- Enable with `outputs.save_predictions=true`.
- Files:
  - `predictions.jsonl` — one JSON per image (raw + clean text, timings, tokens)
  - `predictions.md` — gallery with thumbnails and annotated images
  - `viz/<stem>/result_with_boxes.jpg` — annotated original image
  - `viz/<stem>/result.mmd` — vendor‑style text with `![](images/<idx>.jpg)` references
  - `viz/_thumbs/<stem>.jpg` — thumbnails used in the gallery

Vendor parity outputs
- Use `scripts/deepseek-ocr-infer-one.py` to generate the official artifacts for comparison.
- Typical layout: `<output>/<stem>/{result_with_boxes.jpg,result.mmd,images/}`.
Notes on profiler tables
- CUDA time often aggregates under kernel entries (e.g., `cudaLaunchKernel`, `flash_attn::*`) and may be 0 for many high‑level `aten::*` rows. See `context/summaries/issue-pytorch-profiler-zero-cuda-time.md`.
- Mean ms is `total_time_ms / calls` and helps compare per‑call costs while keeping sort by total.
