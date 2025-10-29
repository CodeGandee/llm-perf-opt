# Artifacts

Run directory
- Hydra writes to `tmp/stage1/<run_id>/`.
- Log file: `llm_profile_runner.log` captures preprocessing, prefill/decode timings, token counts.

Core files
- `report.md` — human summary with aggregates and MFU per stage
- `operators.md` — top‑K operator table from PyTorch profiler
- `metrics.json` — machine‑readable summary

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

