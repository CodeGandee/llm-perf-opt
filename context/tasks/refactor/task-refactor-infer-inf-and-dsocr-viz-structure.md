# Refactor Plan: `infer.max_new_tokens=inf` semantics + DeepSeek‑OCR visualization dir structure

## What to Refactor
- Add support for `infer.max_new_tokens=inf` to mean “generate until stop,” falling back to a model’s intrinsic upper bound if one exists.
- Redesign DeepSeek‑OCR visualization outputs:
  - Per‑image outputs under `<run>/torch_profiler/viz/<md5(full-image-path)>/`.
  - Files: `result_with_boxes.jpg` and `info.json` (includes original image path, result image path, result text, and minimal metadata).
  - Remove `predictions.md` gallery; avoid heavy Markdown generation and large assets duplication.

Scope
- Code: `src/llm_perf_opt/runners/llm_profile_runner.py`, `src/llm_perf_opt/runners/dsocr_session.py`, `src/llm_perf_opt/visualize/annotations.py` (paths and emitters).
- Config: No new keys required for `inf`; reuse `infer.max_new_tokens`. Visualization paths already stage‑relative via `pipeline.torch_profiler.output.visualization.save_dir`.
- Docs: Update configuration guide and quickstarts to document `inf` and the new per‑image outputs.

## Why Refactor
- Usability: Users often want “generate until EOS” (or model-defined stop), not a hard numeric cap.
- Robustness: Where a model has an intrinsic generation cap (e.g., context window), respect it automatically.
- Artifacts at scale: Markdown galleries are heavy and duplicate assets; per‑image JSON + a single annotated image is leaner and easier to consume programmatically.
- Deterministic mapping: Using `md5(full-image-path)` yields a stable, collision‑resistant directory name without leaking arbitrary filenames.

## How to Refactor

1) `infer.max_new_tokens=inf` semantics
- Accept string value "inf" (case‑insensitive) or a sentinel in Hydra overrides to denote “no explicit cap”.
- Interpretation rules:
  - Runner: if `inf`, pass a sentinel (e.g., `None`) to the session.
  - Session: if `max_new_tokens is None`, choose a large internal ceiling, but stop earlier if EOS is produced; never exceed a model‑intrinsic limit if available (e.g., max position embeddings or kv‑cache window; if detectable from config).
  - HF generate: set `max_new_tokens` only when an explicit integer is provided; otherwise loop with incremental decoding until EOS or ceiling.

Before (excerpt)
- llm_profile_runner.py:820
```python
max_new_tokens = int(getattr(cfg, "infer", {}).get("max_new_tokens", 64))
...
res = session.run_inference(
    ..., 
    max_new_tokens=max_new_tokens,
    ...
)
```

After (idea)
```python
raw_mnt = getattr(cfg, "infer", {}).get("max_new_tokens", 64)
max_new_tokens = None if str(raw_mnt).lower() == "inf" else int(raw_mnt)
...
res = session.run_inference(
    ...,
    max_new_tokens=max_new_tokens,  # type: Optional[int]
    ...
)
```

Before (session signature)
- src/llm_perf_opt/runners/dsocr_session.py:78
```python
def run_inference(..., max_new_tokens: int = 64, return_text: bool = False, ...):
```

After (session semantics)
```python
from typing import Optional

def run_inference(..., max_new_tokens: Optional[int] = 64, return_text: bool = False, ...):
    # if None → unlimited until EOS or intrinsic model cap
```

Session generate loop sketch (decode section) — no magic numbers
```python
# if max_new_tokens is None: iterate until EOS or ceiling
from enum import Enum

class _Defaults(Enum):
    # Centralize fallback to avoid magic numbers; consider surfacing as config later
    UNBOUNDED_DECODE_CEILING = 4096

if max_new_tokens is None:
    ceiling = _infer_model_ceiling(self.m_model)
    if ceiling is None:
        ceiling = _Defaults.UNBOUNDED_DECODE_CEILING.value
        logger = logging.getLogger(__name__)
        logger.warning(
            "infer.max_new_tokens=inf: unknown model limit; using fallback ceiling=%d",
            int(ceiling),
        )
    steps = 0
    eos = _resolve_eos_token_id(self.m_tokenizer)  # may be None
    while steps < ceiling:
        out = model.generate(..., max_new_tokens=1, do_sample=False, **gen_kwargs)
        steps += 1
        if eos is not None and eos in out[0, input_len:].tolist():
            break
else:
    out = model.generate(..., max_new_tokens=int(max_new_tokens), **gen_kwargs)
```
Notes
- Do not use inline magic constants. Place conservative fallbacks (e.g., 4096) in a local Enum (or module‑level constant) near the top of the module, and log a WARNING when they are applied.
- `_infer_model_ceiling(model)` can consult `model.config.max_position_embeddings`, `sliding_window`, cache shapes, or return None.
- Keep existing NVTX segmentation (`prefill` then `decode`) intact; for incremental generation, keep `decode_range()` spanning the whole loop.

2) Visualization directory redesign (DeepSeek‑OCR)
- New layout per image: `<stage_dir>/viz/<md5(full-image-path)>/`
  - `result_with_boxes.jpg`: visual overlay from `render_vendor_style`.
  - `info.json`: JSON with fields:
    - `source_image`: absolute path to the source image
    - `result_image`: relative path to `result_with_boxes.jpg`
    - `text_raw`: model output text (with specials)
    - `text_clean`: cleaned text if available (strip specials per model extras)
    - `timings_ms`: subset like `{prefill, decode, vision}` (optional, from run result)

Before (writer)
- src/llm_perf_opt/runners/llm_profile_runner.py:454
```python
_write_predictions_outputs(artifacts_dir, preds, make_gallery, max_images, thumb_width)
# writes torch_profiler/predictions.jsonl and predictions.md + viz/*
```

After (writer contract)
```python
_write_predictions_outputs(pred_dir, vis_dir, preds, make_gallery, max_images, thumb_width)
# Remove predictions.md; write one JSONL + per-image info.json + result_with_boxes.jpg
# Place per-image assets under vis_dir/<md5(full-image-path)>/
```

Per‑image emitter sketch
```python
h = hashlib.md5(str(img_path.resolve()).encode("utf-8")).hexdigest()
img_dir = vis_dir / h
img_dir.mkdir(parents=True, exist_ok=True)
result_img = render_vendor_style(str(img_path), text_raw, str(img_dir))  # returns path to result_with_boxes.jpg
info = {
  "source_image": str(img_path.resolve()),
  "result_image": str(Path(result_img).name),
  "text_raw": text_raw,
  "text_clean": text_clean,
  "timings_ms": {"prefill": prefill_ms, "decode": decode_ms, "vision": vision_ms},
}
(img_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False) + "\n", encoding="utf-8")
```

Removals
- Stop generating `predictions.md` gallery (drop MdUtils dependency in this path).
- Keep `predictions.jsonl` for quick aggregate consumption.

3) Config and documentation
- No new config keys for `inf` — treat string "inf" specially.
- Confirm per‑stage output dirs are still respected (`pipeline.torch_profiler.output.visualization.save_dir=viz` by default).
- Update docs/configuration.md and quickstarts:
  - Show `infer.max_new_tokens=inf` usage.
  - Describe per‑image outputs under `viz/<md5(full-image-path)>/` with `info.json` and `result_with_boxes.jpg`.

## Impact Analysis
- Backward compatibility
  - `infer.max_new_tokens` values that are integers remain unchanged.
  - Markdown gallery removal may affect users relying on `predictions.md`; mitigation: provide transition notes and point to per‑image `info.json` and the aggregate JSONL.
- Performance
  - Incremental generation for `inf` can increase Python overhead; cap with a reasonable ceiling and keep default behavior unchanged for numeric values.
- Storage
  - Per‑image JSON + one annotated image avoids large monolithic Markdown and duplicated thumbnails.

## Expected Outcome
- Users can run “until stop” by setting `infer.max_new_tokens=inf`, with sensible safeguards.
- Visualization outputs are streamlined, machine‑readable, and live under a deterministic, per‑image directory keyed by the full path’s MD5.
- Reduced artifact bloat and simpler downstream consumption.

## References
- Hydra configuration (package mounting): /facebookresearch/hydra
- Transformers generate API: /huggingface/transformers
- Code touch points
  - Runner max_new_tokens handling: src/llm_perf_opt/runners/llm_profile_runner.py:820
  - Session interface + decode loop: src/llm_perf_opt/runners/dsocr_session.py:78
  - Visualization writer: src/llm_perf_opt/runners/llm_profile_runner.py:454
  - Renderer: src/llm_perf_opt/visualize/annotations.py:48, src/llm_perf_opt/visualize/annotations.py:131
