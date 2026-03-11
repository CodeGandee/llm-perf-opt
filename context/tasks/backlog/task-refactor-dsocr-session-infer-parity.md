# Refactor Plan: DeepSeek‑OCR NVTX Inference → Vendor Parity (No `model.infer()`)

## What to Refactor
- Eliminate the temporary fallback to the vendor’s `model.infer(...)` from the Stage‑1 runner visualization path and make our own NVTX‑segmented inference produce vendor‑equivalent detection markup and boxes.
- Align `src/llm_perf_opt/runners/dsocr_session.py` preprocessing, prompt construction, generation, and post‑processing with the reference implementation in:
  - `models/deepseek-ocr/modeling_deepseekocr.py` (see `infer(...)`, `dynamic_preprocess`, `format_messages`, `load_pil_images`, `re_match`, and drawing logic)
  - `models/deepseek-ocr/modeling_deepseekv2.py` (model forward contracts and how image tokens are embedded)

## Why Refactor
- Correctness: Current results sometimes yield only a single `<|ref|>/<|det|>` span while the vendor path produces many. Relying on `model.infer(...)` fixes output quantity but breaks the NVTX benchmarking isolation and mixes vendor I/O side effects into our pipeline.
- Maintainability: Keeping all logic in `dsocr_session.py` removes implicit behaviors hidden inside vendor code and avoids future API drift.
- Performance/Profiling: Preserves our NVTX stage segmentation and avoids additional monolithic calls during decode.

## How to Refactor

1) Audit and reconcile preprocessing contracts (parity with vendor)
- Source of truth: `DeepseekOCRForCausalLM.infer` in `modeling_deepseekocr.py`.
- Actions:
  - Ensure our preprocessing mirrors vendor:
    - Global view: `ImageOps.pad(img, (base_size, base_size), color=mean*255)`, normalize with `(0.5,0.5,0.5)`.
    - Dynamic crops: re‑use our reimplementation of `dynamic_preprocess` to produce `images_crop_raw` and `(w_crop, h_crop)` exactly like vendor.
    - Build `images = [(images_crop_tensor, images_ori_tensor)]` (bf16 on device), identical shapes and dtypes.
    - Build `images_seq_mask` and `images_spatial_crop` with the same semantics.
  - Verify types: `images_seq_mask` should be `torch.bool`; `images_spatial_crop` `torch.long` with shape `[1,2]` or stacked per vendor.

2) Prompt construction equivalence
- The vendor wraps the prompt with `format_messages(..., sft_format='plain')`, which concatenates user/assistant messages (roles ignored). Our current literal string is effectively equivalent, but we will:
  - Add `build_dsocr_prompt(prompt: str) -> str` in `dsocr_session.py` calling the vendor’s `format_messages` with `sft_format='plain'`, to remove any ambiguity and to keep parity if the vendor template changes.
  - Unit‑assert that the output equals our current plain prompt in the default case.

3) Generation policy and stopping
- Vendor decodes full text and then trims terminal stop marker `<｜end▁of▁sentence｜>`. Our current decode is consistent, but to better match vendor behavior:
  - Add a small “decode policy” struct with:
    - `eos_token_id = tokenizer.eos_token_id`
    - `do_sample = False`, `temperature = 0.0` (current defaults)
    - No `top_p/top_k` unless user-specified
    - Keep `max_new_tokens=None` → use `_infer_model_ceiling()` and fall back to a named constant `UNBOUNDED_DECODE_CEILING` (warn once).
  - Do NOT set additional stop criteria that would truncate vendor markup early.

4) Post‑processing: vendor‑equivalent parsing and rendering
- Keep our existing `_parse_spans`/`render_vendor_style` and `write_vendor_result_mmd` which already mirror vendor behavior:
  - Regex matches of `<|ref|>label</|ref|><|det|>[[x1,y1,x2,y2], ...]</|det|>`.
  - Normalize coordinates by 999 to pixel space.
  - Save `crops/` and render an overlay with label text.
  - Ensure `info.json` contains:
    - `boxes: [{x,y,w,h,text}]` where `text` mirrors the segment following the block (matches vendor `process_image_with_refs` intent).
- Add a small helper `postprocess_vendor_markup(text_raw, image_path, out_dir)` to encapsulate these steps for reuse and testing.

5) Remove vendor fallback in the Stage‑1 runner
- Remove the path that calls `session.m_model.infer(...)` and captures stdout. Always use `session.run_inference(...)` text for visualization outputs.
- Keep per‑model visualization toggles in config, but enforce the no‑vendor rule.

6) Validation harness (developer‑only, not user‑facing)
- Add a tiny script/test to compare our NVTX path vs vendor for a small sample:
  - For N images: run `dsocr_session.run_inference(..., return_text=True)` and vendor `model.infer(..., save_results=True)`.
  - Compare number of `boxes` parsed from both paths and the first K boxes’ coordinates within a tolerance.
  - Acceptable variance: ±1 box and ≤2% pixel difference (cropping grid and rounding differences).

7) Incremental rollout & safeguards
- Log a one‑time INFO line “DeepSeek‑OCR: vendor‑parity decode engaged (no vendor infer)” to aid field triage.
- If parsing yields zero boxes across all images, write a WARN with a pointer to the validation harness and the vendor script as a fallback for debugging (not used automatically).

## Before / After (key snippets)

- Before (simplified excerpt from `dsocr_session.run_inference`):

```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
# ... build images, images_seq_mask, images_spatial_crop ...
out = self.m_model.generate(
    **inputs,
    images=images,
    images_seq_mask=images_seq_mask,
    images_spatial_crop=images_spatial_crop,
    attention_mask=attention_mask,
    max_new_tokens=ceiling,
)
text = tokenizer.decode(out[0, input_len:], skip_special_tokens=False)
# parse via _parse_spans(text)
```

- After (proposed):

```python
# 1) Build a vendor‑equivalent prompt
prompt = build_dsocr_prompt("<image>\n<|grounding|>Convert the document to markdown.")

# 2) Prepare inputs with strict parity helpers
inputs, images, images_seq_mask, images_spatial_crop = prepare_dsocr_inputs(
    image_path, base_size, image_size, crop_mode, patch_size=16, downsample_ratio=4,
)

# 3) Decode with vendor‑aligned policy (greedy, EOS only)
ceiling = _infer_model_ceiling() or Defaults.UNBOUNDED_DECODE_CEILING
out = self.m_model.generate(
    **inputs,
    images=images,
    images_seq_mask=images_seq_mask,
    images_spatial_crop=images_spatial_crop,
    attention_mask=attention_mask,
    max_new_tokens=int(ceiling),
    eos_token_id=tokenizer.eos_token_id,
)
text_raw = tokenizer.decode(out[0, input_len:], skip_special_tokens=False)

# 4) Postprocess to boxes + mmd just like vendor
annot_path, boxes = postprocess_vendor_markup(text_raw, image_path, per_image_dir)
```

## Impact Analysis
- Profiling integrity: Preserved. We never call `model.infer(...)` at runtime; all NVTX ranges continue to wrap preprocessing/prefill/decode.
- Output fidelity: Expect many more detection spans (boxes) in `info.json`, closely matching vendor outputs on the same images.
- Risk:
  - Minor divergences in tokenization/masking alignment could still yield fewer boxes on some pages. Mitigation: add the validation harness and log helpful diagnostics for masks/crop shapes.
  - Decode ceiling heuristic might still clip extremely long outputs on some models. Mitigation: log a warning when ceiling fallback is used and surface a config knob for advanced users.

## Expected Outcome
- Visualization outputs align with the vendor script without invoking vendor code.
- `tmp/profile-output/<run>/torch_profiler/viz/<hash>/info.json` contains multiple `boxes` per image for typical documents; `result_with_boxes.jpg` and `result.mmd` reflect the same structure as vendor.
- Nsight Systems/Compute workflows remain unchanged and reliable.

## References
- Vendor implementation
  - models/deepseek-ocr/modeling_deepseekocr.py:703 (infer), 172 (dynamic_preprocess), 230 (format_messages), 267 (load_pil_images)
  - models/deepseek-ocr/modeling_deepseekv2.py: forward and generation plumbing
- Our preprocessing and viz
  - src/llm_perf_opt/runners/dsocr_session.py (NVTX inference)
  - src/llm_perf_opt/visualize/annotations.py (regex + rendering)
- Context7 (Transformers library): /huggingface/transformers

