Run DeepSeek-OCR manual prefill+decode (separate) demo

- Uses the local model under `models/deepseek-ocr` (HF snapshot with remote code).
- Performs prefill (one forward with images) then decode (token-by-token with KV cache).
- Writes outputs and visualizations under `tmp/<YYYYmmdd-HHMMSS>/`.

Example

```
pixi run -e rtx5090 python tests/manual/inference/manual_dsocr_prefill_decode.py \
  --image models/deepseek-ocr/assets/show1.jpg \
  --prompt "<|User|>\nExtract all information from this image and convert them into markdown format.\n<|Assistant|>" \
  --max_new_tokens 512
```

Notes
- For prefill, the script computes visual features and injects them into the sequence following the original model’s grid logic.
- For decode, it omits image args and reuses the KV cache to generate step-by-step.
- If the model returns `<|ref|>...<|/ref|><|det|>...<|/det|>` tags, the script produces cropped images and an overlay image `result_with_boxes.jpg`.

