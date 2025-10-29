# Visualization (Vendor-Style)

We parse `<|ref|>label</|ref|><|det|>[[x1,y1,x2,y2], ...]</|det|>` spans and render overlays matching the vendor implementation.

Implementation: `src/llm_perf_opt/visualize/annotations.py`
- Normalization uses `x/999 * W`, `y/999 * H` exactly like upstream.
- Boxes are outlined; translucent fill is drawn on a separate RGBA overlay and composited.
- For `label == "image"`, cropped regions are saved to `images/<idx>.jpg`.
- We also write a `result.mmd` that replaces image refs with `![](images/<idx>.jpg)` and removes other refs.

Outputs per image
- `viz/<stem>/result_with_boxes.jpg`
- `viz/<stem>/result.mmd`
- `viz/<stem>/images/*.jpg` (if any)

Troubleshooting misaligned boxes
- Ensure you are decoding with special tokens preserved.
- Confirm the input image dimensions used when mapping 0..999 → pixels.
- Compare our `result_with_boxes.jpg` against vendor output from `scripts/deepseek-ocr-infer-one.py`.

