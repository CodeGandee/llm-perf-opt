# DeepSeek‑OCR: NVTX Path Emits Only 1 Box — Vendor Parity Investigation

## Summary
Stage‑1 (NVTX‑segmented) inference often produces only a single `<|ref|>/<|det|>` block per image, while the vendor path (`model.infer(...)`) yields many. Renderer is not at fault; divergence is upstream in how we build the prompt and the image→token schedule that drives visual embedding injection.

## Root Cause (current understanding)
Our NVTX path did not exactly mirror the vendor’s “first pass” generate setup. The model injects image embeddings at positions marked by `images_seq_mask` during the initial generate forward. If we run a separate prefill forward and then call generate, or if attention/position handling differs, downstream decode tends to emit only one detection block.

## Evidence
- Our NVTX path logs (for a 2×2 crop grid) show correct scheduling:
  - `Mask stats | seq_len=703 img_tokens=693 w_crop=2 h_crop=2`
  - `Images shapes | crop=(4, 3, 640, 640) ori=(1, 3, 1024, 1024)`
  - 693 image tokens matches vendor math: 273 (global) + 420 (local crops)
- Vendor `scripts/deepseek-ocr-infer-one.py` on the same images outputs many `<|ref|>/<|det|>` blocks; our NVTX path still outputs 1.
- Removing the separate prefill step (single generate call) is necessary for parity, but an initial attempt triggered a FlashAttention shape assertion — indicating we must also align attention/position inputs.

## Changes Already Made
- Prompt parity
  - Added vendor‑equivalent prompt builder using the vendor “plain” conversation template.
  - File: `src/llm_perf_opt/runners/dsocr_session.py`
- Image token scheduling parity
  - Mirrored vendor scheduling for global + local crops, and ensured `images_seq_mask` length and True‑count match input_ids and visual embeddings.
  - Added runtime logs for sanity (mask stats + shapes).
  - File: `src/llm_perf_opt/runners/dsocr_session.py`
- Generate call alignment
  - Set `use_cache=True`, `temperature=0.0`, `no_repeat_ngram_size=20`, and ensured `images_spatial_crop` is CPU `LongTensor`.
  - Removed `attention_mask` (vendor does not pass it); later we’ll reintroduce parity handling for `position_ids`.
- Vendor fallback removed
  - No longer call `model.infer(...)` inside Stage‑1; NVTX integrity preserved.
  - File: `src/llm_perf_opt/runners/llm_profile_runner.py`
- Inspector
  - `scripts/inspect-deepseek-ocr-infer.py` confirms `infer()` lives at the vendor file cached by Transformers: `~/.cache/huggingface/modules/transformers_modules/DeepSeek-OCR/modeling_deepseekocr.py:703`.

## Remaining Gap
- Despite matched token counts and shapes, our NVTX path still yields a single detection block. The remaining difference is the first‑pass semantics of generate:
  - Vendor performs a single generate call; we historically did a separate prefill pass which disrupts image embedding injection.
  - When switching to single generate, we saw a FlashAttention shape mismatch. That suggests we must pass `attention_mask` (or explicit `position_ids`) as the vendor path implicitly does when deriving positions.

## Plan to Fix
1) Single generate path (vendor parity)
   - Remove the separate prefill forward; perform a single `generate` call like vendor infer.
   - Maintain NVTX segmentation by spanning the “prefill” phase around the initial step inside generate (or document prefill_ms=0 if splitting is impractical).

2) Attention/position parity
   - Pass `attention_mask=torch.ones_like(input_ids)` to allow the model to derive `position_ids` as in vendor path; or explicitly compute vendor‑equivalent `position_ids` and pass them.
   - Keep `images_spatial_crop` on CPU, `images_seq_mask` on the model device.

3) Validate
   - Run on the 3‑image subset and compare counts of `<|ref|>` blocks and boxes against vendor.
   - If still fewer boxes, dump the first K spans (label + coords) from both for side‑by‑side diff.

4) Optional tooling
   - Add a comparator script: run our NVTX path and vendor `model.infer(save_results=True)` on the same images, then print block counts and a preview of spans.

## Repro Commands
- NVTX path (unlimited tokens; small subset):
  - `pixi run python -m llm_perf_opt.runners.llm_profile_runner \
    hydra.run.dir=tmp/profile-output/${now} \
    dataset.subset_filelist=datasets/omnidocbench/subsets/dev-3.txt \
    device=cuda:0 infer.max_new_tokens=inf \
    'pipeline.torch_profiler.activities=[cpu,cuda]' \
    pipeline.torch_profiler.output.prediction.enable=true \
    pipeline.torch_profiler.output.visualization.enable=true \
    pipeline.nsys.enable=false pipeline.ncu.enable=false \
    dataset.sampling.num_epochs=1 dataset.sampling.num_samples_per_epoch=3 dataset.sampling.randomize=false`

- Vendor path (ground truth reference):
  - `pixi run python scripts/deepseek-ocr-infer-one.py \
    -i datasets/omnidocbench/subsets/dev-3.txt -o tmp/vendor-out --device cuda:0`

## File Touchpoints
- NVTX session: `src/llm_perf_opt/runners/dsocr_session.py`
- Stage‑1 runner: `src/llm_perf_opt/runners/llm_profile_runner.py`
- Vendor code (reference): `models/deepseek-ocr/modeling_deepseekocr.py`, `models/deepseek-ocr/modeling_deepseekv2.py`
- Inspector: `scripts/inspect-deepseek-ocr-infer.py`

## Notes
- We added vendor‑style logging to the cached vendor file, but stdout didn’t surface the prints in script runs — the runtime may read from a different module copy. The inspector script identifies the exact file to patch if needed.
- The renderer is already parity‑correct (regex + 999 normalization + crops). Once generate‑time injection is aligned, multiple boxes should appear without vendor fallback.

