**What to Refactor**
- File: src/llm_perf_opt/runners/dsocr_session.py
- Scope:
  - Replace single-call `model.generate(...)` path with explicit two-phase inference: a prefill forward pass and a token-by-token decode loop.
  - Keep existing preprocessing (prompt/image packing) and NVTX stage hooks.
  - Preserve the public API of `DeepSeekOCRSession` where possible (e.g., `from_local()`, `run_inference(...)`).

**Why Refactor**
- Current implementation skips prefill (see dsocr_session.py:394–401) and uses `generate()` to do both prefill and decode (dsocr_session.py:403–441), which prevents separate profiling of prefill vs decode.
- We need distinct NVTX ranges and wall-clock timings for prefill and decode to attribute costs (language prefill, decoding, and the vision stages that occur only during prefill).
- Align runtime behavior with the validated manual script (tests/manual/inference/manual_dsocr_prefill_decode.py), which demonstrates stable separate prefill+decode semantics.

**How to Refactor**
1) Prompt and Inputs
   - Continue normalizing the user prompt to the vendor format. Update to include vendor grounding tokens by default to enable box spans (optional switch if we want to control this).
   - Build `input_ids`, `images`, `images_seq_mask`, `images_spatial_crop` exactly as done today (dsocr_session retains this logic); verify token layout matches the vendor.

2) Prefill Stage (forward pass)
   - Enter `prefill_range()` NVTX range and call `model(...)` directly with:
     - `input_ids`, `attention_mask` (all ones of length T), `use_cache=True`, and the vision args `images`, `images_seq_mask`, `images_spatial_crop`.
   - Record `prefill_ms`. Extract `past_key_values` and last-step logits to seed decoding.

3) Decode Stage (loop)
   - Enter `decode_range()` NVTX range. Initialize `next_input_ids` with the argmax of the last logits from prefill and `past_kv` from prefill.
   - For each step up to `max_new_tokens`:
     - Append one column of ones to `attention_mask` BEFORE calling `prepare_inputs_for_generation(...)` to avoid mask length mismatch (this mirrors the manual script and prevents causal mask errors).
     - Call `prepared = model.prepare_inputs_for_generation(next_input_ids, past_key_values=past_kv, attention_mask=attention_mask, use_cache=True)`.
     - Call `out = model(**prepared, return_dict=True)`, compute next token (greedy or sampling as configured), update `past_kv`, and early-stop on EOS.
   - Record `decode_ms` and number of generated tokens.

4) Output Handling
   - If `return_text=True`, decode the generated tokens into text.
   - Keep existing NVTX vision sub-stage timing aggregation via hooks (sam/clip/projector).
   - Return the same result structure as today, but with accurate `prefill_ms` and `decode_ms`.

5) API Compatibility & Config
   - Maintain `run_inference(...)` signature; interpret `infer` dict to control decode behavior (e.g., `temperature`, `top_p`, `top_k`, `no_repeat_ngram_size`). For greedy parity set `temperature=0.0` by default.
   - Optionally add a parameter to toggle vendor grounding prompt composition (default True) to encourage `<|ref|>/<|det|>` spans for visualization.

6) Tests / Manual Validation
   - Reuse tests/manual/inference/manual_dsocr_prefill_decode.py inputs to validate outputs and ensure no attention/caching mismatches.
   - Ensure visual outputs can be produced downstream (existing visualize/annotations helpers).

**Before/After Snippets**
- Before (single-call generate; dsocr_session.py:394–441):
```python
# Prefill skipped (vendor parity) | prefill_ms measured but no forward
with decode_range():
    out = self.m_model.generate(
        **inputs,
        images=images,
        images_seq_mask=images_seq_mask,
        images_spatial_crop=_spatial,
        max_new_tokens=int(max_new_tokens),
        eos_token_id=..., use_cache=True, temperature=0.0,
        no_repeat_ngram_size=20,
    )
```

- After (explicit prefill + decode):
```python
# Prefill
self._reset_stage_time_accum()
with prefill_range():
    outputs_prefill = self.m_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=True,
        images=images,
        images_seq_mask=images_seq_mask,
        images_spatial_crop=images_spatial_crop,
        return_dict=True,
    )
past_kv = outputs_prefill.past_key_values
last_logits = outputs_prefill.logits[:, -1, :]
next_token = last_logits.argmax(dim=-1)

# Decode
with decode_range():
    next_input_ids = next_token.unsqueeze(1)
    generated = []
    for _ in range(max_new_tokens):
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.size(0), 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        prepared = self.m_model.prepare_inputs_for_generation(
            next_input_ids, past_key_values=past_kv,
            attention_mask=attention_mask, use_cache=True,
        )
        out = self.m_model(**prepared, return_dict=True)
        step_logits = out.logits[:, -1, :]
        next_token = step_logits.argmax(dim=-1)  # or sample per infer config
        generated.append(next_token)
        next_input_ids = next_token.unsqueeze(1)
        past_kv = out.past_key_values
        if eos_id is not None and (next_token == eos_id).all():
            break
```

**Impact Analysis**
- Functional impact:
  - Output tokens may have small differences vs `generate()` defaults if sampling controls differ; mitigate by defaulting to greedy decode (`temperature=0.0`) and honoring `infer` overrides.
  - Attention/caching correctness: risk of shape mismatch (e.g., causal mask). Mitigation: grow `attention_mask` before calling `prepare_inputs_for_generation(...)` (pattern validated in the manual script at tests/manual/inference/manual_dsocr_prefill_decode.py:436–449).
  - Performance: Slight overhead vs `generate()` due to Python loop; acceptable for profiling use case where separation is required.
  - Vision stage timing: Preserved via existing NVTX hooks (`sam`, `clip`, `projector`). Timing only incurred during prefill.

- Compatibility:
  - `run_inference` return fields unchanged: `prefill_ms`, `decode_ms`, `tokens`, `prefill_len`, `vision_ms` (+ sub-stages when present), optional `text`.
  - No changes to constructor or `from_local()` signature.

**Expected Outcome**
- Accurate, separable timings for prefill and decode phases in `DeepSeekOCRSession.run_inference`.
- Equivalent or near-equivalent outputs to the vendor `infer()` path, with the added ability to profile stages independently.
- Continued support for vendor-style visualization of results downstream.

**References**
- Code files
  - src/llm_perf_opt/runners/dsocr_session.py:394–441 (current generate path; prefill skipped at 394–401)
  - tests/manual/inference/manual_dsocr_prefill_decode.py:406–420 (prefill) and 436–459 (decode loop)
  - models/deepseek-ocr/modeling_deepseekocr.py:510–515 (delegation to base model), 620–690 (prepare_inputs_for_generation)
  - src/llm_perf_opt/visualize/annotations.py:20–112, 162–210 (parsing and writing vendor-style outputs)
- Third-party libs (Context7 IDs)
  - /huggingface/transformers
  - /pytorch/pytorch

