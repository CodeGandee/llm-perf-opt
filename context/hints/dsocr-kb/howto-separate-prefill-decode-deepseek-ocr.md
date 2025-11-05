**Purpose**
- Profile DeepSeek-OCR prefill and decode as separate stages by calling different functions: a forward pass for prefill, and a token-by-token loop for decode using the KV cache.

**Quick Summary**
- Prefill: run a single forward on the full prompt with image arguments to build `past_key_values` (KV cache). Visual features are computed and injected only during this step.
- Decode: iterate single-token steps using the cache. Omit image tensors; the model skips visual compute when `seq_len == 1`.

**Code References**
- Image embedding injection gate: `models/deepseek-ocr/modeling_deepseekocr.py:403`
- Visual feature compute + projector + masked scatter: `models/deepseek-ocr/modeling_deepseekocr.py:422`, `426`–`464`, `505`
- Delegate to base transformer forward (builds cache): `models/deepseek-ocr/modeling_deepseekocr.py:510`–`515`
- Generation-time input prep (OCR wrapper): `models/deepseek-ocr/modeling_deepseekocr.py:620`–`690`
- Example generate call wiring image args: `models/deepseek-ocr/modeling_deepseekocr.py:916`–`929`
- Base transformer forward and cache handling: `models/deepseek-ocr/modeling_deepseekv2.py:1540`–`1574`, `1607`–`1637`

**Inputs Overview**
- `input_ids`: tokenized prompt including `<image>` placeholders (full sequence for prefill; 1 last token for decode)
- `images`: list of per-sample tuples `(images_crop, images_ori)`; tensors on CUDA, typically `bfloat16`
- `images_seq_mask`: `BoolTensor [B, T]` marking positions of image tokens for embedding replacement
- `images_spatial_crop`: `LongTensor [B, 2]` with `[width_crop_num, height_crop_num]`
- `attention_mask`: `LongTensor [B, T]` causal mask with 1s for valid tokens

Note: At decode, you can omit `images`, `images_seq_mask`, `images_spatial_crop`. The image branch is gated off because `seq_len == 1` (see line `models/deepseek-ocr/modeling_deepseekocr.py:403`).

**Prefill Only (Build KV Cache)**
- Run the language model forward once on the full prompt + image arguments with `use_cache=True`. This computes and injects visual embeddings and returns `past_key_values` for decode.

```python
# model: DeepseekOCRForCausalLM (HF model)
# Assumes you already constructed: input_ids [B, T], attention_mask [B, T]
# images = [(images_crop, images_ori)] for each batch item
# images_seq_mask [B, T] bool, images_spatial_crop [B, 2] long

model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(
        input_ids=input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        past_key_values=None,
        use_cache=True,
        images=images,  # [(images_crop.cuda(), images_ori.cuda())]
        images_seq_mask=images_seq_mask.cuda(),
        images_spatial_crop=images_spatial_crop.cuda(),
        return_dict=True,
    )

# Cache for decode
past_kv = outputs.past_key_values
# Optionally keep prefill logits
prefill_logits = outputs.logits  # shape [B, T, vocab]
```

Why this is “prefill”
- The image-to-text embedding injection happens only when `seq_len != 1` (line `models/deepseek-ocr/modeling_deepseekocr.py:403`).
- Visual features are computed and injected via `masked_scatter_` (line `models/deepseek-ocr/modeling_deepseekocr.py:505`).
- The base transformer builds the KV cache during this call (lines `models/deepseek-ocr/modeling_deepseekv2.py:1540`–`1574`, `1607`–`1637`).

**Decode Only (Token Loop Using Cache)**
- Starting from the last token, iterate one step at a time, passing `past_key_values` and updating `attention_mask`. Skip image kwargs; the visual path is gated off in decode.

```python
B = input_ids.size(0)
device = input_ids.device

# Seed the loop with the last token of the prefill (or a BOS per your protocol)
next_input_ids = input_ids[:, -1:].contiguous()  # [B, 1]
past_kv = past_kv  # from prefill
attn_mask = attention_mask.clone()  # starts as prefill mask [B, T]

generated_tokens = []
max_new_tokens = 128
eos_id = tokenizer.eos_token_id

for _ in range(max_new_tokens):
    # Prepare inputs for one decode step; omit image args on purpose
    prepared = model.prepare_inputs_for_generation(
        next_input_ids,
        past_key_values=past_kv,
        attention_mask=attn_mask,
        use_cache=True,
        # images=None, images_seq_mask=None, images_spatial_crop=None
    )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**prepared, return_dict=True)

    step_logits = out.logits[:, -1, :]  # [B, vocab]
    next_token = torch.argmax(step_logits, dim=-1)  # or sample
    generated_tokens.append(next_token)

    # Update for next step
    next_input_ids = next_token.unsqueeze(1)  # [B, 1]
    past_kv = out.past_key_values
    attn_mask = torch.cat(
        [attn_mask, torch.ones((B, 1), dtype=attn_mask.dtype, device=device)], dim=1
    )

    # Early stop if everyone hit EOS
    if eos_id is not None and (next_token == eos_id).all().item():
        break

# Final token tensor
generated = torch.stack(generated_tokens, dim=1)  # [B, steps]
```

Notes
- The decode loop uses the wrapper’s `prepare_inputs_for_generation` (OCR variant) which computes `position_ids` and slices inputs using `past_key_values` (lines `models/deepseek-ocr/modeling_deepseekocr.py:620`–`690`).
- Omitting image arguments is safe during decode because the image branch will not execute when `seq_len == 1` (line `models/deepseek-ocr/modeling_deepseekocr.py:403`).

**One-Shot Generate (FYI)**
- If you don’t need separate profiling, you can let `generate()` handle both phases in one call by supplying the image args once:

```python
streamer = None  # or a HF streamer
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    gen_ids = model.generate(
        input_ids.cuda(),
        images=images,
        images_seq_mask=images_seq_mask.cuda(),
        images_spatial_crop=images_spatial_crop.cuda(),
        use_cache=True,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        # temperature=0.0,
        # do_sample=False,
        streamer=streamer,
    )
```

This matches the in-repo example call (lines `models/deepseek-ocr/modeling_deepseekocr.py:916`–`929`).

**Gotchas / Tips**
- Dtypes: the vision stack and injected embeddings are handled in `bfloat16` autocast in the example code.
- Images packing: pass `images` as a Python list of length `B` with tuples `(images_crop, images_ori)`; each is a 4D tensor on CUDA. If there are no crops, use a zero tensor for `images_crop` (see `models/deepseek-ocr/modeling_deepseekocr.py:904`–`909`).
- Masks: ensure `images_seq_mask` aligns exactly with `<image>` token positions in `input_ids` so `masked_scatter_` lands visual embeddings correctly (line `models/deepseek-ocr/modeling_deepseekocr.py:505`).
- Attention mask growth: extend `attention_mask` by one column per decode step so positions/causal mask stay consistent.
- BOS/EOS/IDs: BOS appears to be `0`, EOS in tokenizer config, and a special image token id `128815` is used in packing (see `models/deepseek-ocr/modeling_deepseekocr.py:760`–`836`).

**Why This Separation Works**
- The model explicitly checks `seq_len != 1` before doing any image compute, so image features are only computed during prefill (line `models/deepseek-ocr/modeling_deepseekocr.py:403`). Decode reuses `past_key_values` and processes one token per call, skipping the image branch entirely.

