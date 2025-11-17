**Purpose**
- Provide a repeatable process to enumerate all DeepSeek-OCR modules/ops that need analytical cost models (vision encoder, projector, LLM prefill/decode, custom heads, etc.).

**Quick Summary**
- Use three complementary views of the model:
  - Static analyzer (`scripts/deepseek-ocr-static-analysis.py`) to get per-stage operator summaries.
  - Source code (`models/deepseek-ocr/*.py`) to map those operators to concrete modules and custom layers.
  - The paper (`paper-source/tex/main.tex`) to identify conceptual blocks that might be fused or implemented non‑canonically.
- Record the final list of “analysis modules” (per stage + per op family) in a small JSON/YAML or markdown table under `extern/modelmeter/models/deepseek_ocr/` to drive the analytic models.

**Step 1 – Run static analysis and inspect outputs**
- Invoke the static analyzer to produce a machine‑readable operator breakdown:

```bash
pixi run python scripts/deepseek-ocr-static-analysis.py \
  --device cuda:0 \
  --model "$(pwd)/models/deepseek-ocr" \
  --output tmp/dsocr-static-ops \
  --base-size 1024 \
  --image-size 640 \
  --seq-len 1024 \
  --crop-mode 1 \
  --use-flash-attn 1
```

- Open the generated files:
  - `tmp/dsocr-static-ops/static_compute.json`
  - `tmp/dsocr-static-ops/static_compute.md`
- From `static_compute.json`:
  - Look at `["stages"]` → `["sam" | "clip" | "projector" | "prefill" | "decode"]`.
  - For each stage, inspect the `"operators"` map (keys are op names, values are FLOPs or counts).
  - Collect the *operator families* that appear (e.g., `mm`, `conv2d`, `layer_norm`, `rms_norm`, `flash_attn`, `silu`, `swiglu`, `embedding`, `linear`, `softmax`, etc.).

**Step 2 – Map stage names to concrete modules**
- Open the static analyzer implementation to understand how stages are defined:

```bash
rg "DeepseekOCRStaticAnalyzer" -n src llm_perf_opt
sed -n '1,260p' src/llm_perf_opt/runners/dsocr_analyzer.py
```

- Identify, for each stage (sam/clip/projector/prefill/decode):
  - Which `torch.nn.Module` instances are included in `m_stage_module_map`.
  - How submodules are grouped or excluded (e.g., whether LLM blocks are grouped into a single “prefill” module).
- Write down a mapping like:

```text
sam        -> DeepEncoder.SAM backbone (window attention blocks, early convs)
clip       -> CLIP visual encoder inside DeepEncoder (global attention blocks)
projector  -> Vision-to-text projector & fusion MLP before LLM tokens
prefill    -> DeepSeek-3B MoE blocks for full-sequence processing with images
decode     -> Same LLM blocks but with KV cache and seq_len=1
```

**Step 3 – Inspect model source to enumerate modules**
- Open the DeepSeek-OCR model source:

```bash
sed -n '1,260p' models/deepseek-ocr/modeling_deepseekocr.py
sed -n '1,260p' models/deepseek-ocr/modeling_deepseekv2.py
sed -n '1,260p' models/deepseek-ocr/deepencoder.py
```

- For `deepencoder.py`:
  - Locate the SAM backbone: note key building blocks (e.g., patch embedding, window attention, down/upsampling).
  - Locate the CLIP visual encoder: note transformer block layout, attention/MLP structure, and any non‑standard ops.
  - Identify the *vision-to-text projector*: e.g., convs/linears that compress visual tokens and map to LLM token dimension.
- For `modeling_deepseekocr.py`:
  - Find where image features are computed and injected into the token sequence (search for `images_seq_mask`, `masked_scatter_`, `<image>` ids).
  - Identify wrapper modules or helpers that define the OCR inference path (`infer`, generation helpers, KV cache plumbing).
- For `modeling_deepseekv2.py`:
  - Inspect transformer blocks (attention + MLP):
    - Attention type: standard MHA vs GQA vs grouped-queries; FlashAttention usage.
    - MLP type: SwigLU or other gated activation; hidden size ratios.
  - Note any custom normalization, rotary embeddings, or KV cache layout that differ from a “vanilla” decoder‑only LM.

**Step 4 – Cross-check with vendor infer() path**
- Use `scripts/deepseek-ocr-infer-one.py` to understand which code paths are exercised in real inference:

```bash
sed -n '1,260p' scripts/deepseek-ocr-infer-one.py
```

- Trace `model.infer(...)` (from vendor `trust_remote_code`) into `models/deepseek-ocr/modeling_deepseekocr.py`:
  - Identify preprocessing steps (image tiling/cropping, prompt assembly with `<image>` tokens).
  - Confirm which modules are active during prefill vs decode (e.g., image branch only when `seq_len != 1`).
- This step ensures the module list focuses on *inference‑critical* ops and ignores unused training‑only helpers.

**Step 5 – Use the paper to validate conceptual blocks**
- Open the LaTeX paper and search for architectural descriptions:

```bash
rg "DeepEncoder" paper-source/tex/main.tex
rg "projector" paper-source/tex/main.tex
rg "latent" paper-source/tex/main.tex
```

- From the paper, summarize:
  - DeepEncoder composition (SAM-base + CLIP-large + downsampling compressor).
  - The exact role of the projector (e.g., reducing tokens by 16× before CLIP, mapping to LLM latent tokens).
  - Any *fused* or *conceptual* modules that might correspond to multiple low-level ops (e.g., “optical compressor”, “visual tokenizer”).
- Cross-check these conceptual modules against the Stage → Module mapping and operator families you collected in Steps 1–3.

**Step 6 – Draft the DeepSeek-OCR analysis module catalog**
- Create a small catalog file under `extern/modelmeter/models/deepseek_ocr/`, for example:

```bash
cat > extern/modelmeter/models/deepseek_ocr/module_catalog.md << 'EOF'
EOF
```

- In that catalog, define rows like:

```text
Stage      | Module/Block             | Ops / Families                  | Notes
---------- | ------------------------ | --------------------------------| -----------------------------
sam        | SAM window-attn block    | conv2d, layer_norm, mha, gelu   | From deepencoder.SAM backbone
clip       | CLIP transformer block   | mha, layer_norm, mlp, gelu      | CLIP-large; see deepencoder.py
projector  | Vision projector         | conv2d/linear, pooling          | Reduces tokens, maps to LLM dim
prefill    | LLM transformer block    | gqa_flashattn, swiglu, rms_norm | seq_len > 1, images branch on
decode     | LLM transformer block    | gqa_flashattn, swiglu, rms_norm | seq_len == 1, KV cache only
head       | LM head + softmax        | linear, softmax                 | Vocab projection + softmax
```

- The goal is for this catalog to be the *single source of truth* for which analytic layer models we must implement under `extern/modelmeter/models/deepseek_ocr/layers/`.

**Step 7 – Derive required analytic layer types**
- From the catalog, enumerate analytic layer types needed:
  - Vision: `Conv2d`, `WindowAttention` / `SelfAttention2D`, `LayerNorm`, `Pooling`.
  - LLM: `Embedding`, `GQAFlashattention`/`MHAFlashattention`, `RMSNorm`, `SwiGLU`, `Linear`, `Head` (vocab projection).
  - KV cache: explicit analytic model for cache size and I/O per token.
- Compare this list with existing `extern/modelmeter/layers/*`:
  - Reuse `Linear`, `Embedding`, `Head`, `MHAFlashattention`, `GQAFlashattention`, `RMSNorm`, `Silu`, `SwiGLU` where semantics match.
  - Plan new DeepSeek‑specific variants under `extern/modelmeter/models/deepseek_ocr/layers/` for any custom behavior:
    - e.g., special projector that fuses conv + linear.
    - custom KV‑cache layout or flash‑attn configuration.

**Outcome**
- After following these steps, you will have:
  - A concrete list of DeepSeek-OCR modules/ops grouped by stage.
  - A `module_catalog` file under `extern/modelmeter/models/deepseek_ocr/` capturing that list.
  - A derived checklist of analytic layer types to implement or adapt in `extern/modelmeter/models/deepseek_ocr/layers/`.

