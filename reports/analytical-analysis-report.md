# Analytical Analysis of DeepSeek-OCR Analytic Model

This report summarizes the current analytical (static) performance model for the DeepSeek-OCR stack as implemented under
`extern/modelmeter/models/deepseek_ocr`. It focuses on:
- Theoretical FLOPs and memory usage for key layers and aggregating modules.
- Arithmetic intensity (AI) patterns derived from the analytic formulas.
- How those metrics scale with input size and model configuration.
- Likely performance bottlenecks and gaps in the current analytic coverage.

The analysis is purely model-based (closed-form), using the conventions from:
- `extern/modelmeter/models/deepseek_ocr/LAYER_IMPL_GUIDE.md`
- `extern/modelmeter/models/deepseek_ocr/TENSOR_OR_CUDA_CORE_GUIDE.md`
- `extern/modelmeter/models/deepseek_ocr/LAYER_IO_ESTIMATION_GUIDE.md`

All FLOPs follow the `$2$ FLOPs per MAC` convention, Tensor Core FLOPs are separated from CUDA-core FLOPs, I/O is reported
in terabits, and memory in GB.

---

## Constituent Layers (Modules)

The DeepSeek-OCR analytic model is organized into four main groups:

- **Vision stack (SAM-style image encoder)**
  - `PatchEmbed` (`layers/vision/patch_embed.py`)
  - `Attention` (`layers/vision/attention.py`)
  - `MLPBlock` (`layers/vision/mlp_block.py`)
  - `LayerNorm2d` (`layers/vision/layer_norm2d.py`)
  - `Block` (`layers/vision/block.py`)
  - `ImageEncoderViT` (`layers/vision/image_encoder_vit.py`)

- **CLIP-style vision tower**
  - `CLIPVisionEmbeddings` (`layers/vision/clip_vision_embeddings.py`)
  - `NoTPAttention` (`layers/vision/notp_attention.py`)
  - `NoTPFeedForward` (`layers/vision/notp_feedforward.py`)
  - `NoTPTransformerBlock` (`layers/vision/notp_transformer_block.py`)
  - `NoTPTransformer` (`layers/vision/notp_transformer.py`)
  - `VitModel` (`layers/vision/vit_model.py`)

- **Vision→decoder bridge**
  - `MlpProjector` (`layers/vision/mlp_projector.py`)

- **Decoder / LLaMA stack**
  - `DeepseekV2RMSNorm` (`layers/decoder/deepseek_v2_rms_norm.py`)
  - `LlamaRotaryEmbedding` (`layers/llama/llama_rotary_embedding.py`)
  - `LlamaFlashAttention2` (`layers/llama/llama_flash_attention2.py`)
  - `DeepseekV2MLP` (`layers/decoder/deepseek_v2_mlp.py`)
  - `MoEGate` (`layers/decoder/moe_gate.py`)
  - `DeepseekV2MoE` (`layers/decoder/deepseek_v2_moe.py`)
  - `DeepseekV2DecoderLayer` (`layers/decoder/deepseek_v2_decoder_layer.py`)

- **Root aggregator**
  - `DeepseekOCRModel` (`layers/core/deepseek_ocr_model.py`)

Each of these layers is a `BaseLayer` subclass with a consistent analytic surface:
- Forward/backward Tensor Core and CUDA-core FLOPs.
- Forward/backward I/O (`*_cal_io`) and arithmetic intensity.
- Forward/backward memory footprints (weights, activations, KV cache).
- Optional `verify_by_impl` helpers that compare analytic FLOPs to reference PyTorch implementations.

**Model Architecture Sketch**

At a high level, the DeepSeek-OCR analytic model can be viewed as:

1. **Image path**
   - SAM-style `ImageEncoderViT` consumes preprocessed images (~1024×640, `patch_size=16`).
   - CLIP-style `VitModel` consumes SAM features via `CLIPVisionEmbeddings` and `NoTPTransformer`.
2. **Projection bridge**
   - `MlpProjector` maps concatenated SAM+CLIP features into the decoder embedding space.
3. **Decoder path**
   - A stack of DeepSeek-V2 decoder layers, each modeled by `DeepseekV2DecoderLayer` (RMSNorm → FlashAttention2 →
     RMSNorm → MLP or MoE).
4. **Root analytic wrapper**
   - `DeepseekOCRModel` is intended to aggregate vision + decoder metrics into a single model-level view.

The sections below summarize how each group models FLOPs, I/O, memory, and AI, and how those quantities scale.

---

## Vision Stack (SAM-Style Image Encoder)

### PatchEmbed

**Definition**
- Models Conv2d-based patch embedding from `(B, C_in, H, W)` to `(B, C_out, H_out, W_out)` with `kernel=stride=patch_size`.
- Used both in SAM’s `ImageEncoderViT` and CLIP embeddings when `use_precomputed_patch_embeds=False`.

**Analytical model**
- **Tensor Core FLOPs**: standard Conv2d formula
  - $F_{\mathrm{tc}} \approx 2 \cdot B \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot k^{2} \cdot C_{\text{in}} \cdot C_{\text{out}}$.
- **CUDA-core FLOPs**: `0` (conv is treated as Tensor Core–dominated).
- **Backward FLOPs**: $\approx 2 \times \text{forward}$ for the Tensor Core path; CUDA path remains `0`.
- **I/O (Tb)**:
  - Forward: reads input image + writes patch embeddings.
  - $O_{\mathrm{io}} \propto B \cdot (C_{\text{in}} \cdot H \cdot W + C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}})$.
  - Backward: $\approx 2 \times \text{forward}$ I/O.
- **Memory (GB)**:
  - Parameters: $\text{C\_in} \cdot \text{C\_out} \cdot k^{2}$ fp16 weights.
  - Activations: input + output feature maps.
  - KV cache: `0`.

**Scaling and implications**
- FLOPs and activation memory scale linearly with `B`, `H_out * W_out`, and channels.
- With typical SAM settings (e.g., `image_size=1024`, `patch_size=16`), patch embedding is compute-heavy but
  still lower-intensity than the transformer blocks that follow.

### Attention (vision encoder)

**Definition**
- `Attention` models SAM-style 2D multi-head self-attention over `num_windows` windows of `window_area = H * W` tokens.
- Supports both global (`num_windows=B, window_area=S`) and windowed attention (many small windows).

**Analytical model**
- Treats each window as a batch element:
  - $B_{\text{eff}} = \text{num\_windows}$, $S = \text{window\_area}$, $C = \text{dim}$, $H = \text{num\_heads}$, $d = C / H$.
- **Tensor Core FLOPs (TFLOPs)**:
  - QKV projection: $2 \cdot B_{\text{eff}} \cdot S \cdot C \cdot (3C)$.
  - SDPA core: $2 \cdot B_{\text{eff}} \cdot H \cdot S^{2} \cdot d$ for $QK^{\top}$ plus the same for $AV$, giving $4 \cdot B_{\text{eff}} \cdot H \cdot S^{2} \cdot d$.
  - Output projection: $2 \cdot B_{\text{eff}} \cdot S \cdot C \cdot C$.
  - Optional relative position bias adds a smaller CUDA-core term folded into Tensor Core estimate as a low-order term.
- **CUDA-core FLOPs**: currently modeled as `0` (softmax and minor elementwise ops are neglected for simplicity).
- **Backward FLOPs**: $\approx 2 \times \text{forward}$ tensor-core FLOPs, CUDA-path kept at $0$.
- **I/O (Tb)**:
  - Forward: input ($B_{\text{eff}} \cdot S \cdot C$), QKV ($B_{\text{eff}} \cdot S \cdot 3C$), output ($B_{\text{eff}} \cdot S \cdot C$), fp16.
  - Backward: $\approx 2 \times \text{forward}$ I/O.
- **Memory (GB)**:
  - Parameters: $3C^{2}$ (QKV) + $C^{2}$ (output).
  - Activations: input + QKV + attention buffers.
    - `use_flash_attention=True` collapses the $S^{2}$ term into something closer to $B_{\text{eff}} \cdot S \cdot C$.
    - `False` path approximates several `S × S` fp32 matrices (logits + softmax buffers).
  - KV cache: `0` (vision stack does not maintain a persistent KV cache).

**Scaling and implications**
- FLOPs are dominated by the $\mathcal{O}(B_{\text{eff}} \cdot H \cdot S^{2} \cdot d)$ SDPA term for large windows; windowed attention reduces $S$ but
  increases $B_{\text{eff}}$, trading global $S^{2}$ complexity for more but smaller windows.
- Arithmetic intensity is high for typical SAM settings; attention is compute-heavy and a major consumer of Tensor Core
  FLOPs in the vision stack.

### MLPBlock (vision encoder)

**Definition**
- Two-layer MLP with GELU: `dim → mlp_dim → dim` per token.
- Operates on sequences `(B, S, dim)` after flattening spatial grids.

**Analytical model**
- **Tensor Core FLOPs**:
  - `lin1`: $2 \cdot B \cdot S \cdot \text{dim} \cdot \text{mlp\_dim}$.
  - `lin2`: $2 \cdot B \cdot S \cdot \text{mlp\_dim} \cdot \text{dim}$.
- **CUDA-core FLOPs**:
  - GELU cost: $\text{activation\_flops\_per\_element} \cdot B \cdot S \cdot \text{mlp\_dim}$ (default $\approx 4$ FLOPs per element).
- **Backward**: both Tensor and CUDA paths modeled as $\approx 2 \times \text{forward}$.
- **I/O and memory**:
  - I/O counts input, hidden, and output activations ($B \cdot S \cdot (\text{dim} + \text{mlp\_dim} + \text{dim})$).
  - Parameters: two linear weight matrices (`dim * mlp_dim` and `mlp_dim * dim`).
  - Activations: same shapes as I/O; no KV cache.

**Scaling and implications**
- FLOPs are $\mathcal{O}(B \cdot S \cdot \text{dim} \cdot \text{mlp\_dim})$, linear in sequence length.
- For typical $\text{mlp\_dim} \approx 4 \cdot \text{dim}$, the MLP contributes a similar order of FLOPs as attention per block (for moderate
  $S$), but with higher AI because it lacks the $S^{2}$ term.

### LayerNorm2d

**Definition**
- NCHW LayerNorm used in SAM’s convolutional neck.
- Operates on `(B, C, H, W)`.

**Analytical model**
- CUDA-core-only:
  - Forward FLOPs: $\approx 5 \cdot B \cdot C \cdot H \cdot W$.
  - Backward FLOPs: $\approx 2 \times \text{forward}$.
  - I/O: $\approx 4 \cdot B \cdot C \cdot H \cdot W$ fp16 values counted (multiple passes over activations).
  - Parameters: `2 * C` fp16 values (scale + bias).
  - Activations: input tensor (output is same shape).

**Scaling and implications**
- Low arithmetic intensity relative to attention/MLP.
- Clearly memory/latency bound; a candidate for fusion around conv/MLP where possible.

### Block and ImageEncoderViT (aggregators)

**Block**
- Wraps one `Attention` + one `MLPBlock`, plus two LayerNorms and window partition/unpartition overhead.
- **Tensor Core FLOPs**: sum of child attention + MLP tensor-core FLOPs.
- **CUDA-core FLOPs**:
  - Sum of child CUDA FLOPs + configurable `norm_flops_tflops` to capture norm/window overhead.
  - Backward multiplies the norm overhead by ~2.
- **I/O and memory**: simple sums of child metrics; known to over-estimate peak memory but adequate for relative
  comparisons.

**ImageEncoderViT**
- Composes:
  - One `PatchEmbed`.
  - A stack of `depth` `Block`s with a mixture of windowed and global attention.
  - A convolutional neck (several Conv2d layers + two `LayerNorm2d`).
- **Tensor Core FLOPs**:
  - Sum of patch embedding + all Block tensor-core FLOPs + conv neck FLOPs (modeled as Conv2d).
- **CUDA-core FLOPs**:
  - Sum of Block CUDA FLOPs + two `LayerNorm2d` FLOPs in the neck.
- **Backward FLOPs**:
  - Aggregates child backward FLOPs; conv layers approximated as $\approx 2 \times \text{forward}$.
- **I/O and memory**:
  - Aggregates child I/O and memory, plus explicit conv I/O/weight/activation terms.
  - KV cache is `0` (pure encoder).

**Scaling and implications**
- For a fixed backbone (e.g., SAM-B), encoder cost scales with batch size and image resolution (`image_size` and
  `patch_size`).
- Windowed blocks reduce attention’s $S^{2}$ term at constant grid size by trading global attention for more windows.
- Within DeepSeek-OCR, this encoder is typically executed once per rendered page, so its amortized cost per decoded
  token is small compared to the decoder stack, even though it is compute-heavier than most other non-decoder layers.

---

## CLIP-Style Vision Tower

### CLIPVisionEmbeddings

**Definition**
- CLIP-style patch + position + CLS token embeddings.
- Can either:
  - reuse precomputed SAM patch embeddings (`use_precomputed_patch_embeds=True`), or
  - compute Conv2d patch embeddings internally.

**Analytical model**
- **Tensor Core FLOPs**:
  - $0$ when reusing precomputed embeddings (only adds CLS + positions).
  - Otherwise, delegated entirely to `PatchEmbed`.
- **CUDA-core FLOPs**:
  - Position/CLS additions treated as negligible and modeled as $0$.
- **I/O and memory**:
  - I/O dominated by emitted embeddings: $B \cdot (\text{num\_patches} + 1) \cdot \text{hidden\_size}$.
  - When not precomputed, PatchEmbed I/O/memory is added.
  - Parameter memory includes patch Conv2d weights + CLS + position embeddings.

**Implications**
- In the current DeepSeek-OCR setup, CLIP often reuses SAM features, so analytic FLOPs for this module are near-zero
  (only parameter memory is significant).

### NoTPAttention, NoTPFeedForward, NoTPTransformerBlock, NoTPTransformer, VitModel

**NoTPAttention / NoTPFeedForward**
- Essentially the same analytic structure as vision `Attention` and `MLPBlock` but operating on CLIP token sequences
  (e.g., $S \approx 257$).
- **Attention**: QKV + SDPA + output projections on Tensor Cores, CUDA-core FLOPs ignored; I/O counts input/QKV/output;
  memory includes weights + activations; no KV cache.
- **FeedForward**: dense GEMMs to/from a hidden dimension (QuickGELU); GELU treated as CUDA-core; I/O and memory mirror
  `MLPBlock`.

**NoTPTransformerBlock**
- Aggregates one `NoTPAttention` + one `NoTPFeedForward` + two LayerNorms (norm cost passed via `norm_flops_tflops`).
- Follows the same aggregation pattern as SAM `Block`.

**NoTPTransformer**
- Aggregates a stack of `NoTPTransformerBlock` instances:
  - FLOPs, I/O, and memory are simple sums across blocks.

**VitModel**
- Wraps:
  - `CLIPVisionEmbeddings`,
  - `NoTPTransformer`, and
  - a final LayerNorm (modeled as CUDA-core-only work).
- **Tensor Core FLOPs**: embeddings + transformer tensor-core FLOPs.
- **CUDA-core FLOPs**: transformer CUDA FLOPs + final LayerNorm FLOPs.
- **I/O and memory**: sums of child metrics; final norm I/O/memory is negligible and mostly ignored.

**Scaling and implications**
- CLIP ViT cost scales with the number of CLIP layers, token count (`seq_len`), and hidden size.
- Arithmetic intensity is similar to the SAM transformer: attention and MLPs are compute-heavy; norms are
  memory-oriented.
- In DeepSeek-OCR, CLIP ViT runs once per input image; combined with SAM, vision-side transformer cost is substantial
  but still amortized over many decode steps.

---

## Vision→Decoder Projector

### MlpProjector

**Definition**
- Bridges concatenated SAM+CLIP features into decoder embedding space.
- Supports three modes:
  - `identity` (no-op),
  - `linear` (single `input_dim → output_dim` projection),
  - `mlp_gelu` (stack of `depth` linear layers with GELU).

**Analytical model**
- **Tensor Core FLOPs**:
  - `identity`: $0$.
  - `linear`: $2 \cdot B \cdot N \cdot \text{input\_dim} \cdot \text{output\_dim}$.
  - `mlp_gelu`: first layer `input_dim → output_dim`, plus `depth - 1` layers `output_dim → output_dim`.
- **CUDA-core FLOPs**:
  - Only for `mlp_gelu`, modeling GELU as:
    - $\text{activation\_flops\_per\_element} \cdot B \cdot N \cdot \text{output\_dim} \cdot \text{num\_hidden\_layers}$.
- **I/O and memory**:
  - I/O: read $B \cdot N \cdot \text{input\_dim}$, write $B \cdot N \cdot \text{output\_dim}$ in fp16; backward approximated as $\approx 2 \times \text{forward}$.
  - Parameters: linear weights (and optional hidden layers).
  - Activations: input + output (intermediate activations are folded into the AI approximation).

**Implications**
- For typical settings (e.g., moderate `N` and relatively thin projector), FLOPs are small compared to decoder layers.
- When `mlp_gelu` is deep and `N` is large, projector can become a noticeable but still secondary compute contribution.

---

## Decoder / LLaMA Stack

### DeepseekV2RMSNorm

**Definition**
- RMSNorm used before attention and MLP within each decoder layer.

**Analytical model**
- CUDA-core-only:
  - Forward FLOPs: $\approx 3 \cdot B \cdot S \cdot \text{hidden\_size}$.
  - Backward FLOPs: $\approx 2 \times \text{forward}$.
  - I/O: input + output activations $B \cdot S \cdot \text{hidden\_size}$ counted twice.
  - Parameter memory: one scale vector of length `hidden_size`.

**Implications**
- Low-AI layer; memory-bandwidth-sensitive.
- Its cost per layer is modest compared to attention/MLP but adds up across many decoder layers and both pre-attention
  and pre-MLP norms.

### LlamaRotaryEmbedding

**Definition**
- RoPE primitive modeling cosine/sine embeddings for LLaMA attention.

**Analytical model**
- CUDA-core-only:
  - Forward FLOPs: $\approx 2 \cdot B \cdot (\text{dim} / 2) \cdot S$ for the core frequency-position multiply.
  - Backward: $\approx 2 \times \text{forward}$.
  - I/O: reads/writes cosine and sine embeddings of shape $(B, S, \text{dim})$.
  - Parameter memory: $\approx \text{dim}$ fp16 frequencies.

**Implications**
- Very low compute relative to attention and MLP; mostly relevant for I/O and small parameter memory.
- Included mainly for completeness and to keep attention-layer AI estimates honest when combined with RoPE.

### LlamaFlashAttention2

**Definition**
- Models LLaMA decoder self-attention when `attn_implementation="flash_attention_2"`.
- Uses grouped-query attention via `num_key_value_heads <= num_heads`.

**Analytical model**
- Let $d_{\text{model}} = \text{hidden\_size}$, $H = \text{num\_heads}$, $H_{\text{kv}} = \text{num\_key\_value\_heads}$, $d = d_{\text{model}} / H$, $r = H_{\text{kv}} / H$.
- **Tensor Core FLOPs**:
  - Q: $2 \cdot B \cdot S \cdot d_{\text{model}}^{2}$.
  - K/V: scaled by $r$ each, giving $\approx 2 \cdot B \cdot S \cdot d_{\text{model}}^{2} \cdot 2r$.
  - SDPA core: $\approx 4 \cdot B \cdot H \cdot S^{2} \cdot d$.
  - Output: $2 \cdot B \cdot S \cdot d_{\text{model}}^{2}$.
- **CUDA-core FLOPs**: treated as $0$ (softmax, scaling, masking ignored).
- **Backward FLOPs**: $\approx 2 \times \text{forward}$ Tensor Core FLOPs.
- **I/O and memory**:
  - I/O: counts input, QKV, and output activations.
  - Activations: input + QKV + attention buffers; `use_flash_attention` controls whether a compact $B \cdot S \cdot d_{\text{model}}$
    representation or $S^{2}$ fp32 matrices are assumed.
  - Parameter memory: $3 \cdot d_{\text{model}}^{2} + d_{\text{model}}^{2}$ weights.
  - KV cache: $\approx 2 \cdot B \cdot H_{\text{kv}} \cdot S \cdot d$ fp16, scaled by sequence length; this is the key long-context memory term.

**Scaling and implications**
- FLOPs and KV cache both scale with `S`; attention has an `S^2` compute term and an `O(S)` KV-cache term.
- For long-context prefill, SDPA dominates decoder FLOPs; for autoregressive $S = 1$ decode, KV-cache memory dominates
  but FLOPs shrink to linear in $\text{hidden\_size}$.

### DeepseekV2MLP

**Definition**
- SwiGLU-based MLP used both as the dense decoder MLP and as the per-expert MLP inside MoE blocks.

**Analytical model**
- **Tensor Core FLOPs**:
  - Gate, up, and down projections yield $\approx 6 \cdot B \cdot S \cdot \text{hidden\_size} \cdot \text{intermediate\_size}$.
- **CUDA-core FLOPs**:
  - SwiGLU activation and gating: $\text{activation\_flops\_per\_element} \cdot B \cdot S \cdot \text{intermediate\_size}$.
- **Backward**: Tensor and CUDA FLOPs modeled as $\approx 2 \times \text{forward}$.
- **I/O and memory**:
  - I/O: input + hidden + output activations.
  - Parameters: $\approx 3 \cdot \text{hidden\_size} \cdot \text{intermediate\_size}$.
  - Activations: same as I/O; no KV cache.

**Implications**
- High-AI component; for typical `intermediate_size ~ 8/3 × hidden_size` (DeepSeek-style), MLP compute is comparable to
  or larger than attention compute for moderate `S`.
- A major contributor to decoder Tensor Core FLOPs, especially in non-MoE layers.

### MoEGate and DeepseekV2MoE

**MoEGate**
- Tensor Core FLOPs: $\approx 2 \cdot B \cdot S \cdot \text{hidden\_size} \cdot \text{num\_experts}$ for the expert-scoring projection.
- CUDA-core FLOPs: $\approx 6 \cdot B \cdot S \cdot \text{num\_experts}$ for softmax/top-k/normalization.
- I/O: input ($B \cdot S \cdot h$), logits ($B \cdot S \cdot \text{num\_experts}$), top-k routing weights; gradients approximate $2 \times$.
- Memory: weight matrix $\text{num\_experts} \cdot \text{hidden\_size}$, logits and top-k activations; no KV cache.

**DeepseekV2MoE**
- Composes:
  - one `MoEGate` (routed experts),
  - an aggregated per-expert MLP (`DeepseekV2MLP`) with effective batch $B \cdot k_{\text{active}}$, and
  - optional shared experts modeled as a single wide MLP with increased `intermediate_size`.
- Aggregates FLOPs, I/O, and memory across gate + expert + shared paths.

**Implications**
- For large `num_experts` and non-trivial `k_active`, MoE can considerably increase FLOPs and activation memory
  compared to dense MLP, but does so sparsely per token.
- Analytic model assumes fixed `k_active` and full participation by experts; load imbalance and dynamic routing behavior
  are intentionally ignored for simplicity.

### DeepseekV2DecoderLayer

**Definition**
- One full DeepSeek-V2 decoder block:
  - pre-attention RMSNorm,
  - FlashAttention2,
  - pre-MLP RMSNorm,
  - dense MLP (`DeepseekV2MLP`) or MoE block (`DeepseekV2MoE`).

**Analytical model**
- **FLOPs**:
  - Tensor Core: sum of attention + MLP/MoE tensor-core FLOPs.
  - CUDA core: sum of two RMSNorms + attention CUDA FLOPs (currently `0`) + MLP/MoE CUDA FLOPs.
  - Backward: aggregates backward FLOPs from all sublayers (no global 2× fallback is used; each child provides its own).
- **I/O and memory**:
  - I/O: sum of forward/backward I/O from norms, attention, and MLP/MoE.
  - Memory: sum of weight and activation memory across the same components.
  - KV cache: aggregated from attention + any MLP KV-term (currently `0` for MLP/MoE; all cache resides in attention).

**Implications**
- Per-layer decoder cost is dominated by `LlamaFlashAttention2` and `DeepseekV2MLP/DeepseekV2MoE`.
- RMSNorm and gating overheads are modest in FLOPs but contribute to low-AI, memory-bound work.
- For a full decoder stack with `N` layers, total decoder FLOPs scale linearly with `N`, `B`, and `S` (plus `S^2` from
  attention) and with `hidden_size` and `intermediate_size`.

---

## Root Aggregator: DeepseekOCRModel

**Definition**
- `DeepseekOCRModel` is a thin wrapper that holds:
  - one vision analytic layer (e.g., `ImageEncoderViT` + CLIP/VitModel path collapsed into a single `BaseLayer`), and
  - one decoder analytic layer (e.g., a stack built from `DeepseekV2DecoderLayer`).

**Current analytic coverage**
- **Implemented**:
  - `forward_tensor_core_flops()` returns the sum of vision + decoder tensor-core FLOPs.
- **Not yet implemented**:
  - CUDA-core FLOPs, backward FLOPs, I/O, memory, and arithmetic intensity are not overridden and will fall back to
    `BaseLayer` defaults.

**Implications**
- As of now, the analytic model exposes a clean entry point for *forward Tensor Core FLOPs* at model level but does
  not yet provide full model-wide I/O, memory, or AI metrics.
- These could be added by mirroring the aggregation pattern used in `ImageEncoderViT`, `NoTPTransformer`, and
  `DeepseekV2DecoderLayer`.

---

## Holistic Model Analysis

### Theoretical Metrics and Scaling

**FLOPs**
- **Vision encoder (SAM + CLIP)**:
  - Single-pass cost dominated by transformer blocks (`Attention` + `MLPBlock` / NoTP equivalents) and conv neck.
  - FLOPs scale with:
    - `B` (batch size),
    - $\text{image\_size}^{2} / \text{patch\_size}^{2}$ (number of tokens),
    - depth of the transformer stacks (`depth` for SAM, number of NoTP blocks for CLIP),
    - `dim`, `mlp_ratio`, and `num_heads`.
  - Windowed attention keeps per-block FLOPs manageable by reducing $S$ inside SDPA.
- **Bridge + projector**:
  - FLOPs are typically small compared to vision + decoder, unless `MlpProjector` uses a deep MLP with large `num_tokens`.
- **Decoder**:
  - For each `DeepseekV2DecoderLayer`, FLOPs scale as:
    - $\mathcal{O}(B \cdot S \cdot \text{hidden\_size}^{2})$ for QKV and output projections.
    - $\mathcal{O}(B \cdot H \cdot S^{2} \cdot d)$ for SDPA (prefill regime).
    - $\mathcal{O}(B \cdot S \cdot \text{hidden\_size} \cdot \text{intermediate\_size})$ for MLP or MoE experts.
  - For $L$ decoder layers, total decoder FLOPs are roughly $L$ times the per-layer cost.
  - Autoregressive decode ($S = 1$) eliminates the $S^{2}$ term but repeats layers many times (per token).

**Memory Usage**
- **Parameters**:
  - Dominated by:
    - transformer weights (both vision and decoder),
    - MoE expert MLP weights when `num_experts` and `num_shared_experts` are large,
    - projector weights (moderate).
  - KV cache scales with $B \cdot H_{\text{kv}} \cdot S \cdot d$ and is solely captured in `LlamaFlashAttention2`.
- **Activations**:
  - Largest contributors:
    - attention QKV and attention buffers ($S^{2}$ when not using FlashAttention, otherwise closer to $B \cdot S \cdot d_{\text{model}}$),
    - MLP hidden activations ($B \cdot S \cdot \text{intermediate\_size}$),
    - multi-scale conv features in `ImageEncoderViT` neck.
  - Most aggregator layers (Blocks, Transformers, Encoders) sum sublayer activation memory, which may over-estimate
    peak usage but preserves relative ordering between components.

**Arithmetic Intensity**
- **High-AI components**:
  - Matmul-heavy layers (`PatchEmbed`, `Attention`/`NoTPAttention`, `LlamaFlashAttention2`, `MLPBlock`,
    `DeepseekV2MLP`, expert paths in `DeepseekV2MoE`, `MlpProjector`).
  - These are expected to be compute-bound on Tensor Cores when dimensions are well-aligned.
- **Low-AI components**:
  - Norms (`LayerNorm2d`, `DeepseekV2RMSNorm`), gating (`MoEGate`), RoPE (`LlamaRotaryEmbedding`), and other purely
    CUDA-core layers.
  - These are bandwidth/latency-bound and benefit primarily from fusion and reduced passes over activations.
- **Mixed-AI aggregators**:
  - Blocks and transformers combine both; their AI tends to be closer to the matmul-heavy components because those
    dominate FLOPs even though norms and gating add non-negligible I/O.

---

## Quality of the Analytic Implementation and Gaps

- **Consistency with guides**:
  - All inspected layers follow the `BaseLayer` interface, use Tensor Core vs CUDA-core splits consistent with
    `TENSOR_OR_CUDA_CORE_GUIDE.md`, and model I/O as activation-centric terabit counts per `LAYER_IO_ESTIMATION_GUIDE.md`.
- **Backward modeling**:
- Most primitive layers use a simple $\approx 2 \times \text{forward}$ heuristic for backward FLOPs, I/O, and activation memory.
  - Decoder aggregators (`DeepseekV2DecoderLayer`, `DeepseekV2MoE`) aggregate child backward metrics instead of applying
    a global factor, which is more faithful.
- **Verification coverage**:
  - Nearly all analytic primitives implement `verify_by_impl` helpers that:
    - Load a reference PyTorch module,
    - Measure FLOPs with `torch.utils.flop_counter.FlopCounterMode`,
    - Compensate for missing SDPA instrumentation when necessary (Attention, NoTPAttention, ImageEncoderViT),
    - Assert relative error within a configurable tolerance.
  - This provides good calibration for FLOP formulas, especially at the primitive level.
- **Known approximations / TODOs**:
  - CUDA-core work in attention (softmax, masking) is typically modeled as `0`; this understates total FLOPs slightly
    but does not affect Tensor Core utilization analysis.
  - Aggregated activation memory and I/O are simple sums and ignore overlap/reuse, so peak memory may be
    over-estimated.
  - `DeepseekOCRModel` currently only aggregates forward Tensor Core FLOPs; full model-level I/O, memory, and AI
    metrics remain to be implemented.
  - A few `FIXME` notes remain around SDPA FLOP compensation once PyTorch begins instrumenting fused attention kernels.

Overall, the DeepSeek-OCR analytic layer implementation is coherent, type-safe, and well aligned with the project’s
Tensor Core vs CUDA-core and I/O modeling conventions. Vision, CLIP, projector, and individual decoder blocks are all
covered with closed-form FLOPs, I/O, and memory formulas. The main remaining analytic gap is at the *model root*:
exposing full-wholistic (vision + decoder) metrics beyond forward Tensor Core FLOPs in `DeepseekOCRModel`.
