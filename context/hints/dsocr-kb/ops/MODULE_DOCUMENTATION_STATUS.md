# DeepSeek-OCR Module Documentation Progress

## Completed Documentation (7 modules)

✅ **op-DeepseekOCRConfig.md** - Configuration class
✅ **op-DeepseekOCRModel.md** - Core transformer with vision encoders
✅ **op-DeepseekOCRForCausalLM.md** - Top-level model with LM head
✅ **op-DeepseekV2RMSNorm.md** - RMS normalization layer
✅ **op-DeepseekV2RotaryEmbedding.md** - Base rotary position embeddings
✅ **op-DeepseekV2LinearScalingRotaryEmbedding.md** - Linear RoPE scaling
✅ **op-DeepseekV2DynamicNTKScalingRotaryEmbedding.md** - Dynamic NTK RoPE scaling

## Remaining Modules (27 total)

### Priority 1: Core LLM Modules (12 modules)

#### RoPE (1 remaining)
- **DeepseekV2YarnRotaryEmbedding** (modeling_deepseekv2.py:264-330)
  - Most sophisticated RoPE scaling method
  - Uses YaRN (Yet another RoPE extension) interpolation
  - Best for >8x context extension
  - Key params: beta_fast, beta_slow, mscale, mscale_all_dim
  - FLOP overhead: ~50 extra FLOPs for frequency ramp computation

#### MLP/MoE (3 modules)
- **DeepseekV2MLP** (modeling_deepseekv2.py:381-397)
  - Standard SwiGLU MLP: gate_proj, up_proj, down_proj
  - FLOPs: 3 × 2×B×S×d×d_ff (≈16 TFLOPs for B=1, S=8192, d=1280, d_ff=3584)
  - Memory: 3 linear layers × (d × d_ff) parameters

- **MoEGate** (modeling_deepseekv2.py:400-535)
  - Expert routing with top-k selection
  - Scoring functions: softmax / sigmoid
  - Topk methods: greedy / group_limited_greedy / noaux_tc
  - Auxiliary loss for load balancing
  - FLOPs: 2×B×S×d×n_experts (router) + aux loss computation

- **DeepseekV2MoE** (modeling_deepseekv2.py:559-702)
  - Sparse mixture of experts with optional shared experts
  - Contains: n_routed_experts × DeepseekV2MLP + MoEGate + shared_experts
  - FLOPs: MoEGate + num_experts_per_tok × MLP_FLOPs (sparse activation)
  - Memory: n_routed_experts × MLP params (but only top-k active per token)

#### Attention (2 modules)
- **DeepseekV2Attention** (modeling_deepseekv2.py:721-949)
  - Multi-head Latent Attention with low-rank QKV projections
  - q_lora_rank, kv_lora_rank compression
  - Separate RoPE (qk_rope_head_dim) and non-RoPE (qk_nope_head_dim) components
  - FLOPs: Q/K/V projections + attention computation + output projection
  - Memory: KV cache with low-rank compression

- **DeepseekV2FlashAttention2** (modeling_deepseekv2.py:953-1227)
  - Flash Attention optimized version of DeepseekV2Attention
  - Uses flash_attn_func / flash_attn_varlen_func
  - Same FLOP count but 3-4x faster wall-clock time
  - Memory: Reduced peak activation memory (no materialized attention matrix)

#### Decoder (4 modules)
- **DeepseekV2DecoderLayer** (modeling_deepseekv2.py:1242-1333)
  - Single transformer layer: pre-norm → attention → post-norm → MLP/MoE
  - Chooses attention impl based on config._attn_implementation
  - Chooses MLP vs MoE based on layer_idx and moe_layer_freq
  - FLOPs: Attention + MLP/MoE + 2×RMSNorm
  - Memory: Layer weights + activations + KV cache

- **DeepseekV2Model** (modeling_deepseekv2.py:1452-1637)
  - Full decoder stack: embed_tokens + num_hidden_layers × DecoderLayer + final norm
  - Manages KV cache, attention masks, position IDs
  - FLOPs: Embedding lookup + num_layers × DecoderLayer
  - Memory: All layer params + KV cache + activations

- **DeepseekV2ForCausalLM** (modeling_deepseekv2.py:1640-1860)
  - Adds lm_head (vocab projection) on top of DeepseekV2Model
  - Computes cross-entropy loss for training
  - FLOPs: Model + 2×B×S×d×vocab_size (lm_head projection)
  - Memory: Model + logits (B×S×vocab_size in fp32)

- **DeepseekV2ForSequenceClassification** (modeling_deepseekv2.py:1878-1992)
  - Classification head for non-generation tasks
  - Projects last token hidden state to num_labels
  - Not used in OCR inference pipeline

#### MLP/MoE Helpers (2 modules)
- **AddAuxiliaryLoss** (modeling_deepseekv2.py:538-556)
  - Autograd function for MoE auxiliary loss backprop
  - Zero FLOPs forward, gradient pass-through backward

- **repeat_kv** (modeling_deepseekv2.py:706-717)
  - Helper for GQA (Grouped Query Attention)
  - Repeats KV heads to match Q heads
  - FLOPs: 0 (memory reshape/view)

### Priority 2: Vision Modules (16 modules)

#### Vision Projector (2 modules)
- **MlpProjector** (deepencoder.py:20-186)
  - Maps vision features (2048d) to LLM space (1280d)
  - Multiple architectures: identity, linear, mlp_gelu, downsample variants
  - FLOPs: 2 × N_tokens × input_dim × n_embed (linear projection)
  - Memory: Linear layer params (2048 × 1280 = 2.6M params)

- **LayerNormfp32** (deepencoder.py:190-196)
  - LayerNorm with fp32 accumulation
  - Same as nn.LayerNorm but forces fp32 computation
  - Prevents numerical issues with bf16/fp16 vision features

#### CLIP Vision Encoder (6 modules)
- **CLIPVisionEmbeddings** (deepencoder.py:243-292)
  - Patch embedding + CLS token + absolute position embeddings
  - Conv2d patchification: (3, H, W) → (hidden_size, H/patch_size, W/patch_size)
  - FLOPs: Conv2d (2×C_in×C_out×H×W/patch_size²)

- **NoTPFeedForward** (deepencoder.py:295-309)
  - MLP with QuickGELU activation
  - FLOPs: 2 × N × d × d_ff

- **NoTPAttention** (deepencoder.py:314-371)
  - Multi-head self-attention with scaled_dot_product_attention
  - FLOPs: 4 × N × d² + 2 × N² × d (QKV proj + attention)

- **NoTPTransformerBlock** (deepencoder.py:373-396)
  - ViT block: LayerNorm → Attention → LayerNorm → FFN (pre-norm)
  - FLOPs: Attention + FFN + 2×LayerNorm

- **NoTPTransformer** (deepencoder.py:399-441)
  - Stack of num_layers NoTPTransformerBlocks
  - FLOPs: num_layers × TransformerBlock

- **VitModel** (deepencoder.py:446-511)
  - Full CLIP-L vision encoder: 24 layers, 1024d, 16 heads
  - Uses SAM features as patch embeddings (skip initial conv)
  - FLOPs: ~64.5 GFLOPs per image (see op-DeepseekOCRModel.md)

#### SAM Vision Encoder (6 modules)
- **MLPBlock** (deepencoder.py:572-585)
  - Simple 2-layer MLP for SAM: linear → GELU → linear
  - FLOPs: 2 × 2 × embedding_dim × mlp_dim

- **LayerNorm2d** (deepencoder.py:590-602)
  - LayerNorm for 2D feature maps (NCHW format)
  - Normalizes over channel dimension
  - FLOPs: Similar to standard LayerNorm

- **ImageEncoderViT** (deepencoder.py:606-711)
  - SAM-B encoder: 12 layers, 768d, 12 heads
  - Patch embedding + transformer blocks + convolutional neck
  - FLOPs: ~290 GFLOPs per 1024×1024 image

- **Block** (deepencoder.py:714-777)
  - SAM ViT block with window/global attention
  - Optional relative position embeddings
  - FLOPs: Attention (with rel-pos) + MLP + 2×LayerNorm

- **Attention** (deepencoder.py:780-847)
  - Multi-head attention with 2D relative position bias
  - Window attention for memory efficiency
  - FLOPs: 4×N×d² + 2×N²×d + rel-pos computation

- **PatchEmbed** (deepencoder.py:971-1002)
  - Conv2d patch embedding: (C_in, H, W) → (embed_dim, H', W')
  - FLOPs: 2 × C_in × embed_dim × H × W / (kernel_size²)

#### Vision Helpers (2 modules)
- **get_abs_pos** / **get_abs_pos_sam** (deepencoder.py:199-236, 548-567)
  - Interpolate positional embeddings for arbitrary resolutions
  - Uses bicubic interpolation
  - FLOPs: Interpolation cost (depends on size ratio)

- **add_decomposed_rel_pos** (deepencoder.py:934-968)
  - Compute 2D decomposed relative position bias
  - FLOPs: 2 × einsum operations (q_h×k_h + q_w×k_w)

### Priority 3: Preprocessing Modules (3 modules)

#### Image Transforms (2 modules)
- **BaseTransform** (modeling_deepseekocr.py:306-316)
  - Abstract base class for image transformations
  - Defines interface: __call__, default_shape, set_rng

- **BasicImageTransform** (modeling_deepseekocr.py:319-341)
  - PIL → Tensor + Normalization
  - transforms.Compose([ToTensor(), Normalize(mean, std)])
  - FLOPs: Negligible (per-pixel operations)

#### Text Streaming (1 module)
- **NoEOSTextStreamer** (modeling_deepseekocr.py:343-348)
  - Custom TextStreamer that replaces EOS token with newline
  - Used during generation for real-time output
  - No compute cost

## Summary Statistics

**Total modules documented**: 7 / 34 (20.6%)
**Total modules remaining**: 27 (79.4%)

**By category**:
- OCR wrappers: 3/3 ✅
- LLM core: 4/12 (33%)
- Vision: 0/16 (0%)
- Preprocessing: 0/3 (0%)

**Estimated remaining documentation**:
- Core LLM modules: ~3,500 lines (12 modules × ~290 lines avg)
- Vision modules: ~4,000 lines (16 modules × ~250 lines avg)
- Preprocessing: ~600 lines (3 modules × ~200 lines avg)
- **Total: ~8,100 lines**

## Key Insights from Completed Docs

### FLOP Bottlenecks (per inference)
1. **Vision Encoding**: ~2.5 TFLOPs (SAM + CLIP for 6-patch document)
2. **LM Head Projection**: ~2.15 TFLOPs (1280 → 102400 vocab)
3. **LLM Decoder**: Varies by sequence length
   - Attention: ~2×B×S²×d per layer (prefill)
   - MoE: Sparse, ~num_experts_per_tok × MLP_FLOPs

### Memory Bottlenecks
1. **Logits Tensor**: 3.3 GB (B=1, S=8192, vocab=102400 in fp32)
2. **KV Cache**: Grows with sequence length (mitigated by low-rank compression)
3. **Vision Activations**: ~67 MB per image (transient)
4. **Model Parameters**: Vision (~786 MB) + LLM decoder (varies)

### Optimization Opportunities
1. **Vision**: Run once during prefill, cache not needed
2. **LM Head**: Compute per-token during generation (400 KB vs 3.3 GB)
3. **Attention**: Flash Attention 2 reduces memory 3-4x
4. **MoE**: Sparse activation reduces compute vs dense MLP

## Completion Strategy

To complete the remaining 27 modules efficiently:

1. **Batch similar modules**: Document all RoPE variants together, all CLIP modules together, etc.
2. **Template reuse**: Many modules share structure (e.g., all transformer blocks)
3. **Cross-referencing**: Link related modules extensively
4. **FLOP focus**: Prioritize FLOP/memory analysis for performance-critical modules
5. **Mermaid automation**: Similar modules can reuse diagram templates

**Estimated completion time**: ~8-10 sessions (given current pace of ~7 modules per session)
