**Purpose**
- Catalog the `torch.nn.Module` classes that are *implemented by DeepSeek-OCR’s own code* (under `models/deepseek-ocr`) and exercised by the vendor `infer()` path invoked from `scripts/deepseek-ocr-infer-one.py`.
- Scope is limited to DeepSeek-OCR’s local implementation; we do **not** expand into 3rd‑party internals (Hugging Face, torchvision, etc.), though those may be wrapped or called.

**Entry Point Context**
- `scripts/deepseek-ocr-infer-one.py` loads the model via:
  - `AutoModel.from_pretrained(<repo>/models/deepseek-ocr, trust_remote_code=True, _attn_implementation="flash_attention_2", ...)`
  - Vendor code in `models/deepseek-ocr/modeling_deepseekocr.py` exposes `DeepseekOCRForCausalLM` with a custom `.infer(...)` method.
- At runtime, the effective stack is:
  - OCR wrapper / infer logic
  - DeepSeek LLM core (`DeepseekV2Model` + decoder layers)
  - Vision encoders (`SAM`-style encoder + CLIP-like encoder)
  - Vision→LLM projector
  - Language modeling head

---

**High-Level Module Families (DeepSeek-OCR Local Code)**
- **OCR wrapper / configuration**
  - `DeepseekOCRConfig(DeepseekV2Config)`
  - `DeepseekOCRModel(DeepseekV2Model)`
  - `DeepseekOCRForCausalLM(DeepseekV2ForCausalLM)`
- **LLM core (decoder-only transformer)**
  - `DeepseekV2RMSNorm(nn.Module)`
  - `DeepseekV2RotaryEmbedding(nn.Module)`
  - `DeepseekV2LinearScalingRotaryEmbedding(DeepseekV2RotaryEmbedding)`
  - `DeepseekV2DynamicNTKScalingRotaryEmbedding(DeepseekV2RotaryEmbedding)`
  - `DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding)`
  - `DeepseekV2MLP(nn.Module)`
  - `MoEGate(nn.Module)`
  - `DeepseekV2MoE(nn.Module)`
  - `DeepseekV2Attention(nn.Module)`
  - `DeepseekV2FlashAttention2(DeepseekV2Attention)`
  - `DeepseekV2DecoderLayer(nn.Module)`
  - `DeepseekV2Model(DeepseekV2PreTrainedModel)`  *(core decoder stack; uses HF `PreTrainedModel`, but layers are DeepSeek-defined)*
  - `DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel)`
  - `DeepseekV2ForSequenceClassification(DeepseekV2PreTrainedModel)`  *(present but not used in OCR `infer()` path)*
- **Vision encoders + projector (DeepEncoder)**
  - `MlpProjector(nn.Module)`  *(vision→LLM projector with several configurable variants)*
  - `LayerNormfp32(torch.nn.LayerNorm)`  *(LayerNorm wrapper that accumulates in fp32)*
  - `CLIPVisionEmbeddings(nn.Module)`
  - `NoTPFeedForward(nn.Module)`
  - `NoTPAttention(torch.nn.Module)`
  - `NoTPTransformerBlock(nn.Module)`
  - `NoTPTransformer(nn.Module)`
  - `VitModel(nn.Module)`  *(CLIP-like ViT backbone)*
  - `MLPBlock(nn.Module)`
  - `LayerNorm2d(nn.Module)`
  - `ImageEncoderViT(nn.Module)`  *(SAM-like vision encoder backbone)*
  - `Block(nn.Module)`  *(ViT-style block with window/global attention)*
  - `Attention(nn.Module)`  *(ViT attention with optional 2D relative position embeddings)*
  - `PatchEmbed(nn.Module)`
- **Image preprocessing / transforms**
  - `BaseTransform(ABC)`  *(abstract base, not an `nn.Module` but part of the image pipeline)*
  - `BasicImageTransform(BaseTransform)`  *(wraps torchvision transforms; uses `nn.Identity` and `transforms.Compose` internally)*
- **Streaming / I/O helpers**
  - `NoEOSTextStreamer(TextStreamer)`  *(text streamer used during generation)*

---

**How These Modules Are Wired in the OCR `infer()` Path**
- `DeepseekOCRForCausalLM`
  - Wraps `DeepseekOCRModel` as `.model` and defines:
    - `forward(...)` for standard HF-style generation.
    - `infer(tokenizer, prompt, image_file, output_path, base_size, image_size, crop_mode, ...)` which:
      - Parses/loads images (PIL + cropping/tiling logic).
      - Uses `dynamic_preprocess(...)` and `BasicImageTransform` to build image tensors.
      - Packs image features into `images`, `images_seq_mask`, `images_spatial_crop` kwargs.
      - Calls `self.generate(...)` (via `transformers` generation utilities) so decoding flows through `DeepseekV2Model`.
- `DeepseekOCRModel(DeepseekV2Model)`
  - Adds image-specific components on top of the base decoder:
    - `sam_model = build_sam_vit_b()` → returns `ImageEncoderViT` (SAM-style encoder with `PatchEmbed`, `Block`, `Attention`, `MLPBlock`, `LayerNorm2d`, + conv neck).
    - `vision_model = build_clip_l()` → returns `VitModel` (CLIP-like ViT using `CLIPVisionEmbeddings`, `NoTPTransformer`, `NoTPTransformerBlock`, `NoTPAttention`, `NoTPFeedForward`, `LayerNormfp32`).
    - `projector = MlpProjector(...)` → maps concatenated SAM/CLIP features into the LLM embedding dimension.
    - `image_newline`, `view_seperator` are learnable `nn.Parameter` vectors appended between visual tokens.
  - In `forward(...)`:
    - Computes `inputs_embeds` from tokens.
    - When images are present and not in trivial decode-only case, runs:
      - `sam_model` and `vision_model` on tiled patches and/or full image.
      - `projector` on concatenated features.
      - Packs visual tokens into `inputs_embeds` using `images_seq_mask` + `masked_scatter_`.
    - Delegates to `super().forward(...)` (i.e., `DeepseekV2Model`) for the transformer stack.
- `DeepseekV2Model` / decoder layers
  - Embedding + stack of `DeepseekV2DecoderLayer`:
    - Each decoder layer uses:
      - `DeepseekV2Attention` or `DeepseekV2FlashAttention2` (via `ATTENTION_CLASSES` and `_attn_implementation="flash_attention_2"` from the script).
      - `DeepseekV2MLP` or `DeepseekV2MoE` (which internally uses `MoEGate`).
      - `DeepseekV2RMSNorm` for pre/post-attention normalization.
      - Rotary positional embeddings via one of the `DeepseekV2*RotaryEmbedding` variants, depending on config.
  - `DeepseekV2ForCausalLM` adds:
    - `lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)` for logits projection.

---

**Notes / Non-DeepSeek Dependencies (Not Expanded Here)**
- The DeepSeek-OCR modules above *wrap or call into* several 3rd‑party modules, which are intentionally **not** decomposed here:
  - Hugging Face `transformers`:
    - `PreTrainedModel`, `TextStreamer`, `LlamaAttention`, `LlamaFlashAttention2`, generation utilities.
  - Torch / torchvision:
    - `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, `nn.Embedding`, `torch.nn.functional.scaled_dot_product_attention`, `torchvision.transforms`, etc.
- For analytic modeling and profiling at the DeepSeek-OCR level, treat those as primitive ops or standard layers, and focus on the *DeepSeek-defined* module boundaries listed above.

---

**Constructor Signatures and Parameter Meanings**

Below, each DeepSeek-defined `nn.Module` has a simplified constructor signature with inline comments describing parameters. Only DeepSeek-OCR’s own modules are covered; base classes from `transformers` / `torchvision` are treated as external.

```python
class DeepseekOCRModel(DeepseekV2Model):
    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)                # config: full DeepseekV2Config (LLM dims, heads, MoE, rope, etc.)
        self.sam_model = build_sam_vit_b()      # image encoder (SAM-style ViT), built from DeepSeek’s vision stack
        self.vision_model = build_clip_l()      # CLIP-like vision encoder (VitModel) over SAM features
        n_embed = 1280                          # projector output / LLM embedding size for visual tokens
        self.projector = MlpProjector(          # projects concatenated SAM+CLIP features into size n_embed
            Dict(projector_type="linear", input_dim=2048, n_embed=n_embed)
        )
        self.image_newline = nn.Parameter(...)  # learned separator token between rows/patch groups
        self.view_seperator = nn.Parameter(...) # learned separator token between global/local views
```

```python
class DeepseekOCRForCausalLM(DeepseekV2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)                # config: DeepseekOCRConfig / DeepseekV2Config
        self.model = DeepseekOCRModel(config)   # replace plain DeepseekV2Model with OCR-aware wrapper
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(               # vocab projection head (same shape as base LLM head)
            config.hidden_size, config.vocab_size, bias=False
        )
```

```python
class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        # hidden_size: dimension of last axis to be normalized
        # eps: numerical epsilon added to variance for stability
        ...
```

```python
class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,                       # rotary embedding dimension (typically head_dim)
        max_position_embeddings: int = 2048,  # max sequence length to cache
        base: float = 10000,            # RoPE base frequency
        device=None,                    # optional device for initial buffer allocation
    ):
        ...
```

```python
class DeepseekV2LinearScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        device=None,
        scaling_factor: float = 1.0,    # linearly scales effective context length
    ):
        ...
```

```python
class DeepseekV2DynamicNTKScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        device=None,
        scaling_factor: float = 1.0,    # dynamic scaling factor for extended context (NTK-style)
    ):
        ...
```

```python
class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        device=None,
        scaling_factor: float = 1.0,            # global scaling factor
        original_max_position_embeddings: int = 4096,  # base model context limit
        beta_fast: float = 32,                  # fast-rotation boundary
        beta_slow: float = 1,                   # slow-rotation boundary
        mscale: float = 1,                      # magnitude scaling for selected dims
        mscale_all_dim: float = 0,              # global magnitude scaling (if non-zero)
    ):
        ...
```

```python
class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        config: DeepseekV2Config,      # provides hidden_size, intermediate_size, hidden_act
        hidden_size: int | None = None,# override for input dim (defaults to config.hidden_size)
        intermediate_size: int | None = None,  # override for expansion dim (defaults to config.intermediate_size)
    ):
        ...
```

```python
class MoEGate(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        # config.num_experts_per_tok: top-k experts selected per token
        # config.n_routed_experts: total routed experts (per MoE layer)
        # config.routed_scaling_factor: scales combined expert outputs
        # config.scoring_func: 'softmax' or 'sigmoid' gating nonlinearity
        # config.aux_loss_alpha: weight for load-balancing auxiliary loss
        # config.seq_aux: whether aux loss is per-token or aggregated
        # config.topk_method: 'greedy' / 'group_limited_greedy' / 'noaux_tc' selection strategy
        # config.n_group, config.topk_group: expert grouping for group-limited routing
        ...
```

```python
class DeepseekV2MoE(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        # config.n_routed_experts: number of routed experts
        # config.moe_intermediate_size: per-expert MLP intermediate dim
        # config.ep_size: expert-parallel world size (if >1, distributed experts)
        # config.n_shared_experts: optional dense shared experts in addition to routed ones
        self.gate = MoEGate(config)     # gating module defined above
        self.experts = nn.ModuleList(...)    # collection of DeepseekV2MLP experts
        self.shared_experts = DeepseekV2MLP(...) if config.n_shared_experts else None
```

```python
class DeepseekV2Attention(nn.Module):
    def __init__(
        self,
        config: DeepseekV2Config,           # provides hidden_size, num_attention_heads, kv/q ranks, rope config, etc.
        layer_idx: int | None = None,       # index of this layer (needed for cache handling)
    ):
        # config.attention_dropout: dropout prob on attention weights
        # config.hidden_size: model hidden dim
        # config.num_attention_heads: number of attention heads
        # config.max_position_embeddings, rope_theta, rope_scaling: RoPE configuration
        # config.q_lora_rank, kv_lora_rank: low-rank factors for Q/KV projections (optional)
        # config.qk_rope_head_dim, qk_nope_head_dim, v_head_dim: head sub-dimensions
        ...
```

```python
class DeepseekV2FlashAttention2(DeepseekV2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)      # same params as DeepseekV2Attention
        # _flash_attn_uses_top_left_mask: internal flag for FlashAttention versioning
```

```python
class DeepseekV2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepseekV2Config,      # full transformer config (MoE, attention impl, dims, norm eps, etc.)
        layer_idx: int,               # position of this layer in the stack (0-based)
    ):
        # Chooses attention implementation based on config.use_mla and config._attn_implementation
        # Chooses MLP vs MoE based on config.n_routed_experts, first_k_dense_replace, moe_layer_freq
        self.self_attn = ATTENTION_CLASSES[...](config=config, layer_idx=layer_idx)
        self.mlp = DeepseekV2MoE(config) or DeepseekV2MLP(config)
        self.input_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

```python
class DeepseekV2Model(DeepseekV2PreTrainedModel):
    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        # config.vocab_size: vocabulary size
        # config.hidden_size: embedding / model dimension
        # config.num_hidden_layers: number of decoder layers
        # config.rms_norm_eps: RMSNorm epsilon
        self.embed_tokens = nn.Embedding(      # token embedding
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(           # transformer blocks
            [DeepseekV2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

```python
class DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel):
    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.model = DeepseekV2Model(config)   # core decoder
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(              # LM head mapping hidden → vocab logits
            config.hidden_size, config.vocab_size, bias=False
        )
```

```python
class DeepseekV2ForSequenceClassification(DeepseekV2PreTrainedModel):
    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        # config.num_labels: number of classification labels
        self.num_labels = config.num_labels
        self.model = DeepseekV2Model(config)   # shared decoder backbone
        self.score = nn.Linear(                # classification head over final hidden state
            config.hidden_size, self.num_labels, bias=False
        )
```

```python
class MlpProjector(nn.Module):
    def __init__(self, cfg):
        # cfg.projector_type: 'identity' | 'linear' | 'mlp_gelu' | 'normlayer_downsample_mlp_gelu'
        #                     'downsample_mlp_gelu' | 'low_high_hybrid_split_mlp_gelu'
        #                     'hybrid_split_feature_mlp_gelu' | 'low_high_split_mlp_gelu'
        # cfg.input_dim: input channel dim (or [high_dim, low_dim] for split variants)
        # cfg.n_embed: output embedding dimension
        # cfg.depth: number of MLP layers (where applicable)
        # cfg.mlp_ratio: expansion ratio inside projector MLP (some variants)
        # cfg.downsample_ratio: spatial downsampling factor for *_downsample_* variants
        # cfg.channel_div: fraction of channels assigned to “high” branch (hybrid_split_feature)
        # cfg.token_pooling: whether to pool 2×2 token neighborhoods before projection
        # cfg.conv_fusion_high_low_features: if True, use additional fusion projection
        ...
```

```python
class LayerNormfp32(torch.nn.LayerNorm):
    # Inherits LayerNorm constructor:
    #   normalized_shape, eps=1e-5, elementwise_affine=True
    # and overrides forward to compute in fp32, then cast back to input dtype.
    ...
```

```python
class CLIPVisionEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,   # embedding dimension for patches and CLS token
        image_size: int = 224,     # input image resolution (assumes square)
        patch_size: int = 14,      # patch side length (pixels)
        num_channels: int = 3,     # number of input channels (RGB=3)
    ):
        ...
```

```python
class NoTPFeedForward(nn.Module):
    def __init__(
        self,
        cfg,                       # config with hidden_size, ffn_hidden_size, etc.
        dim: int,                  # input / output dimension
        hidden_dim: int,           # inner MLP dimension
    ):
        ...
```

```python
class NoTPAttention(nn.Module):
    def __init__(self, cfg):
        # cfg.hidden_size: model dim
        # cfg.num_attention_heads: number of heads
        # cfg.seq_length: max sequence length
        # cfg.use_flash_attn: whether to use flash-attention path
        # cfg.attention_dropout: attention dropout probability
        ...
```

```python
class NoTPTransformerBlock(nn.Module):
    def __init__(
        self,
        cfg,                       # vision transformer config (hidden_size, num_attention_heads, ffn_hidden_size, eps)
        layer_id: int,             # 1-based index of this block in the stack
        multiple_of: int = 256,    # unused here (kept for compatibility with some configs)
    ):
        ...
```

```python
class NoTPTransformer(nn.Module):
    def __init__(self, cfg):
        # cfg.num_layers: number of transformer blocks
        self.layers = nn.ModuleList(
            [NoTPTransformerBlock(cfg, layer_id + 1) for layer_id in range(cfg.num_layers)]
        )
```

```python
class VitModel(nn.Module):
    def __init__(
        self,
        cfg,                       # CLIP-like ViT config (hidden_size, image_size, patch_size, etc.)
        freeze_embed: bool = False,# if True, freeze patch embedding parameters
        freeze_pre_norm: bool = False, # if True, freeze pre-layernorm parameters
    ):
        ...
```

```python
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,        # input / output channel dimension
        mlp_dim: int,              # inner MLP dimension
        act: Type[nn.Module] = nn.GELU,  # activation module class
    ):
        ...
```

```python
class LayerNorm2d(nn.Module):
    def __init__(
        self,
        num_channels: int,         # number of channels (C) in NCHW tensors
        eps: float = 1e-6,         # epsilon for numerical stability
    ):
        ...
```

```python
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,      # input image size (assumed square)
        patch_size: int = 16,      # patch size for PatchEmbed
        in_chans: int = 3,         # number of input channels
        embed_dim: int = 768,      # patch embedding dimension
        depth: int = 12,           # number of transformer blocks
        num_heads: int = 12,       # attention heads per block
        mlp_ratio: float = 4.0,    # MLP expansion ratio
        out_chans: int = 256,      # output channels from neck (after convs)
        qkv_bias: bool = True,     # whether to add bias in qkv projections
        norm_layer: Type[nn.Module] = nn.LayerNorm, # normalization layer class
        act_layer: Type[nn.Module] = nn.GELU,       # activation layer class
        use_abs_pos: bool = True,  # if True, use absolute positional embeddings
        use_rel_pos: bool = False, # if True, use relative positional embeddings in attention
        rel_pos_zero_init: bool = True, # if True, init rel-pos params to zero
        window_size: int = 0,      # window size for windowed attention (0 = global)
        global_attn_indexes: Tuple[int, ...] = (), # indices of blocks with global attention
    ):
        ...
```

```python
class Block(nn.Module):
    def __init__(
        self,
        dim: int,                  # channel dimension
        num_heads: int,            # attention heads
        mlp_ratio: float = 4.0,    # MLP expansion ratio
        qkv_bias: bool = True,     # add bias to qkv projections
        norm_layer: Type[nn.Module] = nn.LayerNorm, # normalization layer class
        act_layer: Type[nn.Module] = nn.GELU,       # activation layer class
        use_rel_pos: bool = False, # if True, add relative positional embeddings
        rel_pos_zero_init: bool = True, # zero-init rel-pos embeddings
        window_size: int = 0,      # window size; 0 means global attention
        input_size: Optional[Tuple[int, int]] = None, # spatial size (H, W) for rel-pos tables
    ):
        ...
```

```python
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,                  # channel dimension
        num_heads: int = 8,        # attention heads
        qkv_bias: bool = True,     # whether qkv projection has bias
        use_rel_pos: bool = False, # if True, use 2D relative position embeddings
        rel_pos_zero_init: bool = True, # zero-init rel-pos parameters
        input_size: Optional[Tuple[int, int]] = None, # (H, W) for rel-pos tables
    ):
        ...
```

```python
class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),  # Conv2d kernel size (patch height, width)
        stride: Tuple[int, int] = (16, 16),       # Conv2d stride
        padding: Tuple[int, int] = (0, 0),        # Conv2d padding
        in_chans: int = 3,                        # input channels
        embed_dim: int = 768,                     # output channels / embedding dim
    ):
        ...
```
