# DeepseekV2Model

## What It Is
`DeepseekV2Model` is the complete decoder-only transformer stack for DeepSeek-OCR's LLM component. It combines:
1. **Token embedding layer** to convert input IDs to vectors
2. **40 decoder layers** (`DeepseekV2DecoderLayer`) stacked sequentially
3. **Final RMSNorm** for output normalization
4. **KV cache management** for efficient autoregressive generation
5. **Attention mask handling** (causal masking for autoregression)

This is the core transformer that processes text tokens (including OCR-generated text embeddings from vision) and produces contextualized hidden states. It does NOT include the LM head for next-token prediction - that's added by `DeepseekV2ForCausalLM`.

## Definition
```python
class DeepseekV2Model(DeepseekV2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a DeepseekV2DecoderLayer.
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()
```

## Constructor Information
**Location**: `models/deepseek-ocr/modeling_deepseekv2.py:1452-1480`

**Signature**:
```python
def __init__(self, config: DeepseekV2Config)
```

**Parameters** (from config):
- `vocab_size`: Vocabulary size (default: 32000)
- `hidden_size`: Model hidden dimension (default: 1280)
- `num_hidden_layers`: Number of transformer layers (default: 40)
- `pad_token_id`: Padding token ID (default: 0)
- `_attn_implementation`: "eager" or "flash_attention_2"
- `rms_norm_eps`: Epsilon for final RMSNorm (default: 1e-6)

**Created Components**:

1. **self.embed_tokens**: Token embedding table
   - `nn.Embedding(vocab_size, hidden_size, padding_idx)`
   - Shape: `(32000, 1280)`
   - Parameters: 32,000 × 1,280 = 40,960,000 ≈ 40.96M
   - At bf16: 40.96M × 2 bytes ≈ 82 MB

2. **self.layers**: Stack of decoder layers
   - `ModuleList` of 40 `DeepseekV2DecoderLayer` instances
   - Layer 0: Dense MLP (144 MB)
   - Layers 1-39: MoE (1.87 GB each)
   - Total: 73.07 GB (see op-DeepseekV2DecoderLayer.md)

3. **self.norm**: Final output normalization
   - `DeepseekV2RMSNorm(1280)`
   - Parameters: 1,280
   - At bf16: 2.56 KB

**Total parameters**:
```python
embed_tokens: 40.96M
layers: Layer 0 (72.2M) + Layers 1-39 (937.4M × 39) = 36,600.8M
norm: 1,280
Total: 36,641.8M ≈ 36.64B parameters

At bf16: 36.64B × 2 bytes ≈ 73.28 GB
```

## Module Internals

```mermaid
sequenceDiagram
    participant Input as input_ids<br/>(B, S)
    participant Embed as embed_tokens
    participant PosIDs as Position IDs
    participant Mask as Attention Mask
    participant Layers as Decoder Layers<br/>(×40)
    participant Norm as Final RMSNorm
    participant Output as hidden_states<br/>(B, S, h)

    Note over Input: forward() entry

    Input->>Embed: Lookup embeddings
    Embed-->>PosIDs: inputs_embeds (B, S, 1280)

    alt position_ids not provided
        PosIDs->>PosIDs: Generate position_ids<br/>arange(past_length, seq_len+past_length)
    end

    alt Flash Attention
        Mask->>Mask: Use 2D mask (B, S)<br/>or None if all causal
    else Standard Attention
        Mask->>Mask: Prepare 4D causal mask<br/>(B, 1, S, kv_seq_len)
    end

    PosIDs->>Layers: hidden_states, attention_mask, position_ids

    loop For each of 40 layers
        Layers->>Layers: layer_i(hidden_states, mask, pos, past_kv)
        Layers->>Layers: Update hidden_states<br/>Update KV cache
    end

    Layers-->>Norm: hidden_states (B, S, 1280)
    Norm->>Norm: Final normalization
    Norm-->>Output: normalized hidden_states

    alt use_cache
        Output->>Output: Return past_key_values (40 layer caches)
    end

    alt output_hidden_states
        Output->>Output: Return all_hidden_states (41 states)
    end

    Note over Output: Return BaseModelOutputWithPast
```

## Key Pseudo Code

```python
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None
) -> Union[Tuple, BaseModelOutputWithPast]:
    """
    Full transformer decoder forward pass.

    Args:
        input_ids: (batch, seq_len) input token IDs
        attention_mask: (batch, seq_len) mask for padding tokens
        position_ids: (batch, seq_len) position indices for RoPE
        past_key_values: Cached K, V from previous decoding steps (DynamicCache or list)
        inputs_embeds: (batch, seq_len, hidden_size) pre-computed embeddings (alternative to input_ids)
        use_cache: Whether to return KV cache for next step
        output_attentions: Whether to return attention weights
        output_hidden_states: Whether to return all intermediate hidden states
        return_dict: Whether to return BaseModelOutputWithPast

    Returns:
        BaseModelOutputWithPast with:
            last_hidden_state: (batch, seq_len, hidden_size)
            past_key_values: Updated KV cache
            hidden_states: All layer outputs if output_hidden_states=True
            attentions: All attention weights if output_attentions=True
    """
    # 1. Validate inputs
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Cannot specify both input_ids and inputs_embeds")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("Must specify either input_ids or inputs_embeds")

    # 2. Gradient checkpointing compatibility
    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False  # Incompatible with gradient checkpointing

    # 3. Handle KV cache
    past_key_values_length = 0
    if use_cache:
        # Convert legacy cache format to DynamicCache
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    # 4. Generate position IDs if not provided
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)  # (1, seq_len)

    # 5. Embed tokens
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)  # (B, S, 1280)

    # 6. Prepare attention mask
    if self._use_flash_attention_2:
        # Flash attention uses 2D mask (or None for all-causal)
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    else:
        # Standard attention needs 4D causal mask
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

    # 7. Initialize outputs
    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # 8. Pass through all 40 decoder layers
    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            # Recompute activations during backward pass
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    # 9. Final normalization
    hidden_states = self.norm(hidden_states)

    # 10. Add final hidden state
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    # 11. Convert cache back to legacy format if needed
    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )

    # 12. Return outputs
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
```

## FLOP Count and Memory Usage Impact

### FLOPs (per forward pass)

Assume:
- Input shape: `(B, S)` where B=batch, S=sequence length
- Typical decode: B=1, S=1, cached_context=K=8192
- Typical prefill: B=1, S=8192, K=8192

**Operations**:

1. **Token embedding**: 0 FLOPs (lookup operation)

2. **40 decoder layers**:
   ```
   See op-DeepseekV2DecoderLayer.md for per-layer breakdown

   Decode (S=1):
     Layer 0 (dense): 2.44 GFLOPs
     Layers 1-39 (MoE): 39 × 362.4 GFLOPs = 14.13 TFLOPs
     Total: 14.13 TFLOPs per decode step

   Prefill (S=8192):
     Layer 0 (dense): 10.27 TFLOPs
     Layers 1-39 (MoE): 39 × 12.95 TFLOPs = 505 TFLOPs
     Total: 515 TFLOPs per prefill
   ```

3. **Final RMSNorm**:
   ```
   FLOPs ≈ 3 × B × S × h
   Decode: 3 × 1 × 1 × 1280 ≈ 3.84 KFLOPs
   Prefill: 3 × 1 × 8192 × 1280 ≈ 31.5 MFLOPs
   ```

**Total**:
```
Decode: 14.13 TFLOPs per token
Prefill: 515 TFLOPs for 8192 tokens

Time per decode (RTX 4090, 330 TFLOPS bf16):
  14.13 TFLOPs / 330 TFLOPS ≈ 43 ms (theoretical, assuming 100% MFU)
  Actual: ~80-100 ms (50-60% MFU due to memory bandwidth)

Throughput: ~10-12 tokens/second (single batch)
```

### Memory Usage

#### Parameters:
```
embed_tokens: 82 MB
layers: 73.07 GB (36.6B params)
norm: 2.56 KB
Total: 73.15 GB at bf16
```

#### Activations (per forward pass):

**Decode** (S=1, K=8192):
```
inputs_embeds: 1 × 1 × 1280 × 2 = 2.56 KB
Per layer activations: ~143 MB (see op-DeepseekV2DecoderLayer.md)
Total: 40 × 143 MB ≈ 5.72 GB
```

**Prefill** (S=8192):
```
inputs_embeds: 1 × 8192 × 1280 × 2 = 21 MB
Per layer activations: ~686 MB
Total: 40 × 686 MB ≈ 27.4 GB

With Flash Attention: Reduces attention memory by 70x
Effective: ~10-12 GB
```

#### KV Cache:
```
Per layer: 9.44 MB (K=8192)
Total for 40 layers: 377.6 MB

Scales with context:
  K=16384: 755 MB
  K=32768: 1.51 GB

MLA achieves ~57x reduction vs standard MHA!
```

#### Total inference memory (decode, K=8192):
```
Model parameters: 73.15 GB
Activations: 5.72 GB
KV cache: 377.6 MB
Total: ~79 GB

Fits on single A100 80GB or H100 80GB!
```

## Related Modules
- **Used by**:
  - `DeepseekV2ForCausalLM.model` - adds LM head for generation
  - `DeepseekV2ForSequenceClassification.model` - adds classification head
  - `DeepseekOCRModel.language_model` - integrates with vision encoder
- **Contains**:
  - `nn.Embedding` for token embeddings
  - 40 × `DeepseekV2DecoderLayer` for transformer layers
  - `DeepseekV2RMSNorm` for final normalization
- **Cache management**: Uses `DynamicCache` from transformers for efficient KV caching

## Usage Pattern

```python
from modeling_deepseekv2 import DeepseekV2Model, DeepseekV2Config

config = DeepseekV2Config(
    vocab_size=32000,
    hidden_size=1280,
    num_hidden_layers=40,
    use_mla=True,
    n_routed_experts=160,
)

model = DeepseekV2Model(config)

# Prefill
input_ids = torch.randint(0, 32000, (1, 8192))  # (B, S)
position_ids = torch.arange(8192).unsqueeze(0)

outputs = model(
    input_ids=input_ids,
    position_ids=position_ids,
    use_cache=True,
)

hidden_states = outputs.last_hidden_state  # (1, 8192, 1280)
past_key_values = outputs.past_key_values  # KV cache for 40 layers

# Decode
next_token_id = torch.tensor([[12345]])  # (1, 1)
next_position = torch.tensor([[8192]])

outputs = model(
    input_ids=next_token_id,
    position_ids=next_position,
    past_key_values=past_key_values,  # Reuse cached K, V
    use_cache=True,
)

next_hidden_states = outputs.last_hidden_state  # (1, 1, 1280)
updated_past_key_values = outputs.past_key_values
```

## Key Performance Characteristics

1. **Efficient KV caching**: 377 MB for 8K context (57x smaller than standard attention)
2. **MoE sparsity**: Only 2/160 experts active per token (80x parameter efficiency)
3. **Flash Attention**: 70x memory reduction during prefill
4. **Gradient checkpointing**: Reduces training memory from 27 GB to ~1-2 GB
5. **Pre-norm stability**: Enables stable training of 40-layer deep network

## Optimization Opportunities

1. **Continuous batching**: Process multiple requests with different sequence lengths
2. **Speculative decoding**: Generate multiple tokens per forward pass
3. **KV cache quantization**: Int8/int4 KV cache (2-4x further reduction)
4. **Pipeline parallelism**: Split 40 layers across multiple GPUs
5. **Expert parallelism**: Distribute MoE experts across GPUs

## References
- Original transformer: "Attention Is All You Need" (Vaswani et al., 2017)
- Pre-norm architecture: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
- Multi-head Latent Attention: DeepSeek-V2 paper (Bi et al., 2024)
- Mixture-of-Experts: DeepSeek-MoE paper (Dai et al., 2024)
- Used in: DeepSeek-V2, DeepSeek-V3, DeepSeek-OCR
