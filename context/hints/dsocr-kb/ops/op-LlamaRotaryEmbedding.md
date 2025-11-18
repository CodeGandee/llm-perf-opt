# LlamaRotaryEmbedding

## What It Is
`LlamaRotaryEmbedding` is the Rotary Position Embedding (RoPE) module used by LLaMA‑family models. It produces cosine
and sine tensors that encode token positions and are applied to query and key vectors in attention. Unlike absolute
position embeddings, RoPE encodes positions multiplicatively via rotations in the complex plane, enabling better length
extrapolation and compatibility with streaming/decoding.

This implementation is **config‑driven**: instead of hard‑coding a single RoPE variant, it supports multiple RoPE
types (default, linear scaling, dynamic NTK, YaRN, etc.) through a `rope_scaling` configuration dictionary. It can
update its internal frequency table dynamically when sequences exceed the original training length, and it can reset
to the original frequencies for shorter sequences to preserve numerical precision.

In DeepSeek‑OCR analytic modeling, `LlamaRotaryEmbedding` provides the reference behavior for LLaMA‑style rotary
embeddings consumed by `LlamaFlashAttention2` and other attention layers.

## Definition
```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types may scale attention magnitude
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

## Constructor Information
**Location**: `transformers/models/llama/modeling_llama.py:83-160`

**Signature**:
```python
def __init__(
    self,
    dim: Optional[int] = None,
    max_position_embeddings: int = 2048,
    base: float = 10000,
    device: Optional[torch.device] = None,
    scaling_factor: float = 1.0,
    rope_type: str = "default",
    config: Optional[LlamaConfig] = None,
)
```

**Parameters**:
- `dim`: Embedding dimension (head_dim for RoPE). Typically `hidden_size / num_heads`.
- `max_position_embeddings`: Maximum context length for which RoPE is initially configured.
- `base`: Base frequency for standard RoPE (`10000` by default).
- `device`: Optional device for initial buffer allocation.
- `scaling_factor`: Factor controlling some legacy scaling behaviors (replaced by `rope_scaling` in config).
- `rope_type`: RoPE variant identifier (`"default"`, `"linear"`, `"dynamic"`, `"yarn"`, etc.).
- `config`: Full `LlamaConfig` object; when provided, it drives RoPE behavior through `config.rope_scaling`.

**Created Components**:
- `self.inv_freq`: Registered buffer of inverse frequencies (shape `(dim/2,)`), used to generate phase angles.
- `self.attention_scaling`: Scalar factor applied to `(cos, sin)` to adjust attention magnitude for some variants.
- `self.original_inv_freq`: Copy of the initial frequency table, used to restore original scale when down‑scaling.
- `self.max_seq_len_cached` / `self.original_max_seq_len`: Track current and original maximum sequence lengths.
- `self.rope_init_fn`: Factory from `ROPE_INIT_FUNCTIONS` that computes `(inv_freq, attention_scaling)` for the chosen
  RoPE variant.

## Module Internals

```mermaid
sequenceDiagram
    participant Config as LlamaConfig
    participant RoPE as LlamaRotaryEmbedding
    participant Cache as inv_freq / scaling
    participant Attn as Attention Module

    Note over RoPE: __init__:<br/>Select rope_type & init inv_freq<br/>via ROPE_INIT_FUNCTIONS[rope_type]

    Config-->>RoPE: rope_scaling (optional)
    RoPE->>Cache: register_buffer(inv_freq)

    Attn->>RoPE: forward(x, position_ids)

    alt rope_type contains \"dynamic\"
        RoPE->>RoPE: _dynamic_frequency_update(position_ids, x.device)
        RoPE->>Cache: update inv_freq, max_seq_len_cached
    end

    RoPE->>RoPE: expand inv_freq to (B, dim/2, 1)
    RoPE->>RoPE: compute freqs = inv_freq @ position_ids
    RoPE->>RoPE: emb = concat(freqs, freqs)
    RoPE->>RoPE: cos = cos(emb); sin = sin(emb)
    RoPE->>RoPE: apply attention_scaling

    RoPE-->>Attn: (cos, sin) with shape (B, S, dim)

    Note over Attn: apply_rotary_pos_emb(q, k, cos, sin)
```

## Key Pseudo Code

```python
def __init__(self, dim=None, max_position_embeddings=2048, base=10000,
             device=None, scaling_factor=1.0, rope_type="default",
             config: Optional[LlamaConfig] = None):
    """
    Config‑driven rotary embedding for LLaMA.

    If `config` is provided, RoPE behavior (type, scaling, max length) is derived from
    `config.rope_scaling` and `config.max_position_embeddings`. Legacy arguments remain
    only for backward compatibility.
    """
    # Select rope_type and initial max_seq_len
    if config is None:
        self.rope_type = rope_type
        self.rope_kwargs = {
            "rope_type": rope_type,
            "factor": scaling_factor,
            "dim": dim,
            "base": base,
            "max_position_embeddings": max_position_embeddings,
        }
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
    else:
        self.rope_type = (
            config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            if config.rope_scaling is not None
            else "default"
        )
        self.rope_kwargs = {}
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

    # Initialize frequencies and scaling for this variant
    self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
    inv_freq, self.attention_scaling = self.rope_init_fn(
        config, device, **self.rope_kwargs
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.original_inv_freq = self.inv_freq


def _dynamic_frequency_update(self, position_ids: torch.LongTensor, device: torch.device) -> None:
    """
    Adjust inv_freq when sequences exceed the originally cached length, and restore
    original frequencies when returning to the base regime.
    """
    seq_len = torch.max(position_ids) + 1
    if seq_len > self.max_seq_len_cached:
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, seq_len=seq_len, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = seq_len

    if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
        self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
        self.max_seq_len_cached = self.original_max_seq_len


@torch.no_grad()
def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: Input tensor of shape (B, num_heads, S, head_dim), used only for dtype/device.
        position_ids: (B, S) tensor of token positions.

    Returns:
        cos, sin: Tensors of shape (B, S, head_dim) encoding RoPE angles.
    """
    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)

    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # (B, S, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (B, S, dim)
        cos = emb.cos()
        sin = emb.sin()

    cos = cos * self.attention_scaling
    sin = sin * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

## FLOP Count and Memory Usage Impact

### FLOPs

For a given batch and sequence:
- `B` = batch size
- `S` = sequence length
- `d` = head_dim (`dim`)

Core operations in `forward`:

1. Expand frequencies:
   ```text
   inv_freq_expanded: broadcast only (no FLOPs)
   position_ids_expanded: cast + reshape, O(B × S)
   ```
2. Frequency multiplication:
   ```text
   freqs = (inv_freq_expanded @ position_ids_expanded)
   Shape: (B, d/2, S) → transpose → (B, S, d/2)
   FLOPs ≈ 2 × B × (d/2) × S = B × d × S
   ```
3. Concatenation and trig:
   ```text
   emb = concat(freqs, freqs)        # 0 FLOPs (copy)
   cos = cos(emb); sin = sin(emb)    # ≈ 2 × B × S × d transcendental ops
   ```

Total FLOPs per call:
```text
FLOPs_total ≈ B × d × S + 2 × B × S × d ≈ 3 × B × S × d
```

Example (B=1, S=8192, d=128):
```text
FLOPs_total ≈ 3 × 1 × 8192 × 128 ≈ 3.1M FLOPs
```
This is negligible compared to attention FLOPs (which scale as `O(B × H × S² × d)`).

### Memory Usage

**Parameters / Buffers**:
- `inv_freq`: `(d/2,)` elements (typically a few hundred floats).
- `original_inv_freq`: same shape, used only for dynamic modes.
- `attention_scaling`: scalar.
Overall parameter footprint is **tiny** compared to attention projections.

**Activations**:
- `freqs`: `(B, S, d/2)`
- `emb`: `(B, S, d)` (transient if not cached)
- `cos`, `sin`: `(B, S, d)` each.

Peak activation memory is dominated by `cos` and `sin`, but these are typically reused across many attention heads and
can be shared for all layers that use the same RoPE configuration.

## Related Modules
- **Used by**:
  - `LlamaAttention` and `LlamaFlashAttention2` via `self.rotary_emb`.
  - DeepSeek‑style analytic layers that model RoPE cost (e.g., `LlamaRotaryEmbedding(BaseLayer)`).
- **Depends on**:
  - `ROPE_INIT_FUNCTIONS`: variant‑specific initialization logic.
  - `apply_rotary_pos_emb`: downstream consumer that rotates Q/K.
- **Alternatives / Variants**:
  - `LlamaLinearScalingRotaryEmbedding`: linear‑scaled RoPE (deprecated wrapper).
  - `LlamaDynamicNTKScalingRotaryEmbedding`: dynamic NTK scaling (deprecated wrapper).
  - DeepSeekV2 RoPE variants (`DeepseekV2RotaryEmbedding`, `DeepseekV2YarnRotaryEmbedding`, etc.).

## Usage Pattern

```python
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

config = LlamaConfig(
    hidden_size=4096,
    num_attention_heads=32,
    max_position_embeddings=8192,
    rope_scaling={"rope_type": "dynamic", "factor": 1.0},
)

rope = LlamaRotaryEmbedding(config=config)

B, S, H, d = 1, 2048, config.num_attention_heads, config.hidden_size // config.num_attention_heads
q = torch.randn(B, H, S, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, S, d, device="cuda", dtype=torch.bfloat16)
position_ids = torch.arange(S, device=q.device)[None, :]  # (1, S)

cos, sin = rope(x=q, position_ids=position_ids)           # (B, S, d)
q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)       # same shapes as q, k
```

