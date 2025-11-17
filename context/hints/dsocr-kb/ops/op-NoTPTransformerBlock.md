# NoTPTransformerBlock

## What It Is
`NoTPTransformerBlock` is a single transformer block in CLIP-L, implementing pre-norm architecture: `x + attn(norm(x))` then `x + mlp(norm(x))`.

## Definition
```python
class NoTPTransformerBlock(nn.Module):
    def __init__(self, cfg, layer_id: int, multiple_of=256):
        super().__init__()
        self.dim = cfg.hidden_size  # 1024
        self.self_attn = NoTPAttention(cfg)
        self.mlp = NoTPFeedForward(cfg, dim=cfg.hidden_size, hidden_dim=cfg.ffn_hidden_size)
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        self.layer_id = layer_id

    def forward(self, x):
        residual = self.self_attn.forward(self.layer_norm1(x))
        h = x + residual
        out = h + self.mlp.forward(self.layer_norm2(h))
        return out
```

**Location**: `models/deepseek-ocr/deepencoder.py:373-396`

**Parameters**: `4.19M (attn) + 8.39M (mlp) + 2×1024 (norms) ≈ 12.58M per block`

**FLOPs** (B=1, S=257): `2.83G (attn) + 4.30G (mlp) ≈ 7.13 GFLOPs per block`

## Related Modules
- **Used by**: `NoTPTransformer.layers`
- **Contains**: `NoTPAttention`, `NoTPFeedForward`, 2× LayerNorm

## References
- Pre-norm: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
