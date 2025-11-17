# NoTPAttention

## What It Is
`NoTPAttention` implements multi-head self-attention for CLIP-L (16 heads, 64d per head). It uses PyTorch's `scaled_dot_product_attention` (SDPA) which automatically selects the best implementation (flash attention, memory-efficient attention, or math).

Unlike DeepSeek's MLA, this is standard multi-head attention without low-rank compression.

## Definition
```python
class NoTPAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_attention_heads  # 16
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads  # 64
        self.use_flash_attention = cfg.use_flash_attn

        self.qkv_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
```

## Constructor Information
**Location**: `models/deepseek-ocr/deepencoder.py:314-371`

**Parameters** (CLIP-L):
- `hidden_size`: 1024
- `num_attention_heads`: 16
- `head_dim`: 64
- `use_flash_attn`: True/False

**Parameters**:
```
qkv_proj: 1024 × 3072 = 3,145,728 params
out_proj: 1024 × 1024 = 1,048,576 params
Total: 4,194,304 ≈ 4.19M params ≈ 8.39 MB at bf16
```

## Key Pseudo Code

```python
def forward(self, x):
    """
    Args:
        x: (B, S, hidden_size) = (B, 257, 1024)

    Returns:
        output: (B, S, hidden_size)
    """
    bsz, seqlen, _ = x.shape

    # 1. Project to Q, K, V
    xqkv = self.qkv_proj(x)  # (B, S, 3072)
    xqkv = xqkv.view(bsz, seqlen, 3, self.num_heads, self.head_dim)
    # (B, S, 3, 16, 64)

    # 2. Split Q, K, V
    xq, xk, xv = torch.split(xqkv, 1, dim=2)  # Each (B, S, 1, 16, 64)
    xq = xq.squeeze(2).permute(0, 2, 1, 3)  # (B, 16, S, 64)
    xk = xk.squeeze(2).permute(0, 2, 1, 3)
    xv = xv.squeeze(2).permute(0, 2, 1, 3)

    # 3. Scaled dot-product attention (automatic backend selection)
    output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None)
    # (B, 16, S, 64)

    # 4. Reshape and project
    output = output.permute(0, 2, 1, 3).reshape(bsz, seqlen, -1)  # (B, S, 1024)
    output = self.out_proj(output)

    return output
```

## FLOP Count

**Per forward pass** (B=1, S=257, h=1024, num_heads=16, head_dim=64):
```
qkv_proj: 2 × 1 × 257 × 1024 × 3072 = 1.61 GFLOPs
Attention (Q@K^T + softmax + @V): 2 × 1 × 16 × 257 × 257 × 64 ≈ 0.68 GFLOPs
out_proj: 2 × 1 × 257 × 1024 × 1024 = 0.54 GFLOPs
Total: 2.83 GFLOPs per attention

24 layers: 24 × 2.83 = 67.9 GFLOPs
```

## Memory Usage

**Parameters**: 8.39 MB per attention, 201 MB for 24 layers
**Activations**:
```
qkv: 1 × 257 × 3072 × 2 = 1.58 MB
Attention scores: 1 × 16 × 257 × 257 × 4 = 4.25 MB (fp32)
Output: 526 KB
Peak: ~6 MB per attention
```

## Related Modules
- **Used by**: `NoTPTransformerBlock.self_attn`
- **Alternative**: Flash Attention (fused kernel, 70x memory reduction)

## References
- SDPA: PyTorch 2.0+ automatic attention backend selection
- Flash Attention: "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
