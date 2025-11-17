# MLPBlock

## What It Is
`MLPBlock` is a simple 2-layer MLP with GELU activation used in SAM's transformer blocks. Unlike CLIP's QuickGELU, this uses standard GELU.

## Definition
```python
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))
```

**Location**: `models/deepseek-ocr/deepencoder.py:572-585`

**SAM-B config**: `embedding_dim=768, mlp_dim=3072` (4× expansion)

**Parameters**: `768×3072 + 3072×768 = 4,718,592 ≈ 4.72M per MLP ≈ 9.44 MB at bf16`

**FLOPs** (per forward, B=1, S=4096): `2×1×4096×768×3072 = 19.3 GFLOPs`

## Related Modules
- **Used by**: SAM's `Block` modules
- **Similar to**: `NoTPFeedForward` (CLIP's MLP)

## References
- GELU: "Gaussian Error Linear Units" (Hendrycks & Gimpel, 2016)
- Used in: SAM, ViT, many vision transformers
