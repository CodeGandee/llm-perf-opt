# NoTPFeedForward

## What It Is
`NoTPFeedForward` is a simple 2-layer MLP with QuickGELU activation used in CLIP-L's transformer blocks. It expands the hidden dimension by 4x (1024 → 4096 → 1024), applying non-linear transformations to each token independently.

QuickGELU is a faster approximation of GELU: `x * sigmoid(1.702 * x)` instead of `x * Φ(x)` where Φ is the Gaussian CDF.

## Definition
```python
class NoTPFeedForward(nn.Module):
    def __init__(self, cfg, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        output = self.fc2(quick_gelu(self.fc1(x)))
        return output

@torch.jit.script
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)
```

## Constructor Information
**Location**: `models/deepseek-ocr/deepencoder.py:295-309`

**Parameters**:
- `dim`: Input/output dimension (1024 for CLIP-L)
- `hidden_dim`: Hidden dimension (4096 for CLIP-L, 4× expansion)

**CLIP-L config**: `dim=1024, hidden_dim=4096`

**Parameters**:
```
fc1: 1024 × 4096 = 4,194,304 params
fc2: 4096 × 1024 = 4,194,304 params
Total: 8,388,608 ≈ 8.39M params per MLP ≈ 16.78 MB at bf16
24 layers: 24 × 8.39M = 201.3M params ≈ 403 MB
```

## Key Pseudo Code

```python
def forward(self, x):
    """
    Args:
        x: (B, S, dim) input tokens

    Returns:
        output: (B, S, dim) transformed tokens
    """
    # 1. Expand: dim → hidden_dim
    hidden = self.fc1(x)  # (B, S, 4096)

    # 2. QuickGELU activation
    hidden = hidden * torch.sigmoid(1.702 * hidden)  # Element-wise

    # 3. Contract: hidden_dim → dim
    output = self.fc2(hidden)  # (B, S, 1024)

    return output
```

## FLOP Count

**Per forward pass** (B=1, S=257, dim=1024, hidden_dim=4096):
```
fc1: 2 × 1 × 257 × 1024 × 4096 = 2.15 GFLOPs
QuickGELU: 5 × 1 × 257 × 4096 ≈ 5.26 MFLOPs (sigmoid + multiply)
fc2: 2 × 1 × 257 × 4096 × 1024 = 2.15 GFLOPs
Total: 4.30 GFLOPs per MLP

24 layers: 24 × 4.30 = 103.2 GFLOPs (24% of CLIP-L's 437 GFLOPs)
```

## Memory Usage

**Parameters**: 16.78 MB per MLP, 403 MB for 24 layers
**Activations** (per sample):
```
Input: 1 × 257 × 1024 × 2 = 526 KB
fc1 output: 1 × 257 × 4096 × 2 = 2.10 MB
fc2 output: 526 KB
Peak: ~2.6 MB per MLP
```

## Related Modules
- **Used by**: `NoTPTransformerBlock.mlp`
- **Alternative**: Standard GELU (more accurate, slower)
- **Similar to**: `DeepseekV2MLP` (LLM's feedforward)

## References
- QuickGELU: Used in original CLIP implementation
- GELU: "Gaussian Error Linear Units" (Hendrycks & Gimpel, 2016)
