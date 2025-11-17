# LayerNorm2d

## What It Is
`LayerNorm2d` applies LayerNorm to NCHW format tensors (channels-first), normalizing over the channel dimension. Used in SAM's convolutional neck.

## Definition
```python
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) NCHW tensor

        Returns:
            normalized: (B, C, H, W)
        """
        u = x.mean(1, keepdim=True)  # (B, 1, H, W)
        s = (x - u).pow(2).mean(1, keepdim=True)  # (B, 1, H, W)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
```

**Location**: `models/deepseek-ocr/deepencoder.py:590-602`

**Parameters**: `2 × num_channels` (weight + bias)

**SAM usage**: `LayerNorm2d(256)` and `LayerNorm2d(out_chans)` in neck

**FLOPs**: `~5 × B × C × H × W` per forward

## Related Modules
- **Used by**: `ImageEncoderViT.neck` (SAM's convolutional neck)
- **Alternative**: Standard LayerNorm (for NHWC format)

## References
- LayerNorm: "Layer Normalization" (Ba et al., 2016)
- Used in: SAM, ConvNeXt, many CNN-transformer hybrids
