# PatchEmbed

## What It Is
`PatchEmbed` converts an image into patch embeddings using a Conv2d layer with kernel_size=stride=patch_size. This is the standard approach in ViT-style architectures.

For SAM-B: 1024×1024 image with 16×16 patches → 64×64 grid of 768d tokens.

## Definition
```python
class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input image (e.g., (1, 3, 1024, 1024))

        Returns:
            embeddings: (B, H', W', embed_dim) e.g., (1, 64, 64, 768)
        """
        x = self.proj(x)  # (B, embed_dim, H/stride, W/stride)
        x = x.permute(0, 2, 3, 1)  # (B, H', W', embed_dim) - NHWC format
        return x
```

**Location**: `models/deepseek-ocr/deepencoder.py:971-1002`

**SAM-B config**: `kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768`

**Parameters**: `3 × 16 × 16 × 768 = 589,824 ≈ 0.59M params ≈ 1.18 MB at bf16`

**FLOPs** (1024×1024 image):
```
Conv2d: 2 × B × out_H × out_W × kernel_H × kernel_W × C_in × C_out
= 2 × 1 × 64 × 64 × 16 × 16 × 3 × 768
= 302,514,176 ≈ 303 MFLOPs
```

**Output**: `(B, 64, 64, 768)` for 1024×1024 input

## Related Modules
- **Used by**: `ImageEncoderViT.patch_embed` (SAM)
- **Similar to**: `CLIPVisionEmbeddings.patch_embedding` (CLIP)

## References
- Patch embedding: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- Conv2d for patching: Standard ViT approach
