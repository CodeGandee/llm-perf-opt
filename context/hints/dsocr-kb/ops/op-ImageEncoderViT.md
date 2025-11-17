# ImageEncoderViT

## What It Is
`ImageEncoderViT` is the SAM-B (Segment Anything Model - Base) vision encoder that produces high-resolution visual features. It uses:
1. **Patch embedding** (16×16 patches) for 1024×1024 images
2. **12 transformer blocks** (768d, 12 heads) with window + global attention
3. **Convolutional neck** that progressively downsamples: 768d → 256d → 512d → 1024d

Unlike CLIP which outputs 257 tokens, SAM outputs 64×64 spatial feature maps at 1024d depth.

## Definition
```python
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11),  # Global attention at these layers
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        self.blocks = nn.ModuleList([
            Block(..., window_size=14 if i not in global_attn_indexes else 0) for i in range(depth)
        ])
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),  # 768 → 256
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),  # 256 → 256
            LayerNorm2d(out_chans),
        )
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)  # Downsample 2×
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)  # Downsample 2×

    def forward(self, x):
        x = self.patch_embed(x)  # (B, 64, 64, 768)
        if self.pos_embed is not None:
            x = x + get_abs_pos_sam(self.pos_embed, x.size(1))
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))  # (B, 256, 64, 64)
        x2 = self.net_2(x)    # (B, 512, 32, 32)
        x3 = self.net_3(x2)   # (B, 1024, 16, 16) - or (B, 1024, H/16, W/16) for variable sizes
        return x3
```

**Location**: `models/deepseek-ocr/deepencoder.py:606-711`

**SAM-B config**:
- 12 layers, 768d, 12 heads, 3072 FFN
- 1024×1024 input → 64×64 patches → 16×16 output feature map (1024d)
- Window attention (14×14 local) + global attention at layers [2, 5, 8, 11]

**Parameters**:
```
patch_embed: 3 × 16 × 16 × 768 = 589,824
pos_embed: 64 × 64 × 768 = 3,145,728
12 transformer blocks: ~56.6M
neck (Conv + LayerNorm): ~0.2M
net_2, net_3: ~1.2M
Total: ~61M params ≈ 122 MB at bf16
```

**FLOPs** (per 1024×1024 image):
```
Patch embedding: ~303 MFLOPs
12 transformer blocks: ~40 GFLOPs (window attention reduces complexity)
Neck convolutions: ~2 GFLOPs
Total: ~42 GFLOPs per image
```

**Output for DeepSeek-OCR**:
- Variable-size images are resized preserving aspect ratio
- Typical: 672×896 → processed → (B, 1024, 42, 56) feature map
- Reshaped to (B, 1024, H×W) → transposed to (B, H×W, 1024) → 2352 tokens × 1024d
- Fed as `patch_embeds` to CLIP-L

**Window + Global Attention Pattern**:
- Layers 0, 1, 3, 4, 6, 7, 9, 10: Window attention (14×14 local windows, efficient for large feature maps)
- Layers 2, 5, 8, 11: Global attention (full 64×64, captures long-range dependencies)
- This hybrid reduces FLOPs from O(N²) to ~O(N×window_size²) while maintaining quality

## Related Modules
- **Used by**: `DeepseekOCRModel.vision_tower_high.vision_model_sam`
- **Contains**: `PatchEmbed`, SAM `Block` modules, `LayerNorm2d`
- **Output to**: CLIP-L as `patch_embeds` input

## References
- SAM: "Segment Anything" (Kirillov et al., 2023)
- Window attention: "Swin Transformer" (Liu et al., 2021)
