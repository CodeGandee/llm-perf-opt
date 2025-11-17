# CLIPVisionEmbeddings

## What It Is
`CLIPVisionEmbeddings` converts an input image into a sequence of patch embeddings for the CLIP vision transformer. It combines:
1. **Patch embedding** via Conv2d (14×14 patches)
2. **CLS token** prepended to the sequence
3. **Absolute position embeddings** (learnable, with interpolation for variable resolutions)

For a 224×224 image with 14×14 patches, this produces 16×16=256 patches + 1 CLS = 257 tokens of 1024d each.

In DeepSeek-OCR, this receives pre-computed SAM patch embeddings (1024d) instead of raw pixels, enabling feature reuse.

## Definition
```python
class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size=1024, image_size=224, patch_size=14, num_channels=3):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
```

## Constructor Information
**Location**: `models/deepseek-ocr/deepencoder.py:243-292`

**Parameters**:
- `hidden_size`: Embedding dimension (1024 for CLIP-L)
- `image_size`: Input image size (224 for CLIP)
- `patch_size`: Patch size (14 for CLIP-L, 16 for CLIP-B)
- `num_channels`: Input channels (3 for RGB, 1024 for SAM features)

**CLIP-L config**: `(1024, 224, 14, 3)` → 257 tokens × 1024d

**Parameters**:
```
class_embedding: 1024 params
patch_embedding: 3 × 14 × 14 × 1024 = 602,112 params
position_embedding: 257 × 1024 = 263,168 params
Total: 866,304 ≈ 0.87M params ≈ 1.73 MB at bf16
```

## Key Pseudo Code

```python
def forward(self, pixel_values, patch_embeds):
    """
    Args:
        pixel_values: (B, C, H, W) input image (unused if patch_embeds provided)
        patch_embeds: (B, embed_dim, grid_h, grid_w) pre-computed patches from SAM

    Returns:
        embeddings: (B, num_patches+1, embed_dim) = (B, 257, 1024)
    """
    batch_size = pixel_values.shape[0]

    # 1. Get patch embeddings (from SAM or compute via Conv2d)
    if patch_embeds is not None:
        patch_embeds = patch_embeds  # Use pre-computed SAM features
    else:
        patch_embeds = self.patch_embedding(pixel_values)

    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

    # 2. Prepend CLS token
    class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # (B, 1, embed_dim)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # (B, 257, 1024)

    # 3. Add position embeddings (with interpolation for variable sizes)
    embeddings = embeddings + get_abs_pos(
        self.position_embedding(self.position_ids), embeddings.size(1)
    )

    return embeddings
```

**Position embedding interpolation**:
```python
def get_abs_pos(abs_pos, tgt_size):
    """Interpolate position embeddings to target size (for variable resolutions)."""
    src_size = int(math.sqrt(abs_pos.shape[1] - 1))  # e.g., 16 for 224×224
    tgt_size = int(math.sqrt(tgt_size - 1))          # e.g., 32 for 448×448

    if src_size != tgt_size:
        # Bicubic interpolation of position embeddings
        cls_token, old_pos_embed = abs_pos[:, :1], abs_pos[:, 1:]
        old_pos_embed = old_pos_embed.view(1, src_size, src_size, dim).permute(0, 3, 1, 2)
        new_pos_embed = F.interpolate(old_pos_embed, size=(tgt_size, tgt_size), mode='bicubic')
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1).view(tgt_size * tgt_size, dim)
        return torch.cat([cls_token, new_pos_embed], dim=0).view(1, -1, dim)
    return abs_pos
```

## FLOP Count

**Patch embedding (Conv2d)**: `2 × B × output_H × output_W × kernel_H × kernel_W × C_in × C_out`
```
= 2 × 1 × 16 × 16 × 14 × 14 × 3 × 1024
≈ 303 MFLOPs (only if computing from pixels, not used with SAM)
```

**With SAM features**: 0 FLOPs (patch_embeds provided)

**Position embedding**: 0 FLOPs (lookup + add)

## Memory Usage

**Parameters**: 1.73 MB
**Activations**: `B × 257 × 1024 × 2 = 526 KB` per image

## Related Modules
- **Used by**: `VitModel.embeddings`
- **Input from**: SAM encoder (`patch_embeds`) or raw image
- **Output to**: CLIP transformer blocks

## References
- CLIP: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
- ViT: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
