# VitModel

## What It Is
`VitModel` is the complete CLIP-L vision encoder, combining embeddings + 24 transformer blocks + final normalization. It outputs 257 tokens × 1024d for each input image.

## Definition
```python
class VitModel(nn.Module):
    def __init__(self, cfg, freeze_embed=False, freeze_pre_norm=False):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(
            hidden_size=cfg.hidden_size,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size
        )
        self.transformer = NoTPTransformer(cfg=cfg)
        self.pre_layrnorm = LayerNormfp32(cfg.hidden_size, eps=cfg.get("pre_layernorm_epsilon", 1e-5))

    def forward(self, x, patch_embeds):
        x = self.embeddings(x, patch_embeds)  # (B, 257, 1024)
        hidden_states = self.pre_layrnorm(x)
        output = self.transformer(hidden_states)
        return output
```

**Location**: `models/deepseek-ocr/deepencoder.py:446-511`

**CLIP-L config**: 24 layers, 1024d, 16 heads, 4096 FFN

**Parameters**:
```
embeddings: 0.87M
transformer: 301.92M
pre_layrnorm: 1,024
Total: 302.79M ≈ 606 MB at bf16
```

**FLOPs** (per image):
```
Embeddings: ~303 MFLOPs (if computing from pixels)
Transformer: 171.1 GFLOPs
pre_layrnorm: 2.95 MFLOPs
Total: ~171.4 GFLOPs per image
```

**In DeepSeek-OCR**:
- Receives `patch_embeds` from SAM (1024d features)
- Processes with CLIP-L transformer
- Outputs 257 × 1024d tokens
- Concatenated with SAM output → 257 × 2048d → MlpProjector → 257 × 1280d for LLM

## Related Modules
- **Used by**: `DeepseekOCRModel.vision_tower_high.vision_model`
- **Contains**: `CLIPVisionEmbeddings`, `NoTPTransformer`, `LayerNormfp32`
- **Input from**: SAM's patch embeddings
- **Output to**: Concatenated with SAM, then projected via `MlpProjector`

## References
- CLIP-L: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
- ViT-L/14: Vision Transformer Large with 14×14 patches
