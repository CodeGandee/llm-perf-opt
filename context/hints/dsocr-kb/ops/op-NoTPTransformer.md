# NoTPTransformer

## What It Is
`NoTPTransformer` is the stack of 24 transformer blocks for CLIP-L, implementing the core vision transformer encoder.

## Definition
```python
class NoTPTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_layers = cfg.num_layers  # 24
        self.layers = nn.ModuleList([
            NoTPTransformerBlock(cfg, layer_id + 1) for layer_id in range(self.num_layers)
        ])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
```

**Location**: `models/deepseek-ocr/deepencoder.py:399-441`

**Parameters**: `24 × 12.58M = 301.92M params ≈ 604 MB at bf16`

**FLOPs**: `24 × 7.13 GFLOPs = 171.1 GFLOPs per image`

## Related Modules
- **Used by**: `VitModel.transformer`
- **Contains**: 24 × `NoTPTransformerBlock`

## References
- Transformer stack: "Attention Is All You Need" (Vaswani et al., 2017)
