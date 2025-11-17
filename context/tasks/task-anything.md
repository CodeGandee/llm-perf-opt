analytically analyze the deepseek ocr model, and implement modelmeter layers for it

## Goal
- Figure out the main model components (modules) used in DeepSeek-OCR, its call relationships, calling counts, and per-module operator breakdowns.
- The module-level analysis goes down until we reach pytorch builtin operators (e.g., `torch.nn.Conv2d`, `torch.nn.LayerNorm`, etc.) or well-known custom layers (e.g., FlashAttention).
- This information will be used to build accurate analytic performance and memory models, which will be implemented under `extern/modelmeter/models/deepseek_ocr/`, according to contracts given in `extern/modelmeter/layers/base.py`

## References
- `context/hints/dsocr-kb/about-dynamic-tracing-deepseek-ocr.md`, approaches to dynamically trace DeepSeek-OCR model execution, we prefer the recommended approach metioned there.