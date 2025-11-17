**Purpose**
- Summarize what is and is not realistically possible today for *static tracing* of the DeepSeek-OCR model, and outline practical approaches using PyTorch (`torch.export`, `torch.compile`) and ONNX on *submodules* rather than the full pipeline.

**High-Level Summary**
- There is **no official, end-to-end static export** (ONNX/TensorRT/`torch.compile`) for DeepSeek-OCR from the vendor.
- The model uses:
  - `trust_remote_code` with custom Python logic (`infer`, dynamic cropping, KV-cache handling).
  - FlashAttention2 and other non-standard CUDA/Triton ops.
  - Dynamic control flow (variable crops, image/token packing).
- These features make *whole-model* static tracing brittle. However:
  - You can **statically trace submodules** such as the DeepEncoder (vision) and LLM blocks using PyTorch’s `torch.export` / FX.
  - You may be able to export *some* of those submodules to ONNX if you avoid unsupported ops (e.g., disable flash-attn, avoid dynamic lists).

---

**Option 1 – PyTorch `torch.export` / FX for submodules**

PyTorch 2.5+ provides `torch.export` (and `torch._dynamo.export`) for ahead-of-time graph capture with minimal Python control flow.

Docs:
- `torch.export`: https://pytorch.org/docs/stable/export.html
- `torch.fx`: https://pytorch.org/docs/stable/fx.html

**What works well**
- Isolating pure tensor submodules of DeepSeek-OCR:
  - DeepEncoder vision backbone: SAM encoder + CLIP encoder + conv compressor.
  - Single LLM block (attention + MLP) with KV-cache inputs represented as tensors.
- Using **fixed shapes** (no dynamic crop loops) and avoiding Python containers inside the traced region.

**Example: export the vision encoder**

```python
import torch
from torch.export import export
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession

session = DeepSeekOCRSession.from_local("/abs/path/to/models/deepseek-ocr")
model = session.m_model  # DeepseekOCRForCausalLM (HF wrapper with .model)

# Vendor core is typically nested under .model
core = getattr(model, "model", model)
vision_encoder = core.sam_model  # or a composite wrapper around SAM+CLIP

H = W = 1024
dummy = torch.randn(1, 3, H, W, device=session.device, dtype=session.m_dtype)

graph_module = export(vision_encoder, (dummy,))
print(graph_module)          # FX-style exported graph
graph_module.graph.print_tabular()
```

**Gotchas**
- You may need to:
  - Switch attention implementations to `eager` to avoid FlashAttention in the traced region.
  - Wrap the module in a small adapter that converts any non-tensor inputs (lists/tuples) into tensors.
  - Ensure all branches are shape-stable: no loops whose trip count depends on image size or token count.

This exported graph is useful for:
- Enumerating primitive ops (mm/conv/norm/activation) and their shapes.
- Cross-checking against analytic cost formulas.

---

**Option 2 – `torch.compile` for performance, not structure**

`torch.compile` is designed primarily as a JIT compiler for performance, not as a stable static IR.

Docs:
- `torch.compile`: https://pytorch.org/docs/stable/generated/torch.compile.html

For DeepSeek-OCR:
- You can sometimes wrap submodules like the image encoder:

```python
import torch

vision_encoder = core.sam_model  # example
compiled_encoder = torch.compile(vision_encoder, mode="max-autotune")
out = compiled_encoder(dummy)  # OK for benchmarking
```

- This can reduce kernel launch overhead and improve runtime, but:
  - The internal graph format is **not** meant as a long-lived representation for our analytical tools.
  - Dynamic Python control flow and custom ops (FlashAttention2) still cause graph breaks or fallbacks.

Recommendation:
- Use `torch.compile` only if you want to *benchmark* or sanity‑check performance.
- Prefer `torch.export` / FX for **static introspection** of the computation graph.

---

**Option 3 – ONNX export (very limited and experimental)**

There is currently **no official ONNX export** for DeepSeek-OCR. Any ONNX conversion is ad‑hoc and faces several obstacles:

- **Custom code / trust_remote_code**:
  - `model.infer(...)` orchestrates image loading, cropping, packing, and decoding.
  - Exporters cannot serialize this high‑level Python logic directly.
- **FlashAttention2 and custom CUDA ops**:
  - ONNX exporters do not know how to lower these without custom symbolics.
  - You must disable or replace them (e.g., use PyTorch SDPA) before exporting.
- **Dynamic behavior**:
  - Variable number of crops, changing sequence lengths, and other dynamic patterns complicate static ONNX shapes.

The only realistic ONNX route is:
- Export **narrow subgraphs** with supported ops and fixed shapes, such as:
  - A single CLIP vision transformer block.
  - A single LLM transformer block with standard attention (no flash-attn).

Example sketch (not guaranteed to work without surgery):

```python
import torch
import torch.onnx

block = core.layers[0]  # one transformer block, after disabling flash-attn
dummy_x = torch.randn(1, 128, 4096, device=session.device, dtype=torch.float16)

torch.onnx.dynamo_export(
    block,
    dummy_x,
    export_options=torch.onnx.ExportOptions(dynamic_shapes=False),
).save("dsocr_block.onnx")
```

Even here, you may need to:
- Remove or replace any unsupported ops inside the block.
- Manually inspect the ONNX graph and adjust dynamic dimensions.

For full DeepSeek-OCR (vision + cropping + LLM + KV cache), a single ONNX graph is **not** practical today.

---

**Recommended Strategy for This Project**

For the `llm-perf-opt` analytical pipeline, the most robust static tracing approach is:

1. **Use runtime introspection + FX on submodules**
   - Enumerate modules via `model.named_modules()` and tag them by stage (see `DeepseekOCRStaticAnalyzer` and `DeepSeekOCRSession`).
   - For specific bottleneck blocks (e.g., DeepEncoder, one LLM transformer block), use `torch.export` to grab a static graph with shapes and primitive ops.

2. **Avoid whole-model ONNX / whole-model `torch.compile`**
   - Treat the vendor model as a black box at the `infer(...)` level.
   - Only isolate pure tensor segments you can reason about analytically.

3. **Use analytic formulas for LLM & attention**
   - For prefill/decode transformer computation, rely on closed‑form FLOP/IO formulas (as in `extern/modelmeter/layers/*`) rather than trying to export/trace the entire decoder stack.

4. **Use static analysis outputs as cross-checks**
   - `scripts/deepseek-ocr-static-analysis.py` + `DeepseekOCRStaticAnalyzer` already compute per-stage FLOPs/params/activations via fvcore.
   - FX/`torch.export` graphs should be viewed as a *sanity check* against those numbers, not the primary source of truth.

---

**References**
- DeepSeek-OCR model card (Hugging Face):  
  https://huggingface.co/deepseek-ai/DeepSeek-OCR
- PyTorch `torch.export` docs:  
  https://pytorch.org/docs/stable/export.html
- PyTorch `torch.compile` overview:  
  https://pytorch.org/docs/stable/generated/torch.compile.html
- ONNX export with `torch.onnx.dynamo_export`:  
  https://pytorch.org/docs/stable/onnx_torchscript.html

