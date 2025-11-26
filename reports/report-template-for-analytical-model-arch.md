# <MODEL_NAME> Analytic Architecture and Ops (ModelMeter)

This document is a template for writing an analytic architecture and operator-level report for a specific DNN model (for example, DeepSeek-OCR) backed by ModelMeter analytic layers.
It is meant to capture the static architecture, key analytic layers, tensor shapes, and closed-form FLOP/IO formulas, not runtime performance or MFU.

Replace all `<PLACEHOLDER>` markers with model-specific content when instantiating this template for a concrete model.

## Overview and Scope

**What this section is for**  
Introduce the model at a high level and define the scope of the analytic report.
Explain which analytic implementation is being documented and how it is wired into the broader tooling.

- **Model**: `<MODEL_NAME>` (e.g., DeepSeek-OCR 3B).
- **Analytic implementation root**: `<EXTERN_PACKAGE_ROOT>` (e.g., `extern/modelmeter/models/deepseek_ocr`).
- **Primary goals**:
  - Describe the architecture in terms of major subsystems (vision, encoder/decoder, heads).
  - Document analytic layers and their FLOP/IO formulas.
  - Serve as a reference for future analytic model extensions and verification.

The analytic model is configured via Hydra using `<CONFIG_PATH>` (e.g., `extern/modelmeter/models/<model_id>/configs/<model_id>.yaml`), which typically composes:
- `hf: <hf_config>` – architecture metadata (hidden size, number of layers, heads, MoE layout, etc.).
- `runtime: <runtime_config>` – sequence lengths, batch size, and general analysis parameters.
- `vision: <vision_config>` – model-specific vision tower (if any).
- `decoder` / `encoder`: `<decoder_config>` – text or core network stack.
- `head: <head_config>` – LM head or task-specific projection.
- `model: <model_root_config>` – root analytic factory (e.g., `<model_name>_root.default`).

At a high level, the analytic topology usually follows the vendor/model architecture:
- **Vision branch** (if applicable)
  - Maps images or visual tokens to a sequence of embedding vectors.
- **Core stack** (decoder/encoder or other backbone)
  - Processes token sequences with attention/MLP layers (optionally MoE).
- **Head(s)**
  - Project backbone hidden states into logits or task-specific outputs.

The rest of this document breaks the model into subsystems and analytic layers.

## Layer Inventory by Subpackage (Non-core Layers)

**What this section is for**  
Provide a quick “table of contents” for analytic layers under the `<model_id>/layers` package.
The goal is to give readers a mental map of where each operator lives and how it is grouped (vision/decoder/llama/etc.).

### Vision Layers (`layers/vision`) (optional)

Use this section if the model has a vision tower; otherwise omit it.

- **Patch/Conv embedding** (`<file>.py`): Brief description (e.g., patch embedding Conv2d that maps `(B, C_in, H, W)` to spatial features / tokens).
- **Attention** (`<file>.py`): Vision attention primitive (e.g., SAM-style 2D attention; window vs global).
- **MLP / feed-forward** (`<file>.py`): Vision-side MLP block(s) with expansion ratio, activations.
- **Blocks / transformers** (`<file>.py`): Composite vision transformer blocks aggregating attention + MLP + norms.
- **Norms / small ops** (`<file>.py`): LayerNorm, LayerNorm2d, or similar utility ops.
- **Vision backbone** (`<file>.py`): High-level vision encoder (e.g., ViT/SAM encoder).
- **Projection heads** (`<file>.py`): Projectors from vision feature space into backbone embedding space.
- **Workload helpers** (`<file>.py`): Non-`BaseLayer` utilities for computing view layouts, token counts, or shapes.

For each bullet, the instantiated report should name the actual class and file (e.g., `ImageEncoderViT` in `image_encoder_vit.py`) and give a one-sentence role description.

### Core / Decoder Layers (`layers/decoder` or `layers/core`)

Use this section for the main sequence backbone (decoder, encoder, or similar).

- **MLP / FFN** (`<file>.py`): Dense or SwiGLU MLPs used in the core stack.
- **MoE components** (`<file>.py`): MoE gate and expert MLPs, if applicable.
- **Norms** (`<file>.py`): RMSNorm/LayerNorm used in core layers.
- **Core layer** (`<file>.py`): Composite decoder/encoder block combining attention, MLP/MoE, and norms.
- **Heads** (`<file>.py`): LM head or task-specific projection layers.

As with vision, list concrete classes and their roles when instantiating.

### Shared Primitives (`layers/llama`, `layers/common`, etc.)

Use this section for primitives reused across subsystems (e.g., LLaMA attention, RoPE).

- **Attention** (`<file>.py`): Shared attention primitives (FlashAttention2, standard SDPA, etc.).
- **Positional encodings** (`<file>.py`): RoPE or other positional embedding components.
- **Misc primitives** (`<file>.py`): Any other reusable low-level analytic layers.

## Layer-Level Reference

**What this section is for**  
Provide detailed, reusable templates for describing each analytic `BaseLayer`:
- What it does in the model.
- Torch-style pseudo-code with shapes.
- Closed-form FLOP and IO formulas (forward).

For each real report, fill out one subsection per analytic layer, following the patterns below.

### Vision Layers (example structure)

#### `<VisionLayerName>` (`layers/vision/<file>.py`)

**What it is**  
Short description of the layer’s role and where it sits in the vision tower (e.g., patch embedding, attention block, MLP block).

**Pseudo-code (PyTorch-style)**

```python
# x: <input_shape_example>
x = <torch_ops_on_x>  # <intermediate_shape_comment>
...
y = <final_op>(x)     # <output_shape_comment>
```

Use realistic shapes (e.g., `(B, C, H, W)` or `(B, S, D)`) and comments to show how tensors flow through the layer.

**FLOPs (forward)**  
Define symbolic dimensions, then list formulas.
Example pattern:

- Let `B = <batch_dim>`, `S = <seq_len>`, `D = <hidden_dim>`, `H = <num_heads>`, `d = D / H`, etc.
- Linear/GEMM FLOPs: `F_linear = 2 * B * S * D * D_out`.
- Conv FLOPs: `F_conv = 2 * B * H_out * W_out * K_h * K_w * C_in * C_out`.
- Attention FLOPs: `F_attn ≈ F_qkv + F_qk + F_av + F_out`.

Return a single expression summarizing the dominant FLOPs, e.g.:  
`F_layer ≈ 2 * B * S * D * D_hidden + 2 * B * S * D_hidden * D`.

**I/O (forward)**  
Describe activation traffic in values, then bytes/bits.
Example pattern:

- Input activations: `<count_expr>` values.
- Intermediate activations: `<count_expr>` values.
- Output activations: `<count_expr>` values.
- Total bytes for fp16/bf16: `total_values * 2`.
- Total bits: `total_values * 16`.

Repeat this H5 pattern for each additional vision layer in the instantiated report.

### Core / Decoder Layers (example structure)

#### `<CoreLayerName>` (`layers/decoder/<file>.py`)

**What it is**  
Describe how the core/decoder layer composes attention, MLP/MoE, and norms, and how many times it is repeated in the full model.

**Pseudo-code (PyTorch-style)**

```python
# x: (B, S, D)
h1 = norm1(x)                     # (B, S, D)
attn_out = self_attn(h1, kv_cache)  # (B, S, D)
x = x + attn_out                  # (B, S, D)
h2 = norm2(x)                     # (B, S, D)
mlp_out = mlp_or_moe(h2)          # (B, S, D)
y = x + mlp_out                   # (B, S, D)
```

**FLOPs (forward)**  
Express the layer FLOPs in terms of its subcomponents:

- `F_layer ≈ F_norm1 + F_attn + F_norm2 + F_mlp_or_moe`.
- If the analytic model distinguishes prefill vs decode, note how `S` or KV length changes for each mode.

**I/O (forward)**  
Summarize activation I/O by summing contributions from norms, attention, and MLP/MoE.
Highlight that all activations are typically shaped `(B, S, D)` at this level.

### Shared Primitives (example structure)

#### `<PrimitiveName>` (`layers/llama/<file>.py`)

**What it is**  
Explain what the primitive models (e.g., LLaMA attention, RoPE), and in which higher-level layers it is used.

**Pseudo-code (PyTorch-style)**

```python
# x: <primitive_input_shape>
<intermediate_tensors> = <ops_on_x>   # shapes as comments
...
out = <final_op>(<intermediate_tensors>)  # <output_shape_comment>
```

**FLOPs (forward)**  
List the key GEMM/SDPA/elementwise terms, and how they scale with sequence length, head count, and hidden size.

**I/O (forward)**  
Describe the main activation tensors read/written by the primitive, in values and bytes/bits.

## Notes and Conventions

**What this section is for**  
Clarify global conventions used in the analytic model so per-layer formulas are interpreted consistently.

Recommended content:
- FLOP counting:
  - Use `2 FLOPs per multiply-add (MAC)` for matmuls and convs.
  - Distinguish Tensor Core vs CUDA-core FLOPs (GEMMs/conv/SDPA vs norms/activations/softmax).
- I/O units:
  - `forward_cal_io()` returns terabits (Tb).
  - Activations are assumed fp16/bf16 unless otherwise noted (2 bytes per value).
- Memory metrics:
  - `forward_memory_weight()` in GB for parameters.
  - `forward_memory_activation()` for activations.
  - `forward_memory_kvcache()` for attention KV cache (if applicable).
- Shape parameters:
  - Clearly define `B` (batch), `S` (sequence length), spatial dims (`H`, `W`), and hidden dims (`D`, `d`, `H_heads`).

When instantiating this template, projects should keep these conventions aligned with the actual analytic implementation in `extern/modelmeter/models/<model_id>/layers`.
