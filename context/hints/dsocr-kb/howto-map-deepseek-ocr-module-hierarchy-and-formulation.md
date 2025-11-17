**Purpose**
- Build a hierarchical view of the DeepSeek-OCR model (stage → module → leaf PyTorch ops) and document the mathematical formulation for each module family we want analytic cost models for.

**Quick Summary**
- Use runtime introspection to dump the full `torch.nn.Module` tree for DeepSeek-OCR and tag each module with its stage (`sam`, `clip`, `projector`, `prefill`, `decode`).
- Define “leaf modules” as PyTorch built-ins (e.g., `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, attention blocks) where we stop recursion; everything above them is a composite module.
- For each composite and leaf module family, write down its math in a shared “formulations” document and cross-reference the DeepSeek-OCR paper + source code.

**Step 1 – Load the model for structural introspection**
- Reuse the same loading path as existing tooling (static analyzer, inference script) so we get the exact vendor architecture:

```python
from pathlib import Path
from transformers import AutoModel  # trust_remote_code

repo_root = Path(__file__).resolve().parents[2]
model_path = repo_root / "models" / "deepseek-ocr"

model = AutoModel.from_pretrained(
    str(model_path),
    trust_remote_code=True,
    local_files_only=True,
)
model = model.eval()
```

- Alternatively, use the session abstraction to share config with other tools:

```python
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession

session = DeepSeekOCRSession.from_local(str(model_path))
model = session.m_model  # DeepseekOCRForCausalLM
```

**Step 2 – Dump the full module hierarchy**
- Implement a small utility that walks `named_modules()` and prints an indented tree:

```python
import torch.nn as nn

def print_module_tree(module: nn.Module, prefix: str = "") -> None:
    children = list(module.named_children())
    if not children:
        print(prefix + module.__class__.__name__)
        return
    print(prefix + module.__class__.__name__)
    for name, child in children:
        print(f"{prefix}  ├─ {name}: ", end="")
        print_module_tree(child, prefix + "  ")

print_module_tree(model)  # DeepseekOCRForCausalLM
```

- Redirect output to a file for manual inspection and search:

```bash
python tools/dump_dsocr_tree.py > tmp/dsocr-module-tree.txt
```

- This file is the ground truth for hierarchy: use it to see which logical blocks exist and how they nest (e.g., `model.sam_model.*`, `model.vision_model.*`, `model.layers.[i].self_attn`, `model.layers.[i].mlp`, `lm_head`, etc.).

**Step 3 – Tag modules by stage using the analyzer’s mapping**
- Reuse the stage→module-prefix mapping from `DeepseekOCRStaticAnalyzer`:
  - `sam`       → `model.sam_model`
  - `clip`      → `model.vision_model`
  - `projector` → `model.projector`
  - `prefill`   → `model.embed_tokens`, `model.layers`, `model.norm`
  - `decode`    → `lm_head`

- Write a small helper that, for every module name from `named_modules()`, determines its stage:

```python
STAGE_PREFIXES = {
    "sam": ["model.sam_model"],
    "clip": ["model.vision_model"],
    "projector": ["model.projector"],
    "prefill": ["model.embed_tokens", "model.layers", "model.norm"],
    "decode": ["lm_head"],
}

def infer_stage(module_name: str) -> str | None:
    for stage, prefixes in STAGE_PREFIXES.items():
        if any(module_name.startswith(p) for p in prefixes):
            return stage
    return None
```

- Extend the tree dumper to emit `stage` along with each module, and write a structured artifact:

```python
import json

tree: list[dict[str, str]] = []
for name, mod in model.named_modules():
    stage = infer_stage(name) or "other"
    tree.append({
        "name": name,
        "class": mod.__class__.__name__,
        "stage": stage,
    })

out_path = repo_root / "tmp" / "dsocr-module-tree.json"
out_path.write_text(json.dumps(tree, indent=2))
```

**Step 4 – Define “leaf modules” and cut the hierarchy there**
- For analytical modeling, we only care about PyTorch primitive ops; deeper than that is unnecessary.
- Define leaves as:
  - Modules with no children (`len(list(mod.children())) == 0`).
  - Or modules whose type is in a “builtin primitives” allowlist (even if they internally nest):
    - `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, `nn.RMSNorm` (if used), `nn.Embedding`, `nn.MultiheadAttention` / custom attention blocks, activation functions (`nn.SiLU`, `nn.ReLU`, etc.), pooling.
- Build an index of unique leaf module classes per stage:

```python
import torch.nn as nn
from collections import defaultdict

LEAF_TYPES = {
    nn.Linear,
    nn.Conv2d,
    nn.LayerNorm,
    nn.Embedding,
    nn.SiLU,
    nn.ReLU,
    # add attention / projector classes once identified
}

leaf_classes_by_stage: dict[str, set[str]] = defaultdict(set)

for name, mod in model.named_modules():
    stage = infer_stage(name) or "other"
    is_childless = len(list(mod.children())) == 0
    is_builtin = any(isinstance(mod, t) for t in LEAF_TYPES)
    if is_childless or is_builtin:
        leaf_classes_by_stage[stage].add(mod.__class__.__name__)
```

- This gives you, for each stage, the set of primitive module types that appear at the leaves, which should roughly match the operator families seen in `static_compute.json`.

**Step 5 – Map code to math for each module family**
- For *standard* PyTorch primitives (`nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, etc.), use their official formulations:
  - `Linear`: \( y = x W^T + b \)
  - `Conv2d`: standard cross-correlation over spatial dims with learned kernels.
  - `LayerNorm` / `RMSNorm`: normalization over the last dimension with learned gain (and bias for LayerNorm).
- For *DeepSeek-specific* composites, inspect the source:

```bash
sed -n '1,260p' models/deepseek-ocr/deepencoder.py
sed -n '1,260p' models/deepseek-ocr/modeling_deepseekv2.py
sed -n '1,260p' models/deepseek-ocr/modeling_deepseekocr.py
```

- Identify for each important composite:
  - **SAM block**: how patch embedding, window attention, and feed-forward are wired; what the resolution and channel dimensions are.
  - **CLIP vision transformer block**: self-attention + MLP formulas (standard transformer math).
  - **Projector / optical compressor**: how visual tokens are pooled/reshaped and linearly mapped to latent tokens.
  - **LLM transformer blocks**: attention (GQA/FlashAttention) and MLP (SwiGLU) formulas, including KV-cache layout.

**Step 6 – Cross-check formulations with the paper**
- Use the DeepSeek-OCR paper for authoritative descriptions and notation:

```bash
rg "DeepEncoder" paper-source/tex/main.tex
rg "Contexts Optical Compression" paper-source/tex/main.tex
rg "latent" paper-source/tex/main.tex
```

- Align:
  - Names and symbols (e.g., \(N, n, d_\text{text}, d_\text{latent}\)) with code-level dimensions.
  - The described components (DeepEncoder = SAM-base + CLIP-large + compressor; projector mapping image tokens to latent tokens) with concrete modules in the tree.
  - Any fusion or special behavior (e.g., serial SAM→CLIP with 16× token reduction) with how the modules are wired in code.

**Step 7 – Create a shared “module formulations” document**
- Under `extern/modelmeter/models/deepseek_ocr/`, create a document such as:

```bash
touch extern/modelmeter/models/deepseek_ocr/module_formulations.md
```

- For each module family (grouped by stage), add:

```text
Stage: prefill / decode
Module: GQA FlashAttention block
Hierarchy path: model.layers[i].self_attn

Formulation:
  Q = X W_q, K = X W_k, V = X W_v
  Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V
Notes:
  - Uses grouped-query attention (num_heads != num_kv_heads).
  - Implemented with FlashAttention; IO pattern follows block-tiling.

Stage: prefill / decode
Module: SwiGLU MLP
Hierarchy path: model.layers[i].mlp

Formulation:
  Z = W_up X
  G = W_gate X
  Y = W_down (silu(G) ⊙ Z)
```

- Include references at the bottom:
  - DeepSeek-OCR paper (sections describing DeepEncoder, projector, and compression).
  - PyTorch docs for the standard modules (Linear, Conv2d, LayerNorm, etc.).

**Outcome**
- After following this guide, you will have:
  - A JSON/markdown artifact listing every module name, class, and stage, with leaf modules clearly identified.
  - A companion “module formulations” document that defines the math behind each module family used in DeepSeek-OCR.
  - A solid foundation for designing accurate analytic layer models under `extern/modelmeter/models/deepseek_ocr/layers/`.

