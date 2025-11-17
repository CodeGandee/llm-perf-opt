**Purpose**
- Describe how to build a *dynamic* call graph for DeepSeek-OCR that focuses only on GPU-heavy `forward()` calls, ignoring high-level orchestrating code, and records when and how often each module is invoked.

**High-Level Idea**
- There are two main approaches:
  - **TorchLens-based tracing (recommended)**: use TorchLens to log every operation in a forward pass and derive module-level call counts and graphs from its metadata.
  - **PyTorch-native tooling**: use forward hooks + `torch.profiler` for a fully manual call graph and timing view.
- In both approaches, treat the vendor model as a black box at the script level (e.g., `deepseek-ocr-infer-one.py`) and instrument the **PyTorch modules** inside the loaded model—not the orchestration code.

---

**Approach 1 – TorchLens-based dynamic tracing (recommended)**

TorchLens is vendored under `context/refcode/torchlens/` and available in the Pixi environment. It is the most robust way to obtain a dynamic call graph and visualization for arbitrary PyTorch models (including DeepSeek-OCR).

**Step 1 – Load DeepSeek-OCR and prepare inputs**

Reuse the existing session/analyzer so the tracing setup matches other tooling:

```python
from pathlib import Path
import torch
import torchlens as tl
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession
from llm_perf_opt.runners.dsocr_analyzer import AnalysisConfig, DeepseekOCRStaticAnalyzer

repo_root = Path(__file__).resolve().parents[2]
model_path = repo_root / "models" / "deepseek-ocr"

session = DeepSeekOCRSession.from_local(
    model_path=str(model_path),
    device="cuda:0",
    use_flash_attn=True,
)
model = session.m_model
core = getattr(model, "model", model)  # vendor core

analyzer = DeepseekOCRStaticAnalyzer(session)
config = AnalysisConfig(seq_len=512, base_size=1024, image_size=640)
input_ids, model_kwargs = analyzer.prepare_inputs(config)
```

**Step 2 – Run TorchLens with no activations saved**

TorchLens’ `log_forward_pass` takes `input_args` (tuple/list) and `input_kwargs` (dict). For a call-graph-only view, we want metadata only:

```python
input_args = (input_ids,)
input_kwargs = dict(
    attention_mask=model_kwargs["attention_mask"],
    images=model_kwargs["images"],
    images_seq_mask=model_kwargs["images_seq_mask"],
    images_spatial_crop=model_kwargs["images_spatial_crop"],
    use_cache=True,
    return_dict=True,
)

with torch.no_grad(), torch.autocast("cuda", dtype=session.m_dtype):
    history = tl.log_forward_pass(
        core,
        input_args=input_args,
        input_kwargs=input_kwargs,
        layers_to_save="none",   # metadata only
        vis_opt="none",          # or 'rolled'/'unrolled' if you want a graph
    )
```

See `context/refcode/torchlens/README.md` for more options (e.g., `layers_to_save`, visualization flags).

**Step 3 – Derive module-level call counts and edges**

TorchLens already aggregates module information inside `ModelHistory`. For a per-module “how many times did this run?” view:

```python
from collections import defaultdict

module_call_counts: dict[str, int] = defaultdict(int)

# module_num_passes maps 'module.address' -> number of passes (calls)
for module_name, num_passes in history.module_num_passes.items():
    module_call_counts[module_name] = int(num_passes)
```

To get a more fine-grained view (per operation and its containing module), iterate over layers:

```python
op_call_counts: dict[str, int] = defaultdict(int)

for layer_label in history.layer_labels:
    entry = history[layer_label]
    # containing_modules_origin_nested is e.g. ['model.layers.0.mlp:1', 'model.layers.0:1', 'model:1']
    parents = getattr(entry, "containing_modules_origin_nested", [])
    if parents:
        top = parents[-1]          # innermost module for this op
        op_call_counts[top] += 1
```

You can export these structures to JSON and post-process them into a call graph that aligns with your analytic layer catalog.

**Step 4 – Visualize the graph with Graphviz (optional)**

If Graphviz is installed (Pixi provides it globally), TorchLens can render the computational graph directly:

```python
_ = tl.log_forward_pass(
    core,
    input_args=input_args,
    input_kwargs=input_kwargs,
    layers_to_save=None,      # or 'none' if you only need the graph
    vis_opt="unrolled",       # 'rolled' for a compact view
    vis_outpath="tmp/dsocr-graph.gv",
    vis_fileformat="pdf",
)
```

This produces a PDF/graph file that shows the full operation graph with module nesting, which is useful for sanity-checking your analytic model decomposition.

TorchLens should be your **default choice** for DeepSeek-OCR dynamic tracing; the manual PyTorch approach below is mainly for cases where you need extra customization that TorchLens doesn’t provide.

---

**Approach 2 – Module-level hooks for a Python call graph (manual)**

This approach gives you a module-level call graph (who calls whom, in what order, and how many times).

**Step 1 – Load the model via `DeepSeekOCRSession`**

Reuse the session wrapper so you get the exact configuration and device handling:

```python
from pathlib import Path
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession

repo_root = Path(__file__).resolve().parents[2]
model_path = repo_root / "models" / "deepseek-ocr"

session = DeepSeekOCRSession.from_local(
    model_path=str(model_path),
    device="cuda:0",
    use_flash_attn=True,
)
model = session.m_model           # HF wrapper (DeepseekOCRForCausalLM)
core = getattr(model, "model", model)  # vendor core usually under .model
```

**Step 2 – Build a name → module mapping and GPU-heavy filter**

You need a stable mapping from module objects to their fully qualified names, and a heuristic to keep only GPU-relevant modules.

```python
import torch.nn as nn

name_by_module = {}
for name, mod in core.named_modules():
    name_by_module[mod] = f"model.{name}" if name else "model"

GPU_HEAVY_CLASSES = (
    nn.Linear,
    nn.Conv2d,
    nn.Embedding,
    nn.LayerNorm,
    nn.MultiheadAttention,
    nn.SiLU,
    nn.ReLU,
    # Add DeepSeek-specific classes once identified (e.g., custom attention)
)

def is_gpu_heavy(mod: nn.Module) -> bool:
    if isinstance(mod, GPU_HEAVY_CLASSES):
        return True
    # Heuristic: has CUDA parameters
    try:
        return any(p.is_cuda for p in mod.parameters())
    except Exception:
        return False
```

**Step 3 – Register forward pre/post hooks and track the call stack**

Use hooks to build a call graph and event log:

```python
from collections import defaultdict

call_counts: dict[str, int] = defaultdict(int)
edges: dict[tuple[str, str], int] = defaultdict(int)
events: list[dict] = []
call_stack: list[str] = []

def pre_hook(mod, inputs):
    name = name_by_module.get(mod, mod.__class__.__name__)
    if not is_gpu_heavy(mod):
        return
    parent = call_stack[-1] if call_stack else None
    call_stack.append(name)

    call_counts[name] += 1
    if parent is not None:
        edges[(parent, name)] += 1

    events.append({
        "event": "enter",
        "name": name,
        "parent": parent,
        "index": len(events),
    })

def post_hook(mod, inputs, output):
    if not is_gpu_heavy(mod):
        return
    name = name_by_module.get(mod, mod.__class__.__name__)
    events.append({
        "event": "exit",
        "name": name,
        "index": len(events),
    })
    if call_stack and call_stack[-1] == name:
        call_stack.pop()

hooks = []
for mod in name_by_module.keys():
    if is_gpu_heavy(mod):
        hooks.append(mod.register_forward_pre_hook(pre_hook))
        hooks.append(mod.register_forward_hook(post_hook))
```

**Step 4 – Run a representative inference (prefill + decode)**

You can either:
- Call the full wrapper (e.g., `session.run_inference(...)` if available), or
- Call the HF model with synthetic inputs or a real image, as in `deepseek-ocr-static-analysis.py`.

Example using the static analyzer’s input builder (conceptually):

```python
from llm_perf_opt.runners.dsocr_analyzer import AnalysisConfig, DeepseekOCRStaticAnalyzer

analyzer = DeepseekOCRStaticAnalyzer(session)
config = AnalysisConfig(seq_len=512, base_size=1024, image_size=640)
input_ids, model_kwargs = analyzer.prepare_inputs(config)

with torch.no_grad(), torch.autocast("cuda", dtype=session.m_dtype):
    outputs = model(
        input_ids=input_ids,
        attention_mask=model_kwargs["attention_mask"],
        images=model_kwargs["images"],
        images_seq_mask=model_kwargs["images_seq_mask"],
        images_spatial_crop=model_kwargs["images_spatial_crop"],
        use_cache=True,
        return_dict=True,
    )
```

To get *decode* calls, run a short token loop with `past_key_values`, as described in `howto-separate-prefill-decode-deepseek-ocr.md`.

**Step 5 – Export the call graph and counts**

After the run:

```python
import json

data = {
    "call_counts": {k: int(v) for k, v in call_counts.items()},
    "edges": {f"{src}->{dst}": int(cnt) for (src, dst), cnt in edges.items()},
    "events": events,  # chronological enter/exit events
}

out_path = repo_root / "tmp" / "dsocr-call-graph.json"
out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
```

You can also emit a Graphviz DOT file:

```python
dot_lines = ["digraph dsocr {"] 
for (src, dst), cnt in edges.items():
    dot_lines.append(f'  "{src}" -> "{dst}" [label="{cnt}"];')
dot_lines.append("}")
(repo_root / "tmp" / "dsocr-call-graph.dot").write_text("\n".join(dot_lines))
```

Finally, remove hooks to avoid side effects:

```python
for h in hooks:
    h.remove()
```

This gives you:
- `call_counts`: how many times each GPU-heavy module ran.
- `edges`: parent→child call frequencies (module-level call graph).
- `events`: a chronological list of enter/exit events for ordering analysis.

---

**Approach 3 – `torch.profiler` for kernel-level tracing**

If you want to correlate module calls with CUDA kernel activity, add `torch.profiler`:

Docs:
- https://pytorch.org/docs/stable/profiler.html

Example:

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    with torch.no_grad(), torch.autocast("cuda", dtype=session.m_dtype):
        outputs = model(**inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("tmp/dsocr-profiler-trace.json")
```

This gives:
- Per-op and per-module aggregated CUDA time.
- A Chrome trace you can inspect (CUDA kernels vs modules).

However, the profiler does **not** directly give you a module-to-module call graph; it complements the hook-based call graph with timing info.

---

**Filtering out orchestrating code**

By attaching hooks only to `core` (e.g., `model.model`) and its children, you automatically ignore:
- Scripts like `scripts/deepseek-ocr-infer-one.py`.
- Session helpers and pre/post-processing logic in `DeepSeekOCRSession`.

You can further restrict to LLM or vision stages using the stage→prefix mapping from `DeepseekOCRStaticAnalyzer.m_stage_module_map` (e.g., only modules under `model.sam_model`, `model.vision_model`, `model.projector`, `model.layers`, etc.).

---

**Recommended Usage in This Project**
- Prefer **Approach 1 (TorchLens)**:
  - Use `tl.log_forward_pass` on `core` with representative prefill/decode inputs.
  - Derive module-level call counts and edges from `ModelHistory` (`module_num_passes`, `module_children`, per-layer metadata).
  - Use TorchLens’ Graphviz integration for visual sanity checks.
- Use **Approach 2 (hooks)** only when you need very custom filtering or extra metadata that TorchLens does not expose out of the box.
- Use **Approach 3 (`torch.profiler`)** selectively to:
  - Attach timing and kernel-level information to the most important nodes in your TorchLens- or hook-derived call graph.
- Store resulting artifacts under `tmp/` (e.g., `tmp/dsocr-call-graph.json`, `tmp/dsocr-profiler-trace.json`) and summarize key findings in `extern/modelmeter/models/deepseek_ocr/module_catalog.md` as you refine analytic models.

---

**Best-Practice Tools and Patterns (from broader PyTorch ecosystem)**

- **TorchLens (recommended external tool)**
  - GitHub: `johnmarktaylor91/torchlens`  
  - Paper: *Extracting and visualizing hidden activations and computational graphs of PyTorch models with TorchLens* (Sci Rep, 2023).
  - Provides:
    - Dynamic tracing for *any* PyTorch model (including dynamic graphs).
    - A `log_forward_pass(model, inputs)` API that records every tensor operation and module output.
    - A rich `ModelHistory` object and graph visualization of the forward pass.
  - How to use for DeepSeek-OCR (conceptually):
    - Wrap the loaded DeepSeek-OCR model and inputs in a TorchLens call.
    - Filter TorchLens’ operation list to module-level nodes corresponding to GPU-heavy ops.
    - Aggregate call counts and build a call graph from the returned metadata.
  - This is the closest off-the-shelf solution if you want full-graph introspection with minimal custom code.

- **PyTorch-native hooks (what this guide uses)**
  - Official docs: `nn.Module.register_forward_hook`, `nn.modules.module.register_module_forward_hook` (global).
  - Best practices:
    - Use `model.named_modules()` to get stable names and attach hooks only to modules of interest.
    - Treat hooks as debug/profiling only (remove them after use; avoid modifying outputs).
    - Combine `forward_pre_hook` + `forward_hook` to reconstruct a call stack and get enter/exit events.

- **PyTorch profiler**
  - Official docs: `torch.profiler` with `ProfilerActivity.CPU/CUDA`, `record_shapes=True`, `with_stack=True`.
  - Use it alongside hooks/TorchLens to:
    - Attribute CUDA time to ops/modules.
    - Confirm which “heavy” modules dominate runtime in your call graph.

- **Static tools (FX / export / ONNX)**
  - `torch.fx` and `torch.export` are good for *static graphs* and shape inspection, but they are not designed to produce dynamic call-frequency graphs in eager mode.
  - For DeepSeek-OCR, they are more appropriate for analytic layer modeling (see static-tracing hints) than for module call counting.
