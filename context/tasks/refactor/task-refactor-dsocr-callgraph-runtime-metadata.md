# Refactor Plan: DeepSeek-OCR Callgraph Runtime Metadata

## What to Refactor

- **TorchLens tracing + Option C pipeline for DeepSeek-OCR**:
  - `src/llm_perf_opt/patches/dsocr_torchlens.py`
    - Runtime patches applied to DeepSeek-OCR modules.
    - `DsocrCallgraph` model and JSON loader.
    - Option C grouping helpers (`summarize_stacks`, `summarize_parfor_edges`, `build_grouped_dot_from_option_c`, `build_grouped_mermaid_from_option_c`).
  - `scripts/analytical/dsocr_torchlens_callgraph_option_c.py`
    - Orchestrates TorchLens logging and writes the raw callgraph JSON + runtime metadata JSON.
  - `scripts/analytical/dsocr_parse_callgraph_option_c.py`
    - Consumes the callgraph + runtime metadata and emits grouped DOT/SVG/Mermaid.

The refactor will extend this pipeline so that the **callgraph JSON and downstream artifacts include, for each module node**:

- A structured snapshot of **constructor/config parameters** that influence tensor shapes.
- Representative **input tensor metadata** (shape + dtype).
- Representative **output tensor metadata** (shape + dtype).

## Why Refactor

- The updated design contract (`context/design/dsocr-callgraph-human-friendly.md`) requires:
  - Per-node **constructor parameters** to understand model configuration (e.g., `hidden_size`, `num_heads`, `num_experts`).
  - Per-node **input/output tensor shapes and dtypes** to estimate activation sizes and overall tensor memory footprint.
- Current implementation only captures:
  - Structural information (module types, runtime names, parent/child edges, call counts).
  - Some high-level runtime events (vision inputs, LLM stack size) via `_RUNTIME_LOG`.
- Without shape + dtype + config metadata per node:
  - We cannot answer questions like “how much activation memory is used by `sam_model.blocks.[0-11].attn`?”
  - We cannot reason about how MoE routing or crop grids affect tensor sizes along the graph.
- Adding this metadata will:
  - Turn the callgraph into a **joint structure + memory profile**.
  - Enable downstream tools (reports, web UI, notebooks) to show per-node tensor sizes and cumulative footprints.

## How to Refactor

### Step 1 – Extend callgraph data model and JSON format

- **`DsocrCallgraph`** (in `src/llm_perf_opt/patches/dsocr_torchlens.py`):
  - Add new fields:
    - `constructor_params: Dict[str, Dict[str, Any]]`
    - `input_metadata: Dict[str, Any]`  (per module name, may be a list if multiple inputs)
    - `output_metadata: Dict[str, Any]`
  - Update `load_callgraph_json` to:
    - Load these keys if present in the JSON.
    - Default to `{}` when missing (backwards compatible with existing JSON).

**Before (simplified):**

```python
@dataclass
class DsocrCallgraph:
    call_counts: Dict[str, int]
    edges: Dict[Tuple[str, str], int]
    op_call_counts: Dict[str, int]
    module_children: Dict[str, List[str]]
    module_types: Dict[str, str]
```

**After (simplified):**

```python
@dataclass
class DsocrCallgraph:
    call_counts: Dict[str, int]
    edges: Dict[Tuple[str, str], int]
    op_call_counts: Dict[str, int]
    module_children: Dict[str, List[str]]
    module_types: Dict[str, str]
    constructor_params: Dict[str, Dict[str, Any]]
    input_metadata: Dict[str, Any]
    output_metadata: Dict[str, Any]
```

### Step 2 – Capture constructor/config parameters at runtime

- Add a **lightweight reflection pass** that runs once after DeepSeek-OCR is instantiated and before TorchLens logging:
  - Implement a helper in `dsocr_torchlens.py`, e.g.:

    ```python
    def collect_constructor_params(root: nn.Module) -> Dict[str, Dict[str, Any]]:
        params: Dict[str, Dict[str, Any]] = {}
        for name, module in root.named_modules():
            cfg: Dict[str, Any] = {}
            # Example heuristics:
            if hasattr(module, "hidden_size"):
                cfg["hidden_size"] = int(module.hidden_size)
            if hasattr(module, "num_heads"):
                cfg["num_heads"] = int(module.num_heads)
            # ... add more per-class or generic rules ...
            if cfg:
                params[name] = cfg
        return params
    ```

  - Decide how to **store** this alongside TorchLens JSON:
    - Option A: extend the TorchLens JSON writer (in the Option C tracing script) to merge `constructor_params` into the same JSON file as `call_counts`, `edges`, etc.
    - Option B: write a separate JSON file (e.g. `dsocr-callgraph-node-metadata.json`) and merge it in `load_callgraph_json`.
  - Preferred: **Option A** (single JSON) for easier consumption and alignment with the design contract (“metadata is part of the callgraph JSON”).

### Step 3 – Attach input/output tensor metadata via TorchLens hooks or patches

- We need **per-node input/output metadata** (shapes + dtypes) for `nn.Module` nodes.
- Implementation strategy:

1. **Add a global per-run registry** in `dsocr_torchlens.py`:

   ```python
   @dataclass
   class TensorIOInfo:
       inputs: List[Dict[str, Any]] = field(default_factory=list)
       outputs: List[Dict[str, Any]] = field(default_factory=list)

   _TENSOR_IO: Dict[str, TensorIOInfo] = {}
   ```

2. **Use `register_forward_hook` / `register_forward_pre_hook`** on the DeepSeek-OCR core model before calling `torchlens.log_forward_pass`:

   - Add a helper in `dsocr_torchlens.py`:

     ```python
     def attach_tensor_io_hooks(root: nn.Module) -> List[Any]:
         handles = []
         def _hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
             name = module._module_name  # or use a mapping from id(module) -> runtime name
             info = _TENSOR_IO.setdefault(name, TensorIOInfo())
             info.inputs.append(serialize_tensors(inputs))
             info.outputs.append(serialize_tensors((output,)))
         for name, module in root.named_modules():
             module._module_name = name
             handles.append(module.register_forward_hook(_hook))
         return handles
     ```

   - `serialize_tensors` would:
     - Walk through tuple/dict/list of arguments.
     - For each `torch.Tensor`, capture:
       - `shape` as a list of ints.
       - `dtype` as string (`str(t.dtype)`).
       - Optional: `device`, `requires_grad`.
     - Ignore non-tensor types or record them as `"non_tensor"`.

3. **Coordinate with TorchLens**:
   - We need to ensure our hooks **do not interfere** with TorchLens’s own instrumentation.
   - Practical approach:
     - Attach hooks **before** `torchlens.log_forward_pass`, letting both systems observe the same forward.
     - Hooks must be side-effect-free (read-only, no tensor ops).

4. **Persist input/output metadata**:
   - After the traced run completes:
     - Convert `_TENSOR_IO` into serializable dicts keyed by runtime name.
     - Store them in the callgraph JSON:

     ```python
     callgraph_json["input_metadata"] = {name: info.inputs for name, info in _TENSOR_IO.items()}
     callgraph_json["output_metadata"] = {name: info.outputs for name, info in _TENSOR_IO.items()}
     ```

   - For grouped/pattern nodes (e.g. `sam_model.blocks.[0-11]`), downstream tools can:
     - Aggregate shapes across instances (e.g., confirm they are identical).
     - Compute total tensor sizes by multiplying per-instance sizes by `for N` / `parfor N`.

### Step 4 – Integrate with Option C tracing script

- In `scripts/analytical/dsocr_torchlens_callgraph_option_c.py`:
  - After constructing the `DeepSeekOCRSession` and before `torchlens.log_forward_pass`:
    - Call `collect_constructor_params(session.m_model.model)` and `attach_tensor_io_hooks(session.m_model.model)`.
  - After `torchlens.log_forward_pass` completes:
    - Merge `constructor_params`, `input_metadata`, and `output_metadata` into the callgraph JSON object before writing it.
    - Remove/deactivate hooks by calling `.remove()` on all handles (so we don’t leak hooks into other runs).

Pseudo‑before:

```python
graph = tl.log_forward_pass(core, ...)
graph.to_json(output_path)
```

Pseudo‑after:

```python
handles = attach_tensor_io_hooks(core)
graph = tl.log_forward_pass(core, ...)
for h in handles:
    h.remove()

obj = graph.to_json_dict()
obj["constructor_params"] = collect_constructor_params(core)
obj["input_metadata"] = tensor_io_to_dict()
obj["output_metadata"] = tensor_io_to_dict(outputs=True)
write_json(output_path, obj)
```

### Step 5 – Expose metadata in Option C post-processing

- `load_callgraph_json` will now populate:
  - `constructor_params`, `input_metadata`, `output_metadata`.
- `summarize_stacks` / `summarize_parfor_edges`:
  - Keep them focused on structural aggregation; do not change semantics.
  - Optionally, they may compute derived stats (e.g., total activation size for `sam_model.blocks.[0-11]`) if needed later.
- `build_grouped_dot_from_option_c` / `build_grouped_mermaid_from_option_c`:
  - Labels stay compact (`Class @ pattern for N` and `parfor N`), as per design.
  - If desired, we can add **tooltip metadata** in DOT (e.g., `tooltip="hidden_size=4096, heads=32"`), but this is optional and can be a second phase.
- Leave **visual layout** unchanged; emphasis is on enhancing the underlying JSON.

### Step 6 – Validate and iterate

- Run the full pipeline in the `rtx5090` env:
  - `pixi run -e rtx5090 python scripts/analytical/dsocr_torchlens_callgraph_option_c.py ...`
  - `pixi run -e rtx5090 python scripts/analytical/dsocr_parse_callgraph_option_c.py ...`
- Inspect updated JSON:
  - Confirm `constructor_params`, `input_metadata`, `output_metadata` are present.
  - Check a few key nodes:
    - `sam_model.blocks.0.attn`:
      - Config: `hidden_size`, `num_heads`.
      - I/O shapes: `[B, T, C]` or `[B, H, T, Dh]` depending on implementation.
    - `layers.1.mlp.experts.0`:
      - Config: `intermediate_size`, `num_experts`, `top_k`.
      - I/O shapes consistent across experts.

## Impact Analysis

- **Behavioral impact**:
  - Forward behavior of DeepSeek-OCR must remain unchanged:
    - Hooks are read-only and should not allocate new tensors.
    - Constructor reflection only inspects attributes.
  - TorchLens’s view of the graph should remain structurally identical:
    - We must avoid wrapping modules or altering control flow.

- **Performance impact**:
  - Additional overhead from:
    - Registering hooks on all modules.
    - Serializing I/O shapes/dtypes.
  - Mitigation:
    - Only enable this path in analytical scripts (already the case).
    - Keep metadata minimal: record shapes/dtypes only, avoid full tensor dumps.
    - Optionally limit to a whitelist of module families (e.g., `sam_model.blocks`, `model.layers`, MoE experts) if overhead is too high.

- **Compatibility**:
  - `load_callgraph_json` remains backward compatible:
    - Old JSON files without metadata still load (fields default to `{}`).
  - Downstream scripts (`dsocr_parse_callgraph_option_c.py`) can start using the new fields opportunistically, with guards for missing keys.

- **Risk**:
  - Hook registration could conflict with TorchLens if not done carefully.
  - Large JSON size if we store per-call I/O for every module.
  - Mitigations:
    - Coordinate call order with TorchLens (hooks attached before logging, removed after).
    - Optionally store only the **first observed** I/O shapes per module instance.

## Expected Outcome

- Updated callgraph JSON (e.g., `dsocr-call-graph-torchlens.json`) will contain:

  ```json
  {
    "call_counts": { ... },
    "edges": { ... },
    "module_types": { ... },
    "constructor_params": {
      "sam_model.blocks.0.attn": {
        "hidden_size": 768,
        "num_heads": 12
      },
      "layers.1.mlp.experts.0": {
        "intermediate_size": 16384,
        "num_experts": 64,
        "top_k": 4
      },
      ...
    },
    "input_metadata": {
      "sam_model.blocks.0.attn": [
        { "tensors": [{ "shape": [1, 196, 768], "dtype": "torch.float16" }] }
      ],
      ...
    },
    "output_metadata": {
      "sam_model.blocks.0.attn": [
        { "tensors": [{ "shape": [1, 196, 768], "dtype": "torch.float16" }] }
      ],
      ...
    }
  }
  ```

- Option C grouped views (DOT/SVG/Mermaid) remain visually similar, but:
  - Downstream tools can compute:
    - Per-module and per-family activation memory.
    - Total tensor size for a path or the whole model.
  - Future UI/reporting layers can expose hover tooltips or side panels with config + I/O info.

## References

- Design contract: `context/design/dsocr-callgraph-human-friendly.md`
- Current TorchLens patching + grouping:
  - `src/llm_perf_opt/patches/dsocr_torchlens.py`
  - `scripts/analytical/dsocr_torchlens_callgraph_option_c.py`
  - `scripts/analytical/dsocr_parse_callgraph_option_c.py`
- Existing tasks and profiling context:
  - `context/tasks/001-profile-deepseek-ocr`
  - `context/tasks/003-nvtx-ncu-profiling`
- Third-party library:
  - TorchLens (PyPI): `/torchlens/torchlens` (Context7 library id, if available).

