"""Runtime monkeypatches for DeepSeek-OCR TorchLens tracing.

This module provides small, model-specific patches used only for analytical
tracing (e.g., TorchLens-based call graph extraction). It MUST NOT be imported
in production inference paths.

Design constraints
------------------
- Do not modify vendor files on disk (HF cache or models/deepseek-ocr).
- Do not change the module hierarchy or introduce new nn.Module instances.
- Do not change control flow of the model forward; wrappers must call the
  original forward exactly once.
- Avoid creating new tensor-producing torch ops in wrappers to keep the
  TorchLens call graph structurally identical to the unpatched model.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple
import json
import sys

from attrs import define, field
import torch

RUNTIME_LOG_KEY = "dsocr_torchlens_events"


@define(kw_only=True)
class DsocrTorchlensRuntimeLog:
    """Container for lightweight runtime events emitted by patches."""

    events: List[Dict[str, Any]] = field(factory=list)

    def append(self, event: Dict[str, Any]) -> None:
        self.events.append(event)


_RUNTIME_LOG = DsocrTorchlensRuntimeLog()


def reset_runtime_log() -> None:
    """Clear the in-process runtime log."""

    _RUNTIME_LOG.events.clear()


def get_runtime_log() -> DsocrTorchlensRuntimeLog:
    """Return the in-process runtime log object."""

    return _RUNTIME_LOG


def write_runtime_log(path: Path) -> None:
    """Write the runtime log to a JSON file."""

    data = {
        "events": _RUNTIME_LOG.events,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _patch_quick_gelu() -> None:
    """Replace DeepSeek-OCR's JIT scripted quick_gelu with a plain function.

    This patch avoids TorchLens crashes caused by tensors created in scripted
    code that are not tagged with internal metadata. It preserves the math and
    does not alter the module call graph.
    """

    patched = False
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", "") or ""
        if "deepencoder.py" not in mod_file:
            continue
        quick = getattr(mod, "quick_gelu", None)
        if quick is None:
            continue

        # Avoid double-patching
        if getattr(quick, "_dsocr_torchlens_patched", False):
            return

        def quick_gelu(x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(1.702 * x)

        setattr(quick_gelu, "_dsocr_torchlens_patched", True)
        setattr(mod, "quick_gelu", quick_gelu)
        patched = True
        break

    if not patched:
        # Best-effort; tracing can still proceed but may hit known TorchLens issues.
        print("[dsocr-torchlens] WARNING: quick_gelu patch not applied (module not found)", flush=True)


def _patch_deepseek_ocr_model_forward() -> None:
    """Patch DeepseekOCRModel.forward to emit vision-stage metadata.

    Implementation notes
    --------------------
    - We patch the *class* method so TorchLens sees this as the baseline
      forward and decorates it.
    - The wrapper only reads simple metadata (len(images), crop grid) and does
      not introduce new tensor-producing ops.
    """

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", "") or ""
        if "modeling_deepseekocr.py" not in mod_file:
            continue

        cls = getattr(mod, "DeepseekOCRModel", None)
        if cls is None:
            continue

        orig_forward = getattr(cls, "forward", None)
        if orig_forward is None:
            continue

        if getattr(orig_forward, "_dsocr_torchlens_patched", False):
            return

        @wraps(orig_forward)
        def wrapped_forward(
            self: Any,
            input_ids: Any = None,
            attention_mask: Any = None,
            position_ids: Any = None,
            past_key_values: Any = None,
            inputs_embeds: Any = None,
            use_cache: Any = None,
            output_attentions: Any = None,
            output_hidden_states: Any = None,
            images: Any = None,
            images_seq_mask: Any = None,
            images_spatial_crop: Any = None,
            return_dict: Any = None,
        ) -> Any:
            num_images = len(images) if isinstance(images, list) else 0
            num_crops_total = 0
            try:
                if hasattr(images_spatial_crop, "tolist"):
                    crop_list = images_spatial_crop.tolist()
                    # images_spatial_crop rows are [w_crop, h_crop]
                    for row in crop_list:
                        if isinstance(row, (list, tuple)) and len(row) == 2:
                            w, h = int(row[0]), int(row[1])
                            num_crops_total += max(w * h, 0)
            except Exception:
                num_crops_total = 0

            _RUNTIME_LOG.append(
                {
                    "kind": "vision_inputs",
                    "module": "DeepseekOCRModel",
                    "num_images": int(num_images),
                    "num_crops_total": int(num_crops_total),
                }
            )

            return orig_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
                return_dict=return_dict,
            )

        setattr(wrapped_forward, "_dsocr_torchlens_patched", True)
        setattr(cls, "forward", wrapped_forward)
        break


def _patch_deepseek_ocr_for_causal_lm_forward() -> None:
    """Patch DeepseekOCRForCausalLM.forward to emit LLM stack metadata.

    We only log the number of decoder layers (len(self.model.layers)), relying
    on TorchLens to provide per-layer module call counts and edges.
    """

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", "") or ""
        if "modeling_deepseekocr.py" not in mod_file:
            continue

        cls = getattr(mod, "DeepseekOCRForCausalLM", None)
        if cls is None:
            continue

        orig_forward = getattr(cls, "forward", None)
        if orig_forward is None:
            continue

        if getattr(orig_forward, "_dsocr_torchlens_patched", False):
            return

        @wraps(orig_forward)
        def wrapped_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            num_layers = 0
            num_images = 0
            try:
                model_attr = getattr(self, "model", None)
                if model_attr is not None and hasattr(model_attr, "layers"):
                    num_layers = len(getattr(model_attr, "layers"))
            except Exception:
                num_layers = 0

            try:
                images = kwargs.get("images", None)
                if isinstance(images, list):
                    num_images = len(images)
            except Exception:
                num_images = 0

            _RUNTIME_LOG.append(
                {
                    "kind": "llm_stack",
                    "module": "model.layers",
                    "num_layers": int(num_layers),
                    "num_images": int(num_images),
                }
            )

            return orig_forward(self, *args, **kwargs)

        setattr(wrapped_forward, "_dsocr_torchlens_patched", True)
        setattr(cls, "forward", wrapped_forward)
        break


def apply_runtime_patches() -> None:
    """Apply all runtime patches needed for TorchLens-based tracing.

    This should be called after DeepSeek-OCR's HF modules are imported
    (e.g., after constructing DeepSeekOCRSession) and before calling
    torchlens.log_forward_pass().
    """

    _patch_quick_gelu()
    _patch_deepseek_ocr_model_forward()
    _patch_deepseek_ocr_for_causal_lm_forward()


# ---------------------------------------------------------------------------
# Tensor I/O metadata collection (shapes + dtypes)
# ---------------------------------------------------------------------------


@define(kw_only=True)
class TensorIOInfo:
    """Per-module tensor I/O metadata captured via forward hooks."""

    inputs: List[Dict[str, Any]] = field(factory=list)
    outputs: List[Dict[str, Any]] = field(factory=list)


_TENSOR_IO: Dict[str, TensorIOInfo] = {}


def reset_tensor_io_metadata() -> None:
    """Clear accumulated tensor I/O metadata."""

    _TENSOR_IO.clear()


def _serialize_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    """Convert a tensor into a lightweight, JSON-serializable summary."""

    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def _collect_tensor_summaries(obj: Any) -> List[Dict[str, Any]]:
    """Walk an arbitrary input/output object and collect tensor summaries."""

    summaries: List[Dict[str, Any]] = []

    if isinstance(obj, torch.Tensor):
        summaries.append(_serialize_tensor(obj))
        return summaries

    if isinstance(obj, (list, tuple)):
        for item in obj:
            summaries.extend(_collect_tensor_summaries(item))
        return summaries

    if isinstance(obj, dict):
        for item in obj.values():
            summaries.extend(_collect_tensor_summaries(item))
        return summaries

    return summaries


def _record_tensor_io(name: str, inputs: Tuple[Any, ...], output: Any) -> None:
    """Record input/output tensor metadata for a given module runtime name."""

    info = _TENSOR_IO.setdefault(name, TensorIOInfo())
    input_summaries = _collect_tensor_summaries(inputs)
    output_summaries = _collect_tensor_summaries(output)

    if input_summaries:
        info.inputs.append({"tensors": input_summaries})
    if output_summaries:
        info.outputs.append({"tensors": output_summaries})


def attach_tensor_io_hooks(root: torch.nn.Module) -> List[Any]:
    """Attach forward hooks to capture tensor I/O metadata for all modules.

    Notes
    -----
    - Hooks are read-only and must not allocate new tensors.
    - The caller is responsible for removing all returned handles after the run.
    """

    handles: List[Any] = []

    def make_hook(module_name: str):
        def _hook(module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:  # type: ignore[override]
            _record_tensor_io(module_name, inputs, output)

        return _hook

    for name, module in root.named_modules():
        hook = make_hook(name)
        handle = module.register_forward_hook(hook)
        handles.append(handle)

    return handles


def tensor_io_to_metadata() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return input/output metadata dictionaries suitable for JSON serialization."""

    input_metadata: Dict[str, Any] = {}
    output_metadata: Dict[str, Any] = {}

    for name, info in _TENSOR_IO.items():
        if info.inputs:
            input_metadata[name] = info.inputs
        if info.outputs:
            output_metadata[name] = info.outputs

    return input_metadata, output_metadata


# ---------------------------------------------------------------------------
# Constructor / config parameter collection
# ---------------------------------------------------------------------------


_CONFIG_ATTR_CANDIDATES: Tuple[str, ...] = (
    # Common transformer dims
    "hidden_size",
    "hidden_dim",
    "d_model",
    "embed_dim",
    "intermediate_size",
    "ffn_dim",
    # Attention / heads
    "num_heads",
    "n_heads",
    "num_key_value_heads",
    # Depth / layers
    "num_layers",
    "num_hidden_layers",
    "num_blocks",
    # Convolution / spatial
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    # MoE / experts
    "num_experts",
    "top_k",
)


def _serialize_config_value(value: Any) -> Any:
    """Convert a config attribute into a JSON-serializable primitive."""

    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for item in value:
            if isinstance(item, (int, float, bool, str)):
                out.append(item)
        return out
    return str(value)


def collect_constructor_params(root: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    """Collect shape-relevant constructor/config params for each module.

    The returned mapping is keyed by module runtime name (as in named_modules()).
    Only a curated set of attributes is inspected to avoid pulling in large
    configs; values are converted to JSON-serializable primitives.
    """

    params: Dict[str, Dict[str, Any]] = {}

    for name, module in root.named_modules():
        cfg: Dict[str, Any] = {}
        for attr in _CONFIG_ATTR_CANDIDATES:
            if not hasattr(module, attr):
                continue
            try:
                raw = getattr(module, attr)
            except Exception:
                continue
            serialized = _serialize_config_value(raw)
            cfg[attr] = serialized

        if cfg:
            params[name] = cfg

    return params


# ---------------------------------------------------------------------------
# Call graph post-processing utilities (Option C grouping)
# ---------------------------------------------------------------------------


@define(kw_only=True)
class DsocrCallgraph:
    """Loaded call graph and metadata from TorchLens JSON."""

    call_counts: Dict[str, int] = field()
    edges: Dict[Tuple[str, str], int] = field()
    op_call_counts: Dict[str, int] = field()
    module_children: Dict[str, List[str]] = field()
    module_types: Dict[str, str] = field()
    constructor_params: Dict[str, Dict[str, Any]] = field(factory=dict)
    input_metadata: Dict[str, Any] = field(factory=dict)
    output_metadata: Dict[str, Any] = field(factory=dict)


def load_callgraph_json(path: Path) -> DsocrCallgraph:
    """Load TorchLens call graph JSON produced by dsocr TorchLens scripts."""

    obj = json.loads(path.read_text(encoding="utf-8"))
    raw_edges: Dict[str, int] = obj.get("edges", {})
    edges: Dict[Tuple[str, str], int] = {}
    for key, cnt in raw_edges.items():
        if "->" not in key:
            continue
        parent, child = key.split("->", 1)
        edges[(parent, child)] = int(cnt)

    call_counts = {k: int(v) for k, v in obj.get("call_counts", {}).items()}
    op_call_counts = {k: int(v) for k, v in obj.get("op_call_counts", {}).items()}
    module_children = {
        parent: list(children) for parent, children in obj.get("module_children", {}).items()
    }
    module_types = {k: str(v) for k, v in obj.get("module_types", {}).items()}
    constructor_params = {
        k: dict(v) for k, v in obj.get("constructor_params", {}).items()
    }
    input_metadata = {k: v for k, v in obj.get("input_metadata", {}).items()}
    output_metadata = {k: v for k, v in obj.get("output_metadata", {}).items()}
    return DsocrCallgraph(
        call_counts=call_counts,
        edges=edges,
        op_call_counts=op_call_counts,
        module_children=module_children,
        module_types=module_types,
        constructor_params=constructor_params,
        input_metadata=input_metadata,
        output_metadata=output_metadata,
    )


def load_runtime_metadata(path: Path) -> DsocrTorchlensRuntimeLog:
    """Load runtime metadata JSON produced by write_runtime_log."""

    obj = json.loads(path.read_text(encoding="utf-8"))
    log = DsocrTorchlensRuntimeLog()
    for ev in obj.get("events", []):
        if isinstance(ev, dict):
            log.append(ev)
    return log


def _extract_indices_for_family(call_counts: Dict[str, int], family_prefix: str) -> List[int]:
    """Extract numeric indices for a module family prefix like 'model.layers' or 'sam_model.blocks'."""

    indices: set[int] = set()
    prefix = f"{family_prefix}."
    for name in call_counts.keys():
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        first_token = suffix.split(".", 1)[0]
        if first_token.isdigit():
            indices.add(int(first_token))
    return sorted(indices)


def summarize_stacks(callgraph: DsocrCallgraph) -> List[Dict[str, Any]]:
    """Summarize depth-wise stacks ('for N') for key DeepSeek-OCR module families.

    Each entry includes:
    - family: module family prefix (e.g., 'sam_model.blocks')
    - indices: sorted list of indices
    - for: number of indices
    - class_name: class name of the family modules
    - pattern_runtime: runtime-name pattern with index range (e.g., 'sam_model.blocks.[0-11]')
    """

    families = ["sam_model.blocks", "model.layers"]
    stacks: List[Dict[str, Any]] = []

    for fam in families:
        indices = _extract_indices_for_family(callgraph.call_counts, fam)
        if not indices:
            continue

        # Determine class name from first index, if available
        class_name = "Module"
        first_name = f"{fam}.{indices[0]}"
        if first_name in callgraph.module_types:
            # module_types entries are typically like 'SAMBlock' etc.
            class_name = str(callgraph.module_types[first_name])

        # Build simple [min-max] range pattern
        if indices:
            min_idx = indices[0]
            max_idx = indices[-1]
            if indices == list(range(min_idx, max_idx + 1)):
                idx_pattern = f"[{min_idx}-{max_idx}]"
            else:
                # Fallback: explicit list
                idx_pattern = "[" + ",".join(str(i) for i in indices) + "]"
        else:
            idx_pattern = "[]"

        pattern_runtime = f"{fam}.{idx_pattern}"

        stacks.append(
            {
                "family": fam,
                "indices": indices,
                "for": len(indices),
                "class_name": class_name,
                "pattern_runtime": pattern_runtime,
            }
        )
    return stacks


def summarize_parfor_edges(callgraph: DsocrCallgraph, min_parfor: int = 2) -> List[Dict[str, Any]]:
    """Summarize repeated use of the same module at a given graph level ('parfor N').

    We treat each parent->child pair where the dynamic edge count > min_parfor-1
    as a 'parfor N' usage of the child under that parent.
    """

    parfor: List[Dict[str, Any]] = []
    for (parent, child), cnt in callgraph.edges.items():
        if cnt < min_parfor:
            continue
        parfor.append(
            {
                "parent": parent,
                "child": child,
                "parfor": int(cnt),
            }
        )
    return parfor


def _build_index_pattern(indices: List[int]) -> str:
    """Return a human-friendly index pattern like '[0-11]' or '[0,2,4]'.

    This mirrors the pattern style used in summarize_stacks().
    """

    if not indices:
        return "[]"
    indices = sorted(set(indices))
    min_idx = indices[0]
    max_idx = indices[-1]
    if indices == list(range(min_idx, max_idx + 1)):
        return f"[{min_idx}-{max_idx}]"
    return "[" + ",".join(str(i) for i in indices) + "]"


def build_grouped_dot_from_option_c(
    callgraph: DsocrCallgraph,
    stacks: List[Dict[str, Any]],
    parfor_edges: List[Dict[str, Any]],
) -> str:
    """Build DOT graph from Option C grouped information following the human-friendly contract.

    - Stack nodes: '<ClassName> @ <pattern_runtime> for N'
    - Parfor edges: '<ParentClass> @ <parent> -> <ChildClass> @ <child>' [label='parfor N']
    """

    lines: List[str] = []
    lines.append("digraph dsocr_grouped_option_c {")
    # Arrange call depth left-to-right so that siblings
    # (e.g., many experts or blocks) stack vertically.
    lines.append("  rankdir=LR;")
    lines.append("  nodesep=0.3;")
    lines.append("  ranksep=0.7;")

    # Stack nodes: group families into a single pattern node
    def _module_io_suffix(module_name: str) -> str:
        """Build a short 'in/out' suffix from tensor I/O metadata."""

        parts: List[str] = []

        input_meta = callgraph.input_metadata.get(module_name)
        if isinstance(input_meta, list) and input_meta:
            tensors = input_meta[0].get("tensors") if isinstance(input_meta[0], dict) else None
            if isinstance(tensors, list) and tensors:
                t0 = tensors[0]
                shape = t0.get("shape")
                dtype = t0.get("dtype")
                if shape is not None and dtype is not None:
                    parts.append(f"in {shape} {dtype}")

        output_meta = callgraph.output_metadata.get(module_name)
        if isinstance(output_meta, list) and output_meta:
            tensors = output_meta[0].get("tensors") if isinstance(output_meta[0], dict) else None
            if isinstance(tensors, list) and tensors:
                t0 = tensors[0]
                shape = t0.get("shape")
                dtype = t0.get("dtype")
                if shape is not None and dtype is not None:
                    parts.append(f"out {shape} {dtype}")

        if not parts:
            return ""

        # DOT newline is '\n' inside a quoted string.
        return "\\n" + " | ".join(parts)

    for s in stacks:
        fam = s.get("family", "")
        n_for = int(s.get("for", 0))
        if n_for <= 0:
            continue
        class_name = s.get("class_name", "Module")
        pattern_runtime = s.get("pattern_runtime", fam)

        # Use the first index in the family to derive representative I/O shapes.
        indices = s.get("indices") or []
        io_suffix = ""
        if isinstance(indices, list) and indices:
            first_idx = indices[0]
            first_name = f"{fam}.{first_idx}"
            io_suffix = _module_io_suffix(first_name)

        base_label = f"{class_name} @ {pattern_runtime} for {n_for}"
        label = base_label + io_suffix
        lines.append(f'  "{label}" [shape=box, style=filled, fillcolor=lightgray];')

    # Parfor edges: use class@runtime labels where possible, with I/O info where available.
    def _module_label(name: str) -> str:
        cls = callgraph.module_types.get(name, "Module")
        io_suffix = _module_io_suffix(name)
        return f"{cls} @ {name}{io_suffix}"

    # First, group parfor edges where the parent is the same and the children
    # are different *instances* of the same module class that only differ by a
    # numeric index in their runtime name (e.g., experts.0, experts.1, ...).
    #
    # This implements the "MoE experts should be grouped" behavior and also
    # applies to other families that follow the same pattern.

    # key: (parent_name, child_class, family_prefix, suffix)
    #  - family_prefix: tokens before the varying numeric index
    #  - suffix: tokens after the varying numeric index (may be empty)
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    simple_edges: List[Dict[str, Any]] = []

    for e in parfor_edges:
        parent_raw = e.get("parent", "")
        child_raw = e.get("child", "")
        n = int(e.get("parfor", 0))
        if not parent_raw or not child_raw or n <= 0:
            continue

        tokens = child_raw.split(".")
        idx_positions = [i for i, tok in enumerate(tokens) if tok.isdigit()]
        if not idx_positions:
            # No obvious index position – keep as a simple edge.
            simple_edges.append(e)
            continue

        idx_pos = idx_positions[-1]
        prefix_tokens = tokens[:idx_pos]
        idx_token = tokens[idx_pos]
        suffix_tokens = tokens[idx_pos + 1 :]

        try:
            idx_val = int(idx_token)
        except ValueError:
            # Should not happen due to isdigit() guard, but be safe.
            simple_edges.append(e)
            continue

        family_prefix = ".".join(prefix_tokens) if prefix_tokens else ""
        suffix = ".".join(suffix_tokens) if suffix_tokens else ""
        child_cls = callgraph.module_types.get(child_raw, "Module")
        key = (parent_raw, child_cls, family_prefix, suffix)

        group = grouped.setdefault(
            key,
            {
                "indices": set(),
                "parfor_total": 0,
            },
        )
        group["indices"].add(idx_val)
        group["parfor_total"] += n

    # Emit grouped parfor edges with pattern runtime names on the callee side.
    for (parent_raw, child_cls, family_prefix, suffix), data in grouped.items():
        indices = sorted(data["indices"])
        parfor_total = int(data["parfor_total"])
        if not parent_raw or parfor_total <= 0 or not indices:
            continue

        idx_pattern = _build_index_pattern(indices)

        # Build runtime pattern: '<family_prefix>.[idx_pattern].<suffix?>'
        if family_prefix and suffix:
            child_runtime = f"{family_prefix}.{idx_pattern}.{suffix}"
        elif family_prefix:
            child_runtime = f"{family_prefix}.{idx_pattern}"
        elif suffix:
            child_runtime = f"{idx_pattern}.{suffix}"
        else:
            # Degenerate case; fall back to index pattern alone.
            child_runtime = idx_pattern

        # Derive representative I/O metadata from the first concrete child.
        first_concrete_child = ""
        if family_prefix:
            first_concrete_child = f"{family_prefix}.{indices[0]}"
            if suffix:
                first_concrete_child = f"{first_concrete_child}.{suffix}"

        parent_label = _module_label(parent_raw)
        if first_concrete_child:
            io_suffix = _module_io_suffix(first_concrete_child)
        else:
            io_suffix = ""

        child_label = f"{child_cls} @ {child_runtime}{io_suffix}"
        lines.append(f'  "{parent_label}" -> "{child_label}" [label="parfor {parfor_total}"];')

    # Emit simple (non-groupable) parfor edges as-is.
    for e in simple_edges:
        parent_raw = e.get("parent", "")
        child_raw = e.get("child", "")
        n = int(e.get("parfor", 0))
        if not parent_raw or not child_raw or n <= 0:
            continue
        parent_label = _module_label(parent_raw)
        child_label = _module_label(child_raw)
        lines.append(f'  "{parent_label}" -> "{child_label}" [label="parfor {n}"];')

    lines.append("}")
    return "\n".join(lines)


def build_grouped_mermaid_from_option_c(
    callgraph: DsocrCallgraph,
    stacks: List[Dict[str, Any]],
    parfor_edges: List[Dict[str, Any]],
) -> str:
    """Build a Mermaid graph from Option C grouped information.

    The structure mirrors the DOT output but uses Mermaid syntax:
    - graph LR  (depth left-to-right, siblings vertical)
    - Nodes keyed by stable IDs with human-friendly labels.
    """

    lines: List[str] = []
    # Disable max-width clamping so that wide labels are not truncated
    # inside a constrained container (e.g., MkDocs).
    lines.append("%%{init: {'flowchart': { 'useMaxWidth': false }}}%%")
    lines.append("graph LR")

    # Stable node IDs for Mermaid (labels may contain spaces/symbols).
    label_to_id: Dict[str, str] = {}
    defined_labels: set[str] = set()
    next_id = 0

    def node_id(label: str) -> str:
        nonlocal next_id
        nid = label_to_id.get(label)
        if nid is None:
            nid = f"n{next_id}"
            next_id += 1
            label_to_id[label] = nid
        return nid

    def _io_suffix_for_mermaid(module_name: str) -> str:
        """Return a short HTML line-break suffix with I/O metadata."""

        parts: List[str] = []

        input_meta = callgraph.input_metadata.get(module_name)
        if isinstance(input_meta, list) and input_meta:
            tensors = input_meta[0].get("tensors") if isinstance(input_meta[0], dict) else None
            if isinstance(tensors, list) and tensors:
                t0 = tensors[0]
                shape = t0.get("shape")
                dtype = t0.get("dtype")
                if shape is not None and dtype is not None:
                    parts.append(f"in {shape} {dtype}")

        output_meta = callgraph.output_metadata.get(module_name)
        if isinstance(output_meta, list) and output_meta:
            tensors = output_meta[0].get("tensors") if isinstance(output_meta[0], dict) else None
            if isinstance(tensors, list) and tensors:
                t0 = tensors[0]
                shape = t0.get("shape")
                dtype = t0.get("dtype")
                if shape is not None and dtype is not None:
                    parts.append(f"out {shape} {dtype}")

        if not parts:
            return ""
        # Mermaid supports basic HTML; use <br/> for line breaks.
        return "<br/>" + " | ".join(parts)

    def module_label(name: str) -> str:
        cls = callgraph.module_types.get(name, "Module")
        io_suffix = _io_suffix_for_mermaid(name)
        return f"{cls} @ {name}{io_suffix}"

    def render_label(raw: str) -> str:
        """Return a Mermaid-safe label, keeping it on a single line.

        Combined with `useMaxWidth: false`, this lets nodes grow
        horizontally so the full text remains visible.
        """

        # Basic escaping for quotes/backslashes
        txt = raw.replace("\\", "\\\\").replace('"', '\\"')
        return txt

    # Stack nodes: grouped families with 'for N'
    for s in stacks:
        fam = s.get("family", "")
        n_for = int(s.get("for", 0))
        if n_for <= 0:
            continue
        class_name = s.get("class_name", "Module")
        pattern_runtime = s.get("pattern_runtime", fam)
        base_label = f"{class_name} @ {pattern_runtime} for {n_for}"

        # Use first index as representative for I/O shapes.
        indices = s.get("indices") or []
        if isinstance(indices, list) and indices:
            first_idx = indices[0]
            first_name = f"{fam}.{first_idx}"
            io_suffix = _io_suffix_for_mermaid(first_name)
        else:
            io_suffix = ""

        label = base_label + io_suffix
        nid = node_id(label)
        lines.append(f'  {nid}["{render_label(label)}"];')
        defined_labels.add(label)

    # Group parfor edges the same way as in DOT.
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    simple_edges: List[Dict[str, Any]] = []

    for e in parfor_edges:
        parent_raw = e.get("parent", "")
        child_raw = e.get("child", "")
        n = int(e.get("parfor", 0))
        if not parent_raw or not child_raw or n <= 0:
            continue

        tokens = child_raw.split(".")
        idx_positions = [i for i, tok in enumerate(tokens) if tok.isdigit()]
        if not idx_positions:
            simple_edges.append(e)
            continue

        idx_pos = idx_positions[-1]
        prefix_tokens = tokens[:idx_pos]
        idx_token = tokens[idx_pos]
        suffix_tokens = tokens[idx_pos + 1 :]

        try:
            idx_val = int(idx_token)
        except ValueError:
            simple_edges.append(e)
            continue

        family_prefix = ".".join(prefix_tokens) if prefix_tokens else ""
        suffix = ".".join(suffix_tokens) if suffix_tokens else ""
        child_cls = callgraph.module_types.get(child_raw, "Module")
        key = (parent_raw, child_cls, family_prefix, suffix)

        group = grouped.setdefault(
            key,
            {
                "indices": set(),
                "parfor_total": 0,
            },
        )
        group["indices"].add(idx_val)
        group["parfor_total"] += n

    # Grouped parfor edges with pattern runtimes for the callee.
    for (parent_raw, child_cls, family_prefix, suffix), data in grouped.items():
        indices = sorted(data["indices"])
        parfor_total = int(data["parfor_total"])
        if not parent_raw or parfor_total <= 0 or not indices:
            continue

        parent_label = module_label(parent_raw)
        parent_id = node_id(parent_label)
        idx_pattern = _build_index_pattern(indices)

        if family_prefix and suffix:
            child_runtime = f"{family_prefix}.{idx_pattern}.{suffix}"
        elif family_prefix:
            child_runtime = f"{family_prefix}.{idx_pattern}"
        elif suffix:
            child_runtime = f"{idx_pattern}.{suffix}"
        else:
            child_runtime = idx_pattern

        # Derive representative child I/O from the first concrete child.
        first_concrete_child = ""
        if family_prefix:
            first_concrete_child = f"{family_prefix}.{indices[0]}"
            if suffix:
                first_concrete_child = f"{first_concrete_child}.{suffix}"

        if first_concrete_child:
            io_suffix = _io_suffix_for_mermaid(first_concrete_child)
        else:
            io_suffix = ""

        child_label = f"{child_cls} @ {child_runtime}{io_suffix}"
        child_id = node_id(child_label)
        lines.append(f'  {parent_id} -->|parfor {parfor_total}| {child_id};')

    # Simple (non-groupable) parfor edges.
    for e in simple_edges:
        parent_raw = e.get("parent", "")
        child_raw = e.get("child", "")
        n = int(e.get("parfor", 0))
        if not parent_raw or not child_raw or n <= 0:
            continue
        parent_label = module_label(parent_raw)
        child_label = module_label(child_raw)
        parent_id = node_id(parent_label)
        child_id = node_id(child_label)
        lines.append(f'  {parent_id} -->|parfor {n}| {child_id};')

    # Ensure every label we referenced has a node declaration with its
    # human-readable module name.
    for label, nid in label_to_id.items():
        if label in defined_labels:
            continue
        lines.append(f'  {nid}["{render_label(label)}"];')

    return "\n".join(lines)
