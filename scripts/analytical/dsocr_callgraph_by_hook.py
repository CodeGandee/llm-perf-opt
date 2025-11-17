#!/usr/bin/env python
"""DeepSeek-OCR: module-level call graph via PyTorch forward hooks.

This script builds a dynamic nn.Module call graph for the DeepSeek-OCR model
using PyTorch forward (pre/post) hooks on the vendor core model. It focuses on
GPU-heavy modules and records:

- Per-module call counts.
- Parent→child dynamic edge counts (module-to-module calls).
- A chronological enter/exit event log for debugging.

Outputs are written under ``tmp/`` at the repository root:
- tmp/dsocr-call-graph-hooks.json
- tmp/dsocr-call-graph-hooks.dot

Run with a Pixi environment (e.g., RTX 5090 setup):

    pixi run -e rtx5090 python scripts/analytical/dsocr_callgraph_by_hook.py

The script uses the existing DeepSeekOCRSession + DeepseekOCRStaticAnalyzer
to prepare representative synthetic inputs (no real images required) and then
executes a single forward pass of the HF wrapper model while hooks on the
vendor core collect module-level calling relationships.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn

from llm_perf_opt.runners.dsocr_analyzer import AnalysisConfig, DeepseekOCRStaticAnalyzer
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession
from llm_perf_opt.utils import dsocr_callgraph_parse as cg_parse


def find_repo_root(start: Path) -> Path:
    """Locate the repository root by walking up until pyproject.toml is found."""

    cur = start.resolve()
    for _ in range(10):
        if (cur / "pyproject.toml").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start


def build_inputs(
    session: DeepSeekOCRSession,
    seq_len: int,
    base_size: int,
    image_size: int,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Prepare representative inputs using the existing static analyzer."""

    analyzer = DeepseekOCRStaticAnalyzer(session)
    config = AnalysisConfig(seq_len=seq_len, base_size=base_size, image_size=image_size)
    input_ids, model_kwargs = analyzer.prepare_inputs(config)
    return input_ids, model_kwargs


def build_name_mapping(core: nn.Module) -> Dict[nn.Module, str]:
    """Build module → runtime-name mapping with a stable 'model.*' prefix.

    The vendor core is treated as ``model``; its children are named
    ``model.<subpath>`` based on ``core.named_modules()``.
    """

    name_by_module: Dict[nn.Module, str] = {}
    for name, mod in core.named_modules():
        runtime_name = "model" if not name else f"model.{name}"
        name_by_module[mod] = runtime_name
    return name_by_module


def is_gpu_heavy(mod: nn.Module) -> bool:
    """Heuristic to keep only GPU-relevant modules."""

    gpu_heavy_classes: Tuple[type[nn.Module], ...] = (
        nn.Linear,
        nn.Conv2d,
        nn.Embedding,
        nn.LayerNorm,
        nn.MultiheadAttention,
        nn.SiLU,
        nn.ReLU,
    )

    if isinstance(mod, gpu_heavy_classes):
        return True

    try:
        return any(p.is_cuda for p in mod.parameters())
    except Exception:
        return False


def build_module_metadata(
    name_by_module: Mapping[nn.Module, str],
) -> tuple[Dict[str, str], Dict[str, list[str]]]:
    """Derive module_classes and module_children maps from name_by_module."""

    module_classes: Dict[str, str] = {}
    for mod, name in name_by_module.items():
        module_classes[name] = mod.__class__.__name__

    module_children: Dict[str, list[str]] = defaultdict(list)
    for name in module_classes.keys():
        if name == "model":
            continue
        parent, _, _ = name.rpartition(".")
        if not parent:
            parent = "model"
        module_children[parent].append(name)

    return module_classes, dict(module_children)


def _serialize_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    """Convert a tensor into a lightweight, JSON-serializable summary."""

    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def _collect_tensor_summaries(obj: Any) -> list[Dict[str, Any]]:
    """Walk an arbitrary input/output object and collect tensor summaries."""

    summaries: list[Dict[str, Any]] = []

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


def run_with_hooks(
    core: nn.Module,
    model: nn.Module,
    name_by_module: Mapping[nn.Module, str],
    input_ids: torch.Tensor,
    model_kwargs: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    Dict[str, int],
    Dict[Tuple[str, str], int],
    list[Dict[str, Any]],
    Dict[str, Any],
    Dict[str, Any],
]:
    """Attach hooks to GPU-heavy modules, run a forward pass, and collect call graph + tensor I/O."""

    call_counts: Dict[str, int] = defaultdict(int)
    edges: Dict[Tuple[str, str], int] = defaultdict(int)
    events: list[Dict[str, Any]] = []
    call_stack: list[str] = []
    tensor_io: Dict[str, Dict[str, list[Dict[str, Any]]]] = {}

    def pre_hook(mod: nn.Module, inputs: Tuple[Any, ...]) -> None:  # type: ignore[override]
        if not is_gpu_heavy(mod):
            return
        name = name_by_module.get(mod, mod.__class__.__name__)
        parent = call_stack[-1] if call_stack else None
        call_stack.append(name)

        call_counts[name] += 1
        if parent is not None:
            edges[(parent, name)] += 1

        # Record input tensor metadata for this module call.
        input_summaries = _collect_tensor_summaries(inputs)
        if input_summaries:
            info = tensor_io.setdefault(name, {"inputs": [], "outputs": []})
            info["inputs"].append({"tensors": input_summaries})

        events.append(
            {
                "event": "enter",
                "name": name,
                "parent": parent,
                "index": len(events),
            },
        )

    def post_hook(mod: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:  # type: ignore[override]
        if not is_gpu_heavy(mod):
            return
        name = name_by_module.get(mod, mod.__class__.__name__)
        # Record output tensor metadata for this module call.
        output_summaries = _collect_tensor_summaries(output)
        if output_summaries:
            info = tensor_io.setdefault(name, {"inputs": [], "outputs": []})
            info["outputs"].append({"tensors": output_summaries})

        events.append(
            {
                "event": "exit",
                "name": name,
                "index": len(events),
            },
        )
        if call_stack and call_stack[-1] == name:
            call_stack.pop()

    hooks: list[Any] = []
    try:
        for mod in name_by_module.keys():
            if not is_gpu_heavy(mod):
                continue
            hooks.append(mod.register_forward_pre_hook(pre_hook))
            hooks.append(mod.register_forward_hook(post_hook))

        device_type = device.type
        if device_type == "cuda":
            autocast_ctx: contextlib.AbstractContextManager[Any] = torch.autocast("cuda", dtype=dtype)
        else:
            autocast_ctx = contextlib.nullcontext()

        with torch.no_grad(), autocast_ctx:
            _ = model(
                input_ids=input_ids,
                images=model_kwargs["images"],
                images_seq_mask=model_kwargs["images_seq_mask"],
                images_spatial_crop=model_kwargs["images_spatial_crop"],
                use_cache=True,
                return_dict=True,
            )
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                # Best-effort cleanup; do not fail the run for hook removal issues.
                pass

    input_metadata: Dict[str, Any] = {}
    output_metadata: Dict[str, Any] = {}
    for name, info in tensor_io.items():
        inputs_list = info.get("inputs") or []
        outputs_list = info.get("outputs") or []
        if inputs_list:
            input_metadata[name] = inputs_list
        if outputs_list:
            output_metadata[name] = outputs_list

    return dict(call_counts), dict(edges), events, input_metadata, output_metadata


def write_outputs(
    out_root: Path,
    call_counts: Mapping[str, int],
    edges: Mapping[Tuple[str, str], int],
    events: Iterable[Mapping[str, Any]],
    module_children: Mapping[str, Iterable[str]],
    module_classes: Mapping[str, str],
    input_metadata: Mapping[str, Any],
    output_metadata: Mapping[str, Any],
) -> None:
    """Write JSON and Graphviz DOT outputs to the given directory."""

    out_root.mkdir(parents=True, exist_ok=True)

    edges_flat = {f"{src}->{dst}": int(cnt) for (src, dst), cnt in edges.items()}

    events_list = list(events)

    data: Dict[str, Any] = {
        "call_counts": {name: int(cnt) for name, cnt in call_counts.items()},
        "edges": edges_flat,
        "events": events_list,
        "module_children": {parent: list(children) for parent, children in module_children.items()},
        "module_classes": {name: cls for name, cls in module_classes.items()},
        "input_metadata": {name: meta for name, meta in input_metadata.items()},
        "output_metadata": {name: meta for name, meta in output_metadata.items()},
    }

    json_path = out_root / "dsocr-call-graph-hooks.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Build ungrouped DOT graph with rich node labels including class name and
    # representative I/O shapes. This is the fully expanded call graph.
    dot_lines = ["digraph dsocr_hooks {"]
    dot_lines.append("  rankdir=LR;")

    def _module_io_suffix(name: str) -> str:
        """Build a short 'in/out' suffix from tensor I/O metadata."""

        parts: list[str] = []

        in_meta = input_metadata.get(name)
        if isinstance(in_meta, list) and in_meta:
            tensors = in_meta[0].get("tensors") if isinstance(in_meta[0], dict) else None
            if isinstance(tensors, list) and tensors:
                t0 = tensors[0]
                shape = t0.get("shape")
                dtype = t0.get("dtype")
                if shape is not None and dtype is not None:
                    parts.append(f"in {shape} {dtype}")

        out_meta = output_metadata.get(name)
        if isinstance(out_meta, list) and out_meta:
            tensors = out_meta[0].get("tensors") if isinstance(out_meta[0], dict) else None
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

    node_names: set[str] = set(call_counts.keys())
    for src, dst in edges.keys():
        node_names.add(src)
        node_names.add(dst)

    for name in sorted(node_names):
        cls = module_classes.get(name, "Module")
        io_suffix = _module_io_suffix(name)
        label = f"{cls} @ {name}{io_suffix}"
        dot_lines.append(f'  "{name}" [label="{label}"];')

    for (src, dst), cnt in edges.items():
        dot_lines.append(f'  "{src}" -> "{dst}" [label="{int(cnt)}"];')

    dot_lines.append("}")

    dot_path = out_root / "dsocr-call-graph-hooks.dot"
    dot_path.write_text("\n".join(dot_lines), encoding="utf-8")

    # Build a grouped view (for/parfor) using the shared call graph utilities.
    try:
        callgraph = cg_parse.load_callgraph_json(json_path)
        grouped = cg_parse.compute_grouped_callgraph(
            callgraph=callgraph,
            min_family_size=2,
            min_edge_count=2,
            metadata={
                "source": "hooks",
                "source_json": str(json_path),
            },
        )

        grouped_dot = cg_parse.grouped_to_dot(grouped)
        grouped_dot_path = out_root / "dsocr-call-graph-hooks-grouped.dot"
        grouped_dot_path.write_text(grouped_dot, encoding="utf-8")
    except Exception as exc:
        # Grouped view is best-effort; log the failure but do not break raw call graph generation.
        print(f"[dsocr-hooks] WARNING: grouped call graph generation failed: {exc}")
        grouped_dot_path = None

    # Best-effort SVG generation for quick inspection; dot may not be installed
    # everywhere. Generate SVGs for both ungrouped and grouped graphs when possible.
    try:
        import subprocess

        svg_path = out_root / "dsocr-call-graph-hooks.svg"
        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if grouped_dot_path is not None:
            grouped_svg_path = out_root / "dsocr-call-graph-hooks-grouped.svg"
            subprocess.run(
                ["dot", "-Tsvg", str(grouped_dot_path), "-o", str(grouped_svg_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        # Non-fatal if Graphviz is missing; DOT files remain available.
        pass


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR module-level call graph via PyTorch forward hooks.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to local DeepSeek-OCR model "
        "(default: <repo_root>/models/deepseek-ocr).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for model execution (default: cuda:0).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Representative sequence length for tracing (default: 512).",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=1024,
        help="Global view padding size (default: 1024).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Crop size for local views (default: 640).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <repo_root>/tmp/dsocr-callgraph-hooks).",
    )
    return parser


def main() -> None:
    """Entry point for hook-based DeepSeek-OCR call graph tracing."""

    parser = build_argparser()
    args = parser.parse_args()

    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)

    model_path = (
        Path(args.model_path).resolve()
        if args.model_path is not None
        else (repo_root / "models" / "deepseek-ocr").resolve()
    )

    if not model_path.is_dir():
        raise SystemExit(f"Model path not found: {model_path}")

    session = DeepSeekOCRSession.from_local(
        model_path=str(model_path),
        device=args.device,
        use_flash_attn=True,
    )

    if session.m_model is None or session.m_device is None or session.m_dtype is None:
        raise RuntimeError("DeepSeekOCRSession did not initialize model/device/dtype")

    model = session.m_model
    core = getattr(model, "model", model)

    name_by_module = build_name_mapping(core)
    module_classes, module_children = build_module_metadata(name_by_module)

    input_ids, model_kwargs = build_inputs(
        session=session,
        seq_len=int(args.seq_len),
        base_size=int(args.base_size),
        image_size=int(args.image_size),
    )

    call_counts, edges, events, input_metadata, output_metadata = run_with_hooks(
        core=core,
        model=model,
        name_by_module=name_by_module,
        input_ids=input_ids,
        model_kwargs=model_kwargs,
        device=session.m_device,
        dtype=session.m_dtype,
    )

    out_root = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else repo_root / "tmp" / "dsocr-callgraph-hooks"
    )

    write_outputs(
        out_root=out_root,
        call_counts=call_counts,
        edges=edges,
        events=events,
        module_children=module_children,
        module_classes=module_classes,
        input_metadata=input_metadata,
        output_metadata=output_metadata,
    )


if __name__ == "__main__":
    main()
