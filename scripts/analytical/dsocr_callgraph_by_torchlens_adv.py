"""DeepSeek-OCR TorchLens call graph with runtime loop metadata (Option C).

This script:
- Applies runtime monkeypatches to DeepSeek-OCR for TorchLens compatibility
  and loop metadata logging.
- Runs a TorchLens-based forward trace on the vendor core model.
- Emits:
  - Dynamic call graph JSON/DOT (same shape as dsocr_callgraph_by_torchlens_basic.py)
  - Runtime metadata JSON with coarse loop structure.

NOTE: This script is separate from dsocr_callgraph_by_torchlens_basic.py and does not
modify it. It is intended for analytical use only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple
import contextlib

import torch
import torchlens as tl  # type: ignore[import-untyped]

from llm_perf_opt.runners.dsocr_analyzer import AnalysisConfig, DeepseekOCRStaticAnalyzer
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession
from llm_perf_opt.patches.dsocr_torchlens import (
    apply_runtime_patches,
    attach_tensor_io_hooks,
    collect_constructor_params,
    reset_runtime_log,
    reset_tensor_io_metadata,
    tensor_io_to_metadata,
    write_runtime_log,
)


def find_repo_root(start: Path) -> Path:
    """Locate the repository root by walking up to find pyproject.toml."""

    for current in (start, *start.parents):
        if (current / "pyproject.toml").exists():
            return current
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


def _strip_pass_label(module_with_pass: str) -> str:
    """Strip TorchLens pass suffix from module label: 'name:1' -> 'name'."""

    if ":" in module_with_pass:
        return module_with_pass.split(":", 1)[0]
    return module_with_pass


def compute_callgraph(
    history: Any,
) -> tuple[
    Dict[str, int],
    Dict[Tuple[str, str], int],
    Dict[str, int],
    Dict[str, list[str]],
    Dict[str, str],
]:
    """Derive module-level call counts, edges, and types from TorchLens ModelHistory."""

    module_call_counts: Dict[str, int] = {
        name: int(num_passes) for name, num_passes in history.module_num_passes.items()
    }

    op_call_counts: Dict[str, int] = defaultdict(int)
    edge_call_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for layer_label in history.layer_labels:
        entry = history[layer_label]
        modules_with_pass: Iterable[str] = getattr(entry, "containing_modules_origin_nested", [])
        modules_with_pass = list(modules_with_pass)
        if not modules_with_pass:
            continue

        module_names = [_strip_pass_label(m) for m in modules_with_pass]

        # Attribute each operation to its innermost module.
        innermost = module_names[-1]
        op_call_counts[innermost] += 1

        # Build module-level edges along the nesting chain for this op.
        if len(module_names) > 1:
            parent = module_names[0]
            for child in module_names[1:]:
                edge_call_counts[(parent, child)] += 1
                parent = child

    # Static module hierarchy from TorchLens (no counts)
    module_children: Dict[str, list[str]] = {
        parent: list(children) for parent, children in history.module_children.items()
    }

    # Module types (class names) from TorchLens
    module_types: Dict[str, str] = {
        addr: str(tp) for addr, tp in history.module_types.items()
    }

    return module_call_counts, edge_call_counts, op_call_counts, module_children, module_types


def write_callgraph_outputs(
    out_root: Path,
    module_call_counts: Mapping[str, int],
    edge_call_counts: Mapping[Tuple[str, str], int],
    op_call_counts: Mapping[str, int],
    module_children: Mapping[str, Iterable[str]],
    module_types: Mapping[str, str],
    constructor_params: Mapping[str, Mapping[str, Any]] | None = None,
    input_metadata: Mapping[str, Any] | None = None,
    output_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write JSON and Graphviz DOT call graph outputs to the given directory."""

    out_root.mkdir(parents=True, exist_ok=True)

    edges_flat = {
        f"{src}->{dst}": int(cnt) for (src, dst), cnt in edge_call_counts.items()
    }

    data: Dict[str, Any] = {
        "call_counts": {k: int(v) for k, v in module_call_counts.items()},
        "edges": edges_flat,
        "op_call_counts": {k: int(v) for k, v in op_call_counts.items()},
        "module_children": {
            parent: list(children) for parent, children in module_children.items()
        },
        "module_types": {k: str(v) for k, v in module_types.items()},
    }

    if constructor_params is not None:
        data["constructor_params"] = {k: dict(v) for k, v in constructor_params.items()}
    if input_metadata is not None:
        data["input_metadata"] = dict(input_metadata)
    if output_metadata is not None:
        data["output_metadata"] = dict(output_metadata)

    json_path = out_root / "dsocr-call-graph-torchlens.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    dot_lines = ["digraph dsocr_torchlens {"]
    for (src, dst), cnt in edge_call_counts.items():
        dot_lines.append(f'  "{src}" -> "{dst}" [label="{int(cnt)}"];')
    dot_lines.append("}")

    dot_path = out_root / "dsocr-call-graph-torchlens.dot"
    dot_path.write_text("\n".join(dot_lines), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="TorchLens-based dynamic module call graph for DeepSeek-OCR (with runtime loop metadata).",
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
        help="Output directory (default: <repo_root>/tmp/dsocr-torchlens-callgraph).",
    )
    return parser


def main() -> None:
    """Entry point for dynamic TorchLens tracing with runtime metadata."""

    parser = build_argparser()
    args = parser.parse_args()

    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)

    model_path = (
        Path(args.model_path).resolve()
        if args.model_path is not None
        else (repo_root / "models" / "deepseek-ocr").resolve()
    )

    session = DeepSeekOCRSession.from_local(
        model_path=str(model_path),
        device=args.device,
        use_flash_attn=False,
    )

    # Runtime patches: TorchLens compatibility + loop metadata.
    apply_runtime_patches()

    if session.m_model is None:
        raise RuntimeError("DeepSeekOCRSession did not initialize model")

    model = session.m_model
    core = getattr(model, "model", model)

    input_ids, model_kwargs = build_inputs(
        session=session,
        seq_len=int(args.seq_len),
        base_size=int(args.base_size),
        image_size=int(args.image_size),
    )

    images_value = model_kwargs["images"]
    if isinstance(images_value, (list, tuple)):
        images_for_torchlens: list[Any] = []
        for item in images_value:
            if isinstance(item, tuple):
                images_for_torchlens.append(list(item))
            else:
                images_for_torchlens.append(item)
    else:
        images_for_torchlens = images_value  # type: ignore[assignment]

    input_args = (input_ids,)
    input_kwargs = {
        "images": images_for_torchlens,
        "images_seq_mask": model_kwargs["images_seq_mask"],
        "images_spatial_crop": model_kwargs["images_spatial_crop"],
    }

    device = session.m_device
    device_type = device.type if isinstance(device, torch.device) else "cuda"
    dtype = session.m_dtype or torch.bfloat16

    if device_type == "cuda":
        autocast_ctx = torch.autocast("cuda", dtype=dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    # Reset per-run metadata and attach tensor I/O hooks.
    reset_runtime_log()
    reset_tensor_io_metadata()
    tensor_io_handles = attach_tensor_io_hooks(core)

    with torch.no_grad(), autocast_ctx:
        history = tl.log_forward_pass(
            core,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save="none",
            vis_opt="none",
        )

    # Remove hooks now that tracing is complete.
    for handle in tensor_io_handles:
        try:
            handle.remove()
        except Exception:
            # Best-effort cleanup; do not fail the run for hook removal issues.
            pass

    module_call_counts, edge_call_counts, op_call_counts, module_children, module_types = compute_callgraph(history)

    out_root = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else repo_root / "tmp" / "dsocr-torchlens-callgraph"
    )

    # Collect constructor/config params and tensor I/O metadata.
    constructor_params = collect_constructor_params(core)
    input_metadata, output_metadata = tensor_io_to_metadata()

    write_callgraph_outputs(
        out_root=out_root,
        module_call_counts=module_call_counts,
        edge_call_counts=edge_call_counts,
        op_call_counts=op_call_counts,
        module_children=module_children,
        module_types=module_types,
        constructor_params=constructor_params,
        input_metadata=input_metadata,
        output_metadata=output_metadata,
    )

    # Write runtime loop/stack metadata.
    write_runtime_log(out_root / "dsocr-callgraph-runtime-metadata.json")


if __name__ == "__main__":
    main()
