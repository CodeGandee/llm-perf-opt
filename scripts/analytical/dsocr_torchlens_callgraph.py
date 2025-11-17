"""TorchLens-based dynamic call graph for DeepSeek-OCR.

This script runs a single representative forward pass of the DeepSeek-OCR
model using TorchLens and exports a module-level call graph focused on the
vendor model (no orchestration code).

Outputs are written under ``tmp/`` at the repository root:
- tmp/dsocr-call-graph-torchlens.json
- tmp/dsocr-call-graph-torchlens.dot

Run via Pixi:
    pixi run dsocr-torchlens-callgraph
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple
import contextlib
import sys

import torch
import torchlens as tl  # type: ignore[import-untyped]

from llm_perf_opt.runners.dsocr_analyzer import AnalysisConfig, DeepseekOCRStaticAnalyzer
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession


def find_repo_root(start: Path) -> Path:
    """Locate the repository root by walking up to find pyproject.toml."""

    for current in (start, *start.parents):
        if (current / "pyproject.toml").exists():
            return current
    return start


def patch_deepseek_quick_gelu() -> None:
    """Monkey-patch DeepSeek-OCR quick_gelu to a plain function.

    The vendor implementation uses a torch.jit.script'ed helper, which can
    produce tensors that TorchLens has not tagged with internal metadata.
    Replacing it with a standard PyTorch function keeps numerics but ensures
    TorchLens can see the underlying ops.
    """

    patched = False
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        quick = getattr(mod, "quick_gelu", None)
        if quick is None:
            continue
        mod_file = getattr(mod, "__file__", "") or ""
        if "deepencoder" not in mod_file:
            continue

        def quick_gelu(x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(1.702 * x)

        setattr(mod, "quick_gelu", quick_gelu)
        patched = True
        break

    if not patched:
        # Best-effort; if this fails we still attempt tracing.
        print("[dsocr-torchlens] WARNING: quick_gelu patch not applied (module not found)")


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
) -> tuple[Dict[str, int], Dict[Tuple[str, str], int], Dict[str, int], Dict[str, list[str]]]:
    """Derive module-level call counts and edges from TorchLens ModelHistory.

    Returns
    -------
    module_call_counts : dict
        Module name -> number of forward passes (from module_num_passes).
    edge_call_counts : dict
        (parent_module, child_module) -> dynamic edge count derived from
        per-layer module nesting.
    op_call_counts : dict
        Module name -> number of operations whose innermost module is this
        module (rough proxy for per-op workload).
    module_children : dict
        Static module hierarchy parent -> list[child] as recorded by TorchLens.
    """

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

    return module_call_counts, edge_call_counts, op_call_counts, module_children


def write_outputs(
    out_root: Path,
    module_call_counts: Mapping[str, int],
    edge_call_counts: Mapping[Tuple[str, str], int],
    op_call_counts: Mapping[str, int],
    module_children: Mapping[str, Iterable[str]],
    module_classes: Mapping[str, str],
) -> None:
    """Write JSON and Graphviz DOT outputs to the given directory."""

    out_root.mkdir(parents=True, exist_ok=True)

    edges_flat = {
        f"{src}->{dst}": int(cnt) for (src, dst), cnt in edge_call_counts.items()
    }

    data = {
        "call_counts": {k: int(v) for k, v in module_call_counts.items()},
        "edges": edges_flat,
        "op_call_counts": {k: int(v) for k, v in op_call_counts.items()},
        "module_children": {
            parent: list(children) for parent, children in module_children.items()
        },
        "module_classes": {name: cls for name, cls in module_classes.items()},
    }

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
        description="TorchLens-based dynamic module call graph for DeepSeek-OCR.",
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
    """Entry point for dynamic TorchLens tracing."""

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

    # Runtime patch to make TorchLens compatible with DeepSeek-OCR in this environment.
    patch_deepseek_quick_gelu()

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

    input_args = (input_ids,)

    images_value = model_kwargs["images"]
    # TorchLens mutates input_kwargs tensors in-place when moving to devices.
    # Convert inner image tuples to lists so its nested_assign utility can update them.
    if isinstance(images_value, (list, tuple)):
        images_for_torchlens: list[Any] = []
        for item in images_value:
            if isinstance(item, tuple):
                images_for_torchlens.append(list(item))
            else:
                images_for_torchlens.append(item)
    else:
        images_for_torchlens = images_value  # type: ignore[assignment]

    input_kwargs = {
        "images": images_for_torchlens,
        "images_seq_mask": model_kwargs["images_seq_mask"],
        "images_spatial_crop": model_kwargs["images_spatial_crop"],
    }

    device = session.m_device
    device_type = device.type if isinstance(device, torch.device) else "cuda"
    dtype = session.m_dtype or torch.bfloat16

    autocast_ctx: contextlib.AbstractContextManager[Any]
    if device_type == "cuda":
        autocast_ctx = torch.autocast("cuda", dtype=dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    with torch.no_grad(), autocast_ctx:
        history = tl.log_forward_pass(
            core,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save="none",
            vis_opt="none",
        )

    module_call_counts, edge_call_counts, op_call_counts, module_children = compute_callgraph(history)

    out_root = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else repo_root / "tmp" / "dsocr-torchlens-callgraph"
    )

    # Map TorchLens module names to PyTorch class names using the core model's
    # named_modules registry.
    module_classes: Dict[str, str] = {}
    for name, mod in core.named_modules():
        if not name:
            continue
        module_classes[name] = mod.__class__.__name__

    write_outputs(
        out_root=out_root,
        module_call_counts=module_call_counts,
        edge_call_counts=edge_call_counts,
        op_call_counts=op_call_counts,
        module_children=module_children,
        module_classes=module_classes,
    )


if __name__ == "__main__":
    main()
