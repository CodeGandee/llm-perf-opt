#!/usr/bin/env python
# ruff: noqa: E402
"""DeepSeek-OCR: TorchInfo Static Component Analysis

This script uses ``torchinfo`` to run a static structural analysis of the
DeepSeek-OCR model and derive per-stage parameter / MAC (mult-add) counts.

It complements the fvcore-based analysis in ``scripts/deepseek-ocr-static-analysis.py``
by reusing the same session + input preparation but leveraging TorchInfo's
layer-wise summary.

Usage
-----
Run with the RTX 5090 Pixi environment:

    pixi run -e rtx5090 python scripts/analytical/dsocr_find_static_components.py \\
        --device cuda:0

Optional arguments allow you to control image/sequence dimensions and output:

    pixi run -e rtx5090 python scripts/analytical/dsocr_find_static_components.py \\
        --device cuda:0 \\
        --base-size 1024 \\
        --image-size 640 \\
        --seq-len 1024 \\
        --depth 4

Outputs
-------
- Prints the full TorchInfo summary table to stdout.
- Prints an aggregated per-stage breakdown (SAM, CLIP, projector, prefill, decode)
  derived from TorchInfo's leaf layers.
- Optionally writes the same information to disk via ``--output``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
from torchinfo import ModelStatistics, summary


def find_repo_root(start: Path) -> Path:
    """Locate the repository root by walking up until ``pyproject.toml``."""

    for current in (start, *start.parents):
        if (current / "pyproject.toml").exists():
            return current
    return start


here = Path(__file__).resolve()
repo_root = find_repo_root(here)
sys.path.insert(0, str(repo_root / "src"))

from llm_perf_opt.runners.dsocr_analyzer import AnalysisConfig, DeepseekOCRStaticAnalyzer  # noqa: E402
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession  # noqa: E402
from llm_perf_opt.utils.torchinfo_export import TorchinfoJSONExporter  # noqa: E402


@dataclass
class StageTorchinfoStats:
    """Aggregated TorchInfo statistics for a single logical stage."""

    stage_name: str
    params: int = 0
    macs: int = 0

    def to_readable(self) -> str:
        params_m = self.params / 1e6
        macs_g = self.macs / 1e9
        return f"{self.stage_name.upper():9s} | params={params_m:8.2f}M  macs={macs_g:8.2f}G"


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="TorchInfo-based static component analysis for DeepSeek-OCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model and device
    parser.add_argument(
        "--model",
        type=str,
        default=str(repo_root / "models" / "deepseek-ocr"),
        help="Local model path (default: <repo_root>/models/deepseek-ocr).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device specifier for analysis (default: cuda:0).",
    )
    parser.add_argument(
        "--use-flash-attn",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable FlashAttention in the session (default: 1).",
    )

    # Input / analysis configuration (kept consistent with dsocr_analyzer)
    parser.add_argument(
        "--base-size",
        type=int,
        default=1024,
        help="Global padded image size used to build representative inputs (default: 1024).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Local crop size used to build representative inputs (default: 640).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Representative sequence length for textual tokens (default: 1024).",
    )
    parser.add_argument(
        "--crop-mode",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable dynamic crops when preparing synthetic inputs (default: 1).",
    )

    # TorchInfo controls
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Maximum module depth for TorchInfo summary (default: 3).",
    )
    parser.add_argument(
        "--col-width",
        type=int,
        default=25,
        help="Column width for TorchInfo summary table (default: 25).",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output directory. If set, writes torchinfo-summary.txt "
            "and torchinfo-stages.json (default: no files written)."
        ),
    )

    return parser


def _aggregate_by_stage(
    stats: ModelStatistics,
    named_modules: Mapping[int, str],
    stage_module_map: Mapping[str, Iterable[str]],
) -> Dict[str, StageTorchinfoStats]:
    """Aggregate TorchInfo layer statistics into logical stages.

    Parameters
    ----------
    stats:
        TorchInfo ModelStatistics object returned by ``summary``.
    named_modules:
        Mapping from ``id(module)`` to its fully-qualified name as seen in
        ``model.named_modules()``.
    stage_module_map:
        Mapping from stage_name -> list of module name prefixes, taken from
        ``DeepseekOCRStaticAnalyzer.m_stage_module_map``.
    """

    out: Dict[str, StageTorchinfoStats] = {
        stage: StageTorchinfoStats(stage_name=stage) for stage in stage_module_map.keys()
    }

    for layer in stats.summary_list:
        # Only attribute stats to leaf layers to avoid double-counting.
        if not getattr(layer, "is_leaf_layer", False):
            continue

        module = getattr(layer, "module", None)
        if module is None:
            continue

        mod_name = named_modules.get(id(module))
        if mod_name is None:
            continue

        for stage_name, prefixes in stage_module_map.items():
            for prefix in prefixes:
                if mod_name == prefix or mod_name.startswith(f"{prefix}."):
                    stage_stats = out[stage_name]
                    stage_stats.params += int(getattr(layer, "num_params", 0) or 0)
                    stage_stats.macs += int(getattr(layer, "macs", 0) or 0)
                    break

    return out


def main() -> int:
    """Run TorchInfo static analysis and optionally write outputs."""

    parser = build_argparser()
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"ERROR: Model path not found: {model_path}")
        print("Please ensure the DeepSeek-OCR model is available locally.")
        return 1

    print("=" * 80)
    print("DeepSeek-OCR TorchInfo Static Component Analysis")
    print("=" * 80)
    print(f"\nModel path: {model_path}")
    print(f"Device: {args.device}")
    print("\nInput configuration:")
    print(f"  Base size:   {int(args.base_size)}")
    print(f"  Image size:  {int(args.image_size)}")
    print(f"  Seq length:  {int(args.seq_len)}")
    print(f"  Crop mode:   {bool(args.crop_mode)}")
    print("\nTorchInfo configuration:")
    print(f"  Depth:       {int(args.depth)}")
    print(f"  Col width:   {int(args.col_width)}")
    print(f"  FlashAttn:   {bool(args.use_flash_attn)}")

    # Optional output directory
    output_dir: Path | None
    if args.output is not None:
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    # Step 1: Initialize session
    print("\n" + "-" * 80)
    print("Step 1: Initializing DeepSeekOCRSession...")
    print("-" * 80)
    try:
        session = DeepSeekOCRSession.from_local(
            model_path=str(model_path),
            device=str(args.device),
            use_flash_attn=bool(args.use_flash_attn),
        )
        if session.m_model is None:
            raise RuntimeError("DeepSeekOCRSession did not initialize model")
        print("✓ Session initialized successfully")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"✗ Failed to initialize session: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    # Step 2: Prepare representative inputs using the existing analyzer.
    print("\n" + "-" * 80)
    print("Step 2: Preparing representative inputs...")
    print("-" * 80)
    try:
        analyzer = DeepseekOCRStaticAnalyzer(session)
        cfg = AnalysisConfig(
            image_h=int(args.base_size),
            image_w=int(args.base_size),
            base_size=int(args.base_size),
            image_size=int(args.image_size),
            seq_len=int(args.seq_len),
            crop_mode=bool(args.crop_mode),
            use_analytic_fallback=True,
            use_synthetic_inputs=True,
        )
        input_ids, model_kwargs = analyzer.prepare_inputs(cfg)
        print(
            f"✓ Inputs prepared | input_len={input_ids.shape[-1]} "
            f"images_shape={tuple(model_kwargs['images'][0][0].shape)}",
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"✗ Failed to prepare inputs: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    # Step 3: Run TorchInfo summary on the vendor model core.
    print("\n" + "-" * 80)
    print("Step 3: Running TorchInfo summary (this may take a while)...")
    print("-" * 80)

    model = session.m_model
    core = getattr(model, "model", model)

    # Build input_data matching the vendor forward signature.
    input_data: Dict[str, Any] = {
        "input_ids": input_ids,
        "images": model_kwargs["images"],
        "images_seq_mask": model_kwargs["images_seq_mask"],
        "images_spatial_crop": model_kwargs["images_spatial_crop"],
    }

    try:
        stats = summary(
            core,
            input_data=input_data,
            depth=int(args.depth),
            col_width=int(args.col_width),
            col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
            row_settings=("depth", "var_names"),
            device=session.m_device or torch.device(args.device),
            verbose=0,
        )
        print("✓ TorchInfo summary completed\n")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"✗ TorchInfo summary failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    # Print the full torchinfo table.
    print(stats)

    # Step 4: Aggregate by logical stage using analyzer's mapping.
    print("\n" + "-" * 80)
    print("Step 4: Aggregating TorchInfo stats by stage...")
    print("-" * 80)

    # Map module objects to their fully-qualified names for aggregation.
    named_modules: Dict[int, str] = {id(mod): name for name, mod in core.named_modules()}

    stage_stats = _aggregate_by_stage(
        stats=stats,
        named_modules=named_modules,
        stage_module_map=analyzer.m_stage_module_map,
    )

    # Build detailed per-layer representations (flat + hierarchical).
    exporter = TorchinfoJSONExporter.from_model(stats=stats, model=core)
    layers_flat = exporter.layers_flat()
    hierarchy = exporter.hierarchy()

    total_params = int(getattr(stats, "total_params", 0))
    total_macs = int(getattr(stats, "total_mult_adds", 0) or 0)

    print("\nPer-stage aggregated stats (leaf layers only):")
    for stage_name in ["sam", "clip", "projector", "prefill", "decode"]:
        if stage_name not in stage_stats:
            continue
        s = stage_stats[stage_name]
        print("  " + s.to_readable())

    print("\nOverall totals (from TorchInfo):")
    print(f"  Total params: {total_params / 1e6:8.2f}M")
    print(f"  Total MACs:   {total_macs / 1e9:8.2f}G")

    # Step 5: Optional output to disk.
    if output_dir is not None:
        print("\n" + "-" * 80)
        print(f"Step 5: Writing outputs to {output_dir} ...")
        print("-" * 80)

        try:
            txt_path = output_dir / "torchinfo-summary.txt"
            stages_json_path = output_dir / "torchinfo-stages.json"
            layers_json_path = output_dir / "torchinfo-layers.json"

            txt_path.write_text(str(stats), encoding="utf-8")

            stages_payload = {
                "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "model_path": str(model_path),
                "device": str(args.device),
                "base_size": int(args.base_size),
                "image_size": int(args.image_size),
                "seq_len": int(args.seq_len),
                "crop_mode": bool(args.crop_mode),
                "depth": int(args.depth),
                "total_params": total_params,
                "total_macs": total_macs,
                "stages": {name: asdict(stats_obj) for name, stats_obj in stage_stats.items()},
            }
            stages_json_path.write_text(json.dumps(stages_payload, indent=2), encoding="utf-8")

            layers_payload = {
                "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "model_path": str(model_path),
                "device": str(args.device),
                "base_size": int(args.base_size),
                "image_size": int(args.image_size),
                "seq_len": int(args.seq_len),
                "crop_mode": bool(args.crop_mode),
                "depth": int(args.depth),
                "total_params": total_params,
                "total_macs": total_macs,
                "layers_flat": layers_flat,
                "hierarchy": hierarchy,
            }
            layers_json_path.write_text(json.dumps(layers_payload, indent=2), encoding="utf-8")

            print(f"✓ Wrote TorchInfo summary text: {txt_path}")
            print(f"✓ Wrote per-stage JSON summary: {stages_json_path}")
            print(f"✓ Wrote per-layer JSON summary: {layers_json_path}")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"✗ Failed to write outputs: {exc}")
            import traceback

            traceback.print_exc()
            return 1

    print("\n" + "=" * 80)
    print("✓ TorchInfo static component analysis completed.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
