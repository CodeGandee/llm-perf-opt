#!/usr/bin/env python
# ruff: noqa: E402
"""DeepSeek-OCR: Per-Stage Static Computational Analysis

This script runs comprehensive static analysis on the DeepSeek-OCR model using
fvcore to compute parameters, FLOPs, and activations for each architectural stage.

Usage
-----
Run with Pixi environment:

    pixi run python scripts/deepseek-ocr-static-analysis.py --device cuda:1

    pixi run python scripts/deepseek-ocr-static-analysis.py \
      --device cuda:0 \
      --output tmp/my-analysis \
      --base-size 1024 \
      --seq-len 512

Arguments
---------
--device <device>           Device spec (default: cuda:1)
--model <path>              Local model path (default: <repo_root>/models/deepseek-ocr)
--output <path>             Output directory (default: tmp/static-analysis-<timestamp>)
--base-size <int>           Base padded size for analysis (default: 1024)
--image-size <int>          Image tile size for analysis (default: 640)
--seq-len <int>             Representative sequence length (default: 1024)
--crop-mode <0/1>           Enable dynamic crops (default: 1)
--use-flash-attn <0/1>      Enable FlashAttention (default: 1)

Outputs
-------
Creates in output directory:
- static_compute.json       Machine-readable report
- static_compute.md         Human-readable markdown report with tables

Notes
-----
- Analysis uses synthetic inputs by default (no real images needed)
- FLOPs are shape-dependent; values based on specified dimensions
- Vision stages (SAM/CLIP/projector) use fvcore tracing
- LLM stages (prefill/decode) use analytical formulas
- For most accurate results, use representative dimensions matching your workload
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from llm_perf_opt.runners.dsocr_analyzer import (
    AnalysisConfig,
    DeepseekOCRStaticAnalyzer,
)
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession
from llm_perf_opt.profiling.export import (
    write_static_compute_json,
    write_static_compute_markdown,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run per-stage static analysis on DeepSeek-OCR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model and device
    parser.add_argument(
        "--model",
        type=str,
        default=str(repo_root / "models" / "deepseek-ocr"),
        help="Local model path (default: <repo_root>/models/deepseek-ocr)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device spec (default: cuda:1)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: tmp/static-analysis-<timestamp>)",
    )

    # Analysis configuration
    parser.add_argument(
        "--base-size",
        type=int,
        default=1024,
        help="Base padded size for vision analysis (default: 1024)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Image tile size for vision analysis (default: 640)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Representative sequence length for LLM analysis (default: 1024)",
    )
    parser.add_argument(
        "--crop-mode",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable dynamic crops (default: 1)",
    )
    parser.add_argument(
        "--use-flash-attn",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable FlashAttention (default: 1)",
    )

    return parser.parse_args()


def main():
    """Run static analysis and save reports."""
    args = parse_args()

    # Setup output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = repo_root / "tmp" / f"static-analysis-{timestamp}"
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DeepSeek-OCR Static Computational Analysis")
    print("=" * 80)
    print(f"\nModel path: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output dir: {output_dir}")
    print("\nAnalysis configuration:")
    print(f"  Base size: {args.base_size}")
    print(f"  Image size: {args.image_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Crop mode: {bool(args.crop_mode)}")
    print(f"  FlashAttention: {bool(args.use_flash_attn)}")

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nERROR: Model path not found: {model_path}")
        print("Please ensure the model is downloaded to the specified path")
        return 1

    # Step 1: Initialize session
    print("\n" + "-" * 80)
    print("Step 1: Initializing DeepSeekOCRSession...")
    print("-" * 80)

    try:
        session = DeepSeekOCRSession.from_local(
            model_path=str(model_path),
            device=args.device,
            use_flash_attn=bool(args.use_flash_attn),
        )
        print("✓ Session initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize session: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 2: Create analyzer
    print("\n" + "-" * 80)
    print("Step 2: Creating DeepseekOCRStaticAnalyzer...")
    print("-" * 80)

    try:
        analyzer = DeepseekOCRStaticAnalyzer(session)
        print("✓ Analyzer created successfully")
        print(f"  Stage-module mapping: {list(analyzer.m_stage_module_map.keys())}")
    except Exception as e:
        print(f"✗ Failed to create analyzer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Configure analysis
    print("\n" + "-" * 80)
    print("Step 3: Configuring analysis...")
    print("-" * 80)

    config = AnalysisConfig(
        image_h=args.base_size,
        image_w=args.base_size,
        base_size=args.base_size,
        image_size=args.image_size,
        seq_len=args.seq_len,
        crop_mode=bool(args.crop_mode),
        use_analytic_fallback=True,
        use_synthetic_inputs=True,
    )
    print("✓ Configuration created")

    # Step 4: Generate report
    print("\n" + "-" * 80)
    print("Step 4: Generating static analysis report...")
    print("-" * 80)
    print("(This may take a few minutes depending on model size and fvcore tracing)")

    try:
        report = analyzer.generate_report(config)
        print("✓ Report generated successfully")
    except Exception as e:
        print(f"✗ Failed to generate report: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 5: Display results
    print("\n" + "-" * 80)
    print("Step 5: Report Summary")
    print("-" * 80)

    total = report.get("total", {})
    print("\nTotal Model:")
    print(f"  Parameters: {float(total.get('params', 0)) / 1e6:.2f}M")
    print(f"  FLOPs: {float(total.get('flops', 0)) / 1e9:.2f}G")
    print(f"  Activations: {float(total.get('activations', 0)) / 1e6:.2f}M")

    print("\nPer-Stage Breakdown:")
    stages = report.get("stages", {})
    for stage_name in ["sam", "clip", "projector", "prefill", "decode"]:
        if stage_name not in stages:
            continue
        stage_data = stages[stage_name]
        params_m = float(stage_data.get("params", 0)) / 1e6
        flops_g = float(stage_data.get("flops", 0)) / 1e9
        flops_iso_g = (
            float(stage_data["flops_isolated"]) / 1e9
            if stage_data.get("flops_isolated") is not None
            else None
        )
        flops_ana_g = (
            float(stage_data["flops_analytic"]) / 1e9
            if stage_data.get("flops_analytic") is not None
            else None
        )

        print(f"\n  {stage_name.upper()}:")
        print(f"    Params: {params_m:.2f}M")
        print(f"    FLOPs (extracted): {flops_g:.2f}G")
        if flops_iso_g is not None:
            print(f"    FLOPs (isolated): {flops_iso_g:.2f}G")
        if flops_ana_g is not None:
            print(f"    FLOPs (analytic): {flops_ana_g:.2f}G")

        # Show top operators
        ops = stage_data.get("operators", {})
        if ops:
            top_ops = sorted(ops.items(), key=lambda x: float(x[1]), reverse=True)[:3]
            if top_ops:
                print(f"    Top operators: {', '.join(op for op, _ in top_ops)}")

    # Step 6: Write outputs
    print("\n" + "-" * 80)
    print("Step 6: Writing output files...")
    print("-" * 80)

    try:
        json_path = output_dir / "static_compute.json"
        md_path = output_dir / "static_compute.md"

        write_static_compute_json(report, json_path)
        print(f"✓ JSON written: {json_path}")

        write_static_compute_markdown(report, md_path)
        print(f"✓ Markdown written: {md_path}")

    except Exception as e:
        print(f"✗ Failed to write outputs: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Success
    print("\n" + "=" * 80)
    print("✓ Static analysis completed successfully!")
    print("=" * 80)
    print(f"\nOutputs available in: {output_dir}")
    print("  - static_compute.json")
    print("  - static_compute.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
