#!/usr/bin/env python3
"""Quick test script for DeepseekOCRStaticAnalyzer.

This script demonstrates how to use the static analyzer class with a
DeepSeekOCRSession to generate comprehensive static compute reports.

Usage:
    python test_static_analyzer.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_perf_opt.runners.dsocr_analyzer import (
    DeepseekOCRStaticAnalyzer,
    AnalysisConfig,
)
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession
from llm_perf_opt.profiling.export import (
    write_static_compute_json,
    write_static_compute_markdown,
)


def main():
    """Run static analysis test."""
    print("=" * 80)
    print("DeepseekOCRStaticAnalyzer Test")
    print("=" * 80)

    # Configuration
    model_path = Path.cwd() / "models" / "deepseek-ocr"
    output_dir = Path.cwd() / "tmp" / "static_analysis_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"ERROR: Model path not found: {model_path}")
        print("Please ensure the model is downloaded to models/deepseek-ocr")
        return 1

    print(f"\nModel path: {model_path}")
    print(f"Output dir: {output_dir}")

    # 1. Initialize session
    print("\n" + "-" * 80)
    print("Step 1: Initializing DeepSeekOCRSession...")
    print("-" * 80)

    try:
        session = DeepSeekOCRSession.from_local(
            model_path=str(model_path),
            device="cuda:0",
            use_flash_attn=True,
        )
        print("✓ Session initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize session: {e}")
        return 1

    # 2. Create analyzer
    print("\n" + "-" * 80)
    print("Step 2: Creating DeepseekOCRStaticAnalyzer...")
    print("-" * 80)

    try:
        analyzer = DeepseekOCRStaticAnalyzer(session)
        print("✓ Analyzer created successfully")
        print(f"  Stage-module mapping: {list(analyzer.m_stage_module_map.keys())}")
    except Exception as e:
        print(f"✗ Failed to create analyzer: {e}")
        return 1

    # 3. Configure analysis
    print("\n" + "-" * 80)
    print("Step 3: Configuring analysis...")
    print("-" * 80)

    config = AnalysisConfig(
        image_h=1024,
        image_w=1024,
        base_size=1024,
        image_size=640,
        seq_len=512,  # Shorter for quick test
        crop_mode=True,
        use_analytic_fallback=True,
        use_synthetic_inputs=True,
    )
    print("✓ Configuration created:")
    print(f"  Base size: {config.base_size}")
    print(f"  Image size: {config.image_size}")
    print(f"  Seq len: {config.seq_len}")
    print(f"  Crop mode: {config.crop_mode}")
    print(f"  Synthetic inputs: {config.use_synthetic_inputs}")

    # 4. Generate report
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

    # 5. Display results
    print("\n" + "-" * 80)
    print("Step 5: Report Summary")
    print("-" * 80)

    total = report.get("total", {})
    print(f"\nTotal Model:")
    print(f"  Parameters: {float(total.get('params', 0)) / 1e6:.2f}M")
    print(f"  FLOPs: {float(total.get('flops', 0)) / 1e9:.2f}G")
    print(f"  Activations: {float(total.get('activations', 0)) / 1e6:.2f}M")

    print(f"\nPer-Stage Breakdown:")
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

    # 6. Write outputs
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
    print("✓ Test completed successfully!")
    print("=" * 80)
    print(f"\nOutputs available in: {output_dir}")
    print("  - static_compute.json")
    print("  - static_compute.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
