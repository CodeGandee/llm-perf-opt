#!/usr/bin/env python
"""What-if analysis for DeepSeek-OCR analytic models.

This script loads an AnalyticModelReport JSON artifact produced by the
DeepSeek-OCR analytic runner and recomputes projected runtimes under
alternative device peak TFLOPs or batch-size scaling assumptions.

Example
-------
    python scripts/analytical/dsocr_analytic_what_if.py \\
        --report tmp/profile-output/<run_id>/static_analysis/analytic_model/report.json \\
        --peak-tflops 300.0 \\
        --batch-scale 2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_perf_opt.profiling.hw import get_device_name, get_peak_tflops


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="What-if analysis over DeepSeek-OCR AnalyticModelReport artifacts.",
    )
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to AnalyticModelReport JSON file (report.json).",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help=(
            "Target device peak TFLOPs (bf16). If omitted, the current device "
            "from llm_perf_opt.profiling.hw is used."
        ),
    )
    parser.add_argument(
        "--batch-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplicative factor to apply to per-module FLOPs/time to "
            "simulate a different effective batch size (default: 1.0)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report_path = Path(args.report).expanduser().resolve()
    if not report_path.is_file():
        raise SystemExit(f"ERROR: report JSON not found: {report_path}")

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("ERROR: report JSON must be a JSON object.")

    if args.peak_tflops is not None:
        peak_tflops = float(args.peak_tflops)
    else:
        device_name = get_device_name()
        peak_tflops = get_peak_tflops(device_name, precision="bf16")

    if peak_tflops <= 0.0:
        raise SystemExit("ERROR: peak_tflops must be positive for what-if analysis.")

    batch_scale = float(args.batch_scale)
    report_id = raw.get("report_id", "<unknown>")
    print(f"Loaded AnalyticModelReport: report_id={report_id}")
    print(f"Using peak_tflops={peak_tflops:.3f} (bf16), batch_scale={batch_scale:.3f}")
    print()
    print("Per-module projections:")
    print("{:<40s} {:>12s} {:>12s}".format("module_id", "time_ms", "flops_TF"))

    metrics = raw.get("module_metrics", [])
    if not isinstance(metrics, list):
        raise SystemExit("ERROR: AnalyticModelReport.module_metrics must be a list.")

    total_time_ms = 0.0
    total_flops_tflops = 0.0
    for snap in metrics:
        if not isinstance(snap, dict):
            continue
        module_id = str(snap.get("module_id", ""))
        flops_val = float(snap.get("total_flops_tflops", 0.0))
        io_val = float(snap.get("total_io_tb", 0.0))

        flops = flops_val * batch_scale
        time_ms = 1000.0 * flops / peak_tflops
        total_time_ms += time_ms
        total_flops_tflops += flops
        print(f"{module_id:<40s} {time_ms:12.3f} {flops:12.3f}")

    print()
    print("Aggregated projections:")
    print(f"- total_flops_tflops: {total_flops_tflops:.3f}")
    print(f"- total_time_ms:      {total_time_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
