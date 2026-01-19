"""CLI helpers for Wan2.1 analytic reports (report.json)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cattrs import Converter

from llm_perf_opt.data.analytic_common import ScalarParam
from llm_perf_opt.data.wan2_1_analytic import Wan2_1AnalyticModelReport
from llm_perf_opt.visualize.wan2_1_analytic_summary import (
    aggregate_category_flops,
    top_k_layers_by_flops,
    total_flops_tflops,
)


def _build_converter() -> Converter:
    conv = Converter()
    conv.register_structure_hook(ScalarParam, lambda v, _: v)
    return conv


def load_report(path: str) -> Wan2_1AnalyticModelReport:
    """Load a Wan2.1 `report.json` into a typed domain object."""

    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    return _build_converter().structure(payload, Wan2_1AnalyticModelReport)


def _print_top_layers(report: Wan2_1AnalyticModelReport, *, k: int) -> None:
    total = total_flops_tflops(report)
    rows = top_k_layers_by_flops(report, k=k, leaf_only=True)
    print(f"Top {k} layers by FLOPs (leaf modules):")
    for idx, (module, snap) in enumerate(rows, start=1):
        share = float(snap.total_flops_tflops) / total if total > 0.0 else 0.0
        print(f"{idx:2d}. {module.module_id}  flops={float(snap.total_flops_tflops):.6f} TFLOPs  share={share:.4f}")


def _print_top_categories(report: Wan2_1AnalyticModelReport, *, k: int) -> None:
    total = total_flops_tflops(report)
    totals = aggregate_category_flops(report, leaf_only=True)
    rows = sorted(totals.items(), key=lambda kv: (-float(kv[1]), kv[0]))[:k]
    print(f"Top {k} categories by FLOPs (leaf modules):")
    for idx, (cat_id, flops) in enumerate(rows, start=1):
        share = float(flops) / total if total > 0.0 else 0.0
        print(f"{idx:2d}. {cat_id}  flops={float(flops):.6f} TFLOPs  share={share:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Wan2.1 analytic report tools.")
    parser.add_argument("--report", type=str, required=True, help="Path to report.json")
    parser.add_argument("--compare", type=str, default=None, help="Optional second report.json to diff against")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for layer/category summaries")
    args = parser.parse_args()

    try:
        report_a = load_report(args.report)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to load report: {exc}", file=sys.stderr)
        return 1

    print(f"Report: {args.report}")
    print(f"  report_id={report_a.report_id}  workload={report_a.workload.profile_id}  total={total_flops_tflops(report_a):.6f} TFLOPs")
    _print_top_layers(report_a, k=int(args.top_k))
    print("")
    _print_top_categories(report_a, k=min(int(args.top_k), 10))

    if args.compare:
        try:
            report_b = load_report(args.compare)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: failed to load compare report: {exc}", file=sys.stderr)
            return 1
        total_a = total_flops_tflops(report_a)
        total_b = total_flops_tflops(report_b)
        ratio = (total_b / total_a) if total_a > 0.0 else float("inf")
        print("")
        print("Comparison:")
        print(f"  A: {args.report}  total={total_a:.6f} TFLOPs")
        print(f"  B: {args.compare}  total={total_b:.6f} TFLOPs")
        print(f"  ratio B/A={ratio:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
