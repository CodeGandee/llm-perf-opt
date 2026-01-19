"""Contract-oriented CLI wrapper for Wan2.1 analytic modeling.

This entry point accepts arguments mirroring the `Wan2_1AnalyticRequest`
contract and dispatches an analytic run via
`llm_perf_opt.runners.wan2_1_analyzer`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from llm_perf_opt.contracts.models import Wan2_1AnalyticRequest
from llm_perf_opt.runners.wan2_1_analyzer import Wan2_1AnalyzerConfig, Wan2_1StaticAnalyzer
from llm_perf_opt.utils.paths import wan2_1_analytic_dir


def _parse_overrides(values: list[str] | None) -> dict[str, int] | None:
    if not values:
        return None
    out: dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = int(v)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan2.1 analytic modeling (contract-oriented wrapper).")
    parser.add_argument("--model-id", type=str, default="wan2.1-t2v-14b")
    parser.add_argument("--model-variant", type=str, default="t2v-14b")
    parser.add_argument("--workload-profile-id", type=str, default="wan2-1-512p")
    parser.add_argument("--profile-run-id", type=str, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--run-id", type=str, default=None, help="Optional output run id (defaults to a timestamp).")
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="Workload override in key=value form (e.g., height=720). May be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    overrides_dict = _parse_overrides(args.override)

    _ = Wan2_1AnalyticRequest(
        model_id=args.model_id,
        model_variant=args.model_variant,
        workload_profile_id=args.workload_profile_id,
        overrides=overrides_dict,
        profile_run_id=args.profile_run_id,
        force_rebuild=bool(args.force_rebuild),
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    overrides: list[str] = []
    if overrides_dict:
        for k, v in overrides_dict.items():
            overrides.append(f"workload.overrides.{k}={int(v)}")

    analyzer = Wan2_1StaticAnalyzer()
    try:
        report = analyzer.run(
            cfg=Wan2_1AnalyzerConfig(workload_profile_id=args.workload_profile_id, run_id=run_id),
            overrides=overrides,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Wan2.1 analytic modeling failed: {exc}", file=sys.stderr)
        return 1

    out_dir = wan2_1_analytic_dir(report.report_id)
    print("Wan2.1 analytic modeling accepted:")
    print(f"  report_id={report.report_id}")
    print(f"  artifacts_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
