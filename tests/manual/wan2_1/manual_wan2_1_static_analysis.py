"""Manual run: generate Wan2.1 static analysis artifacts under tmp/profile-output/.

This script is a convenience wrapper around the Wan2.1 analyzer runner for ad-hoc inspection.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from llm_perf_opt.runners.wan2_1_analyzer import Wan2_1AnalyzerConfig, Wan2_1StaticAnalyzer
from llm_perf_opt.utils.paths import wan2_1_analytic_dir, wan2_1_report_path


def _parse_args() -> argparse.Namespace:
    """Parse CLI args for the manual static analysis script."""

    parser = argparse.ArgumentParser(description="Manual Wan2.1 static analysis (writes report artifacts under tmp/profile-output/).")
    parser.add_argument("--run-id", type=str, default=None, help="Run id used for tmp/profile-output/<run_id>/.")
    parser.add_argument("--workload", type=str, default="wan2-1-512p", help="Workload profile id (e.g., wan2-1-ci-tiny, wan2-1-512p).")
    return parser.parse_args()


def main() -> int:
    """Run the analyzer and print output artifact paths."""

    args = _parse_args()
    run_id = args.run_id or datetime.now().strftime("wan2-1-manual-%Y%m%d-%H%M%S")

    analyzer = Wan2_1StaticAnalyzer()
    report = analyzer.run(
        cfg=Wan2_1AnalyzerConfig(workload_profile_id=str(args.workload), run_id=str(run_id)),
        overrides=[],
    )

    out_dir = Path(wan2_1_analytic_dir(report.report_id))
    print("Wan2.1 static analysis complete.")
    print(f"  report_id={report.report_id}")
    print(f"  artifacts_dir={out_dir}")
    print(f"  report_json={wan2_1_report_path(report.report_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
