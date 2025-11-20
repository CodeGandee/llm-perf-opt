#!/usr/bin/env python
"""Manual analytic-model export for DeepSeek-OCR.

This script runs the DeepSeek-OCR analytic CLI and verifies that a standalone
``AnalyticModelReport`` JSON/YAML pair is written under::

    tmp/profile-output/<run_id>/static_analysis/analytic_model/

The intent is to mimic external capacity-planning tools that consume the
machine-readable analytic model without relying on runtime traces.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from llm_perf_opt.utils.paths import analytic_model_dir, workspace_root


def main() -> int:
    root = Path(workspace_root())
    run_id = datetime.now().strftime("manual-analytic-model-%Y%m%d-%H%M%S")

    model_dir = root / "models" / "deepseek-ocr"
    if not model_dir.exists():
        print(f"ERROR: model path not found: {model_dir}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "llm_perf_opt.runners.dsocr_analyzer",
        "--mode",
        "analytic",
        "--model",
        str(model_dir),
        "--device",
        "cuda:0",
        "--workload-profile-id",
        "dsocr-standard-v1",
        "--run-id",
        run_id,
    ]
    print("Running analytic CLI for model export:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: analytic CLI exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    analytic_dir = Path(analytic_model_dir(run_id))
    json_path = analytic_dir / "report.json"
    yaml_path = analytic_dir / "report.yaml"

    if not analytic_dir.is_dir():
        print(f"ERROR: analytic output directory not found: {analytic_dir}", file=sys.stderr)
        return 1

    missing = [p for p in (json_path, yaml_path) if not p.exists()]
    if missing:
        print("ERROR: expected analytic model artifacts are missing:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        return 1

    # Basic structural validation: ensure the JSON payload looks like an AnalyticModelReport.
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    required_keys = {"report_id", "model", "workload", "modules", "module_metrics"}
    missing_keys = sorted(required_keys - set(payload.keys()))
    if missing_keys:
        print(
            f"ERROR: analytic model JSON is missing required keys: {', '.join(missing_keys)}",
            file=sys.stderr,
        )
        return 1

    print("✓ AnalyticModelReport export verified:")
    print(f"  - {json_path}")
    print(f"  - {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

