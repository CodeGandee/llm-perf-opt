#!/usr/bin/env python
"""Manual analytic performance report for DeepSeek-OCR.

This script runs:

    python -m llm_perf_opt.runners.dsocr_analyzer --mode analytic

against a local DeepSeek-OCR checkpoint and verifies that JSON/YAML
analytic artifacts and Markdown layer docs are written under:

    tmp/profile-output/<run_id>/static_analysis/analytic_model/
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from llm_perf_opt.utils.paths import analytic_model_dir, workspace_root


def main() -> int:
    root = Path(workspace_root())
    run_id = datetime.now().strftime("manual-analytic-%Y%m%d-%H%M%S")

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
    print("Running analytic CLI:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: analytic CLI exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    analytic_dir = Path(analytic_model_dir(run_id))
    layers_dir = analytic_dir / "layers"
    json_path = analytic_dir / "report.json"
    yaml_path = analytic_dir / "report.yaml"

    if not analytic_dir.is_dir():
        print(f"ERROR: analytic output directory not found: {analytic_dir}", file=sys.stderr)
        return 1

    missing = [p for p in (json_path, yaml_path, layers_dir) if not p.exists()]
    if missing:
        print("ERROR: expected analytic artifacts are missing:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print("✓ Analytic performance report artifacts found:")
    print(f"  - {json_path}")
    print(f"  - {yaml_path}")
    print(f"  - {layers_dir} (Markdown docs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

