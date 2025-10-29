"""Manual script: run Stage 1 profiling benchmark (US1).

This script invokes the Hydra-based Stage 1 runner to produce a profiling
report with NVTX segmentation, PyTorch operator summary, aggregates, and MFU.

Usage
-----
Run with Pixi (recommended):

    pixi run python tests/manual/manual_stage1_benchmark.py

Configuration via environment variables (optional):

- ``STAGE1_DEVICE``: CUDA device string (default: ``cuda:0``)
- ``STAGE1_REPEATS``: Number of repeated runs (default: ``3``)
- ``STAGE1_SUBSET_FILELIST``: Relative (to dataset root) or absolute path to a
  newline-delimited list of images to use (default: unset → fallback globs)

Notes
-----
- The runner writes artifacts under ``tmp/stage1/<run_id>/`` in the repository
  root (report.md, operators.md, metrics.json).
- This script follows the manual-script guide: global-scope execution without a
  ``__main__`` guard and minimal use of helper functions to ease notebook use.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional


def _find_repo_root(start: Path) -> Path:
    """Return the repository root (directory containing ``pyproject.toml``).

    Falls back to ``start`` if no parent contains the file.
    """

    cur = start.resolve()
    for _ in range(10):  # bounded ascent
        if (cur / "pyproject.toml").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start


def _latest_run_dir(stage1_dir: Path) -> Optional[Path]:
    """Return the most recent run directory under ``stage1_dir`` (by name).

    Returns None if no run directories exist.
    """

    if not stage1_dir.is_dir():
        return None
    runs = [p for p in stage1_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    # run_id format is YYYYmmdd-HHMMSS → lexicographic max is latest
    return sorted(runs, key=lambda p: p.name)[-1]


# Resolve paths and config
_SCRIPT_DIR = Path(__file__).parent
_ROOT = _find_repo_root(_SCRIPT_DIR)

DEVICE = os.environ.get("STAGE1_DEVICE", "cuda:0")
REPEATS = os.environ.get("STAGE1_REPEATS", "3")
SUBSET = os.environ.get("STAGE1_SUBSET_FILELIST")


# Compose Hydra command
cmd = [
    sys.executable,
    "-m",
    "llm_perf_opt.runners.llm_profile_runner",
    "dataset=omnidocbench",
    "model/deepseek_ocr/arch@model=deepseek_ocr.default",
    "model/deepseek_ocr/infer@infer=deepseek_ocr.default",
    f"repeats={REPEATS}",
    f"device={DEVICE}",
]
if SUBSET:
    cmd.append(f"dataset.subset_filelist={SUBSET}")

print("[manual] Running Stage 1 benchmark with command:\n  ", " ".join(cmd))


# Execute from repo root so dataset/model relative paths resolve as expected
subprocess.run(cmd, check=True, cwd=_ROOT)


# Locate latest artifacts
stage1_dir = _ROOT / "tmp" / "stage1"
latest = _latest_run_dir(stage1_dir)
if latest is None:
    print("[manual] No run directory was created under tmp/stage1")
else:
    report = latest / "report.md"
    operators = latest / "operators.md"
    metrics = latest / "metrics.json"
    print("[manual] Artifacts:")
    print("  report:", report)
    print("  operators:", operators)
    print("  metrics:", metrics)
    if report.is_file():
        print("\n[manual] Report (head):")
        try:
            print("\n".join(report.read_text(encoding="utf-8").splitlines()[:20]))
        except Exception as exc:  # pragma: no cover - display aid
            print("[manual] Could not preview report:", exc)
