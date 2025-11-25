"""Manual script: run Stage 1 profiling benchmark in-process (no subprocess).

This variant composes Hydra config in-process and calls the Phase 3 runner
helpers directly. It is suitable for interactive environments (e.g., Jupyter)
and regular CLI execution via Pixi.

Usage
-----
Run with Pixi (from anywhere):

    pixi run python tests/manual/manual_stage1_benchmark_inproc.py

Environment overrides (optional):

- ``STAGE1_DEVICE``: CUDA device string (default: ``cuda:0``)
- ``STAGE1_REPEATS``: Number of repeats (default: ``3``)
- ``STAGE1_SUBSET_FILELIST``: Relative (to dataset root) or absolute path to a
  newline-delimited list of images to use (default: unset → fallback globs)

Notes
-----
- Artifacts are written under ``tmp/stage1/<run_id>/`` at the repo root.
- Global-scope execution with small helper functions for readability.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from torch.profiler import ProfilerActivity, profile  # type: ignore[attr-defined]

from llm_perf_opt.profiling.export import top_n_operators  # type: ignore[import-untyped]
from llm_perf_opt.profiling.hw import get_device_name, get_peak_tflops  # type: ignore[import-untyped]
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession  # type: ignore[import-untyped]
from llm_perf_opt.runners.llm_profile_runner import (  # type: ignore[import-untyped]
    ImageRun,
    _collect_operator_records,
    _iter_images,
    _summarize_runs,
    _write_outputs,
)


def _find_repo_root(start: Path) -> Path:
    """Return the repo root (dir containing ``pyproject.toml``), else ``start``."""

    cur = start.resolve()
    for _ in range(10):
        if (cur / "pyproject.toml").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start


def _latest_run_dir(stage1_dir: Path) -> Path | None:
    """Return the latest run directory under ``stage1_dir`` by name ordering."""

    if not stage1_dir.is_dir():
        return None
    runs = [p for p in stage1_dir.iterdir() if p.is_dir()]
    return (sorted(runs, key=lambda p: p.name)[-1] if runs else None)


# Resolve repo root robustly for both scripts and notebooks
try:
    _HERE = Path(__file__).parent
except NameError:  # __file__ is undefined in some notebooks
    _HERE = Path.cwd()

_ROOT = _find_repo_root(_HERE)
os.chdir(_ROOT)


# Environment overrides
DEVICE = os.environ.get("STAGE1_DEVICE", "cuda:0")
REPEATS = os.environ.get("STAGE1_REPEATS", "3")
SUBSET = os.environ.get("STAGE1_SUBSET_FILELIST")


# Compose Hydra config in-process (re-entrant for notebooks)
GlobalHydra.instance().clear()
with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "conf")):
    overrides: list[str] = [
        f"device={DEVICE}",
        f"repeats={REPEATS}",
        f"model.path={( _ROOT / 'models/deepseek-ocr').resolve()}",
        f"dataset.root={( _ROOT / 'datasets/omnidocbench/source-data').resolve()}",
    ]
    if SUBSET:
        overrides.append(f"dataset.subset_filelist={SUBSET}")
    cfg = compose(config_name="config", overrides=overrides)


# Build session
session = DeepSeekOCRSession.from_local(
    model_path=cfg.model.path,
    device=cfg.device,
    use_flash_attn=bool(cfg.use_flash_attn),
)


# Discover images
images = list(
    _iter_images(
        cfg.dataset.root,
        list(cfg.dataset.fallback_patterns),
        cfg.dataset.get("subset_filelist"),
    )
)
if not images:
    raise RuntimeError(f"No images found in dataset root: {cfg.dataset.root}")


# Peak TFLOPs (coarse table)
peak = get_peak_tflops(get_device_name(), str(cfg.model.dtype))


# Representative operator profile
activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

operator_records: list[dict[str, Any]] = []
try:
    decoder_prompt = str(
        getattr(getattr(cfg, "infer", {}), "decoder_prompt", "<image>\n<|grounding|>Convert the document to markdown."),
    )
    with profile(activities=activities) as prof:  # type: ignore[call-arg]
        _ = session.run_inference(
            image_path=str(images[0]),
            prompt=decoder_prompt,
            max_new_tokens=int(getattr(cfg, "infer", {}).get("max_new_tokens", 64)),
        )
    operator_records = _collect_operator_records(prof)
except Exception:
    operator_records = []


# Repeats across dataset
runs: list[ImageRun] = []
max_new_tokens = int(getattr(cfg, "infer", {}).get("max_new_tokens", 64))
for i in range(int(cfg.repeats)):
    img = images[i % len(images)]
    res = session.run_inference(
        image_path=str(img),
        prompt=decoder_prompt,
        max_new_tokens=max_new_tokens,
    )
    runs.append(
        ImageRun(
            image_path=str(img),
            prefill_ms=float(res.get("prefill_ms", 0.0)),
            decode_ms=float(res.get("decode_ms", 0.0)),
            tokens=int(res.get("tokens", 0)),
        )
    )


# Summarize and write outputs
run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
artifacts_dir = _ROOT / "tmp" / "stage1" / run_id
summary = _summarize_runs(runs, getattr(session, "m_model", None), peak)
_write_outputs(artifacts_dir, summary, top_n_operators(operator_records, n=20), top_k=20)


# Display artifact pointers and a quick report preview
report = artifacts_dir / "report.md"
operators = artifacts_dir / "operators.md"
metrics = artifacts_dir / "metrics.json"

print("[manual-inproc] Artifacts:")
print("  report:", report)
print("  operators:", operators)
print("  metrics:", metrics)

if report.is_file():
    print("\n[manual-inproc] Report (head):")
    try:
        print("\n".join(report.read_text(encoding="utf-8").splitlines()[:20]))
    except Exception as exc:  # pragma: no cover - preview aid
        print("[manual-inproc] Could not preview report:", exc)
