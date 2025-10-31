"""Stage 1 profiling runner (Hydra entry).

Implements Phase 3 (US1) of the Stage 1 profiling workflow:

- Uses Hydra to configure dataset/model/runtime.
- Executes DeepSeek‑OCR via :class:`~llm_perf_opt.runners.dsocr_session.DeepSeekOCRSession`.
- Wraps one representative run in PyTorch Profiler (CPU+CUDA) to collect
  operator‑level statistics.
- Aggregates repeated runs and estimates MFU (model‑level and per‑stage).
- Writes artifacts (report.md, metrics.json, operators.md) under tmp/stage1/<run_id>/.

This module is intentionally lightweight and delegates reusable helpers to the
``profiling`` package. It aims to be clear and single‑purpose for Stage 1.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.profiler import ProfilerActivity, profile  # type: ignore[attr-defined]

from hydra.core.hydra_config import HydraConfig
from llm_perf_opt.profiling.aggregate import mean_std
from llm_perf_opt.profiling.export import (
    top_n_operators,
    write_operator_markdown,
    write_stakeholder_summary,
    OperatorRecord,
)
from llm_perf_opt.profiling.hw import get_device_name, get_peak_tflops, write_env_json
from llm_perf_opt.profiling.mfu import compute_stage_mfu
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession
from llm_perf_opt.runners.dsocr_analyzer import DeepseekOCRStaticAnalyzer, AnalysisConfig
from PIL import Image  # type: ignore[import-untyped]
from llm_perf_opt.visualize.annotations import render_vendor_style, write_vendor_result_mmd


# -----------------------------
# Filesystem and input helpers
# -----------------------------

def _read_filelist(root: str, filelist: str) -> list[Path]:
    """Read a newline‑delimited file list; resolve relative entries to ``root``.

    Parameters
    ----------
    root : str
        Dataset root directory.
    filelist : str
        Path to a newline‑separated list of file paths (absolute or relative to ``root``).

    Returns
    -------
    list[Path]
        Absolute paths to files that exist. Non‑existent entries are filtered out.
    """

    fp = Path(filelist)
    if not fp.is_absolute():
        # Prefer resolving relative to the original working directory (Hydra runtime.cwd)
        # so users can pass repo-relative paths even when Hydra chdir is enabled.
        try:
            from hydra.core.hydra_config import HydraConfig as _HC  # local import to avoid cycles
            runtime_cwd = Path(_HC.get().runtime.cwd)
            cand = runtime_cwd / filelist
            fp = cand if cand.exists() else (Path(root) / filelist)
        except Exception:
            fp = Path(root) / filelist
    lines = [ln.strip() for ln in fp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: list[Path] = []
    for ln in lines:
        p = Path(ln)
        p = p if p.is_absolute() else (Path(root) / ln)
        try:
            rp = p.resolve()
            if rp.exists():
                out.append(rp)
        except Exception:
            continue
    return out


def _iter_images(root: str, fallback_patterns: list[str], subset_filelist: str | None) -> Iterable[Path]:
    """Yield image paths from a subset filelist or fallback glob patterns.

    If ``subset_filelist`` is provided, it is read relative to ``root`` (unless
    absolute). Otherwise, glob patterns relative to ``root`` are used.
    """

    if subset_filelist:
        return _read_filelist(root, subset_filelist)
    rp = Path(root)
    out: list[Path] = []
    for pat in fallback_patterns:
        out.extend(sorted(rp.glob(pat)))
    # De‑dup while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return [pp.resolve() for pp in uniq]


# -----------------------------
# Profiling and aggregation
# -----------------------------

@dataclass
class ImageRun:
    image_path: str
    prefill_ms: float
    decode_ms: float
    tokens: int
    prefill_len: int
    vision_ms: float
    sam_ms: float = 0.0
    clip_ms: float = 0.0
    projector_ms: float = 0.0


def _collect_operator_records(prof: Any) -> list[OperatorRecord]:
    """Extract operator‑level summaries from a PyTorch profiler object.

    Returns a list of dicts with keys: ``op_name``, ``total_time_ms``,
    ``cuda_time_ms``, ``calls``.
    """

    records: list[OperatorRecord] = []
    try:
        for evt in prof.key_averages(group_by_input_shape=False):  # type: ignore[attr-defined]
            # Some events may lack CUDA time; use 0.0
            total_ms = (
                float(
                    getattr(evt, "self_cpu_time_total", 0.0)
                    + getattr(evt, "cpu_time_total", 0.0)
                )
                / 1000.0
            )
            cuda_ms = (
                float(
                    getattr(evt, "self_cuda_time_total", 0.0)
                    + getattr(evt, "cuda_time_total", 0.0)
                )
                / 1000.0
            )
            calls = int(getattr(evt, "count", 0))
            name = str(getattr(evt, "key", getattr(evt, "name", "")))
            if not name:
                continue
            mean_ms = (total_ms / max(calls, 1)) if total_ms > 0 else 0.0
            records.append(
                {
                    "op_name": name,
                    "total_time_ms": max(total_ms, 0.0),
                    "cuda_time_ms": max(cuda_ms, 0.0),
                    "calls": max(calls, 0),
                    "mean_ms": float(mean_ms),
                }
            )
    except Exception:
        # Fail‑open: return what we collected
        pass
    return records


def _infer_model_dims(model: object) -> tuple[int, int, int]:
    """Best‑effort extraction of (d_model, d_ff, n_layers) from a HF model config.

    Falls back to (1024, 4096, 24) if unavailable.
    """

    try:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            d_model = int(getattr(cfg, "hidden_size", getattr(cfg, "d_model", 1024)))
            d_ff = int(getattr(cfg, "intermediate_size", getattr(cfg, "ffn_dim", 4096)))
            n_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 24)))
            return d_model, d_ff, n_layers
    except Exception:
        pass
    return 1024, 4096, 24


def _summarize_runs(runs: list[ImageRun], model_obj: object, peak_tflops: float, ctx_len_mode: str = "auto", ctx_len_fixed: int | None = None, vision_flops: float | None = None, model_window: int | None = None) -> dict:
    """Compute aggregates and MFU estimates from per‑image runs.

    Model‑level MFU is computed from decode throughput and an analytical
    FLOPs/token estimate. Per‑stage MFU provides ``decode`` using the same
    logic and sets ``prefill`` to 0.0 as a conservative placeholder for Stage 1.
    """

    if not runs:
        raise ValueError("No runs to summarize")

    prefill_vals = [r.prefill_ms for r in runs]
    decode_vals = [r.decode_ms for r in runs]
    tokens_vals = [float(max(r.tokens, 1)) for r in runs]
    dec_tokens_per_s = [tok / (ms / 1000.0) for tok, ms in zip(tokens_vals, decode_vals)]

    prefill_mean, prefill_std = mean_std(prefill_vals)
    decode_mean, decode_std = mean_std(decode_vals)
    tokens_mean, tokens_std = mean_std(tokens_vals)
    tps_mean, tps_std = mean_std(dec_tokens_per_s)

    d_model, d_ff, n_layers = _infer_model_dims(model_obj)
    # Use averages across runs for a representative MFU calculation
    prefill_len_mean = int(mean_std([r.prefill_len for r in runs])[0])
    tokens_mean_int = int(tokens_mean)
    # Compute per-stage MFU with improved context selection
    stage = compute_stage_mfu(
        prefill_ms=prefill_mean,
        decode_ms=decode_mean,
        vision_ms=float(mean_std([r.vision_ms for r in runs])[0]),
        prefill_len=prefill_len_mean,
        new_tokens=tokens_mean_int,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        peak_tflops=peak_tflops,
        ctx_len_mode=ctx_len_mode,
        ctx_len_fixed=ctx_len_fixed,
        model_window=model_window,
        vision_flops=vision_flops,
    )

    # Stage-wise timing aggregates (include only present stages)
    def _stage_stats(vals: list[float]) -> tuple[float, float] | None:
        if any(v > 0.0 for v in vals):
            return mean_std(vals)
        return None

    stage_ms: dict[str, dict] = {}
    for key, vals in (
        ("prefill", prefill_vals),
        ("decode", decode_vals),
        ("vision", [r.vision_ms for r in runs]),
        ("sam", [getattr(r, "sam_ms", 0.0) for r in runs]),
        ("clip", [getattr(r, "clip_ms", 0.0) for r in runs]),
        ("projector", [getattr(r, "projector_ms", 0.0) for r in runs]),
    ):
        stats = _stage_stats([float(v) for v in vals])
        if stats is not None:
            m, s = stats
            stage_ms[key] = {"mean": m, "std": s}

    return {
        "aggregates": {
            "prefill_ms": {"mean": prefill_mean, "std": prefill_std},
            "decode_ms": {"mean": decode_mean, "std": decode_std},
            "tokens": {"mean": tokens_mean, "std": tokens_std},
            "tokens_per_s": {"mean": tps_mean, "std": tps_std},
            "stage_ms": stage_ms,
        },
        "mfu_model_level": float(stage.model_level),
        "mfu_per_stage": {"prefill": float(stage.prefill), "decode": float(stage.decode), "vision": float(stage.vision)},
        "model_dims": {"d_model": d_model, "d_ff": d_ff, "n_layers": n_layers},
        "peak_tflops": peak_tflops,
    }


def _write_outputs(artifacts_dir: Path, summary: dict, operator_records: list[OperatorRecord], top_k: int = 20) -> None:
    """Write report.md, operators.md, and metrics.json under ``artifacts_dir``.

    Parameters
    ----------
    artifacts_dir : Path
        Destination directory (created if missing).
    summary : dict
        Summary dictionary from ``_summarize_runs``.
    operator_records : list[dict]
        Raw operator records to export as a Markdown top‑K table.
    top_k : int, default=20
        Number of operator rows to include.
    """

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Operators
    operators_md_path = artifacts_dir / "operators.md"
    write_operator_markdown(operator_records, str(operators_md_path), top_k=top_k)

    # Report
    report_path = artifacts_dir / "report.md"
    aggr = summary.get("aggregates", {})
    mfu_model = float(summary.get("mfu_model_level", 0.0))
    mfu_stages = summary.get("mfu_per_stage", {})
    lines = [
        "# Stage 1 Profiling Report",
        "",
        f"- Timestamp: {datetime.utcnow().isoformat()}Z",
        f"- Device: {get_device_name()}",
        f"- Peak TFLOPs (est.): {float(summary.get('peak_tflops', 0.0)):.2f}",
        "",
        "## Aggregates",
        (
            "- Prefill ms: "
            f"mean={aggr.get('prefill_ms',{}).get('mean',0):.3f}, "
            f"std={aggr.get('prefill_ms',{}).get('std',0):.3f}"
        ),
        (
            "- Decode ms: "
            f"mean={aggr.get('decode_ms',{}).get('mean',0):.3f}, "
            f"std={aggr.get('decode_ms',{}).get('std',0):.3f}"
        ),
        f"- Tokens: mean={aggr.get('tokens',{}).get('mean',0):.1f}, std={aggr.get('tokens',{}).get('std',0):.1f}",
        (
            "- Tokens/s: "
            f"mean={aggr.get('tokens_per_s',{}).get('mean',0):.3f}, "
            f"std={aggr.get('tokens_per_s',{}).get('std',0):.3f}"
        ),
        "",
        "## MFU",
        f"- Model-level MFU: {mfu_model:.6f}",
        (
            "- Per-stage MFU: "
            f"vision={float(mfu_stages.get('vision',0.0)):.6f}, "
            f"prefill={float(mfu_stages.get('prefill',0.0)):.6f}, "
            f"decode={float(mfu_stages.get('decode',0.0)):.6f}"
        ),
        "",
        "## Operators",
        f"See operators table: {operators_md_path.name}",
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Metrics (JSON)
    metrics_path = artifacts_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Stakeholder summary (US2)
    aggr = summary.get("aggregates", {}) if isinstance(summary.get("aggregates"), dict) else {}
    prefill_mean = float(aggr.get("prefill_ms", {}).get("mean", 0.0))
    decode_mean = float(aggr.get("decode_ms", {}).get("mean", 0.0))
    tps_mean = float(aggr.get("tokens_per_s", {}).get("mean", 0.0))
    mfu_model = float(summary.get("mfu_model_level", 0.0))
    mfu_stages = summary.get("mfu_per_stage", {}) if isinstance(summary.get("mfu_per_stage"), dict) else {}
    mfu_decode = float(mfu_stages.get("decode", 0.0))
    mfu_prefill = float(mfu_stages.get("prefill", 0.0))
    mfu_vision = float(mfu_stages.get("vision", 0.0))

    stage_msgs: dict[str, str] = {}
    if decode_mean >= max(1.0, prefill_mean) * 1.2:
        stage_msgs["decode"] = (
            f"Decode dominates runtime (≈ {decode_mean:.1f} ms per run). "
            f"Tokens/s ≈ {tps_mean:.2f}; MFU(decode) ≈ {mfu_decode:.6f}."
        )
    elif prefill_mean >= max(1.0, decode_mean) * 1.2:
        stage_msgs["prefill"] = (
            f"Prefill dominates runtime (≈ {prefill_mean:.1f} ms per run). "
            f"MFU(prefill) ≈ {mfu_prefill:.6f}."
        )
    else:
        stage_msgs["decode"] = (
            f"Decode and prefill comparable (decode ≈ {decode_mean:.1f} ms). "
            f"Tokens/s ≈ {tps_mean:.2f}; MFU(decode) ≈ {mfu_decode:.6f}."
        )
        stage_msgs["prefill"] = f"Prefill ≈ {prefill_mean:.1f} ms; MFU(prefill) ≈ {mfu_prefill:.6f}."
    if mfu_vision > 0.0:
        stage_msgs["vision"] = f"Vision compute contributes to MFU ≈ {mfu_vision:.6f}."
    stage_msgs["model"] = f"Model-level MFU ≈ {mfu_model:.6f}."

    stakeholder_path = artifacts_dir / "stakeholder_summary.md"
    try:
        stats_payload = {
            "aggregates": aggr,
            "mfu_model": mfu_model,
            "mfu_stages": mfu_stages,
            "peak_tflops": float(summary.get("peak_tflops", 0.0)),
            "device": get_device_name(),
        }
        write_stakeholder_summary(
            str(stakeholder_path),
            top_ops=top_n_operators(operator_records, n=top_k),
            stage_takeaways=stage_msgs,
            stats=stats_payload,
        )
    except Exception:
        # Fail-open: if writing summary fails, continue without blocking US1 artifacts
        pass


def _write_static_compute(artifacts_dir: Path, static_compute: dict) -> None:
    """Write static compute report to JSON and Markdown using mdutils."""

    try:
        # JSON
        sp = artifacts_dir / "static_compute.json"
        with sp.open("w", encoding="utf-8") as jf:
            json.dump(static_compute, jf, indent=2)

        # Markdown
        mp = artifacts_dir / "static_compute.md"
        file_base = str(mp)[:-3]
        md = MdUtils(file_name=file_base)
        md.new_header(level=1, title="Static Compute Report (best-effort)")
        params = static_compute.get("params", {}) if isinstance(static_compute.get("params"), dict) else {}
        md.new_list(
            items=[
                f"Params total: {int(params.get('total', 0))}",
                f"Params trainable: {int(params.get('trainable', 0))}",
            ]
        )
        decode = static_compute.get("decode", {}) if isinstance(static_compute.get("decode"), dict) else {}
        vision = static_compute.get("vision", {}) if isinstance(static_compute.get("vision"), dict) else {}
        prefill = static_compute.get("prefill", {}) if isinstance(static_compute.get("prefill"), dict) else {}
        md.new_header(level=2, title="Vision")
        md.new_list(items=[f"FLOPs (approx): {float(vision.get('flops', 0.0)):.3e}"])
        md.new_header(level=2, title="Prefill")
        md.new_list(
            items=[
                f"FLOPs per seq (approx): {float(prefill.get('flops_per_seq', 0.0)):.3e}",
                f"FLOPs/token avg (approx): {float(prefill.get('flops_per_token_avg', 0.0)):.3e}",
            ]
        )
        md.new_header(level=2, title="Decode")
        if "flops_per_token_proj_only" in decode:
            md.new_list(items=[f"FLOPs/token (proj only, fvcore): {float(decode.get('flops_per_token_proj_only', 0.0)):.3e}"])
        md.new_paragraph(str(static_compute.get("notes", "")))
        md.create_md_file()
    except Exception:
        pass


def _clean_prediction_text(text: str, strip_special: bool) -> str:
    """Return a human-friendlier variant of the model output text.

    If ``strip_special`` is true, remove common end tokens and trim whitespace,
    while preserving structural tokens like table tags.
    """

    if not isinstance(text, str):
        return ""
    s = text
    # Known EOS marker from model impl
    s = s.replace("<｜end▁of▁sentence｜>", "").strip()
    if strip_special:
        # Minimal cleanup: remove stray nulls and excessive whitespace
        s = s.replace("\x00", "").strip()
    return s


def _write_predictions_outputs(
    pred_dir: Path,
    vis_dir: Path,
    preds: list[dict],
    make_gallery: bool,
    max_images: int | None,
    thumb_width: int,
) -> None:
    """Write predictions.jsonl and per-image visualization artifacts.

    - predictions.jsonl → `pred_dir/predictions.jsonl`
    - Per-image: `vis_dir/<md5(full-image-path)>/result_with_boxes.jpg` and `info.json`
      (no `predictions.md` gallery).
    """

    # JSONL
    pred_dir.mkdir(parents=True, exist_ok=True)
    pj = pred_dir / "predictions.jsonl"
    with pj.open("w", encoding="utf-8") as f:
        for rec in preds:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    if not make_gallery:
        return

    import hashlib as _hashlib
    vis_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for rec in preds:
        if max_images is not None and count >= int(max_images):
            break
        img_path = Path(str(rec.get("image", "")))
        text_raw = str(rec.get("text_raw", ""))
        text_clean = str(rec.get("text_clean", ""))
        pre_ms = float(rec.get("prefill_ms", 0.0))
        dec_ms = float(rec.get("decode_ms", 0.0))
        vis_ms = float(rec.get("vision_ms", 0.0))
        try:
            h = _hashlib.md5(str(img_path.resolve()).encode("utf-8")).hexdigest()
        except Exception:
            h = _hashlib.md5(str(img_path).encode("utf-8")).hexdigest()
        per_image_dir = vis_dir / h
        per_image_dir.mkdir(parents=True, exist_ok=True)

        result_img_name = None
        result_mmd_name = None
        if img_path.is_file():
            try:
                out_annotated, boxes = render_vendor_style(str(img_path), text_raw, str(per_image_dir))
                result_img_name = Path(out_annotated).name
            except Exception:
                result_img_name = None
                boxes = []
            # Write result.mmd referencing crops/
            try:
                out_mmd = write_vendor_result_mmd(text_raw, str(per_image_dir))
                result_mmd_name = Path(out_mmd).name
            except Exception:
                result_mmd_name = None

        info = {
            "source_image": str(img_path.resolve()) if img_path.is_absolute() else str(img_path),
            "result_image": result_img_name,
            "text_raw": text_raw,
            "result_mmd": result_mmd_name,
            "timings_ms": {"prefill": pre_ms, "decode": dec_ms, "vision": vis_ms},
            "boxes": boxes,
        }
        try:
            (per_image_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        count += 1


def _write_inputs_yaml(artifacts_dir: Path, images: list[Path], dataset_root: str, subset_filelist: str | None) -> None:
    """Write inputs.yaml listing absolute image paths with basic metadata using OmegaConf."""

    records: list[dict] = []
    for p in images:
        try:
            st = p.stat()
            width = height = None
            try:
                with Image.open(p) as im:
                    width, height = im.size
            except Exception:
                width = height = None
            records.append(
                {
                    "path": str(p.resolve()),
                    "bytes": int(getattr(st, "st_size", 0)),
                    "width": int(width) if width is not None else None,
                    "height": int(height) if height is not None else None,
                }
            )
        except Exception:
            continue
    payload = {
        "dataset_root": str(dataset_root),
        "subset_filelist": str(subset_filelist) if subset_filelist else None,
        "count": len(records),
        "images": records,
    }
    yml = OmegaConf.to_yaml(payload)
    (artifacts_dir / "inputs.yaml").write_text(yml, encoding="utf-8")


def _write_assumptions_md(artifacts_dir: Path, cfg: DictConfig) -> None:
    """Write assumptions.md using mdutils (no raw string writes)."""

    file_base = str(artifacts_dir / "assumptions")
    md = MdUtils(file_name=file_base)
    md.new_header(level=1, title="Run Assumptions")
    md.new_list(
        items=[
            f"Device: {getattr(cfg, 'device', 'cuda:0')}",
            f"Repeats: {int(getattr(cfg, 'repeats', 1))}",
            "Batch size: 1",
            f"Model path: {getattr(getattr(cfg, 'model', {}), 'path', '')}",
        ]
    )

    md.new_header(level=2, title="Decoding Params")
    infer = getattr(cfg, "infer", {})
    md.new_list(
        items=[
            f"max_new_tokens: {int(getattr(infer, 'max_new_tokens', 64))}",
            f"temperature: {float(getattr(infer, 'temperature', 0.0))}",
            f"no_repeat_ngram_size: {int(getattr(infer, 'no_repeat_ngram_size', 0))}",
            f"do_sample: {bool(getattr(infer, 'do_sample', False))}",
            f"context_len_mode: {str(getattr(infer, 'context_len_mode', 'auto'))}",
            f"context_len_fixed: {int(getattr(infer, 'context_len_fixed', 0)) or 'null'}",
        ]
    )

    md.new_header(level=2, title="Preprocess Params")
    pre = getattr(getattr(cfg, "model", {}), "preprocess", {})
    md.new_list(
        items=[
            f"enable: {bool(getattr(pre, 'enable', True))}",
            f"base_size: {int(getattr(pre, 'base_size', 1024))}",
            f"image_size: {int(getattr(pre, 'image_size', 640))}",
            f"crop_mode: {bool(getattr(pre, 'crop_mode', False))}",
            f"patch_size: {int(getattr(pre, 'patch_size', 16))}",
            f"downsample_ratio: {int(getattr(pre, 'downsample_ratio', 4))}",
        ]
    )

    md.new_header(level=2, title="Profiling Settings (PyTorch rep)")
    prof = getattr(getattr(cfg, "pipeline", {}), "torch_profiler", {})
    acts = ",".join([str(x) for x in list(getattr(prof, "activities", ["cpu", "cuda"]))])
    md.new_list(
        items=[
            f"activities: [{acts}]",
            f"rep_max_new_tokens: {int(getattr(prof, 'rep_max_new_tokens', 64))}",
            f"warmup_rounds: {int(getattr(prof, 'warmup_rounds', 0))}",
            f"warmup_synthetic: {bool(getattr(prof, 'warmup_synthetic', True))}",
        ]
    )

    md.new_header(level=2, title="Dataset Selection")
    ds = getattr(cfg, "dataset", {})
    md.new_list(
        items=[
            f"root: {getattr(ds, 'root', '')}",
            f"subset_filelist: {getattr(ds, 'subset_filelist', 'null')}",
        ]
    )

    md.create_md_file()


# -----------------------------
# Hydra entry
# -----------------------------

@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI orchestrator
    """Hydra entry point for Stage 1 profiling.

    The runner selects a representative image for operator profiling and then
    performs repeated runs across the chosen dataset to compute aggregates and
    MFU estimates. Artifacts are saved under ``tmp/stage1/<run_id>/``.
    """

    # Enable warnings capture so they surface in Hydra log file
    logging.captureWarnings(True)
    logger = logging.getLogger(__name__)

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

    # Prepare output dirs: resolve absolute run dir and pipeline subdirs
    run_dir_cfg = Path(HydraConfig.get().run.dir)
    base_cwd = Path(HydraConfig.get().runtime.cwd)
    run_root = run_dir_cfg if run_dir_cfg.is_absolute() else (base_cwd / run_dir_cfg)
    torch_out_dir = run_root / "torch_profiler"
    static_out_dir = run_root / "static_analysis"
    # Use torch_profiler as the main artifacts_dir for this runner
    artifacts_dir = torch_out_dir
    # Set up a file logger for easier debugging of runs
    try:
        torch_out_dir.mkdir(parents=True, exist_ok=True)
        static_out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(artifacts_dir / "llm_profile_runner.log", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        fh.setFormatter(fmt)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(fh)
    except Exception:
        pass
    logger.info(
        "Stage1 profiling start | device=%s repeats=%s model=%s dataset_root=%s artifacts_dir=%s",
        cfg.device,
        int(cfg.repeats),
        cfg.model.path,
        cfg.dataset.root,
        str(artifacts_dir),
    )
    logger.info("Discovered %d images for dataset", len(images))

    # Device peak TFLOPs (coarse table)
    device_name = get_device_name()
    precision = str(getattr(cfg.model, "dtype", "bf16"))
    peak = get_peak_tflops(device_name, precision)

    # Optional warmup rounds (gate by pipeline.torch_profiler.enable)
    tp_cfg = getattr(getattr(cfg, "pipeline", {}), "torch_profiler", {})
    # Prefer `enable`; fallback to `enabled` for preset compatibility
    prof_enabled = bool(getattr(tp_cfg, "enable", getattr(tp_cfg, "enabled", True)))
    warmup_rounds = int(getattr(tp_cfg, "warmup_rounds", 0)) if prof_enabled else 0
    warmup_synth = bool(getattr(tp_cfg, "warmup_synthetic", True))
    if warmup_rounds > 0:
        logger.info("Warmup: rounds=%d synthetic=%s", warmup_rounds, warmup_synth)
        from PIL import Image  # type: ignore[import-untyped]
        tmp_img = artifacts_dir / "_warmup.png"
        if warmup_synth:
            base_size = int(getattr(cfg.model, "preprocess", {}).get("base_size", 1024))
            Image.new("RGB", (base_size, base_size), color=(127, 127, 127)).save(tmp_img)
            wimg = str(tmp_img)
        else:
            wimg = str(images[0])
        for _ in range(warmup_rounds):
            try:
                _ = session.run_inference(
                    image_path=wimg,
                    prompt="<image>\\n<|grounding|>Convert the document to markdown.",
                    max_new_tokens=8,
                    preprocess=dict(
                        enable=bool(getattr(cfg.model, "preprocess", {}).get("enable", True)),
                        base_size=int(getattr(cfg.model, "preprocess", {}).get("base_size", 1024)),
                        image_size=int(getattr(cfg.model, "preprocess", {}).get("image_size", 640)),
                        crop_mode=bool(getattr(cfg.model, "preprocess", {}).get("crop_mode", False)),
                        patch_size=int(getattr(cfg.model, "preprocess", {}).get("patch_size", 16)),
                        downsample_ratio=int(getattr(cfg.model, "preprocess", {}).get("downsample_ratio", 4)),
                    ),
                )
            except Exception:
                break

    # Representative operator profile on the first image (gate by profiling.enabled)
    operator_records: list[OperatorRecord] = []
    if prof_enabled:
        rep_image = str(images[0])
        prof_cfg = tp_cfg
        sel_acts = [str(x).lower() for x in list(getattr(prof_cfg, "activities", ["cpu", "cuda"]))]
        activities = []
        if "cpu" in sel_acts:
            activities.append(ProfilerActivity.CPU)
        if "cuda" in sel_acts and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        try:
            # Keep rep profile bounded; honor profiling config caps and switches
            rep_cap = int(getattr(prof_cfg, 'rep_max_new_tokens', 64))
            raw_mnt = getattr(cfg, "infer", {}).get("max_new_tokens", 64)
            rep_max_new = rep_cap if str(raw_mnt).lower() == "inf" else int(min(int(raw_mnt), rep_cap))
            record_shapes = bool(getattr(prof_cfg, "record_shapes", False))
            profile_memory = bool(getattr(prof_cfg, "profile_memory", False))
            with_stack = bool(getattr(prof_cfg, "with_stack", False))
            with profile(activities=activities if activities else [ProfilerActivity.CPU], record_shapes=record_shapes, profile_memory=profile_memory, with_stack=with_stack) as prof:  # type: ignore[call-arg]
                logger.info("Profiling representative image: %s", rep_image)
                _ = session.run_inference(
                    image_path=rep_image,
                    prompt="<image>\n<|grounding|>Convert the document to markdown.",
                    max_new_tokens=rep_max_new,
                    preprocess=dict(
                        enable=bool(getattr(cfg.model, "preprocess", {}).get("enable", True)),
                        base_size=int(getattr(cfg.model, "preprocess", {}).get("base_size", 1024)),
                        image_size=int(getattr(cfg.model, "preprocess", {}).get("image_size", 640)),
                        crop_mode=bool(getattr(cfg.model, "preprocess", {}).get("crop_mode", False)),
                        patch_size=int(getattr(cfg.model, "preprocess", {}).get("patch_size", 16)),
                        downsample_ratio=int(getattr(cfg.model, "preprocess", {}).get("downsample_ratio", 4)),
                    ),
                )
                # Ensure all CUDA work is complete so GPU timings are captured by the profiler
                try:
                    if torch.cuda.is_available() and any(a == ProfilerActivity.CUDA for a in activities):
                        torch.cuda.synchronize()
                except Exception:
                    pass
            operator_records = _collect_operator_records(prof)
            logger.info("Collected %d operator records", len(operator_records))
        except Exception:
            operator_records = []  # Fail‑open for environments without profiler
    else:
        logger.info("profiling.enabled=false: skipping PyTorch representative profiling and warmup")

    # Repeated runs across dataset
    runs: list[ImageRun] = []
    repeats = int(cfg.repeats)
    raw_mnt = getattr(cfg, "infer", {}).get("max_new_tokens", 64)
    max_new_tokens: int | None = None if str(raw_mnt).lower() == "inf" else int(raw_mnt)
    images_iter: Iterator[Path] = iter(images)
    # Per-stage output configuration (preferred)
    tp_stage = getattr(getattr(cfg, "pipeline", {}), "torch_profiler", {})
    tp_out = getattr(tp_stage, "output", {}) if tp_stage else {}
    pred_cfg = getattr(tp_out, "prediction", {}) if tp_out else {}
    vis_cfg = getattr(tp_out, "visualization", {}) if tp_out else {}

    save_preds = bool(getattr(pred_cfg, "enable", False))
    make_gallery = bool(getattr(vis_cfg, "enable", True))
    # Default subdirs when enabled and save_dir is omitted
    _pred_dir_raw = getattr(pred_cfg, "save_dir", None)
    _vis_dir_raw = getattr(vis_cfg, "save_dir", None)
    pred_dir_cfg = str(_pred_dir_raw) if (_pred_dir_raw not in (None, "null")) else ("pred" if save_preds else ".")
    vis_dir_cfg = str(_vis_dir_raw) if (_vis_dir_raw not in (None, "null")) else ("viz" if make_gallery else ".")

    # Model-specific extras under per-stage output
    extra = getattr(tp_out, "extra", {}) if tp_out else {}
    dso_extra = getattr(extra, "deepseek_ocr", {}) if extra else {}
    dso_pred = getattr(dso_extra, "prediction", {}) if dso_extra else {}
    dso_vis = getattr(dso_extra, "visualization", {}) if dso_extra else {}

    strip_special = bool(getattr(dso_pred, "strip_special_tokens", False))
    max_images = dso_vis.get("max_images", 16)
    max_images = None if max_images in (None, "null") else int(max_images)
    thumb_w = int(dso_vis.get("thumbnail_width", 480))

    # Back-compat fallbacks (legacy outputs.* keys)
    if not save_preds:
        try:
            save_preds = bool(getattr(getattr(cfg, "outputs", {}), "save_predictions", False))
        except Exception:
            save_preds = False
    if not strip_special:
        try:
            strip_special = bool(getattr(getattr(getattr(cfg, "outputs", {}), "predictions", {}), "strip_special_tokens", False))
        except Exception:
            strip_special = False
    if not isinstance(max_images, int):
        try:
            _vc = getattr(getattr(cfg, "outputs", {}), "visualization", {})
            _mi = _vc.get("max_images", 16)
            max_images = None if _mi in (None, "null") else int(_mi)
            thumb_w = int(_vc.get("thumbnail_width", thumb_w))
            make_gallery = bool(_vc.get("enable", make_gallery))
        except Exception:
            pass

    # Resolve prediction/visualization directories relative to the stage dir
    from pathlib import Path as _P
    pred_dir = _P(pred_dir_cfg) if _P(pred_dir_cfg).is_absolute() else (torch_out_dir / pred_dir_cfg)
    vis_dir = _P(vis_dir_cfg) if _P(vis_dir_cfg).is_absolute() else (torch_out_dir / vis_dir_cfg)
    preds_for_outputs: list[dict] = []
    if save_preds:
        logger.info("Saving predictions to %s", str(artifacts_dir / "predictions.jsonl"))
    for i in range(repeats):
        # Cycle through images if repeats > len(images)
        try:
            img = next(images_iter)
        except StopIteration:
            images_iter = iter(images)
            img = next(images_iter)
        logger.info("Repeat %d/%d | image=%s", i + 1, repeats, str(img))
        res = session.run_inference(
            image_path=str(img),
            prompt="<image>\n<|grounding|>Convert the document to markdown.",
            max_new_tokens=max_new_tokens,
            return_text=save_preds,
            preprocess=dict(
                enable=bool(getattr(cfg.model, "preprocess", {}).get("enable", True)),
                base_size=int(getattr(cfg.model, "preprocess", {}).get("base_size", 1024)),
                image_size=int(getattr(cfg.model, "preprocess", {}).get("image_size", 640)),
                crop_mode=bool(getattr(cfg.model, "preprocess", {}).get("crop_mode", False)),
                patch_size=int(getattr(cfg.model, "preprocess", {}).get("patch_size", 16)),
                downsample_ratio=int(getattr(cfg.model, "preprocess", {}).get("downsample_ratio", 4)),
            ),
            infer=dict(
                temperature=float(getattr(cfg, "infer", {}).get("temperature", 0.0)),
                max_new_tokens=int(getattr(cfg, "infer", {}).get("max_new_tokens", max_new_tokens)),
                no_repeat_ngram_size=int(getattr(cfg, "infer", {}).get("no_repeat_ngram_size", 0)),
                do_sample=bool(getattr(cfg, "infer", {}).get("do_sample", False)),
            ),
        )
        runs.append(
            ImageRun(
                image_path=str(img),
                prefill_ms=float(res.get("prefill_ms", 0.0)),
                decode_ms=float(res.get("decode_ms", 0.0)),
                tokens=int(res.get("tokens", 0)),
                prefill_len=int(res.get("prefill_len", 0)),
                vision_ms=float(res.get("vision_ms", 0.0)),
                sam_ms=float(res.get("sam_ms", 0.0)),
                clip_ms=float(res.get("clip_ms", 0.0)),
                projector_ms=float(res.get("projector_ms", 0.0)),
            )
        )
        if save_preds and isinstance(res.get("text"), str):
            text_raw = str(res.get("text"))
            text_clean = _clean_prediction_text(text_raw, strip_special=strip_special)
            preds_for_outputs.append(
                {
                    "image": str(img),
                    "text_raw": text_raw,
                    "text_clean": text_clean,
                    "prefill_ms": float(res.get("prefill_ms", 0.0)),
                    "decode_ms": float(res.get("decode_ms", 0.0)),
                    "vision_ms": float(res.get("vision_ms", 0.0)),
                    "tokens": int(res.get("tokens", 0)),
                    "tokens_per_s": (
                        (float(res.get("tokens", 0)) / (float(res.get("decode_ms", 1.0)) / 1000.0))
                        if float(res.get("decode_ms", 0.0)) > 0
                        else 0.0
                    ),
                }
            )

    # Summarize + outputs
    # Static compute via analyzer (preferred) with fallback
    static_report: dict = {}
    try:
        # We'll fill seq_len after we compute prefill length mean; compute summary first to get it
        pass
    except Exception:
        pass

    summary = _summarize_runs(
        runs,
        getattr(session, "m_model", None),
        peak,
        ctx_len_mode=str(getattr(cfg, "infer", {}).get("context_len_mode", "auto")),
        ctx_len_fixed=int(getattr(cfg, "infer", {}).get("context_len_fixed", 0)) or None,
        vision_flops=None,
        model_window=None,
    )

    # Compute improved MFU using static analyzer (guarded by unified pipeline config)
    sa_cfg = getattr(getattr(cfg, "pipeline", {}), "static_analysis", {})
    if bool(getattr(sa_cfg, "enable", True)):
        try:
            pre_cfg = getattr(getattr(cfg, "model", {}), "preprocess", {})
            prefill_len_mean = int(summary.get("aggregates", {}).get("tokens", {}).get("mean", 0) or 0)
            # Use actual input prefill length from runs if available
            try:
                prefill_len_mean = int(mean_std([r.prefill_len for r in runs])[0])
            except Exception:
                pass
            analyzer = DeepseekOCRStaticAnalyzer(session)
            aconf = AnalysisConfig(
                image_h=int(pre_cfg.get("base_size", 1024)),
                image_w=int(pre_cfg.get("base_size", 1024)),
                base_size=int(pre_cfg.get("base_size", 1024)),
                image_size=int(pre_cfg.get("image_size", 640)),
                seq_len=max(prefill_len_mean, 1),
                crop_mode=bool(pre_cfg.get("crop_mode", True)),
                patch_size=int(pre_cfg.get("patch_size", 16)),
                downsample_ratio=int(pre_cfg.get("downsample_ratio", 4)),
                use_analytic_fallback=bool(getattr(sa_cfg, "use_analytic_fallback", True)),
                use_synthetic_inputs=bool(getattr(sa_cfg, "use_synthetic_inputs", True)),
            )
            static_report = analyzer.generate_report(aconf)
            # Extract stage flops
            stages = static_report.get("stages", {}) if isinstance(static_report.get("stages"), dict) else {}
            def _stage_flops(name: str) -> float:
                st = stages.get(name, {}) if isinstance(stages.get(name), dict) else {}
                # Prefer analytic for decode (per-token), otherwise flops
                if name == "decode":
                    return float(st.get("flops_analytic", st.get("flops", 0.0)) or 0.0)
                if name == "prefill":
                    return float(st.get("flops_analytic", st.get("flops", 0.0)) or 0.0)
                return float(st.get("flops", 0.0) or 0.0)

            prefill_flops_total = _stage_flops("prefill")
            decode_flops_per_token = _stage_flops("decode")
            vision_flops_total = _stage_flops("sam") + _stage_flops("clip") + _stage_flops("projector")

            aggr = summary.get("aggregates", {}) if isinstance(summary.get("aggregates"), dict) else {}
            pf_ms = float(aggr.get("prefill_ms", {}).get("mean", 0.0))
            dc_ms = float(aggr.get("decode_ms", {}).get("mean", 0.0))
            vn_ms = float(aggr.get("stage_ms", {}).get("vision", {}).get("mean", 0.0)) if isinstance(aggr.get("stage_ms", {}), dict) else 0.0
            toks_mean = float(aggr.get("tokens", {}).get("mean", 0.0))

            # MFU calculations (TFLOPs utilization)
            def _mfu(flops: float, ms: float) -> float:
                if ms <= 0.0 or peak <= 0.0:
                    return 0.0
                achieved_tflops = (flops / 1e12) / (ms / 1000.0)
                return float(achieved_tflops / peak)

            mfu_prefill = _mfu(prefill_flops_total, pf_ms)
            mfu_decode = _mfu(decode_flops_per_token * max(toks_mean, 1.0), dc_ms)
            mfu_vision = _mfu(vision_flops_total, vn_ms) if vn_ms > 0.0 else 0.0
            # Overall model-level MFU across prefill+decode (avoid double counting vision)
            total_flops = prefill_flops_total + decode_flops_per_token * max(toks_mean, 1.0)
            total_time_s = (pf_ms + dc_ms) / 1000.0 if (pf_ms + dc_ms) > 0 else 0.0
            mfu_model = (total_flops / 1e12) / total_time_s / peak if total_time_s > 0 and peak > 0 else summary.get("mfu_model_level", 0.0)

            # Update summary MFUs with improved estimates
            summary["mfu_per_stage"] = {
                "prefill": float(mfu_prefill),
                "decode": float(mfu_decode),
                "vision": float(mfu_vision),
            }
            summary["mfu_model_level"] = float(mfu_model)

            # Write detailed static compute report (JSON+MD) if allowed
            try:
                if bool(getattr(sa_cfg, "write_reports", True)):
                    from llm_perf_opt.profiling.export import write_static_compute_json, write_static_compute_markdown
                    write_static_compute_json(static_report, static_out_dir / "static_compute.json")
                    write_static_compute_markdown(static_report, static_out_dir / "static_compute.md")
            except Exception:
                pass
        except Exception:
            # Fall back to previous simple static report path
            static_compute = {}
            try:
                static_compute = session.estimate_static_compute()
                _write_static_compute(static_out_dir, static_compute)
            except Exception:
                pass
    if save_preds and preds_for_outputs:
        _write_predictions_outputs(
            pred_dir,
            vis_dir,
            preds_for_outputs,
            make_gallery=make_gallery,
            max_images=max_images,
            thumb_width=thumb_w,
        )
    aggr = summary.get("aggregates", {})
    logger.info(
        (
            "Aggregates | prefill_ms=%.3f±%.3f decode_ms=%.3f±%.3f tokens=%.1f±%.1f tps=%.3f±%.3f"
        ),
        float(aggr.get("prefill_ms", {}).get("mean", 0.0)),
        float(aggr.get("prefill_ms", {}).get("std", 0.0)),
        float(aggr.get("decode_ms", {}).get("mean", 0.0)),
        float(aggr.get("decode_ms", {}).get("std", 0.0)),
        float(aggr.get("tokens", {}).get("mean", 0.0)),
        float(aggr.get("tokens", {}).get("std", 0.0)),
        float(aggr.get("tokens_per_s", {}).get("mean", 0.0)),
        float(aggr.get("tokens_per_s", {}).get("std", 0.0)),
    )
    logger.info(
        "MFU | model=%.6f decode=%.6f",
        float(summary.get("mfu_model_level", 0.0)),
        float(summary.get("mfu_per_stage", {}).get("decode", 0.0)),
    )
    _write_outputs(artifacts_dir, summary, top_n_operators(operator_records, n=20), top_k=20)
    logger.info(
        "Wrote artifacts | report=%s operators=%s metrics=%s",
        str(artifacts_dir / "report.md"),
        str(artifacts_dir / "operators.md"),
        str(artifacts_dir / "metrics.json"),
    )

    # US3: Reproducibility artifacts
    try:
        write_env_json(str(artifacts_dir / "env.json"))
    except Exception:
        pass
    try:
        _write_inputs_yaml(
            artifacts_dir,
            images,
            dataset_root=str(cfg.dataset.root),
            subset_filelist=str(cfg.dataset.get("subset_filelist")) if cfg.dataset.get("subset_filelist") else None,
        )
    except Exception:
        pass
    try:
        _write_assumptions_md(artifacts_dir, cfg)
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
