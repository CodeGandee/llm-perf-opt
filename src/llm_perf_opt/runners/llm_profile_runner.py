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
from omegaconf import DictConfig
import torch
from torch.profiler import ProfilerActivity, profile  # type: ignore[attr-defined]

from hydra.core.hydra_config import HydraConfig
from llm_perf_opt.profiling.aggregate import mean_std
from llm_perf_opt.profiling.export import top_n_operators, write_operator_markdown
from llm_perf_opt.profiling.hw import get_device_name, get_peak_tflops
from llm_perf_opt.profiling.mfu import estimate_decode_flops_per_token, mfu as mfu_ratio
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession
from PIL import Image  # type: ignore[import-untyped]
from llm_perf_opt.visualize.annotations import render_vendor_style, write_vendor_result_mmd
from mdutils.mdutils import MdUtils  # type: ignore[import-untyped]


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


def _collect_operator_records(prof: Any) -> list[dict]:
    """Extract operator‑level summaries from a PyTorch profiler object.

    Returns a list of dicts with keys: ``op_name``, ``total_time_ms``,
    ``cuda_time_ms``, ``calls``.
    """

    records: list[dict] = []
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
            records.append(
                {
                    "op_name": name,
                    "total_time_ms": max(total_ms, 0.0),
                    "cuda_time_ms": max(cuda_ms, 0.0),
                    "calls": max(calls, 0),
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


def _summarize_runs(runs: list[ImageRun], model_obj: object, peak_tflops: float) -> dict:
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
    ctx_len = 512  # Stage 1 default; refine in later stages with actual decode context
    flops_per_token = estimate_decode_flops_per_token(d_model, d_ff, n_layers, ctx_len)
    mfu_model = mfu_ratio(tokens_per_s=tps_mean, flops_per_token=flops_per_token, peak_tflops=peak_tflops)
    mfu_per_stage = {
        "prefill": 0.0,  # placeholder (Stage 1)
        "decode": mfu_ratio(tokens_per_s=tps_mean, flops_per_token=flops_per_token, peak_tflops=peak_tflops),
    }

    return {
        "aggregates": {
            "prefill_ms": {"mean": prefill_mean, "std": prefill_std},
            "decode_ms": {"mean": decode_mean, "std": decode_std},
            "tokens": {"mean": tokens_mean, "std": tokens_std},
            "tokens_per_s": {"mean": tps_mean, "std": tps_std},
        },
        "mfu_model_level": mfu_model,
        "mfu_per_stage": mfu_per_stage,
        "model_dims": {"d_model": d_model, "d_ff": d_ff, "n_layers": n_layers, "ctx_len": ctx_len},
        "peak_tflops": peak_tflops,
    }


def _write_outputs(artifacts_dir: Path, summary: dict, operator_records: list[dict], top_k: int = 20) -> None:
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
    artifacts_dir: Path,
    preds: list[dict],
    make_gallery: bool,
    max_images: int | None,
    thumb_width: int,
) -> None:
    """Write predictions.jsonl and an optional Markdown gallery with thumbnails."""

    # JSONL
    pj = artifacts_dir / "predictions.jsonl"
    with pj.open("w", encoding="utf-8") as f:
        for rec in preds:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    if not make_gallery:
        return

    # Gallery (Markdown) with local thumbnails for portability
    # Organize vendor-style annotated assets per image under viz/<stem>/
    viz_root = artifacts_dir / "viz"
    viz_root.mkdir(parents=True, exist_ok=True)
    thumb_dir = viz_root / "_thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    md = MdUtils(file_name=str((artifacts_dir / "predictions").as_posix()))
    md.new_header(level=1, title="Predictions Gallery")

    count = 0
    for rec in preds:
        if max_images is not None and count >= int(max_images):
            break
        img_path = Path(str(rec.get("image", "")))
        text_raw = str(rec.get("text_raw", ""))
        text_clean = str(rec.get("text_clean", ""))
        # Render vendor-style annotations (result_with_boxes.jpg + images/) per-image subdir
        annotated_img_rel = None
        if img_path.is_file():
            try:
                per_image_dir = viz_root / img_path.stem
                per_image_dir.mkdir(parents=True, exist_ok=True)
                out_annotated = render_vendor_style(str(img_path), text_raw, str(per_image_dir))
                # Also write vendor-style result.mmd next to annotated image
                try:
                    _ = write_vendor_result_mmd(text_raw, str(per_image_dir))
                except Exception:
                    pass
                annotated_img_rel = out_annotated.relative_to(artifacts_dir)
            except Exception:
                annotated_img_rel = None
        if img_path.is_file():
            try:
                im = Image.open(img_path).convert("RGB")
                w, h = im.size
                if w > thumb_width:
                    ratio = thumb_width / float(w)
                    im = im.resize((thumb_width, int(h * ratio)))
                thumb_name = img_path.stem + ".jpg"
                thumb_path = thumb_dir / thumb_name
                im.save(thumb_path, format="JPEG", quality=90)
                rel_thumb = thumb_path.relative_to(artifacts_dir)
                md.new_header(level=2, title=img_path.name)
                md.new_paragraph(f"![{img_path.name}]({rel_thumb.as_posix()})")
                if annotated_img_rel is not None:
                    md.new_paragraph("Annotated (with boxes)")
                    md.new_paragraph(f"![annotated]({annotated_img_rel.as_posix()})")
            except Exception:
                md.new_header(level=2, title=img_path.name)
        else:
            md.new_header(level=2, title=img_path.name)

        md.new_paragraph("**Prediction (clean)**")
        md.new_line("```text")
        md.new_line(text_clean)
        md.new_line("```")
        md.new_paragraph("Raw (with specials)")
        md.new_line("```text")
        md.new_line(text_raw)
        md.new_line("```")
        count += 1

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

    # Prepare output dir (Hydra run dir configured to Stage 1 artifacts path)
    artifacts_dir = Path(HydraConfig.get().run.dir)
    # Set up a file logger for easier debugging of runs
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
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

    # Representative operator profile on the first image
    operator_records: list[dict] = []
    rep_image = str(images[0])
    prof_cfg = getattr(cfg, "profiling", {})
    sel_acts = [str(x).lower() for x in list(prof_cfg.get("activities", ["cpu", "cuda"]))]
    activities = []
    if "cpu" in sel_acts:
        activities.append(ProfilerActivity.CPU)
    if "cuda" in sel_acts and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    try:
        # Keep rep profile bounded; honor profiling config caps and switches
        rep_cap = int(getattr(getattr(cfg, 'profiling', {}), 'rep_max_new_tokens', 64))
        rep_max_new = int(min(int(getattr(cfg, "infer", {}).get("max_new_tokens", 64)), rep_cap))
        record_shapes = bool(prof_cfg.get("record_shapes", False))
        profile_memory = bool(prof_cfg.get("profile_memory", False))
        with_stack = bool(prof_cfg.get("with_stack", False))
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
        operator_records = _collect_operator_records(prof)
        logger.info("Collected %d operator records", len(operator_records))
    except Exception:
        operator_records = []  # Fail‑open for environments without profiler

    # Repeated runs across dataset
    runs: list[ImageRun] = []
    repeats = int(cfg.repeats)
    max_new_tokens = int(getattr(cfg, "infer", {}).get("max_new_tokens", 64))
    images_iter: Iterator[Path] = iter(images)
    save_preds = bool(getattr(cfg, "outputs", {}).get("save_predictions", False))
    strip_special = bool(getattr(cfg.outputs, "predictions", {}).get("strip_special_tokens", False)) if hasattr(cfg, "outputs") else False
    viz_cfg = getattr(cfg.outputs, "visualization", {}) if hasattr(cfg, "outputs") else {}
    make_gallery = bool(viz_cfg.get("enable", True))
    max_images = viz_cfg.get("max_images", 16)
    max_images = None if max_images in (None, "null") else int(max_images)
    thumb_w = int(viz_cfg.get("thumbnail_width", 480))
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
                    "tokens": int(res.get("tokens", 0)),
                    "tokens_per_s": (
                        (float(res.get("tokens", 0)) / (float(res.get("decode_ms", 1.0)) / 1000.0))
                        if float(res.get("decode_ms", 0.0)) > 0
                        else 0.0
                    ),
                }
            )

    # Summarize + outputs
    summary = _summarize_runs(runs, getattr(session, "m_model", None), peak)
    if save_preds and preds_for_outputs:
        _write_predictions_outputs(
            artifacts_dir,
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


if __name__ == "__main__":  # pragma: no cover
    main()
