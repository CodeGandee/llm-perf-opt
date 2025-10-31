"""Shared inference engine for pipeline stages.

Provides a reusable dataset loop and outputs writer used by both the
`torch_profiler` and `direct_inference` stages. The engine does not start
any external profilers; stage-specific runners can supply hooks if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
import random
from contextlib import nullcontext

from omegaconf import DictConfig, OmegaConf  # type: ignore[import-untyped]

from llm_perf_opt.profiling.aggregate import mean_std
from llm_perf_opt.profiling.hw import get_device_name, get_peak_tflops
from llm_perf_opt.profiling.mfu import compute_stage_mfu
from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession
from llm_perf_opt.visualize.annotations import render_vendor_style, write_vendor_result_mmd


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


def _read_filelist(root: str, filelist: str) -> list[Path]:
    fp = Path(filelist)
    if not fp.is_absolute():
        try:
            from hydra.core.hydra_config import HydraConfig as _HC  # local import
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
    if subset_filelist:
        return _read_filelist(root, subset_filelist)
    rp = Path(root)
    out: list[Path] = []
    for pat in fallback_patterns:
        out.extend(sorted(rp.glob(pat)))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return [pp.resolve() for pp in uniq]


def _require_int_max_new_tokens(raw_mnt: object) -> int:
    if raw_mnt is None:
        raise ValueError(
            "infer.max_new_tokens must be an integer; got null/None. "
            "Specify an explicit integer (e.g., 8192)."
        )
    if isinstance(raw_mnt, str) and str(raw_mnt).strip().lower() == "inf":
        raise ValueError(
            "infer.max_new_tokens must be an integer; 'inf' is not supported. "
            "Choose an explicit value like 8192."
        )
    try:
        import math as _math
        if isinstance(raw_mnt, float):
            if _math.isinf(raw_mnt) or not raw_mnt.is_integer():
                raise ValueError("infer.max_new_tokens must be an integer (no decimals/inf).")
            return int(raw_mnt)
    except Exception:
        pass
    try:
        return int(raw_mnt)
    except Exception:
        raise ValueError("infer.max_new_tokens must be an integer; received an invalid value.")


def _clean_prediction_text(text: str, strip_special: bool) -> str:
    if not isinstance(text, str):
        return ""
    s = text.replace("<｜end▁of▁sentence｜>", "").strip()
    if strip_special:
        s = s.replace("\x00", "").strip()
    return s


def _write_predictions_outputs(
    pred_dir: Path,
    vis_dir: Path,
    preds: list[dict],
    make_gallery: bool,
    max_images: int | None,
) -> None:
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
        pre_ms = float(rec.get("prefill_ms", 0.0))
        dec_ms = float(rec.get("decode_ms", 0.0))
        vis_ms = float(rec.get("vision_ms", 0.0))
        try:
            h = _hashlib.md5(str(img_path.resolve()).encode("utf-8")).hexdigest()
        except Exception:
            h = _hashlib.md5(str(img_path).encode("utf-8")).hexdigest()
        per_image_dir = vis_dir / h
        per_image_dir.mkdir(parents=True, exist_ok=True)

        if img_path.is_file():
            try:
                out_annotated, boxes = render_vendor_style(str(img_path), text_raw, str(per_image_dir))
                _ = out_annotated  # annotated image path (not used further here)
            except Exception:
                boxes = []
            try:
                _ = write_vendor_result_mmd(text_raw, str(per_image_dir))
            except Exception:
                pass

        info = {
            "source_image": str(img_path.resolve()) if img_path.is_absolute() else str(img_path),
            "text_raw": text_raw,
            "timings_ms": {"prefill": pre_ms, "decode": dec_ms, "vision": vis_ms},
            "boxes": boxes,
        }
        try:
            (per_image_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        count += 1


def _summarize_runs(
    runs: list[ImageRun],
    model_obj: object,
    peak_tflops: float,
    ctx_len_mode: str = "auto",
    ctx_len_fixed: int | None = None,
    vision_flops: float | None = None,
    model_window: int | None = None,
) -> dict:
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

    # Infer model dims best-effort
    try:
        cfg = getattr(model_obj, "config", None)
        if cfg is not None:
            d_model = int(getattr(cfg, "hidden_size", getattr(cfg, "d_model", 1024)))
            d_ff = int(getattr(cfg, "intermediate_size", getattr(cfg, "ffn_dim", 4096)))
            n_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 24)))
        else:
            raise AttributeError
    except Exception:
        d_model, d_ff, n_layers = 1024, 4096, 24

    prefill_len_mean = int(mean_std([r.prefill_len for r in runs])[0])
    tokens_mean_int = int(tokens_mean)
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


class ProfilingHooks:
    """Optional callbacks/contexts for profiling stages.

    For direct inference, leave defaults (no-ops). Torch profiler runners can
    supply a context manager via `context_provider` if needed.
    """

    def on_epoch_start(self, epoch: int) -> None:  # pragma: no cover - optional hook
        return None

    def on_epoch_end(self, epoch: int) -> None:  # pragma: no cover - optional hook
        return None

    def on_iter_start(self, idx: int) -> None:  # pragma: no cover - optional hook
        return None

    def on_iter_end(self, idx: int) -> None:  # pragma: no cover - optional hook
        return None

    def context_provider(self):  # pragma: no cover - optional hook
        return nullcontext()


def run_stage_dataset(
    cfg: DictConfig,
    session: DeepSeekOCRSession,
    stage_name: str,
    stage_out_dir: Path,
    stage_tmp_dir: Path,
    logger: logging.Logger,
    hooks: Optional[ProfilingHooks] = None,
) -> tuple[list[ImageRun], list[dict], dict]:
    """Run dataset inference for a pipeline stage and write outputs.

    Returns runs, predictions list (for outputs), and the summary dict.
    """

    hooks = hooks or ProfilingHooks()

    # Discover images
    images_all = list(
        _iter_images(
            cfg.dataset.root,
            list(cfg.dataset.fallback_patterns),
            cfg.dataset.get("subset_filelist"),
        )
    )
    if not images_all:
        raise RuntimeError(f"No images found in dataset root: {cfg.dataset.root}")

    # Output config for this stage
    stg = getattr(getattr(cfg, "pipeline", {}), stage_name, {})
    out_cfg = getattr(stg, "output", {})
    pred_cfg = getattr(out_cfg, "prediction", {})
    vis_cfg = getattr(out_cfg, "visualization", {})
    extra_dsocr = getattr(getattr(out_cfg, "extra", {}), "deepseek_ocr", {})

    save_preds = bool(pred_cfg.get("enable", False))
    pred_dirname = pred_cfg.get("save_dir", None) or "pred"
    vis_enable = bool(vis_cfg.get("enable", True))
    vis_dirname = vis_cfg.get("save_dir", None) or "viz"

    strip_special = bool(getattr(extra_dsocr, "prediction", {}).get("strip_special_tokens", False))
    max_images = getattr(getattr(extra_dsocr, "visualization", {}), "max_images", 16)

    # Sampling config
    sam = getattr(getattr(cfg, "dataset", {}), "sampling", {})
    n_per_epoch_raw = sam.get("num_samples_per_epoch", None)
    n_per_epoch: int | None = None if n_per_epoch_raw in (None, "null") else int(n_per_epoch_raw)
    n_epochs = int(sam.get("num_epochs", 1))
    rand = bool(sam.get("randomize", False))
    seed = sam.get("seed", None)
    rng = random.Random(int(seed)) if seed not in (None, "null") else random.Random()

    # infer settings
    max_new_tokens = _require_int_max_new_tokens(getattr(getattr(cfg, "infer", {}), "max_new_tokens", 64))
    _infer_cfg = getattr(cfg, "infer", {})
    _infer_kwargs = dict(
        temperature=float(_infer_cfg.get("temperature", 0.0)),
        no_repeat_ngram_size=int(_infer_cfg.get("no_repeat_ngram_size", 0)),
        do_sample=bool(_infer_cfg.get("do_sample", False)),
    )

    # Prepare dirs
    stage_out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = stage_out_dir / str(pred_dirname)
    vis_dir = stage_out_dir / str(vis_dirname)
    stage_tmp_dir.mkdir(parents=True, exist_ok=True)

    runs: list[ImageRun] = []
    preds_for_outputs: list[dict] = []

    for ep in range(max(1, n_epochs)):
        hooks.on_epoch_start(ep)
        order = images_all[:]
        if rand:
            rng.shuffle(order)
        if n_per_epoch is None:
            selected = order
        else:
            if not rand:
                start = (ep * n_per_epoch) % max(1, len(order))
                selected = (order[start:] + order[:start])[: min(n_per_epoch, len(order))]
            else:
                selected = order[: min(n_per_epoch, len(order))]
        for idx, img in enumerate(selected):
            hooks.on_iter_start(idx)
            with hooks.context_provider():
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
                    infer=_infer_kwargs,
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
            hooks.on_iter_end(idx)
        hooks.on_epoch_end(ep)

    # Summarize
    device_name = get_device_name()
    precision = str(getattr(cfg.model, "dtype", "bf16"))
    peak = get_peak_tflops(device_name, precision)
    summary = _summarize_runs(
        runs,
        getattr(session, "m_model", None),
        peak,
        ctx_len_mode=str(getattr(cfg, "infer", {}).get("context_len_mode", "auto")),
        ctx_len_fixed=int(getattr(cfg, "infer", {}).get("context_len_fixed", 0)) or None,
        vision_flops=None,
        model_window=int(getattr(cfg, "infer", {}).get("model_window", 0)) or None,
    )

    # Outputs
    if save_preds:
        _write_predictions_outputs(
            pred_dir,
            vis_dir,
            preds_for_outputs,
            make_gallery=bool(vis_enable),
            max_images=(None if getattr(getattr(out_cfg, "visualization", {}), "max_images", 16) in (None, "null") else int(getattr(getattr(out_cfg, "visualization", {}), "max_images", 16))),
        )

    return runs, preds_for_outputs, summary

