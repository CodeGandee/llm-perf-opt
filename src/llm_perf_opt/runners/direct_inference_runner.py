"""Direct Inference Runner (no profiling).

Hydra entry that runs dataset inference without any profiling overhead.
Outputs are written under `${hydra.run.dir}/direct_inference/` with the same
prediction/visualization schema used by other stages.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession
from llm_perf_opt.runners.inference_engine import run_stage_dataset


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI orchestrator
    logging.captureWarnings(True)
    logger = logging.getLogger(__name__)

    di_cfg = getattr(getattr(cfg, "pipeline", {}), "direct_inference", {})
    if not bool(getattr(di_cfg, "enable", False)):
        logger.info("pipeline.direct_inference.enable=false; nothing to do")
        return

    # Parse dtype from config
    dtype_str = str(getattr(cfg.model, "dtype", "bf16")).lower()
    dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    model_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # Build session
    session = DeepSeekOCRSession.from_local(
        model_path=cfg.model.path,
        device=cfg.device,
        use_flash_attn=bool(cfg.use_flash_attn),
        dtype=model_dtype,
    )

    # Resolve run root and stage dirs
    run_dir_cfg = Path(HydraConfig.get().run.dir)
    base_cwd = Path(HydraConfig.get().runtime.cwd)
    run_root = run_dir_cfg if run_dir_cfg.is_absolute() else (base_cwd / run_dir_cfg)
    stage_out_dir = run_root / "direct_inference"
    stage_tmp_dir = run_root / "tmp" / "direct_inference"
    stage_out_dir.mkdir(parents=True, exist_ok=True)
    stage_tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Direct inference start | device=%s model=%s dataset_root=%s out=%s",
        cfg.device,
        cfg.model.path,
        cfg.dataset.root,
        str(stage_out_dir),
    )

    runs, preds, summary = run_stage_dataset(
        cfg=cfg,
        session=session,
        stage_name="direct_inference",
        stage_out_dir=stage_out_dir,
        stage_tmp_dir=stage_tmp_dir,
        logger=logger,
        hooks=None,
    )

    aggr = summary.get("aggregates", {}) if isinstance(summary.get("aggregates"), dict) else {}
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


if __name__ == "__main__":  # pragma: no cover
    main()

