"""Tests for normal vs flash analytic vision variants in DeepSeek-OCR.

These tests verify two properties:

- At the layer level, :class:`NoTPAttention` uses the ``use_flash_attention``
  flag only for memory/I/O modeling (FLOPs are identical).
- At the model level, the analytic vision composites wired as
  ``vision_normal`` and ``vision_flash`` produce identical FLOPs but
  different I/O and activation memory for the same workload.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from modelmeter.models.common import StageCost
from modelmeter.models.deepseek_ocr.layers.core.deepseek_ocr_model import DeepseekOCRModel
from modelmeter.models.deepseek_ocr.layers.vision.notp_attention import NoTPAttention


def test_notp_attention_flash_vs_normal_memory_differs() -> None:
    """Layer-level NoTPAttention should change memory/I/O but not FLOPs."""

    common_kwargs = {
        "hidden_size": 1024,
        "num_heads": 16,
        "seq_len": 256,
        "batch_size": 1,
    }
    attn_normal = NoTPAttention(use_flash_attention=False, **common_kwargs)
    attn_flash = NoTPAttention(use_flash_attention=True, **common_kwargs)

    flops_normal = attn_normal.forward_tensor_core_flops() or 0.0
    flops_flash = attn_flash.forward_tensor_core_flops() or 0.0
    io_normal = attn_normal.forward_cal_io() or 0.0
    io_flash = attn_flash.forward_cal_io() or 0.0
    mem_normal = attn_normal.forward_memory_activation() or 0.0
    mem_flash = attn_flash.forward_memory_activation() or 0.0

    # FLOPs should be identical; memory/I/O should be strictly lower
    # for the flash-attention variant.
    assert flops_normal == flops_flash
    assert io_normal > io_flash
    assert mem_normal > mem_flash


def _load_analytic_cfg() -> DictConfig:
    """Load the base DeepSeek-OCR analytic Hydra config."""

    # tests/unit/deepseek_ocr/... -> repo root at parents[3]
    root = Path(__file__).resolve().parents[3]
    config_dir = root / "extern" / "modelmeter" / "models" / "deepseek_ocr" / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg: DictConfig = compose(config_name="deepseek_ocr")
    return cfg


def _build_vision_model(cfg: DictConfig, *, use_flash: bool) -> DeepseekOCRModel:
    """Instantiate DeepseekOCRModel with a specific vision composite."""

    cfg_local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))

    model_cfg = getattr(cfg_local, "model", None)
    if model_cfg is not None:
        has_normal = hasattr(model_cfg, "vision_layer_normal")
        has_flash = hasattr(model_cfg, "vision_layer_flash")
        if use_flash and has_flash:
            model_cfg.vision_layer = model_cfg.vision_layer_flash
        elif not use_flash and has_normal:
            model_cfg.vision_layer = model_cfg.vision_layer_normal

    model: DeepseekOCRModel = instantiate(cfg_local.model)
    # Use the standard crops-mode vision path.
    model.set_vision_mode("crops")
    return model


def test_vision_normal_vs_flash_stagecost_memory_differs() -> None:
    """Root vision composites should differ in memory/I/O but not FLOPs."""

    cfg = _load_analytic_cfg()

    model_normal = _build_vision_model(cfg, use_flash=False)
    model_flash = _build_vision_model(cfg, use_flash=True)

    batch_size = int(cfg.runtime.batch_size)

    model_normal.start_vision(batch_size=batch_size)
    cost_normal: StageCost = model_normal.get_forward_cost()

    model_flash.start_vision(batch_size=batch_size)
    cost_flash: StageCost = model_flash.get_forward_cost()

    flops_normal = float(cost_normal.flops_tflops)
    flops_flash = float(cost_flash.flops_tflops)
    io_normal = float(cost_normal.io_tb)
    io_flash = float(cost_flash.io_tb)
    mem_normal = float(cost_normal.activations_gb)
    mem_flash = float(cost_flash.activations_gb)

    # FLOPs should be identical for normal vs flash analytic vision;
    # flash attention should reduce I/O and activation memory.
    assert flops_normal == flops_flash
    assert io_normal > io_flash
    assert mem_normal > mem_flash
