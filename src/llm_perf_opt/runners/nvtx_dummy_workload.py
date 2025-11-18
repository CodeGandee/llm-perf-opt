"""NVTX‑annotated dummy workload for Nsight range replay.

Hydra entrypoint that forwards a tiny ShallowResNet on synthetic input to
exercise NVTX ranges (stem, residual, head, and conv subranges).
"""

from __future__ import annotations

from typing import Any

import torch
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from llm_perf_opt.dnn_models.factory import get_model


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:  # pragma: no cover - tiny workload
    try:
        device = str(getattr(cfg, "device", "cpu"))
    except Exception:
        device = "cpu"
    try:
        bs = int(getattr(getattr(cfg, "infer", {}), "batch_size", 4))
    except Exception:
        bs = 4

    model: Any = get_model("dummy_shallow_resnet", device=device)
    x = torch.randn(bs, 3, 64, 64, device=device)
    # Warmup
    try:
        model.warmup(device=device)
    except Exception:
        pass
    # Few forwards to ensure ranges present
    for _ in range(2):
        _ = model(x)


if __name__ == "__main__":  # pragma: no cover
    main()
