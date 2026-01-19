"""Factory helpers for dummy models used in tests and manual profiling."""

from __future__ import annotations

from typing import Literal

import torch

from llm_perf_opt.dnn_models.shallow_resnet import ShallowResNet

ModelName = Literal["dummy_shallow_resnet"]


def get_model(name: ModelName, device: str | None = None) -> torch.nn.Module:
    """Return a dummy model instance by name.

    Args:
        name: Supported model identifier (currently only ``"dummy_shallow_resnet"``).
        device: Optional device string (e.g., ``"cuda:0"``) to move the model to.

    Returns:
        An instantiated ``torch.nn.Module``.
    """

    if name == "dummy_shallow_resnet":
        model = ShallowResNet()
    else:  # pragma: no cover - explicit guard for future extensions
        raise ValueError(f"Unknown dummy model: {name}")
    if device is not None:
        model.to(device)
    return model
