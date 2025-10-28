"""Hardware utilities for profiling metadata and peak throughput lookup.

Functions
---------
get_device_name
    Resolve a CUDA device name, or 'cpu' if unavailable.
get_peak_tflops
    Coarse table-based TFLOPs lookup with env override.
capture_env
    Minimal environment snapshot for reproducibility.
"""

from __future__ import annotations

import os
import torch


def get_device_name(index: int = 0) -> str:
    """Return the CUDA device name (or 'cpu' if CUDA unavailable).

    Parameters
    ----------
    index : int, default=0
        CUDA device index.

    Returns
    -------
    str
        The device name or 'cpu'.
    """

    return torch.cuda.get_device_name(index) if torch.cuda.is_available() else "cpu"


def get_peak_tflops(device_name: str, precision: str = "bf16") -> float:
    """Return a coarse theoretical peak TFLOPs for known devices.

    Parameters
    ----------
    device_name : str
        Friendly GPU device name.
    precision : str, default='bf16'
        Numeric precision context (informational only in this stub).

    Returns
    -------
    float
        TFLOPs value. Falls back to ``MFU_PEAK_TFLOPS`` env var, or ``100`` if unset.
    """

    table = {
        "NVIDIA H100": 990.0,
        "NVIDIA A100": 312.0,
        "NVIDIA GeForce RTX 4090": 330.0,
    }
    for key, val in table.items():
        if key in device_name:
            return val
    try:
        return float(os.environ.get("MFU_PEAK_TFLOPS", "100"))
    except Exception:
        return 100.0


def capture_env() -> dict:
    """Capture minimal environment information for reproducibility.

    Returns
    -------
    dict
        Includes GPU name, CUDA version (if available), Torch and Transformers versions.
    """

    info = {
        "gpu": get_device_name(),
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }
    try:
        import transformers as H  # type: ignore[import-not-found,import-untyped]

        info["transformers"] = H.__version__
    except Exception:  # pragma: no cover - optional dependency path
        info["transformers"] = None
    return info
