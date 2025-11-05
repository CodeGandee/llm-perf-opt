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
    """Return theoretical peak TFLOPs for known devices based on precision.

    Parameters
    ----------
    device_name : str
        Friendly GPU device name.
    precision : str, default='bf16'
        Numeric precision: 'fp32', 'fp16', 'bf16', or 'tf32'.
        For 'bf16', returns FP16 Tensor Core performance.

    Returns
    -------
    float
        TFLOPs value. Falls back to ``MFU_PEAK_TFLOPS`` env var, or ``100`` if unset.

    Notes
    -----
    FP16/BF16 values are dense Tensor Core performance (not sparse).
    FP32 values are CUDA core performance.
    Data sources: NVIDIA official specs, TechPowerUp, Tom's Hardware (2025).
    See context/hints/nv-profile-kb/peak-tflops-reference.md for details.
    """

    # Nested table: device -> precision -> TFLOPS
    table = {
        # Data Center GPUs
        "NVIDIA H100 SXM": {
            "fp32": 67.0,
            "fp16": 1979.0,  # Tensor Core
            "bf16": 1979.0,  # Tensor Core
            "tf32": 989.0,   # Tensor Core
        },
        "NVIDIA H100 PCIe": {
            "fp32": 51.0,
            "fp16": 1513.0,
            "bf16": 1513.0,
            "tf32": 756.0,
        },
        "NVIDIA H200": {
            "fp32": 67.0,
            "fp16": 1979.0,
            "bf16": 1979.0,
            "tf32": 989.0,
        },
        "NVIDIA A100": {
            "fp32": 19.5,
            "fp16": 312.0,
            "bf16": 312.0,
            "tf32": 156.0,
        },
        "NVIDIA A800": {
            "fp32": 19.5,
            "fp16": 312.0,
            "bf16": 312.0,
            "tf32": 156.0,
        },
        # Consumer GPUs
        "NVIDIA GeForce RTX 4090": {
            "fp32": 82.6,
            "fp16": 661.0,  # Tensor Core
            "bf16": 661.0,  # Tensor Core
        },
        # RTX 50‑series (Blackwell)
        # Sources:
        # - NVIDIA RTX Blackwell GPU Architecture (public PDF; GB202 specs)
        # - Puget Systems AI review indicating ~209.5 TFLOPS (BF16/FP16 dense) for RTX 5090
        #   https://www.pugetsystems.com/labs/articles/nvidia-geforce-rtx-5090-amp-5080-ai-review/
        # Approx FP32 computed from CUDA cores (21,760) * 2.407 GHz * 2 FLOPs / 1e12 ≈ 104.8 TFLOPS.
        "NVIDIA GeForce RTX 5090": {
            "fp32": 104.8,
            "fp16": 209.5,  # Tensor Core (dense)
            "bf16": 209.5,  # Tensor Core (dense)
        },
        "NVIDIA GeForce RTX 3090": {
            "fp32": 35.6,
            "fp16": 285.0,  # Tensor Core (sparse spec; use with caution)
            "bf16": 285.0,  # Tensor Core
        },
    }

    # Normalize precision to lowercase
    prec = precision.lower()
    if prec not in ("fp32", "fp16", "bf16", "tf32"):
        prec = "bf16"  # default to bf16 if unrecognized

    # Find matching device (substring match)
    for key, prec_dict in table.items():
        if key in device_name:
            # Return value for requested precision, fallback to bf16, then first available
            if prec in prec_dict:
                return prec_dict[prec]
            elif "bf16" in prec_dict:
                return prec_dict["bf16"]
            else:
                return next(iter(prec_dict.values()))

    # No match found, try environment variable
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


def write_env_json(path: str) -> None:
    """Write environment snapshot to a JSON file.

    Parameters
    ----------
    path : str
        Destination file path for the JSON content.
    """

    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(capture_env(), f, indent=2)
