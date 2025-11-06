"""Dummy models used for testing and verification.

This package provides small, deterministic models that emit NVTX ranges and
produce short‑lived kernels suitable for Nsight Compute/Sys manual testing.
"""

from llm_perf_opt.dnn_models.factory import get_model

__all__ = ["get_model"]
