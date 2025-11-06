"""Dummy models used for testing and verification.

This package provides small, deterministic models that emit NVTX ranges and
produce short‑lived kernels suitable for Nsight Compute/Sys manual testing.
"""

from .factory import get_model

__all__ = ["get_model"]

