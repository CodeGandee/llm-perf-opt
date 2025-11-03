"""Vendor tool availability checks for Stage 2.

This module provides small helpers to ensure Nsight Systems/Compute are
available on PATH and raise clear, user-friendly errors otherwise.
"""

from __future__ import annotations

import shutil


class ToolNotFoundError(RuntimeError):
    """Raised when a required external tool is not available on PATH."""


def ensure_nsys() -> str:
    """Return `nsys` executable path or raise a friendly error.

    Returns
    -------
    str
        Resolved path to the `nsys` executable.

    Raises
    ------
    ToolNotFoundError
        If `nsys` is not available in PATH.
    """

    path = shutil.which("nsys")
    if not path:
        raise ToolNotFoundError(
            "Nsight Systems (nsys) not found in PATH. Install it and verify with 'nsys --version'."
        )
    return path


def ensure_ncu() -> str:
    """Return `ncu` executable path or raise a friendly error.

    Returns
    -------
    str
        Resolved path to the `ncu` executable.

    Raises
    ------
    ToolNotFoundError
        If `ncu` is not available in PATH.
    """

    path = shutil.which("ncu")
    if not path:
        raise ToolNotFoundError(
            "Nsight Compute (ncu) not found in PATH. Install it and verify with 'ncu --version'."
        )
    return path
