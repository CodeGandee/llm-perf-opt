from __future__ import annotations

import shutil


class ToolNotFoundError(RuntimeError):
    pass


def ensure_nsys() -> str:
    """Return `nsys` executable path or raise a friendly error.

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

