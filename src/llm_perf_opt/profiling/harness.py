"""Profiling harness utilities.

This module provides simple NVTX range helpers and can be extended with
PyTorch profiler wrappers.

Functions
---------
nvtx_range
    Context manager that pushes/pops an NVTX range.
"""

from __future__ import annotations

from contextlib import contextmanager
import nvtx  # type: ignore[import-untyped]


@contextmanager
def nvtx_range(name: str):
    """Push/pop an NVTX range with the given name.

    Parameters
    ----------
    name : str
        The range name to appear in NVTX-aware tools.

    Examples
    --------
    >>> with nvtx_range("prefill"):
    ...     pass
    """

    nvtx.push_range(name)
    try:
        yield
    finally:
        nvtx.pop_range()
