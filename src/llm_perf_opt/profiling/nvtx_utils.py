"""NVTX helper ranges for stage segmentation.

Defines convenience context managers for ``prefill`` and ``decode`` NVTX
ranges used across runners and sessions for profiler gating.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
import nvtx  # type: ignore[import-untyped]


@contextmanager
def prefill_range() -> Iterator[None]:
    """NVTX range labeled 'prefill'."""

    nvtx.push_range("prefill")
    try:
        yield
    finally:
        nvtx.pop_range()


@contextmanager
def decode_range() -> Iterator[None]:
    """NVTX range labeled 'decode'."""

    nvtx.push_range("decode")
    try:
        yield
    finally:
        nvtx.pop_range()
