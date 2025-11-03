"""NVTX helper ranges for stage segmentation.

Defines convenience context managers for `prefill` and `decode` NVTX ranges,
and Stage 2 domain-labeled ranges (``LLM@prefill``, ``LLM@decode_all``) for
Nsight integration.
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


@contextmanager
def llm_prefill() -> Iterator[None]:
    """NVTX range labeled 'LLM@prefill' (Stage 2)."""

    nvtx.push_range("LLM@prefill")
    try:
        yield
    finally:
        nvtx.pop_range()


@contextmanager
def llm_decode_all() -> Iterator[None]:
    """NVTX range labeled 'LLM@decode_all' (Stage 2)."""

    nvtx.push_range("LLM@decode_all")
    try:
        yield
    finally:
        nvtx.pop_range()
