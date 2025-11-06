"""Data models for NVTX range–scoped Nsight Compute reporting.

This module defines light‑weight ``attrs`` data classes used to represent
regions and their aggregated profiling summaries. Business logic lives in
specialized helpers; these classes serve as typed containers suitable for
serialization.

Classes
-------
NCUProfileRegion
    A named NVTX region with optional nesting and scope qualifiers.
NCUProfileRegionReport
    Aggregated metrics and kernel counts for a single region.
"""

from __future__ import annotations

from typing import Optional
from attrs import define


@define(kw_only=True)
class NCUProfileRegion:
    """A named NVTX region profiled by Nsight Compute.

    Attributes
    ----------
    name : str
        Region display name (e.g., ``'A'``, ``'A::A1'``).
    parent : str | None, optional
        Optional parent region name for nesting (derived via ``::``).
    depth : int
        Nesting depth where 0 denotes a root region.
    process : str | None, optional
        Process identifier if available in multi‑process runs.
    device : str | None, optional
        Device identifier (e.g., ``'cuda:0'``).
    """

    name: str
    parent: Optional[str] = None
    depth: int = 0
    process: Optional[str] = None
    device: Optional[str] = None


@define(kw_only=True)
class NCUProfileRegionReport:
    """Aggregated metrics and kernel summaries for a region.

    Attributes
    ----------
    region : NCUProfileRegion
        The region metadata this report summarizes.
    total_ms : float
        Inclusive time attributed to this region in milliseconds.
    kernel_count : int
        Number of kernel invocations attributed to this region.
    sections_path : str | None, optional
        Path to imported sections text (if generated).
    csv_path : str | None, optional
        Path to raw CSV export for this region (if generated).
    markdown_path : str | None, optional
        Path to per‑region Markdown (if generated).
    json_path : str | None, optional
        Path to per‑region JSON (if generated).
    """

    region: NCUProfileRegion
    total_ms: float = 0.0
    kernel_count: int = 0
    sections_path: Optional[str] = None
    csv_path: Optional[str] = None
    markdown_path: Optional[str] = None
    json_path: Optional[str] = None

