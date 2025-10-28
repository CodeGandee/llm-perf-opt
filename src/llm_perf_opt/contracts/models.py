"""Profiling contracts (attrs models).

This module defines request/response schemas for profiling workflows. These
mirror OpenAPI schemas in
`specs/001-profile-deepseek-ocr/contracts/openapi.yaml` and the Python
contracts design in `specs/001-profile-deepseek-ocr/contracts/python-contracts.md`.

Classes
-------
LLMProfileRequest
    Request payload for starting a profiling run.
LLMProfileAccepted
    Acknowledgement for a queued/running profiling request.
OperatorSummary
    Aggregated operator-level metrics (time, calls).
Stats
    Simple mean/std pair used in aggregates.
LLMProfileReportSummary
    Public-facing summary of a profiling run.

Notes
-----
All filesystem paths are absolute; validators enforce this where applicable.
"""

from __future__ import annotations

import os
from typing import Dict, Literal

from attrs import define, field
from attrs.validators import instance_of


def _abs_path(_: object, attr: object, value: str) -> None:
    """Enforce absolute paths for string fields.

    Parameters
    ----------
    _ : object
        Unused instance reference from attrs validation protocol.
    attr : object
        Attribute metadata (may provide ``name``).
    value : str
        The value to validate.

    Raises
    ------
    ValueError
        If ``value`` is not an absolute path.
    """

    if not os.path.isabs(value):
        name = getattr(attr, "name", "path")
        raise ValueError(f"{name} must be an absolute path")


@define(kw_only=True)
class LLMProfileRequest:
    """Inputs for a profiling run.

    Examples
    --------
    >>> LLMProfileRequest(model_path="/abs/models/dsocr", input_dir="/abs/data/samples")
    LLMProfileRequest(...)
    """

    model_path: str = field(validator=[instance_of(str), _abs_path])
    input_dir: str = field(validator=[instance_of(str), _abs_path])
    repeats: int = field(default=3, validator=[instance_of(int)])
    use_flash_attn: bool = field(default=True, validator=[instance_of(bool)])
    device: str = field(default="cuda:0", validator=[instance_of(str)])
    max_new_tokens: int = field(default=64, validator=[instance_of(int)])


@define(kw_only=True)
class LLMProfileAccepted:
    """Acknowledgement of a profiling request being queued or running."""

    run_id: str = field(validator=[instance_of(str)])
    status: Literal["queued", "running"] = field(validator=[instance_of(str)])
    artifacts_dir: str = field(validator=[instance_of(str), _abs_path])


@define(kw_only=True)
class OperatorSummary:
    """Aggregated operator measurement record."""

    op_name: str = field(validator=[instance_of(str)])
    total_time_ms: float = field(validator=[instance_of(float)])
    cuda_time_ms: float = field(validator=[instance_of(float)])
    calls: int = field(validator=[instance_of(int)])


@define(kw_only=True)
class Stats:
    """Mean and standard deviation for a metric."""

    mean: float = field(validator=[instance_of(float)])
    std: float = field(validator=[instance_of(float)])


@define(kw_only=True)
class LLMProfileReportSummary:
    """Summary of a profiling run for external consumption."""

    run_id: str = field(validator=[instance_of(str)])
    mfu_model_level: float = field(validator=[instance_of(float)])
    mfu_per_stage: Dict[str, float] = field(factory=dict)
    top_operators: list[OperatorSummary] = field(factory=list)
    aggregates: Dict[str, Stats] = field(factory=dict)
    notes: str = field(validator=[instance_of(str)])
