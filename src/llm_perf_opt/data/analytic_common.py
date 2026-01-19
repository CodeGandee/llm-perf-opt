"""Shared analytic report schema types.

This module defines common `attrs`-based data models used by analytic report
artifacts across multiple models (for example DeepSeek-OCR and Wan2.1).
The goal is to keep report consumers stable while allowing model-specific
wrappers to evolve independently.
"""

from __future__ import annotations

import os
from typing import Union

from attrs import Attribute, define, field
from attrs.validators import instance_of

ScalarParam = Union[int, float, str, bool]


def _validate_absolute_path(_instance: object, attribute: Attribute[str], value: str) -> None:
    """Ensure a path field is an absolute path."""

    if not os.path.isabs(value):
        raise ValueError(f"{attribute.name} must be an absolute path, got {value!r}")


def _validate_non_negative_float(_instance: object, attribute: Attribute[float], value: float) -> None:
    """Ensure a float metric is non-negative."""

    if value < 0.0:
        raise ValueError(f"{attribute.name} must be non-negative, got {value!r}")


def _validate_non_negative_int(_instance: object, attribute: Attribute[int], value: int) -> None:
    """Ensure an integer metric is non-negative."""

    if value < 0:
        raise ValueError(f"{attribute.name} must be non-negative, got {value!r}")


@define(kw_only=True)
class AnalyticModuleNode:
    """Node in the analytic module hierarchy."""

    module_id: str = field(validator=[instance_of(str)])
    name: str = field(validator=[instance_of(str)])
    qualified_class_name: str = field(validator=[instance_of(str)])
    stage: str = field(validator=[instance_of(str)])
    parent_id: str | None = field(default=None)
    children: list[str] = field(factory=list)
    repetition: str = field(default="none", validator=[instance_of(str)])
    repetition_count: int | None = field(default=None)
    constructor_params: dict[str, ScalarParam] = field(factory=dict)


@define(kw_only=True)
class OperatorCategory:
    """Logical group for operators (e.g., conv2d, attention)."""

    category_id: str = field(validator=[instance_of(str)])
    display_name: str = field(validator=[instance_of(str)])
    description: str = field(validator=[instance_of(str)])
    match_classes: list[str] = field(factory=list)


@define(kw_only=True)
class OperatorMetrics:
    """Per-operator-category metric breakdown within a module."""

    category_id: str = field(validator=[instance_of(str)])
    calls: int = field(validator=[instance_of(int), _validate_non_negative_int])
    flops_tflops: float = field(validator=[instance_of(float), _validate_non_negative_float])
    io_tb: float = field(validator=[instance_of(float), _validate_non_negative_float])
    share_of_module_flops: float = field(validator=[instance_of(float), _validate_non_negative_float])


@define(kw_only=True)
class ModuleMetricsSnapshot:
    """Aggregated analytic metrics for a single module under a workload."""

    module_id: str = field(validator=[instance_of(str)])
    profile_id: str = field(validator=[instance_of(str)])
    calls: int = field(validator=[instance_of(int), _validate_non_negative_int])
    total_time_ms: float = field(validator=[instance_of(float), _validate_non_negative_float])
    total_flops_tflops: float = field(validator=[instance_of(float), _validate_non_negative_float])
    total_io_tb: float = field(validator=[instance_of(float), _validate_non_negative_float])
    memory_weights_gb: float = field(validator=[instance_of(float), _validate_non_negative_float])
    memory_activations_gb: float = field(validator=[instance_of(float), _validate_non_negative_float])
    memory_kvcache_gb: float = field(validator=[instance_of(float), _validate_non_negative_float])
    share_of_model_time: float = field(validator=[instance_of(float), _validate_non_negative_float])
    operator_breakdown: list[OperatorMetrics] = field(factory=list)


__all__ = [
    "ScalarParam",
    "_validate_absolute_path",
    "_validate_non_negative_float",
    "_validate_non_negative_int",
    "AnalyticModuleNode",
    "OperatorCategory",
    "OperatorMetrics",
    "ModuleMetricsSnapshot",
]
