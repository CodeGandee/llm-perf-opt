"""Domain models for Wan2.1 analytic modeling.

This module defines `attrs`-based data classes for Wan2.1 analytic reports.
It keeps the Wan-specific model spec and workload profile separate from the
shared analytic schema types (module tree and per-module metrics snapshots).
"""

from __future__ import annotations

from attrs import Attribute, define, field
from attrs.validators import instance_of

from llm_perf_opt.data.analytic_common import (
    AnalyticModuleNode,
    ModuleMetricsSnapshot,
    OperatorCategory,
    _validate_absolute_path,
    _validate_non_negative_float,
)


def _validate_positive_int(_instance: object, attribute: Attribute[int], value: int) -> None:
    """Ensure an integer field is positive."""

    if value <= 0:
        raise ValueError(f"{attribute.name} must be positive, got {value!r}")


@define(kw_only=True)
class Wan2_1ModelSpec:
    """Model specification for Wan2.1 analytic reports."""

    model_id: str = field(validator=[instance_of(str)])
    model_variant: str = field(validator=[instance_of(str)])
    config_path: str = field(
        validator=[instance_of(str), _validate_absolute_path],
        metadata={"help": "Absolute path to the Wan2.1 model config metadata used for analytic parameters."},
    )
    hidden_size: int = field(validator=[instance_of(int), _validate_positive_int])
    num_layers: int = field(validator=[instance_of(int), _validate_positive_int])
    num_attention_heads: int = field(validator=[instance_of(int), _validate_positive_int])
    head_dim: int = field(validator=[instance_of(int), _validate_positive_int])
    mlp_intermediate_size: int = field(validator=[instance_of(int), _validate_positive_int])
    vae_downsample_factor: int = field(validator=[instance_of(int), _validate_positive_int])
    patch_size: int = field(validator=[instance_of(int), _validate_positive_int])
    latent_channels: int = field(validator=[instance_of(int), _validate_positive_int])
    notes: str = field(default="", validator=[instance_of(str)])


@define(kw_only=True)
class Wan2_1WorkloadProfile:
    """Synthetic workload profile for Wan2.1 analytic reports."""

    profile_id: str = field(validator=[instance_of(str)])
    description: str = field(validator=[instance_of(str)])
    batch_size: int = field(validator=[instance_of(int), _validate_positive_int])
    num_frames: int = field(validator=[instance_of(int), _validate_positive_int])
    height: int = field(validator=[instance_of(int), _validate_positive_int])
    width: int = field(validator=[instance_of(int), _validate_positive_int])
    num_inference_steps: int = field(validator=[instance_of(int), _validate_positive_int])
    text_len: int = field(validator=[instance_of(int), _validate_positive_int])


@define(kw_only=True)
class Wan2_1AnalyticModelReport:
    """Full analytic model report for a Wan2.1 workload."""

    report_id: str = field(validator=[instance_of(str)])
    model: Wan2_1ModelSpec = field()
    workload: Wan2_1WorkloadProfile = field()
    modules: list[AnalyticModuleNode] = field(factory=list)
    operator_categories: list[OperatorCategory] = field(factory=list)
    module_metrics: list[ModuleMetricsSnapshot] = field(factory=list)
    profile_run_id: str | None = field(default=None)
    predicted_total_time_ms: float = field(default=0.0, validator=[instance_of(float), _validate_non_negative_float])
    notes: str = field(default="", validator=[instance_of(str)])
    layer_docs_dir: str | None = field(default=None)


__all__ = ["Wan2_1ModelSpec", "Wan2_1WorkloadProfile", "Wan2_1AnalyticModelReport"]
