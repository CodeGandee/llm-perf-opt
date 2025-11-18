"""Domain models for DeepSeek-OCR analytic modeling.

This module defines attrs-based data classes that represent the
DeepSeek-OCR analytic model domain, including model and workload
metadata, module hierarchy, operator categories, per-module metrics,
and static operator lists derived from TorchInfo artifacts.

Classes
-------
DeepSeekOCRModelSpec
    Model configuration and variant metadata.
OCRWorkloadProfile
    Synthetic workload profile parameters (for example, dsocr-standard-v1).
AnalyticModuleNode
    Node in the analytic module hierarchy.
OperatorCategory
    Operator family definition used for grouping metrics.
ModuleMetricsSnapshot
    Aggregated analytic metrics for a single module under a workload.
OperatorMetrics
    Per-operator-category metric breakdown within a module.
AnalyticModelReport
    Full analytic model report for a DeepSeek-OCR workload.
OperatorSpec
    Static operator entry as parsed from TorchInfo artifacts.
TargetOperatorList
    Collection of TorchInfo-derived operators and artifact paths.
"""

from __future__ import annotations

import os
from typing import Literal, Union

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
class DeepSeekOCRModelSpec:
    """Model specification for DeepSeek-OCR analytic reports."""

    model_id: str = field(validator=[instance_of(str)])
    model_variant: str = field(validator=[instance_of(str)])
    hf_revision: str | None = field(default=None)
    config_path: str = field(
        validator=[instance_of(str), _validate_absolute_path],
        metadata={"help": "Absolute path to the DeepSeek-OCR model config"},
    )
    hidden_size: int = field(validator=[instance_of(int)])
    intermediate_size: int = field(validator=[instance_of(int)])
    num_layers: int = field(validator=[instance_of(int)])
    num_attention_heads: int = field(validator=[instance_of(int)])
    vision_backbone: Literal["sam_vit_b", "clip_vit_l", "other"] = field(validator=[instance_of(str)])
    uses_moe: bool = field(validator=[instance_of(bool)])
    notes: str = field(default="", validator=[instance_of(str)])


@define(kw_only=True)
class OCRWorkloadProfile:
    """Synthetic workload profile (for example, ``dsocr-standard-v1``)."""

    profile_id: str = field(validator=[instance_of(str)])
    description: str = field(validator=[instance_of(str)])
    seq_len: int = field(validator=[instance_of(int)])
    base_size: int = field(validator=[instance_of(int)])
    image_size: int = field(validator=[instance_of(int)])
    crop_mode: bool = field(validator=[instance_of(bool)])
    max_new_tokens: int = field(validator=[instance_of(int)])
    doc_kind: Literal["text_heavy", "mixed_layout", "image_rich", "synthetic"] = field(
        validator=[instance_of(str)],
    )
    num_pages: int = field(validator=[instance_of(int)])


@define(kw_only=True)
class AnalyticModuleNode:
    """Node in the analytic module hierarchy."""

    module_id: str = field(validator=[instance_of(str)])
    name: str = field(validator=[instance_of(str)])
    qualified_class_name: str = field(validator=[instance_of(str)])
    stage: Literal["vision", "projector", "prefill", "decode", "other"] = field(validator=[instance_of(str)])
    parent_id: str | None = field(default=None)
    children: list[str] = field(factory=list)
    repetition: Literal["none", "for", "parfor"] = field(default="none", validator=[instance_of(str)])
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
    share_of_module_flops: float = field(
        validator=[instance_of(float), _validate_non_negative_float],
    )


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
    share_of_model_time: float = field(
        validator=[instance_of(float), _validate_non_negative_float],
    )
    operator_breakdown: list[OperatorMetrics] = field(factory=list)


@define(kw_only=True)
class AnalyticModelReport:
    """Full analytic model report for DeepSeek-OCR."""

    report_id: str = field(validator=[instance_of(str)])
    model: DeepSeekOCRModelSpec = field()
    workload: OCRWorkloadProfile = field()
    modules: list[AnalyticModuleNode] = field(factory=list)
    operator_categories: list[OperatorCategory] = field(factory=list)
    module_metrics: list[ModuleMetricsSnapshot] = field(factory=list)
    profile_run_id: str | None = field(default=None)
    predicted_total_time_ms: float = field(validator=[instance_of(float), _validate_non_negative_float])
    measured_total_time_ms: float | None = field(default=None)
    predicted_vs_measured_ratio: float | None = field(default=None)
    notes: str = field(default="", validator=[instance_of(str)])
    layer_docs_dir: str = field(
        validator=[instance_of(str), _validate_absolute_path],
        metadata={"help": "Absolute path to generated per-layer Markdown docs"},
    )


@define(kw_only=True)
class OperatorSpec:
    """Static operator entry as parsed from TorchInfo."""

    class_name: str = field(validator=[instance_of(str)])
    class_name_qualified: str = field(validator=[instance_of(str)])
    is_pytorch_builtin: bool = field(validator=[instance_of(bool)])
    is_custom: bool = field(validator=[instance_of(bool)])
    children_classes: list[str] = field(factory=list)
    default_category_id: str = field(default="", validator=[instance_of(str)])


@define(kw_only=True)
class TargetOperatorList:
    """Static operator snapshot derived from TorchInfo artifacts."""

    snapshot_id: str = field(validator=[instance_of(str)])
    source_artifact_dir: str = field(
        validator=[instance_of(str), _validate_absolute_path],
        metadata={"help": "Absolute path to directory containing TorchInfo artifacts"},
    )
    layers_md_path: str = field(
        validator=[instance_of(str), _validate_absolute_path],
        metadata={"help": "Absolute path to torchinfo-unique-layers.md"},
    )
    layers_json_path: str = field(
        validator=[instance_of(str), _validate_absolute_path],
        metadata={"help": "Absolute path to torchinfo-unique-layers.json"},
    )
    stages_json_path: str = field(
        validator=[instance_of(str), _validate_absolute_path],
        metadata={"help": "Absolute path to torchinfo-stages.json"},
    )
    operators: list[OperatorSpec] = field(factory=list)
