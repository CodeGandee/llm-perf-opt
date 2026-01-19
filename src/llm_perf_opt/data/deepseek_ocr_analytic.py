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

from typing import Dict, Iterable, Literal

from attrs import define, field
from attrs.validators import instance_of

from llm_perf_opt.data.analytic_common import (
    AnalyticModuleNode,
    ModuleMetricsSnapshot,
    OperatorCategory,
    OperatorMetrics,
    _validate_absolute_path,
    _validate_non_negative_float,
)

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


def categorize_operator_class_name(class_name_qualified: str) -> str:
    """Return a coarse operator category id for a qualified class name.

    This helper performs a best-effort classification based on the fully
    qualified class name derived from TorchInfo or analytic modules. The
    mapping is intentionally simple and stable so downstream capacity-planning
    tools can group operators into a small number of families without
    depending on framework-internal package layouts.

    Examples
    --------
    >>> categorize_operator_class_name("torch.nn.modules.conv.Conv2d")
    'conv2d'
    >>> categorize_operator_class_name("torch.nn.modules.linear.Linear")
    'linear'
    >>> categorize_operator_class_name("torch.nn.modules.activation.GELU")
    'activation'
    """

    name = class_name_qualified.lower()
    if "conv" in name:
        return "conv2d"
    if "linear" in name or ".lm_head" in name:
        return "linear"
    if "norm" in name:
        return "norm"
    if "attention" in name or "attn" in name:
        return "attention"
    if "embedding" in name:
        return "embedding"
    if "pool" in name:
        return "pool"
    if "dropout" in name:
        return "dropout"
    if any(act in name for act in ("gelu", "relu", "silu", "sigmoid", "tanh", "softmax")):
        return "activation"
    return "other"


def build_operator_categories_from_target_list(
    target_ops: TargetOperatorList,
) -> list[OperatorCategory]:
    """Derive :class:`OperatorCategory` instances from a TargetOperatorList.

    The resulting categories are used to drive capacity-planning exports.
    Each operator in ``target_ops.operators`` is assigned a ``default_category_id``
    based on its qualified class name, and the category's ``match_classes``
    list is populated with all matched ``class_name_qualified`` values.
    """

    categories: Dict[str, OperatorCategory] = {}

    for op in target_ops.operators:
        category_id = categorize_operator_class_name(op.class_name_qualified)
        op.default_category_id = category_id

        category = categories.get(category_id)
        if category is None:
            display = category_id.replace("_", " ").title()
            description = f"Operators belonging to the {display} family."
            category = OperatorCategory(
                category_id=category_id,
                display_name=display,
                description=description,
                match_classes=[],
            )
            categories[category_id] = category

        # Avoid duplicates while preserving input order as much as possible.
        if op.class_name_qualified not in category.match_classes:
            category.match_classes.append(op.class_name_qualified)

    # Ensure there is always an "other" bucket available for fallbacks.
    if "other" not in categories:
        categories["other"] = OperatorCategory(
            category_id="other",
            display_name="Other",
            description="Operators that do not match a more specific family.",
            match_classes=[],
        )

    return list(categories.values())


def build_operator_category_index(
    categories: Iterable[OperatorCategory],
) -> Dict[str, str]:
    """Build a lookup from qualified class name to category id."""

    index: Dict[str, str] = {}
    for category in categories:
        for cls in category.match_classes:
            if cls not in index:
                index[cls] = category.category_id
    return index


__all__ = [
    "DeepSeekOCRModelSpec",
    "OCRWorkloadProfile",
    # Shared-schema compatibility re-exports.
    "AnalyticModuleNode",
    "OperatorCategory",
    "OperatorMetrics",
    "ModuleMetricsSnapshot",
    # DeepSeek-OCR report wrapper and TorchInfo-derived operator metadata.
    "AnalyticModelReport",
    "OperatorSpec",
    "TargetOperatorList",
    # Operator categorization helpers.
    "categorize_operator_class_name",
    "build_operator_categories_from_target_list",
    "build_operator_category_index",
]
