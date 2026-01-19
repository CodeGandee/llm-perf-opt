"""Contract models (attrs-based schemas).

This module defines request/response schemas for profiling workflows and
DeepSeek-OCR analytic modeling. These mirror OpenAPI schemas in

- ``specs/001-profile-deepseek-ocr/contracts/openapi.yaml`` and
  ``specs/001-profile-deepseek-ocr/contracts/python-contracts.md``; and
- ``specs/001-deepseek-ocr-modelmeter/contracts/openapi.yaml`` and
  ``specs/001-deepseek-ocr-modelmeter/contracts/python-contracts.md``.
  ``specs/004-wan2-1-analytic-model/contracts/openapi.yaml`` and
  ``specs/004-wan2-1-analytic-model/contracts/python-contracts.md``.

Notes
-----
- All filesystem paths are absolute; validators enforce this where applicable.
- Analytic modeling contracts are light-weight views over the internal
  domain models defined in :mod:`llm_perf_opt.data.deepseek_ocr_analytic`.
"""

from __future__ import annotations

import os
from typing import Dict, Literal

from attrs import define, field
from attrs.validators import instance_of

from llm_perf_opt.data.analytic_common import AnalyticModuleNode, ModuleMetricsSnapshot, OperatorCategory
from llm_perf_opt.data.deepseek_ocr_analytic import AnalyticModelReport


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


# ---------------------------------------------------------------------------
# DeepSeek-OCR analytic modeling contracts
# ---------------------------------------------------------------------------


@define(kw_only=True)
class DeepSeekOCRAnalyticRequest:
    """Request to build or refresh the DeepSeek-OCR analytic model."""

    model_id: str = field(
        validator=[instance_of(str)],
        metadata={"help": "Canonical model id (e.g., deepseek-ai/DeepSeek-OCR)"},
    )
    model_variant: str = field(
        validator=[instance_of(str)],
        metadata={"help": "Internal model variant (e.g., deepseek-ocr-v1-base)"},
    )
    workload_profile_id: str = field(
        validator=[instance_of(str)],
        metadata={"help": "Workload profile id (e.g., dsocr-standard-v1)"},
    )
    profile_run_id: str | None = field(
        default=None,
        metadata={"help": "Optional Stage1/Stage2 profiling run id for later validation"},
    )
    force_rebuild: bool = field(
        default=False,
        validator=[instance_of(bool)],
        metadata={"help": "Force recomputation even if a matching report already exists"},
    )


@define(kw_only=True)
class DeepSeekOCRAnalyticAccepted:
    """Acknowledgement of an analytic modeling request."""

    report_id: str = field(validator=[instance_of(str)])
    status: Literal["queued", "running"] = field(validator=[instance_of(str)])


@define(kw_only=True)
class AnalyticModuleSummary:
    """Top-level summary record for a single analytic module."""

    module_id: str = field(validator=[instance_of(str)])
    name: str = field(validator=[instance_of(str)])
    stage: Literal["vision", "projector", "prefill", "decode", "other"] = field(
        validator=[instance_of(str)],
    )
    share_of_model_time: float = field(validator=[instance_of(float)])
    total_time_ms: float = field(validator=[instance_of(float)])
    total_flops_tflops: float = field(validator=[instance_of(float)])
    memory_activations_gb: float = field(validator=[instance_of(float)])


@define(kw_only=True)
class DeepSeekOCRAnalyticReportSummary:
    """Summary view over an :class:`AnalyticModelReport`."""

    report_id: str = field(validator=[instance_of(str)])
    model_variant: str = field(validator=[instance_of(str)])
    workload_profile_id: str = field(validator=[instance_of(str)])
    predicted_total_time_ms: float = field(validator=[instance_of(float)])
    measured_total_time_ms: float | None = field(
        default=None,
        metadata={
            "help": (
                "Optional measured runtime; typically None in the current "
                "development stage."
            ),
        },
    )
    predicted_vs_measured_ratio: float | None = field(
        default=None,
        metadata={
            "help": (
                "Optional predicted/measured ratio; reserved for future "
                "runtime comparison."
            ),
        },
    )
    top_modules: list[AnalyticModuleSummary] = field(factory=list)
    notes: str = field(validator=[instance_of(str)])


@define(kw_only=True)
class DeepSeekOCRAnalyticModel:
    """Contract view of the full DeepSeek-OCR analytic model."""

    # For the full-model view we simply expose the internal AnalyticModelReport
    # as-is. This keeps the web/CLI contract aligned with the domain model
    # while still allowing a dedicated type name in OpenAPI specs.
    report: AnalyticModelReport = field()


# ---------------------------------------------------------------------------
# Wan2.1 analytic modeling contracts
# ---------------------------------------------------------------------------


@define(kw_only=True)
class Wan2_1AnalyticRequest:
    """Request to build or refresh the Wan2.1 analytic model."""

    model_id: str = field(validator=[instance_of(str)])
    model_variant: str = field(validator=[instance_of(str)])
    workload_profile_id: str = field(validator=[instance_of(str)])
    overrides: dict[str, int] | None = field(default=None)
    profile_run_id: str | None = field(default=None)
    force_rebuild: bool = field(default=False, validator=[instance_of(bool)])


@define(kw_only=True)
class Wan2_1AnalyticAccepted:
    """Acknowledgement of a Wan2.1 analytic modeling request."""

    report_id: str = field(validator=[instance_of(str)])
    status: Literal["queued", "running"] = field(validator=[instance_of(str)])


@define(kw_only=True)
class Wan2_1ModelSpec:
    """Contract view of the Wan2.1 model spec."""

    model_id: str = field(validator=[instance_of(str)])
    model_variant: str = field(validator=[instance_of(str)])
    config_path: str = field(validator=[instance_of(str), _abs_path])
    hidden_size: int = field(validator=[instance_of(int)])
    num_layers: int = field(validator=[instance_of(int)])
    num_attention_heads: int = field(validator=[instance_of(int)])
    head_dim: int = field(validator=[instance_of(int)])
    mlp_intermediate_size: int = field(validator=[instance_of(int)])
    vae_downsample_factor: int = field(validator=[instance_of(int)])
    patch_size: int = field(validator=[instance_of(int)])
    latent_channels: int = field(validator=[instance_of(int)])
    notes: str = field(validator=[instance_of(str)])


@define(kw_only=True)
class Wan2_1WorkloadProfile:
    """Contract view of a Wan2.1 workload profile."""

    profile_id: str = field(validator=[instance_of(str)])
    description: str = field(validator=[instance_of(str)])
    batch_size: int = field(validator=[instance_of(int)])
    num_frames: int = field(validator=[instance_of(int)])
    height: int = field(validator=[instance_of(int)])
    width: int = field(validator=[instance_of(int)])
    num_inference_steps: int = field(validator=[instance_of(int)])
    text_len: int = field(validator=[instance_of(int)])


@define(kw_only=True)
class Wan2_1AnalyticModel:
    """Contract view of the Wan2.1 analytic model report."""

    report_id: str = field(validator=[instance_of(str)])
    model: Wan2_1ModelSpec = field()
    workload: Wan2_1WorkloadProfile = field()
    modules: list[AnalyticModuleNode] = field(factory=list)
    operator_categories: list[OperatorCategory] = field(factory=list)
    module_metrics: list[ModuleMetricsSnapshot] = field(factory=list)
    profile_run_id: str | None = field(default=None)
