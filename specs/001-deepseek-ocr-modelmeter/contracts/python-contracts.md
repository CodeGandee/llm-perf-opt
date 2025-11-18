# Python Contracts Design (DeepSeek‑OCR Analytic Modeling)

This document specifies Python‑oriented contracts that mirror the OpenAPI schemas in
`/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/contracts/openapi.yaml` for DeepSeek‑OCR analytic
modeling. Prefer `attrs` for Python data models per the repository convention; use `cattrs` for (de)serialization.

## Goals

- Provide strongly typed, documented `attrs` models for analytic modeling requests and responses.
- Keep names stable and generic enough to allow future DeepSeek‑OCR variants.
- Align with domain models in `data-model.md` while exposing a summary‑oriented view for external callers.

## Package Location

- Recommended module path for implementation: `src/llm_perf_opt/contracts/models.py` (shared with Stage 1) or a nearby
  module such as `src/llm_perf_opt/contracts/analytic.py` if separation is preferred.
- Schemas map 1:1 to `components.schemas` entries in `openapi.yaml`.

## Schemas

### DeepSeekOCRAnalyticRequest

- model_id: str — Canonical model identifier (e.g., `"deepseek-ai/DeepSeek-OCR"`).
- model_variant: str — Internal variant name (e.g., `"deepseek-ocr-v1-base"`).
- workload_profile_id: str — Workload profile id (e.g., `"dsocr-standard-v1"`).
- profile_run_id: str | None — Optional Stage 1/2 profiling run id to validate against.
- force_rebuild: bool = False — Force recomputation even if a matching analytic report already exists.

### DeepSeekOCRAnalyticAccepted

- report_id: str — Identifier for the analytic report (used in subsequent GETs).
- status: Literal["queued", "running"] — Queuing status for long‑running builds.

### AnalyticModuleSummary

- module_id: str — Module identifier from the analytic model.
- name: str — Human‑readable module name.
- stage: Literal["vision", "projector", "prefill", "decode", "other"] — Stage attribution.
- share_of_model_time: float — Fraction of total runtime assigned to this module (0..1+).
- total_time_ms: float — Predicted runtime for this module.
- total_flops_tflops: float — Predicted FLOPs (TFLOPs) for this module.
- memory_activations_gb: float — Predicted activation memory for this module.

### DeepSeekOCRAnalyticReportSummary

- report_id: str — Identifier for the analytic model.
- model_variant: str — Variant name (mirrors request).
- workload_profile_id: str — Workload profile id (mirrors request).
- predicted_total_time_ms: float — Predicted end‑to‑end runtime (theoretical, analytic estimate).
- measured_total_time_ms: float | None — Optional measured runtime for future validation; may be omitted in this development stage.
- predicted_vs_measured_ratio: float | None — Optional predicted/measured ratio (None if no measurement or validation is deferred).
- top_modules: list[AnalyticModuleSummary] — Top modules by runtime share.
- notes: str — Stakeholder‑oriented notes and caveats.

### DeepSeekOCRModelSpec (contract view)

- model_id: str
- model_variant: str
- hf_revision: str | None
- hidden_size: int
- intermediate_size: int
- num_layers: int
- num_attention_heads: int
- vision_backbone: Literal["sam_vit_b", "clip_vit_l", "other"]
- uses_moe: bool

### OCRWorkloadProfile (contract view)

- profile_id: str
- description: str
- seq_len: int
- base_size: int
- image_size: int
- crop_mode: bool
- max_new_tokens: int
- doc_kind: Literal["text_heavy", "mixed_layout", "image_rich", "synthetic"]
- num_pages: int

### DeepSeekOCRAnalyticModel (contract view)

- report_id: str
- model: DeepSeekOCRModelSpec
- workload: OCRWorkloadProfile
- modules: list[AnalyticModuleNode]
- operator_categories: list[OperatorCategory]
- module_metrics: list[ModuleMetricsSnapshot]
- profile_run_id: str | None  — Optional profiling run id used only when runtime comparison is enabled in a later phase.

### AnalyticModuleNode (contract view)

- module_id: str
- name: str
- qualified_class_name: str
- stage: Literal["vision", "projector", "prefill", "decode", "other"]
- parent_id: str | None
- children: list[str]
- repetition: Literal["none", "for", "parfor"]
- repetition_count: int | None

### OperatorCategory (contract view)

- category_id: str
- display_name: str
- description: str
- match_classes: list[str]

### ModuleMetricsSnapshot (contract view)

- module_id: str
- profile_id: str
- calls: int
- total_time_ms: float
- total_flops_tflops: float
- total_io_tb: float
- memory_weights_gb: float
- memory_activations_gb: float
- memory_kvcache_gb: float
- share_of_model_time: float
- operator_breakdown: list[OperatorMetrics]

### OperatorMetrics (contract view)

- category_id: str
- calls: int
- flops_tflops: float
- io_tb: float
- share_of_module_flops: float

## Attrs Models (reference implementation)

```python
from __future__ import annotations
from typing import Literal
from attrs import define, field
from attrs.validators import instance_of


@define(kw_only=True)
class DeepSeekOCRAnalyticRequest:
    """Request to build or refresh the DeepSeek-OCR analytic model."""

    model_id: str = field(validator=[instance_of(str)], metadata={"help": "Canonical model id (e.g., deepseek-ai/DeepSeek-OCR)"})
    model_variant: str = field(validator=[instance_of(str)], metadata={"help": "Internal model variant (e.g., deepseek-ocr-v1-base)"})
    workload_profile_id: str = field(validator=[instance_of(str)], metadata={"help": "Workload profile id (e.g., dsocr-standard-v1)"})
    profile_run_id: str | None = field(default=None, metadata={"help": "Optional Stage1/Stage2 profiling run id"})
    force_rebuild: bool = field(default=False, validator=[instance_of(bool)], metadata={"help": "Force recomputation even if cached"})


@define(kw_only=True)
class DeepSeekOCRAnalyticAccepted:
    report_id: str = field(validator=[instance_of(str)])
    status: Literal["queued", "running"] = field(validator=[instance_of(str)])


@define(kw_only=True)
class AnalyticModuleSummary:
    module_id: str = field(validator=[instance_of(str)])
    name: str = field(validator=[instance_of(str)])
    stage: Literal["vision", "projector", "prefill", "decode", "other"] = field(validator=[instance_of(str)])
    share_of_model_time: float = field(validator=[instance_of(float)])
    total_time_ms: float = field(validator=[instance_of(float)])
    total_flops_tflops: float = field(validator=[instance_of(float)])
    memory_activations_gb: float = field(validator=[instance_of(float)])


@define(kw_only=True)
class DeepSeekOCRAnalyticReportSummary:
    report_id: str = field(validator=[instance_of(str)])
    model_variant: str = field(validator=[instance_of(str)])
    workload_profile_id: str = field(validator=[instance_of(str)])
    predicted_total_time_ms: float = field(validator=[instance_of(float)])
    measured_total_time_ms: float | None = field(default=None, metadata={"help": "Optional measured runtime; typically None in this development stage"})
    predicted_vs_measured_ratio: float | None = field(default=None, metadata={"help": "Optional predicted/measured ratio; reserved for future runtime comparison"})
    top_modules: list[AnalyticModuleSummary] = field(factory=list)
    notes: str = field(validator=[instance_of(str)])
```

## Serialization with cattrs

Use `cattrs` to convert to/from plain dicts for JSON IO, similar to Stage 1 contracts:

```python
import json
from cattrs import Converter

converter = Converter()

req = DeepSeekOCRAnalyticRequest(
    model_id="deepseek-ai/DeepSeek-OCR",
    model_variant="deepseek-ocr-v1-base",
    workload_profile_id="dsocr-standard-v1",
)

payload = converter.unstructure(req)
json_payload = json.dumps(payload)
roundtrip_obj = converter.structure(payload, DeepSeekOCRAnalyticRequest)
```

## Conventions

- Follow the `attrs` guidance in `magic-context/instructions/attrs-usage-guide.md`.
- Use `@define(kw_only=True)` and `field(..., metadata={"help": "..."})`.
- Keep field names stable and snake_case to match OpenAPI properties.
- When extending contracts, add optional fields with sensible defaults to preserve compatibility.
