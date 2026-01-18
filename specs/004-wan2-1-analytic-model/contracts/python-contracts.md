# Python Contracts Design (Wan2.1 Analytic Modeling)

This document specifies Python-oriented contracts that mirror the OpenAPI schemas in `/data1/huangzhe/code/llm-perf-opt/specs/004-wan2-1-analytic-model/contracts/openapi.yaml` for Wan2.1 analytic modeling. Prefer `attrs` for Python data models per the repository convention; use `cattrs` for (de)serialization.

## Goals

- Provide strongly typed, documented `attrs` models for analytic modeling requests and responses.
- Keep names stable and generic enough to allow future Wan2.1 variants.
- Align with the domain entities described in `../data-model.md` while exposing a summary-oriented view for callers.

## Package Location

- Recommended module path for implementation: `src/llm_perf_opt/contracts/wan2_1.py` (or a shared `src/llm_perf_opt/contracts/models.py` if the project prefers a single module).
- Schemas map 1:1 to `components.schemas` entries in `openapi.yaml`.

## Schemas

### Wan2_1AnalyticRequest

- model_id: str
- model_variant: str
- workload_profile_id: str
- overrides: dict[str, int] | None (batch_size, num_frames, height, width, num_inference_steps, text_len)
- profile_run_id: str | None
- force_rebuild: bool = False

### Wan2_1AnalyticAccepted

- report_id: str
- status: Literal["queued", "running"]

### Wan2_1ModelSpec (contract view)

- model_id: str
- model_variant: str
- config_path: str (absolute path)
- hidden_size: int
- num_layers: int
- num_attention_heads: int
- head_dim: int
- mlp_intermediate_size: int
- vae_downsample_factor: int
- patch_size: int
- latent_channels: int
- notes: str

### Wan2_1WorkloadProfile (contract view)

- profile_id: str
- description: str
- batch_size: int
- num_frames: int
- height: int
- width: int
- num_inference_steps: int
- text_len: int

### Wan2_1AnalyticModel (contract view)

- report_id: str
- model: Wan2_1ModelSpec
- workload: Wan2_1WorkloadProfile
- modules: list[AnalyticModuleNode]
- operator_categories: list[OperatorCategory]
- module_metrics: list[ModuleMetricsSnapshot]
- profile_run_id: str | None

## Attrs Models (reference implementation)

```python
from __future__ import annotations

from typing import Literal

from attrs import define, field
from attrs.validators import instance_of


@define(kw_only=True)
class Wan2_1AnalyticRequest:
    model_id: str = field(validator=[instance_of(str)])
    model_variant: str = field(validator=[instance_of(str)])
    workload_profile_id: str = field(validator=[instance_of(str)])
    overrides: dict[str, int] | None = field(default=None)
    profile_run_id: str | None = field(default=None)
    force_rebuild: bool = field(default=False, validator=[instance_of(bool)])


@define(kw_only=True)
class Wan2_1AnalyticAccepted:
    report_id: str = field(validator=[instance_of(str)])
    status: Literal["queued", "running"] = field(validator=[instance_of(str)])
```

## Serialization with cattrs

Use `cattrs` to convert to/from plain dicts for JSON IO:

```python
import json

from cattrs import Converter

converter = Converter()

req = Wan2_1AnalyticRequest(
    model_id="wan2.1-t2v-14b",
    model_variant="t2v-14b",
    workload_profile_id="wan2-1-ci-tiny",
)

payload = converter.unstructure(req)
json_payload = json.dumps(payload)
roundtrip_obj = converter.structure(payload, Wan2_1AnalyticRequest)
```
