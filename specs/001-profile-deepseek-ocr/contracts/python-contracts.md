# Python Contracts Design (Profiling API)

This document specifies Python-oriented contracts that mirror the OpenAPI schemas in `openapi.yaml` for LLM profiling requests and responses. Prefer `attrs` for Python data models per the repository convention; use `cattrs` for (de)serialization.

## Goals
- Provide strongly-typed, documented `attrs` models for request/response payloads
- Keep names generic (no Stage1… types) while reflecting Stage 1 capabilities
- Enforce absolute-path requirements where applicable
- Keep stable field names and support forward-compatible extensions via optional fields

## Package Location

- Recommended module path for implementation: `src/llm_perf_opt/contracts/models.py`
- These schemas map 1:1 to `components.schemas` entries in OpenAPI.

## Schemas

### LLMProfileRequest
- model_path: str — Absolute path or local HF repo directory for DeepSeek‑OCR
- input_dir: str — Absolute path to image directory (10–20 images)
- repeats: int = 3 — Number of repeated passes for aggregation (>=1)
- use_flash_attn: bool = True — Enable FlashAttention where available
- device: str = "cuda:0" — Target device
- max_new_tokens: int = 64 — Cap for text generation fallback (>=1)

### LLMProfileAccepted
- run_id: str — Server-assigned identifier for the run
- status: Literal["queued", "running"] — Queuing status
- artifacts_dir: str — Absolute path where artifacts will be written (under repo `tmp/`)

### LLMProfileReportSummary
- run_id: str — Identifier matching the accepted response
- mfu_model_level: float — Overall MFU ratio (0..1+)
- mfu_per_stage: Dict[str, float] — Per-stage MFU (e.g., {"prefill": 0.12, "decode": 0.31})
- top_operators: list[OperatorSummary]
- aggregates: Dict[str, Stats] — Aggregates like mean/std for timings and throughput
- notes: str — Stakeholder summary

#### OperatorSummary
- op_name: str
- total_time_ms: float
- cuda_time_ms: float
- calls: int

#### Stats
- mean: float
- std: float

## Attrs Models (reference implementation)

```python
from __future__ import annotations
from typing import Dict, Literal
from attrs import define, field
from attrs.validators import instance_of
import os

def _abs_path(_: object, attr: "attrs.Attribute[str]", value: str) -> None:  # type: ignore[name-defined]
    if not os.path.isabs(value):
        raise ValueError(f"{attr.name} must be an absolute path")

def _ge(min_value: int):
    def _validator(_: object, attr: "attrs.Attribute[int]", value: int) -> None:  # type: ignore[name-defined]
        if value < min_value:
            raise ValueError(f"{attr.name} must be >= {min_value}")
    return _validator

@define(kw_only=True)
class LLMProfileRequest:
    """Profile an LLM run over an image set.

    Fields align with OpenAPI LLMProfileRequest. Paths must be absolute.
    """
    model_path: str = field(validator=[instance_of(str), _abs_path], metadata={"help": "Absolute path or local HF repo"})
    input_dir: str = field(validator=[instance_of(str), _abs_path], metadata={"help": "Absolute path to image directory (10–20 images)"})
    repeats: int = field(default=3, validator=[instance_of(int), _ge(1)], metadata={"help": ">=1 repetitions for aggregation"})
    use_flash_attn: bool = field(default=True, validator=[instance_of(bool)], metadata={"help": "Enable FlashAttention if available"})
    device: str = field(default="cuda:0", validator=[instance_of(str)], metadata={"help": "Target device (e.g., cuda:0)"})
    max_new_tokens: int = field(default=64, validator=[instance_of(int), _ge(1)], metadata={"help": "Max new tokens for fallback generation"})

@define(kw_only=True)
class LLMProfileAccepted:
    run_id: str = field(validator=[instance_of(str)])
    status: Literal["queued", "running"] = field(validator=[instance_of(str)])
    artifacts_dir: str = field(validator=[instance_of(str), _abs_path])

@define(kw_only=True)
class OperatorSummary:
    op_name: str = field(validator=[instance_of(str)])
    total_time_ms: float = field(validator=[instance_of(float)])
    cuda_time_ms: float = field(validator=[instance_of(float)])
    calls: int = field(validator=[instance_of(int), _ge(0)])

@define(kw_only=True)
class Stats:
    mean: float = field(validator=[instance_of(float)])
    std: float = field(validator=[instance_of(float)])

@define(kw_only=True)
class LLMProfileReportSummary:
    run_id: str = field(validator=[instance_of(str)])
    mfu_model_level: float = field(validator=[instance_of(float)])
    mfu_per_stage: Dict[str, float] = field(factory=dict)
    top_operators: list[OperatorSummary] = field(factory=list)
    aggregates: Dict[str, Stats] = field(factory=dict)
    notes: str = field(validator=[instance_of(str)])
```

## Serialization with cattrs

Use `cattrs` to convert to/from plain dicts for JSON IO:

```python
import json
from cattrs import Converter

converter = Converter()

req = LLMProfileRequest(
    model_path="/abs/path/models/deepseek-ocr",
    input_dir="/abs/path/data/samples",
)

# dict for API call
payload = converter.unstructure(req)
json_payload = json.dumps(payload)

# back to objects from a dict
roundtrip_obj = converter.structure(payload, LLMProfileRequest)
```

## Conventions
- Follow `attrs` guide in `magic-context/instructions/attrs-usage-guide.md`
- Use `@define(kw_only=True)` and `field(..., metadata={"help": "..."})`
- Validate absolute paths explicitly; validate numeric ranges with custom validators
- Keep field names stable and snake_case to match OpenAPI properties

## Mapping to OpenAPI
- LLMProfileRequest ↔ components.schemas.LLMProfileRequest
- LLMProfileAccepted ↔ components.schemas.LLMProfileAccepted
- LLMProfileReportSummary ↔ components.schemas.LLMProfileReportSummary

## Notes
- Prefer `attrs` for domain and contract models; use Pydantic only when required by a web framework.
- When adding detail (e.g., operator memory footprint), add optional fields with sensible defaults to preserve compatibility.
