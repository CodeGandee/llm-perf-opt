# Data Model: DeepSeek‑OCR Stage 1 Profiling

This document defines domain entities, fields, and relationships for Stage 1 artifacts. Models should be implemented with `attrs` for internal data. Web/contract schemas align with `specs/001-profile-deepseek-ocr/contracts/python-contracts.md`.

## Entities

1) ModelUnderTest
- id: string (human name or path)
- model_path: string (absolute path or HF id resolved locally)
- dtype: enum {bf16, fp16, fp32}
- device: string (e.g., cuda:0)
- use_flash_attn: bool
- n_layers: int (if available)
- d_model: int (if available)
- d_ff: int (if available)
- n_heads: int (if available)
- encoder_type: enum {ViT, CNN, Unknown}

2) InputImage
- path: string (absolute)
- width: int
- height: int
- category: enum {text_heavy, mixed_layout, image_rich, unknown}

3) InputSet
- id: string (name)
- root_dir: string (absolute)
- count: int
- images: list[InputImage]

4) StageTiming
- stage: enum {prefill, decode}
- elapsed_ms: float
- tokens: int (generated tokens for decode; proxy tokens for prefill if used)
- throughput_toks_per_s: float

5) OperatorSummary (contracts-aligned)
- op_name: string
- total_time_ms: float
- cuda_time_ms: float
- calls: int

6) RunMeta
- run_id: string (uuid)
- created_at: datetime (iso8601)
- repeats: int
- repo_root: string (absolute)
- gpu_name: string
- compute_capability: string
- peak_tflops_bf16: float
- torch_version: string
- transformers_version: string
- cuda_version: string

7) Stats (contracts-aligned)
- mean: float
- std: float

8) LLMProfileReport (domain-rich)
- run_id: string (uuid)
- model: ModelUnderTest
- input_set: InputSet
- timings: list[StageTiming]
- operators_topk: list[OperatorSummary] (sorted by total_time_ms)
- mfu_model_level: float (0..1)
- mfu_per_stage: dict[stage -> float]
- aggregates: dict[metric -> {mean: float, std: float}]
- notes: string (stakeholder summary)
- meta: RunMeta

9) LLMProfileReportSummary (contracts)
- run_id: string
- mfu_model_level: float
- mfu_per_stage: dict[stage -> float]
- top_operators: list[OperatorSummary]
- aggregates: dict[metric -> Stats]
- notes: string

## Relationships

- LLMProfileReport references exactly one ModelUnderTest and one InputSet.
- StageTiming has one of two stages and contributes to `mfu_per_stage`.
- OperatorSummary entries are used for top‑K listings; attribution to stages may be inferred from NVTX context if available.

## Validation Rules

- `elapsed_ms > 0`, `tokens >= 0`, `throughput_toks_per_s >= 0`.
- `images` in InputSet must be non‑empty (10–20 recommended).
- `mfu_*` in [0, 1.5] (guardrails; >1 implies estimator error).
- Paths must be absolute.

## Contracts Alignment

Contracts in `/specs/001-profile-deepseek-ocr/contracts/` define request/response shapes using the following names (see `python-contracts.md`):
- LLMProfileRequest
- LLMProfileAccepted
- LLMProfileReportSummary

Domain models above intentionally use the same naming for shared concepts (OperatorSummary, Stats) and introduce `LLMProfileReport` for the full internal artifact. Conversion between domain and contracts should be done via `cattrs`.
