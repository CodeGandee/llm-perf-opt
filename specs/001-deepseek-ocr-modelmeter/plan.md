# Implementation Plan: DeepSeek-OCR Analytic Modeling in ModelMeter

**Branch**: `[001-deepseek-ocr-modelmeter]` | **Date**: 2025-11-18 | **Spec**: [/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/spec.md]
**Input**: Feature specification from `/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement an analytic performance and memory model for DeepSeek-OCR within ModelMeter so that internal engineers can
generate human-readable and machine-readable breakdowns of time, memory, and operator mix across the model’s major
components for a standard, parameterized OCR workload.

The primary implementation surface is the `BaseLayer` analytic interface in
`extern/modelmeter/layers/base.py`. For each target DeepSeek-OCR layer or operator (derived from the TorchInfo/static
analysis artifacts), we will implement a dedicated class under
`extern/modelmeter/models/deepseek_ocr/layers/` that subclasses `BaseLayer` and provides closed-form theoretical
implementations of all required metrics (forward/backward Tensor Core and CUDA core FLOPs, I/O volume, arithmetic
intensity, and weight/activation/KV-cache memory footprints).

These DeepSeek-OCR-specific analytic layer classes collectively map DeepSeek-OCR modules and PyTorch operators into
ModelMeter’s analytic layer space, using the TorchInfo-derived operator list and callgraph artifacts in
`/workspace/code/llm-perf-opt/reports/20211117-dsorc-op-analysis/static-20251118-130533/` as the primary static
reference, and integrate with existing runners in `src/llm_perf_opt/runners` to keep measurement flows consistent.

The primary outputs of the implemented scripts will be:
- structured analytic model artifacts (JSON and/or YAML) that encode the DeepSeek-OCR module hierarchy, per-module and
  per-operator metrics, and workload metadata; and
- human-readable Markdown documentation for the analyzed layers/operators, explaining each layer’s definition, the
  theoretical FLOPs/I/O/memory formulas implemented in the corresponding `BaseLayer` subclass, and the modeling
  assumptions and rationale.

## Technical Context

<!-- Technical context resolved in Phase 0; see research.md for supporting rationale. -->

**Language/Version**: Python 3.11 (Pixi-managed environment, CUDA 12.4 toolchain via `pixi run -e rtx5090 …`)
**Primary Dependencies**: PyTorch, ModelMeter (`extern/modelmeter`), Hydra/omegaconf, attrs, TorchInfo
**Storage**: Filesystem-only artifacts under `reports/` and `tmp/profile-output/` (no external database or message bus)
**Testing**: pytest for unit/integration tests plus manual flows under `/workspace/code/llm-perf-opt/tests/manual`; add
feature-specific manual scripts for DeepSeek-OCR analytic reports and export paths
**Target Platform**: Linux server with NVIDIA GPU (RTX 5090-class) using the `rtx5090` Pixi environment
**Project Type**: Single Python package (`src/llm_perf_opt`) with external analytic models in `extern/modelmeter`
**Performance Goals**: Provide a theoretically grounded analytic model for a single DeepSeek-OCR forward pass
(`forward()`) that estimates FLOPs, I/O, and memory footprint per module and operator category, such that internal
experts can reason about performance without executing any additional profiling runs
**Constraints**: Theoretical analysis is implemented as a separate module that does not run as part of Stage 1/Stage 2
profiling and does not change profiling overhead; no automated numeric comparison against measured runtimes or hardware
scheduling effects is required in this development stage
**Scale/Scope**: Single-model integration (`deepseek-ocr-v1-base`) with a focus on forward-pass analytic estimates
only; runtime scheduling/overhead/optimization effects are explicitly out of scope

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Pythonic clarity & docstrings: All new/changed public APIs/classes include
  NumPy‑style docstrings and illustrative examples.
- Typed, linted, formatted: All new code is fully type‑annotated; `mypy` passes
  with zero errors; `ruff` lint/format passes with zero errors.
- OO discipline for functional classes: Member vars prefixed with `m_`; use
  properties for read‑only access and `set_xxx()` for mutations; constructors
  argument‑free with `from_xxx()` factories where needed.
- Data models: Use `attrs` (default) with `@define(kw_only=True)` and `field`
  metadata, or `pydantic` for web schemas; no business logic in models.
- Runtime environment declared: Execution context specified (Pixi preferred;
  otherwise venv). Commands in this plan use that environment explicitly.
- Testing plan: Manual tests provided for major functionality with file paths
  under `tests/manual/…`. Automated tests only if requested or critical; specify
  locations under `tests/unit/…` and `tests/integration/…` if included.

Gate Evaluation (pre-design): PASS. No Constitution violations are expected for this feature: we will use Pixi-managed
Python 3.11 environments, fully type-annotated and `ruff`/`mypy`-clean code, attrs-based data models, and manual tests
for major DeepSeek-OCR analytic workflows, with automated tests added only where critical.

Gate Evaluation (post-design): PASS. Phase 1 artifacts (research, data models, contracts, quickstart, and agent
context updates) align with the constitution: data models are attrs-based and data-only, runtime assumptions are
expressed via Pixi commands, and no unresolved clarifications or justified exceptions remain.

## Project Structure

### Documentation (this feature)

```text
/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/
├── plan.md              # Implementation plan for DeepSeek-OCR analytic modeling (this file)
├── research.md          # Phase 0 research output (clarifications, decisions, alternatives)
├── data-model.md        # Phase 1 data model for DeepSeek-OCR analytic entities
├── quickstart.md        # Phase 1 workflow and usage guide for this feature
├── contracts/           # Phase 1 API/contract definitions (e.g., OpenAPI or schema files)
└── tasks.md             # Phase 2 task breakdown (owned by /speckit.tasks, not this command)
```

### Source Code (repository root)

```text
/workspace/code/llm-perf-opt/
├── extern/modelmeter/
│   ├── layers/                      # Generic analytic layer interfaces and reusable ops (e.g., BaseLayer)
│   ├── models/
│   │   └── deepseek_ocr/
│   │       ├── layers/              # DeepSeek-OCR-specific analytic BaseLayer subclasses for target ops
│   │       └── README.md
│   └── analysis/
├── src/llm_perf_opt/
│   ├── runners/
│   │   ├── dsocr_analyzer.py        # DeepSeek-OCR static analysis and helper entry points (optional integration)
│   │   ├── dsocr_session.py         # DeepSeek-OCR session orchestration
│   │   └── deep_profile_runner.py   # Stage-2 profiling flows (unchanged; not extended by this feature)
│   ├── data/                        # Shared attrs-based data models for profiling and analytic artifacts
│   ├── contracts/                   # Contract conversion utilities for analytic/profiling outputs
│   ├── profiling/                   # Existing Nsight/torch-profiler orchestration and export logic
│   ├── utils/                       # Utilities (e.g., TorchInfo export, callgraph parsing)
│   └── visualize/                   # Visualization helpers for annotated timelines and summaries
├── scripts/analytical/              # Helper scripts to derive static information (TorchInfo, callgraphs, etc.)
│   └── dsocr_find_static_components.py
├── conf/model/deepseek_ocr/         # Hydra configs for DeepSeek-OCR architecture, inference, and outputs
├── reports/20211117-dsorc-op-analysis/
│   └── static-20251118-130533/      # TorchInfo-derived operator list and callgraph artifacts
└── tests/
    ├── unit/                        # Existing unit tests (e.g., kernels export, stakeholder summaries)
    └── manual/
        ├── inference/               # Manual DeepSeek-OCR inference workflows
        ├── ncu/                     # Manual Nsight Compute runs
        └── stage2_profile/          # Manual deep profiling flows
```

**Structure Decision**: Single Python package rooted at `/workspace/code/llm-perf-opt/src/llm_perf_opt` with external
analytic models in `/workspace/code/llm-perf-opt/extern/modelmeter`. This feature primarily extends
`extern/modelmeter/layers/base.py` and `extern/modelmeter/models/deepseek_ocr/layers/` to implement theoretical
forward/backward analytic layers for each target DeepSeek-OCR operator. The `src/llm_perf_opt` package may contribute
reusable helpers (data models, contracts, static analysis utilities) that can support other models in future, but it
is not the main implementation surface for this feature. Helper scripts under
`/workspace/code/llm-perf-opt/scripts/analytical/` (for example, `dsocr_find_static_components.py`) are used to extract
static information from the model but are considered supporting tooling rather than core implementation.

### Target TorchInfo layers → DeepSeek-OCR layer directories

For this feature, the target layer list comes from
`reports/20211117-dsorc-op-analysis/static-20251118-130533/torchinfo-unique-layers.{md,json}` (`num_unique_layers = 29`).
The JSON `parents` field exposes a natural hierarchy:

- `DeepseekOCRModel` is the root model.
- Vision encoder / projector modules live under `transformers_modules.<sha>.deepencoder.*`.
- Decoder / language-model modules live under `transformers_modules.<sha>.modeling_deepseekv2.*`.
- LLaMA attention primitives live under `transformers.models.llama.modeling_llama.*`.

We reflect this hierarchy in the DeepSeek-OCR analytic layer package by grouping `BaseLayer` subclasses into
subdirectories under `extern/modelmeter/models/deepseek_ocr/layers/`:

```text
extern/modelmeter/models/deepseek_ocr/layers/
├── __init__.py
├── core/
│   └── deepseek_ocr_model.py         # DeepseekOCRModel(BaseLayer)
├── vision/
│   ├── attention.py                  # Attention(BaseLayer)
│   ├── block.py                      # Block(BaseLayer)
│   ├── clip_vision_embeddings.py     # CLIPVisionEmbeddings(BaseLayer)
│   ├── image_encoder_vit.py          # ImageEncoderViT(BaseLayer)
│   ├── layer_norm2d.py               # LayerNorm2d(BaseLayer)
│   ├── mlp_block.py                  # MLPBlock(BaseLayer)
│   ├── mlp_projector.py              # MlpProjector(BaseLayer)
│   ├── notp_attention.py             # NoTPAttention(BaseLayer)
│   ├── notp_feedforward.py           # NoTPFeedForward(BaseLayer)
│   ├── notp_transformer.py           # NoTPTransformer(BaseLayer)
│   ├── notp_transformer_block.py     # NoTPTransformerBlock(BaseLayer)
│   ├── patch_embed.py                # PatchEmbed(BaseLayer)
│   └── vit_model.py                  # VitModel(BaseLayer)
├── decoder/
│   ├── deepseek_v2_decoder_layer.py  # DeepseekV2DecoderLayer(BaseLayer)
│   ├── deepseek_v2_mlp.py            # DeepseekV2MLP(BaseLayer)
│   ├── deepseek_v2_moe.py            # DeepseekV2MoE(BaseLayer)
│   ├── deepseek_v2_rms_norm.py       # DeepseekV2RMSNorm(BaseLayer)
│   └── moe_gate.py                   # MoEGate(BaseLayer)
└── llama/
    ├── llama_flash_attention2.py     # LlamaFlashAttention2(BaseLayer)
    └── llama_rotary_embedding.py     # LlamaRotaryEmbedding(BaseLayer)
```

- File names use `snake_case` and class names mirror the original module names (CamelCase), consistent with
  `extern/modelmeter/layers/`.
- The `core/`, `vision/`, `decoder/`, and `llama/` folders correspond directly to the high-level parents observed in
  `torchinfo-unique-layers.json` (`DeepseekOCRModel`, `deepencoder.*`, `modeling_deepseekv2.*`, and
  `modeling_llama.*` respectively).

PyTorch builtins from the TorchInfo layer list (`GELU`, `SiLU`, `ModuleList`, `Sequential`, `Conv2d`, `Linear`,
`LayerNorm`, `Embedding`) will be analyzed via shared analytic implementations in `extern/modelmeter/layers/` rather
than DeepSeek-specific subclasses. The DeepSeek-OCR analytic model will still reference these operator categories and
export per-operator metrics and Markdown docs (for example, `layers/linear.md`, `layers/conv2d.md`, `layers/gelu.md`)
but the underlying formulas will be defined in reusable generic layers, not duplicated under
`extern/modelmeter/models/deepseek_ocr/layers/`.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No Constitution violations are currently anticipated for this feature; this section will remain empty unless a justified
exception to the gates in the Constitution Check is introduced in later phases.

## Phase 0: Outline & Research

Unknowns (from Technical Context):
- Acceptable overhead budget for additional analytic modeling work on top of Stage 1/Stage 2.
- Scope of supported DeepSeek-OCR variants and how to handle future model changes.
- Standard OCR workload definition (synthetic parameters) and how it ties to existing analysis tooling.

Research Tasks:
- Define the `dsocr-standard-v1` workload profile (seq_len, base_size, image_size, crop_mode, max_new_tokens) using
  existing DeepSeek-OCR hints and analyzer defaults.
- Establish an explicit overhead budget (percentage and absolute cap) for analytic modeling relative to Stage 1/2.
- Decide how TorchInfo artifacts under `reports/20211117-dsorc-op-analysis/static-20251118-130533/` should be treated
  (canonical snapshot vs regenerated per run).
- Choose an integration pattern between `llm_perf_opt` runners and ModelMeter (`extern/modelmeter`).

Output: `/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/research.md` (decisions with rationale and
alternatives).

## Phase 1: Design & Contracts

Deliverables:
- Analytic layer catalog and formulas
  - Define the target operator/layer list for DeepSeek-OCR (from TorchInfo/static artifacts) that will be modeled via
    `BaseLayer` subclasses.
  - For each operator category (for example, convolution, linear, layer norm, attention, embedding), specify the
    closed-form formulas needed to implement all `BaseLayer` metrics: forward/backward Tensor Core and CUDA core FLOPs,
    forward/backward I/O volume, forward/backward arithmetic intensity, and forward/backward weight/activation/KV-cache
    memory footprints.
  - Document how DeepSeek-OCR-specific shapes and call counts will be passed into these layer classes.
- `/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/data-model.md`
  - Define `DeepSeekOCRModelSpec`, `OCRWorkloadProfile`, `AnalyticModuleNode`, `OperatorCategory`,
    `ModuleMetricsSnapshot`, `AnalyticModelReport`, and `TargetOperatorList` entities.
  - Align with attrs-based implementations to be added under `src/llm_perf_opt/data/`.
- `/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/contracts/openapi.yaml`
  - OpenAPI paths for starting an analytic modeling run and fetching summary/full analytic model payloads.
- `/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/contracts/python-contracts.md`
  - Python-side contract models (attrs) for requests/responses, aligned with OpenAPI and domain data models.
- `/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/contracts/MAPPING.md`
  - Mapping from API endpoints to local CLI behavior (Pixi commands and artifact locations).
- `/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/quickstart.md`
  - Environment, static TorchInfo regeneration, analytic modeling CLI sketch, and manual validation steps.
- Agent context:
  - Update agent-specific context (including `AGENTS.md`) via
    `.specify/scripts/bash/update-agent-context.sh codex` after Technical Context is finalized.

## Phase 2: Readiness Check (stop here)

Re-evaluate Constitution Gates after Phase 1:
- Confirm Technical Context has no remaining `NEEDS CLARIFICATION` markers.
- Confirm data models use attrs (or pydantic where web schemas require) with no embedded business logic.
- Confirm runtime environment and commands are expressed in Pixi form (no system Python).
- Confirm manual test locations for major functionality are documented under `tests/manual/deepseek_ocr/…`.

If any gate remains violated, justify it in Complexity Tracking or adjust the design before progressing to
implementation and `/speckit.tasks` task breakdown.
