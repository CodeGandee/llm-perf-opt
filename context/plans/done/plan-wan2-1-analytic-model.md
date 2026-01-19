# Plan: Wan2.1 Analytic Model (ModelMeter-style)

## HEADER
- **Purpose**: Implement a DeepSeek-OCR-style analytic model for Wan2.1 (starting with Wan2.1-T2V-14B) using ModelMeter `BaseLayer` + `StageCostMixin`, with Hydra configs and reusable analytic report data models.
- **Status**: Done
- **Date**: 2026-01-16
- **Last updated**: 2026-01-19
- **Completed**: 2026-01-19
- **Dependencies**:
  - extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py
  - extern/modelmeter/models/deepseek_ocr/configs/deepseek_ocr.yaml
  - extern/modelmeter/models/common/stage_cost.py
  - src/llm_perf_opt/runners/dsocr_analyzer.py
  - src/llm_perf_opt/data/deepseek_ocr_analytic.py
  - models/wan2.1-t2v-14b/source-data/config.json
- **Target**: Developers implementing analytic performance models and Stage-1/Stage-2 profiling integrations.

---

## 1. Purpose and Outcome

Implement a first-class analytic model for Wan2.1 that mirrors the DeepSeek-OCR analytic approach: composable `BaseLayer` modules, Hydra-driven configs, and structured report artifacts. Success looks like:

- A `modelmeter.models.wan2_1` package that can instantiate an analytic Wan2.1-T2V-14B model via Hydra and return consistent FLOPs/I/O/memory estimates for representative workloads.
- Shared analytic report data structures that are reusable across DeepSeek-OCR and Wan2.1, minimizing duplicate schema code while allowing model-specific extensions.
- A lightweight “analysis runner” (initially analogous to `DeepseekOCRStaticAnalyzer`, but Wan-focused) that produces a `*.json` report and optional Markdown summaries under `tmp/profile-output/<run_id>/static_analysis/`.
- A verification strategy that checks internal invariants (shape math, scaling with resolution/frames/steps) and (optionally) compares selected submodule FLOPs against a PyTorch reference when feasible.

Assumptions (explicit, revise as needed):

- v1 targets Wan2.1-T2V-14B “diffusion transformer” core (DiT-like blocks) as the primary cost driver, with optional modeling of text encoder and VAE encode/decode as coarse-grained layers.
- Workload knobs we support in v1: `num_frames`, `height`, `width`, `num_inference_steps`, `text_len` (default 512 from `config.json`), and batch size.
- Wan diffusion does not use KV-cache in the LLM sense; any attention caching is either absent or not modeled (KV memory should be 0 for the diffusion blocks in the analytic model).

## 2. Implementation Approach

### 2.1 High-level flow

1. Introduce a new ModelMeter analytic package `extern/modelmeter/models/wan2_1/` that mirrors the DeepSeek-OCR structure: `layers/`, `configs/`, and `scripts/verify/`.
2. Create a small set of analytic layers that capture Wan2.1 compute structure with existing ModelMeter layer primitives wherever possible (e.g., `Linear`, `SelfAttention`, `RMSNorm`), adding Wan-specific layers only when necessary to model missing behaviors (e.g., modulation/AdaLN variants, video tokenization geometry).
3. Define Hydra configs under `extern/modelmeter/models/wan2_1/configs/` with groups similar to DeepSeek-OCR: `hf/` (architecture metadata), `runtime/` (workload knobs), `transformer/` (block prototype + depth), `text_encoder/`, `vae/`, and `model/` (root aggregator).
4. Refactor analytic report data models into shared modules so both DeepSeek-OCR and Wan2.1 reuse the same `AnalyticModuleNode`, `OperatorCategory`, `ModuleMetricsSnapshot`, `OperatorMetrics`, and `TargetOperatorList` types; keep model-specific `ModelSpec` and `WorkloadProfile` types separate.
5. Add a Wan-focused analyzer entrypoint in `src/llm_perf_opt/runners/` that instantiates the Wan analytic model, traverses its module tree, emits a structured report, and optionally writes a short Markdown summary (mirroring the output conventions of DeepSeek-OCR static analysis).

### 2.2 Sequence diagram (steady-state usage)

```mermaid
sequenceDiagram
  participant Dev as Developer<br/>CLI user
  participant HY as Hydra<br/>compose+instantiate
  participant MM as ModelMeter<br/>wan2_1 analytic
  participant AM as Wan2.1<br/>AnalyticModel
  participant AN as llm_perf_opt<br/>WanAnalyzer
  participant OUT as tmp/profile-output<br/>static_analysis

  Dev->>AN: run wan analysis<br/>(hydra overrides)
  AN->>HY: compose config<br/>(wan2_1_t2v_14b)
  HY->>MM: instantiate cfg.model
  MM->>AM: build layer tree<br/>(tokens, steps, dims)
  AN->>AM: evaluate costs<br/>(per-module + total)
  AN->>OUT: write report.json<br/>+ summary.md
```

## 3. Files to Modify or Add

- **extern/modelmeter/models/wan2_1/__init__.py** add package exports and config-root helpers (mirrors deepseek_ocr).
- **extern/modelmeter/models/wan2_1/layers/** new analytic layers for Wan components (root aggregator, transformer block, optional text encoder and VAE coarse layers).
- **extern/modelmeter/models/wan2_1/configs/** Hydra config root for Wan2.1 analytic model composition (entry config + config groups).
- **extern/modelmeter/models/__init__.py** export Wan model(s) if we want them importable from `modelmeter.models` (optional, consistent with other models).
- **src/llm_perf_opt/data/analytic_common.py** new shared analytic data models extracted from `deepseek_ocr_analytic.py`.
- **src/llm_perf_opt/data/deepseek_ocr_analytic.py** refactor to import shared types from `analytic_common.py` (API-compatible re-exports as needed).
- **src/llm_perf_opt/data/wan2_1_analytic.py** Wan-specific `ModelSpec` and `WorkloadProfile` plus `AnalyticModelReport` specialization using shared node/metric types.
- **src/llm_perf_opt/runners/wan2_1_analyzer.py** new runner that instantiates Wan analytic model and writes artifacts under `static_analysis/`.
- **tests/unit/** add unit tests for Wan analytic geometry and scaling behavior, plus tests covering shared analytic data models migration.
- **docs/** (optional) short how-to for running Wan analytic analysis (similar to DeepSeek-OCR docs/hints).

## 4. TODOs (Implementation Steps)

- [X] **Define Wan2.1 scope v1** core-only (diffusion transformer) analytic scope with workloads `wan2-1-ci-tiny`, `wan2-1-512p`, `wan2-1-720p` (see `specs/004-wan2-1-analytic-model/research.md:1` and `specs/004-wan2-1-analytic-model/quickstart.md:1`).
- [X] **Add shared analytic data models** implemented in `src/llm_perf_opt/data/analytic_common.py:1` with DeepSeek-OCR refactor in `src/llm_perf_opt/data/deepseek_ocr_analytic.py:1`.
- [X] **Create Wan model spec/workload models** implemented in `src/llm_perf_opt/data/wan2_1_analytic.py:1` and contract types in `src/llm_perf_opt/contracts/models.py:1`.
- [X] **Implement Wan token geometry helper** implemented in `extern/modelmeter/models/wan2_1/layers/geometry.py:1` with tests in `tests/unit/wan2_1/test_geometry.py:1`.
- [X] **Implement Wan analytic layers (minimal set)** implemented under `extern/modelmeter/models/wan2_1/layers/core/wan2_1_dit_model.py:1` and `extern/modelmeter/models/wan2_1/layers/transformer/wan2_1_transformer_block.py:1`.
  - Root aggregator layer (`Wan2_1T2VModel`) using `StageCostMixin` and composing sublayers.
  - Transformer block prototype layer(s) modeling per-layer attention and FFN using existing primitives (`SelfAttention`, `Linear`, `RMSNorm`, optional modulation layer).
  - Step-scaling wrapper for diffusion iterations (total cost = per-step cost × `num_inference_steps`; memory should not scale with steps).
- [X] **Hydra configs for Wan2.1** implemented under `extern/modelmeter/models/wan2_1/configs/wan2_1_t2v_14b.yaml:1` and group dirs under `extern/modelmeter/models/wan2_1/configs/:1`.
- [X] **Wan analyzer runner** implemented in `src/llm_perf_opt/runners/wan2_1_analyzer.py:1` (writes `report.json` + `summary.md` under `tmp/profile-output/<run_id>/static_analysis/wan2_1/`).
- [X] **Verification hooks (v1)** implemented via invariants and tests in `tests/unit/wan2_1/test_report_invariants.py:1`, `tests/unit/wan2_1/test_hotspots.py:1`, and integration checks in `tests/integration/wan2_1/test_wan2_1_analyzer_report.py:1`.
  - Costs are non-negative and finite.
  - FLOPs and I/O scale linearly with `num_inference_steps`.
  - FLOPs scale with token count and layer count as expected.
  - KV-cache memory is 0 (or explicitly justified) for diffusion blocks.
- [X] **Optional reference verification** implemented under `extern/modelmeter/models/wan2_1/scripts/verify/run_verify_layers.py:1` and `extern/modelmeter/models/wan2_1/scripts/verify/run_verify_end2end.py:1`.
- [X] **Unit tests** implemented under `tests/unit/data/:1`, `tests/unit/utils/:1`, and `tests/unit/wan2_1/:1`, plus integration tests under `tests/integration/wan2_1/:1`.
- [X] **Docs** updated in `docs/running.md:1`, `specs/004-wan2-1-analytic-model/quickstart.md:1`, and `extern/modelmeter/models/wan2_1/scripts/verify/README.md:1`.
