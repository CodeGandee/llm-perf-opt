# Phase 0 Research: DeepSeek‑OCR Analytic Modeling in ModelMeter

This document resolves clarifications and captures best‑practice choices for the DeepSeek‑OCR analytic modeling feature.
It focuses on (a) overhead and scope unknowns in the Technical Context, (b) technology‑specific best practices, and
(c) integration patterns between `llm_perf_opt` and ModelMeter.

## Unknowns and Decisions

1) Additional overhead budget for analytic modeling

- Decision: Treat analytic modeling as a light‑weight layer on top of existing Stage 1/Stage 2 and static analysis
  flows. For the **standard OCR workload**, any new runtime instrumentation or data collection added specifically for
  analytic modeling MUST keep additional wall‑clock overhead **≤ 10%** of the baseline Stage 1/Stage 2 measurement
  runtime, with an absolute soft cap of **30 seconds** per run. Purely offline processing of static artifacts
  (e.g., TorchInfo JSON, callgraph exports) is not counted against this runtime budget but must still keep end‑to‑end
  report generation within the 5‑minute target in SC‑001.
- Rationale: The spec requires that analytic reports be available within 5 minutes and be usable alongside existing
  measurement workflows. A 10% overhead budget is tight enough to guard against heavy new tracing while leaving room
  for modest extra passes (e.g., module enumeration, serialization, or callgraph parsing).
- Alternatives considered:
  - ≤ 5% overhead: Stronger guarantee but unrealistic if we need occasional extra runs (e.g., reusing dynamic
    callgraph hooks) and may discourage useful instrumentation.
  - No explicit budget: Risks accidental 2–3× slowdowns when analytic modeling logic grows, conflicting with SC‑001.

2) Model variants and workload diversity assumptions

- Decision: For this feature, **DeepSeek‑OCR** refers to the vendor checkpoint `deepseek-ai/DeepSeek-OCR` loaded via
  `trust_remote_code=True` with the default model configuration. We name this internal variant
  `deepseek-ocr-v1-base`. All analytic modeling, calibration, and validation in this plan target this variant only.
  The data model will include fields for `model_id` and `model_variant` so future DeepSeek‑OCR variants (e.g., larger
  encoders or updated decoders) can be added without changing existing contracts, but they are out of scope for this
  feature. Workload diversity is handled by treating the “standard OCR workload” as a named profile plus optional
  alternative profiles; this feature is responsible for defining **one canonical standard profile** and ensuring the
  analytic model is calibrated for it.
- Rationale: The spec’s key entities and success criteria assume a stable model definition and a specific workload.
  Trying to support multiple DeepSeek‑OCR variants with a single analytic model in the first iteration would require
  additional calibration and validation effort that is not needed to unlock the primary value (P1/P2 stories).
- Alternatives considered:
  - Multi‑variant analytic model in one step: Higher upfront complexity and calibration burden; better handled as
    follow‑up work once the base pipeline is proven.
  - Treat DeepSeek‑OCR as architecture‑agnostic and ignore variants: Would make the analytic model misleading if the
    underlying architecture changes significantly.

3) Standard OCR workload parameterization

- Decision: Define a reproducible **standard OCR workload** as a synthetic profile based on the existing DeepSeek‑OCR
  analysis presets:
  - Logical document characteristics:
    - Single‑page document with mixed layout (text + simple graphics), moderate content density.
    - Target sequence length `seq_len = 512` tokens (including image tokens and text), matching the default
      `AnalysisConfig` used for dynamic tracing.
  - Image preprocessing parameters (aligned with the “Gundam” / analysis setting):
    - `base_size = 1024`
    - `image_size = 640`
    - `crop_mode = True` (dynamic crops enabled)
  - Generation settings:
    - Greedy decoding (`temperature = 0.0`).
    - `max_new_tokens` fixed to a standard value (e.g., 512) for analytic calibration; actual Stage 1/Stage 2 scripts
      may use larger limits as long as they log the effective generated length.
  - Workload identifier:
    - Canonical ID: `dsocr-standard-v1`.
  The workload will be expressed as an `OCRWorkloadProfile` entity (see `data-model.md`) and referenced by all analytic
  reports, rather than relying on pinned image files.
- Rationale: This profile ties directly to existing hints and analysis tools (`AnalysisConfig(seq_len=512, base_size=
  1024, image_size=640)` in `dsocr_analyzer`) while respecting the spec’s requirement for a fully parameterized,
  synthetic workload. Using a named profile allows future variants (e.g., multi‑page or higher‑resolution workloads)
  without invalidating current results.
- Alternatives considered:
  - Base settings without crops (`base_size=image_size=1024, crop_mode=False`): Simpler but less representative of
    complex document layouts where DeepSeek‑OCR shines.
  - Real image datasets: Explicitly rejected by the spec in favor of parameterized synthetic workloads.

4) Source of truth for target operator list and module hierarchy

- Decision: Use the static TorchInfo artifacts under
  `/workspace/code/llm-perf-opt/reports/20211117-dsorc-op-analysis/static-20251118-130533/` as the **single source of
  truth** for the target operator and module list:
  - `torchinfo-unique-layers.json` / `.md` define the set of module classes and their children.
  - `torchinfo-layers.json` / `torchinfo-stages.json` provide per‑stage breakdowns and shapes.
  The analytic model will treat this snapshot as canonical for `deepseek-ocr-v1-base`. Regeneration via
  `scripts/analytical/dsocr_find_static_components.py` is only required when DeepSeek‑OCR code changes (e.g., new
  modules or upgraded checkpoints); such regeneration should be opt‑in and documented, not part of the normal run
  path.
- Rationale: Rerunning heavy static analysis on every analytic modeling invocation would violate the overhead budget
  and introduce instability. A checked‑in, documented snapshot provides a stable contract for which modules and
  operators must be modeled.
- Alternatives considered:
  - Always rerun TorchInfo analysis: More “live” but high cost and brittle if vendor code changes.
  - Manually curated operator list: Easier to drift from the actual model implementation.

5) Mapping DeepSeek‑OCR modules to ModelMeter layers

- Decision: Model DeepSeek‑OCR with a **two‑tier mapping**:
  - Tier 1 (domain modules): Medium‑granularity modules such as `ImageEncoderViT`, `VitModel`, `MlpProjector`,
    `DeepseekV2DecoderLayer`, `DeepseekV2MoE`, etc., defined in a new
    `extern/modelmeter/models/deepseek_ocr/layers/` package. Each domain module exposes structured parameters
    (hidden sizes, number of heads, patch sizes, etc.) and composes underlying analytic primitives.
  - Tier 2 (primitives): Reuse existing ModelMeter primitives (`Linear`, `Embedding`, `MHAFlashattention`, `RMSNorm`,
    `SwiGLU`, etc.) for PyTorch built‑ins and well‑known custom layers. New primitives are only added when the operator
    cannot be expressed as a composition of existing ones without losing essential cost characteristics.
  The mapping from TorchInfo classes to ModelMeter layers will be maintained as a declarative table (e.g., JSON or
  Python mapping) rather than ad‑hoc conditionals scattered through the code.
- Rationale: A two‑tier approach aligns with the spec’s requirement for medium‑granularity modules with leaf nodes at
  built‑in/custom layers, while keeping the analytic code maintainable and reusable across models that share blocks.
- Alternatives considered:
  - Only primitive layers (no domain modules): Would make it hard to attribute costs to high‑level components (visual
    encoder, decoder block, projector) in reports.
  - Only domain modules (no primitives): Would prevent reusing ModelMeter primitives and make it difficult to reason
    about operator‑level breakdowns.

6) Integration pattern between `llm_perf_opt` and ModelMeter

- Decision: Keep analytic modeling as a **pure library integration** with clear boundaries:
  - `src/llm_perf_opt/runners/dsocr_analyzer.py` and related helpers are responsible for:
    - Obtaining static analysis artifacts (TorchInfo, callgraphs).
    - Building input structures (e.g., module hierarchies, operator counts) for the analytic model.
    - Converting analytic outputs into project data models (`attrs`‑based) under `src/llm_perf_opt/data/`.
  - `extern/modelmeter/models/deepseek_ocr/` is responsible for:
    - Implementing `DeepseekOCRAnalyticModel` and associated analytic layer classes.
    - Providing a small public API that accepts a `DeepSeekOCRConfig`‑like description and workload profile and
      returns module‑ and operator‑level cost estimates.
  - Contracts and user‑facing schemas (CLI/service) will live in `src/llm_perf_opt/contracts` and
    `specs/001-deepseek-ocr-modelmeter/contracts/`.
  No new cross‑package side effects (e.g., global state, environment mutation) are introduced; ModelMeter remains
  import‑only and stateless for a given configuration.
- Rationale: This separation leverages existing project structure (runners + data + contracts) and keeps ModelMeter
  reusable beyond `llm_perf_opt`. It also simplifies testing by allowing analytic models to be exercised with purely
  synthetic inputs.
- Alternatives considered:
  - Embedding analytic logic directly in runners: Faster to prototype but harder to reuse and test independently.
  - Exposing ModelMeter only via a CLI: Would complicate integration with existing Python workflows in `llm_perf_opt`.

7) Testing and validation strategy for analytic models

- Decision: For this development stage, validation will focus on theoretical analysis and manual (human) review rather
  than automated runtime comparison:
  - Synthetic consistency:
    - Unit‑level tests for domain analytic layers (e.g., DeepSeek decoder block) that verify monotonicity and scale
      with respect to key parameters (sequence length, hidden size, number of heads).
    - Golden‑value tests for small, toy configurations where FLOPs/IO can be derived analytically.
  - Manual expert review:
    - At least two internal performance or ML engineers will review the DeepSeek‑OCR analytic model (module hierarchy,
      operator mapping, and cost formulas) and validate that the results are internally consistent and plausible given
      the known architecture and TorchInfo/static analysis outputs.
    - Automated comparison against measured runtimes (Stage 1/Stage 2 traces or Nsight metrics) is explicitly out of
      scope for this development stage and will be treated as a follow‑up enhancement.
  Manual tests for this feature will be added under
  `/workspace/code/llm-perf-opt/tests/manual/deepseek_ocr/manual_deepseek_ocr_analytic_model.py` (exact name to be
  confirmed in Phase 1) and integrated into the docs quickstart; these tests will focus on producing analytic artifacts
  suitable for human inspection rather than enforcing numeric thresholds vs measured timings.
- Rationale: This phase is intended to establish a robust theoretical analytic model and data structures. Deferring
  automated runtime comparisons keeps the work tractable while still enabling expert review of correctness. Runtime
  validation can be added once Stage 2 profiling flows are mature and stable for DeepSeek‑OCR.
- Alternatives considered:
  - Combining theoretical modeling with full empirical calibration (fit all parameters to Nsight traces) in this phase:
    higher complexity and risk of overfitting to a single hardware setup.
  - Purely analytic validation with no human review: insufficient guardrails given the complexity of DeepSeek‑OCR and
    the importance of interpretability for internal stakeholders.

## Consolidated Choices (TL;DR)

- Overhead: Additional analytic modeling runtime overhead ≤ 10% of Stage 1/Stage 2 baselines for the standard OCR
  workload; offline static processing still respects the 5‑minute report goal.
- Scope: Target `deepseek-ocr-v1-base` only; design data models to allow future DeepSeek‑OCR variants and additional
  workload profiles.
- Standard workload: Synthetic profile `dsocr-standard-v1` with `seq_len=512`, `base_size=1024`, `image_size=640`,
  `crop_mode=True`, greedy decoding, and fixed `max_new_tokens` for calibration.
- Operator source: TorchInfo static artifacts under
  `reports/20211117-dsorc-op-analysis/static-20251118-130533/` are canonical for the target operator list and module
  hierarchy.
- Mapping: Two‑tier mapping from DeepSeek‑OCR modules to ModelMeter domain layers and primitives, expressed via a
  declarative mapping table.
- Integration: Library‑style integration where `llm_perf_opt` runners orchestrate inputs/outputs and
  `extern/modelmeter/models/deepseek_ocr/` implements the analytic model.
- Validation: Synthetic unit tests and golden‑value checks, combined with manual expert review of analytic artifacts;
  automated runtime comparison is deferred to a later phase.
