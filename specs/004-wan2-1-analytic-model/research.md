# Research: Wan2.1 Analytic FLOP Modeling (ModelMeter-style)

This research consolidates the key implementation decisions needed to deliver a Wan2.1-T2V-14B analytic FLOP model that can be verified layer-by-layer and end-to-end (within the analytic scope) against a reference measurement with a ≤5% error budget.

## Decisions

- Decision: Use PyTorch `torch.utils.flop_counter.FlopCounterMode` as the primary reference FLOP measurement.
  Rationale: This is already the reference mechanism used by existing ModelMeter verification scripts in this repository (for example DeepSeek-OCR end-to-end checks), and it provides a deterministic, automatable baseline for comparisons.
  Alternatives considered: Nsight Compute FLOP metrics (heavier workflow and not naturally layer-by-layer); vendor-reported FLOPs (often unavailable or inconsistently defined); hand-derived FLOP tables only (no executable verification loop).

- Decision: Define “end-to-end” for verification as “diffusion transformer core forward across `num_inference_steps`” (core-only) for v1, with optional coarse-grained modeling of text encoder and VAE that does not participate in the ≤5% accuracy gate unless explicitly enabled.
  Rationale: The feature’s accuracy gate is about matching model compute; the diffusion transformer dominates typical workloads and is the component where layer-by-layer accounting is meaningful and stable across implementations, while end-to-end full pipelines (text encoder + VAE + scheduler) vary by integration and may be unavailable in CI.
  Alternatives considered: Full pipeline end-to-end (requires a complete vendor pipeline and assets; fragile across implementations); DiT-only but without a declared scope (ambiguous verification target).

- Decision: Achieve “layer-by-layer” matching by verifying at the transformer-block level and at major subcomponents within each block (attention + feed-forward + modulation/norm), using reference PyTorch mirrors and/or vendor modules when available.
  Rationale: Block-level accounting provides stable identifiers and meaningful hotspots, while still being granular enough to catch analytic mismatches; per-operator FLOP tables from `FlopCounterMode` can be used as an additional diagnostic when module granularity is insufficient.
  Alternatives considered: Only end-to-end totals (can hide compensating errors); extremely fine-grained operator-level matching as the primary gate (high maintenance and sensitive to implementation changes).

- Decision: Use the FLOP counting convention implied by `FlopCounterMode` for verification (for example, matrix multiply counted as `2 * M * N * K` for multiply+add).
  Rationale: The goal is to match the chosen reference measurement within tolerance; elementwise operators are often counted as zero by `FlopCounterMode`, so verification should not require counting them precisely.
  Alternatives considered: “True math FLOPs” including elementwise/norm costs (hard to reconcile with `FlopCounterMode` and unlikely to change optimization decisions).

- Decision: Keep an explicit “torch-visible” compatibility mode in analytic layers (via `StageCostMixin._set_ignore_torch_unsupported_flop_count(True)`) for cases where the reference implementation uses fused kernels that are not attributed by `FlopCounterMode`.
  Rationale: Existing ModelMeter analytic models use this pattern to reconcile analytic formulas with what PyTorch can observe; it reduces the risk of being blocked by reference implementation details.
  Alternatives considered: Hard-coding omissions in all analytic FLOP formulas (loses “full analytic” fidelity and makes future changes harder to reason about).

- Decision: Compute video token geometry from model metadata when possible, and otherwise use documented defaults with explicit disclosure in the report metadata.
  Rationale: Token geometry drives the dominant FLOP terms in diffusion transformers; making it a single, tested helper reduces duplication and prevents silent mismatches.
  Alternatives considered: Scatter token math across layers (harder to audit); assume fixed tokens per resolution with no config linkage (breaks when model variants change).

- Decision: Standardize artifact layout for Wan2.1 static analysis under `/data1/huangzhe/code/llm-perf-opt/tmp/profile-output/<run_id>/static_analysis/wan2_1/`.
  Rationale: Aligns with the repository’s “filesystem-only artifacts” convention and keeps Wan2.1 outputs discoverable and comparable with other static-analysis reports.
  Alternatives considered: New top-level output dirs (fragments run context); storing artifacts outside `tmp/` (violates repo conventions).

## Constraints & Practices

- External model reference is machine-local and must not be committed: use `models/wan2.1-t2v-14b/bootstrap.sh` to create `models/wan2.1-t2v-14b/source-data` as a symlink to `${LLM_MODELS_ROOT}/Wan2.1-T2V-14B` (or `${WAN21_T2V_14B_PATH}`).
- Verification scripts should be runnable via `pixi run -e rtx5090 ...` and should skip (with clear messaging) when the local Wan2.1 reference model is not available.
- Report schemas should reuse shared `attrs` types (module nodes, operator categories, module metrics snapshots) to avoid duplicating DeepSeek-OCR schema logic.

## Open Items resolved here

- Reference mechanism: use `FlopCounterMode` (primary) with optional vendor import path when local code is available.
- Verification scope: v1 “end-to-end” accuracy gate applies to the diffusion transformer core across steps for the standard workload set.
- Geometry: implement one canonical token-count helper and unit-test monotonic scaling with frames and resolution.
