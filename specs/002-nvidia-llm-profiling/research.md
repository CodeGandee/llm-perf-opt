# Research & Decisions — Stage 2 NVIDIA Deep Profiling

Date: 2025-10-29  
Branch: 002-nvidia-llm-profiling  
Spec: specs/002-nvidia-llm-profiling/spec.md

## Profiling Tooling

- Decision: Use Nsight Systems (nsys) for timeline + NVTX ranges; Nsight Compute (ncu) for kernel metrics (utilization, occupancy, memory throughput).
- Rationale: nsys captures end-to-end flows with NVTX; ncu provides kernel-level counters essential for attribution and bottleneck analysis.
- Alternatives considered: CUPTI custom collectors (more engineering), PyTorch profiler only (insufficient kernel counters).

## Overhead Target & Modes

- Decision: Default deep profiling overhead target ≤ 25% over baseline; provide a lighter mode (reduced counters, sampling) if exceeded.
- Rationale: Ensures usability on developer workstations. Lighter mode preserves directional insights.
- Alternatives considered: No cap (risk long runs, poor UX); strict ≤ 10% (too restrictive for kernel metrics).

## Trace Size & Retention

- Decision: Cap per-run artifacts at ~2 GB; keep last 5 runs per stage; compress older runs.
- Rationale: Prevents disk bloat while keeping a short history for comparison.
- Alternatives considered: Unlimited retention (risk disk exhaustion); single-run retention (hurts comparability).

## Representative Inputs

- Decision: Use a small, fixed input bundle that triggers both prefill and decode, plus the vision preprocessing path. Keep total run < 30 minutes.
- Rationale: Must exercise the critical kernels without excessive runtime.
- Alternatives considered: Full dataset (too slow); tiny toy sample (misses realistic kernel mix).

## Testing Approach (Exporters)

- Decision: Add minimal pytest coverage for table generation (column headers, sorting by total, mean ms calculation), leave kernel counter correctness to manual review/ground truth.
- Rationale: Catch regressions in formatting and aggregation logic with low effort.
- Alternatives considered: Full e2e automated validation (heavy fixtures, high maintenance).

## Data Model Reuse

- Decision: Reuse Stage 1 domain/contract models (StageTiming, OperatorSummary/OperatorRecord, LLMProfileReport aggregates). Introduce only `KernelRecord` and optionally extend `LLMProfileReport` with `kernels_topk`.
- Rationale: Avoid duplication and reduce migration cost; keep Stage 2 focused on kernel attribution.
- Alternatives considered: New parallel model set (adds maintenance burden and risk of divergence).

## Multi-GPU / MIG Scope

- Decision: Default scope is single GPU. Document guidance for device selection and MIG; multi-GPU support as future work.
- Rationale: Keeps Stage 2 focused; complexity grows with multi-GPU synchronization.
- Alternatives considered: Multi-GPU in Stage 2 (risks scope creep and delays).
