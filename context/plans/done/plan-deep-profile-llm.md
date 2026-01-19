# Deep Profiling Plan: LLM Inference (Python‑first, NVIDIA GPU)

## HEADER
- **Status**: Done
- **Completed**: 2026-01-19

Purpose
- Establish a consistent, Python‑first methodology to analyze LLM inference
  performance on NVIDIA GPUs, enabling clear bottleneck identification and
  evidence‑based design decisions.

## Stage 1: Early Profiling and MFU

- Stage boundaries (NVTX ranges)
  - What: consistent labels for critical phases such as prefill and decode.
  - Why: makes timelines from multiple tools interpretable and comparable; lets
    you isolate first‑token latency vs steady‑state token latency.

- Static model analysis (e.g., fvcore/ptflops)
  - What: parameter counts, layerwise FLOPs estimates, and activation sizes
    derived from model structure and representative input shapes.
  - Why: establishes an analytical FLOPs‑per‑token baseline, highlights
    complexity hotspots by module, and provides priors for early MFU estimates
    before dynamic profiling.

- Operator‑level timeline (PyTorch Profiler)
  - What: per‑op CPU/CUDA time, tensor shapes, and memory activity.
  - Why: pinpoints the heaviest operators and call paths; clarifies whether
    overhead is in CPU scheduling, CUDA kernels, or allocations.

- Early model‑level MFU estimate (before kernel deep‑dive)
  - What: a coarse achieved‑FLOPs and MFU estimate at the model level by
    combining measured token throughput with an analytical FLOPs‑per‑token
    estimate based on model dimensions (attention + MLP). Optionally refine
    using available GEMM shape summaries if present.
  - Why: provides an early directional MFU signal to prioritize where to look
    next, without waiting for Nsight Compute; helps triage compute‑ vs
    memory‑bound hypotheses sooner.

- Per‑stage MFU (NVTX‑segmented)
  - What: MFU estimates computed separately for NVTX‑defined stages (e.g.,
    prefill vs decode) using stage‑specific throughput and FLOPs estimates; can
    be corroborated later with tool views filtered to those ranges.
  - Why: prefill and decode often exhibit different compute/memory behavior;
    per‑stage MFU highlights which phase under‑utilizes hardware and focuses
    optimization effort.

## Stage 2: Deep Tooling and Corroboration

- Whole‑program timeline (Nsight Systems)
  - What: end‑to‑end CPU↔GPU flow, kernel overlaps, memcpys, launch behavior,
    and concurrency.
  - Why: reveals idle gaps, serialization, and transfer bottlenecks that do not
    surface in op‑only views.

- Kernel deep‑dive (Nsight Compute)
  - What: achieved vs peak throughput, warp stall reasons, occupancy,
    DRAM/L2 activity, and Tensor Core utilization for hot kernels.
  - Why: determines compute‑ vs memory‑bound behavior and whether kernels are
    using the hardware’s efficient paths (e.g., Tensor Cores) effectively.

- GEMM shape visibility (cuBLAS logs)
  - What: M/N/K dimensions and selected algorithms for GEMMs.
  - Why: enables achieved FLOPs and MFU estimation and informs whether matmul
    shapes align with hardware‑friendly tile choices.

- Power, clocks, utilization (NVML)
  - What: time‑series of GPU power, clocks, and utilization.
  - Why: detects throttling, power caps, or clock drops that confound kernel
    analysis and explain performance variance.

Outcomes
- A coherent artifact set (operator timeline, whole‑program trace, kernel
  metrics, optional GEMM shapes and power) sufficient to explain where time is
  spent and why.
- A brief diagnosis that maps observations to model/runtime design choices
  (e.g., memory‑bound attention due to low L2 hit rate; under‑utilized Tensor
  Cores due to shapes/dtypes).

Non‑Goals
- No implementation details or commands here; this plan defines what to collect
  and why, not how to instrument or run tools.
