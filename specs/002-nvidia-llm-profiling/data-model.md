# Data Model — Stage 2 NVIDIA Deep Profiling

## Entities

- ProfilingSession
  - id: string (run identifier)
  - model_target: ModelTarget
  - started_at: datetime
  - finished_at: datetime
  - environment: object (hardware, software, cuda)
  - config: object (profiling flags, inputs)
  - artifacts: list[string] (paths)

- ModelTarget
  - name: string (e.g., deepseek-ocr)
  - variant: string (e.g., base/large)
  - parameters: string/int (num params)
  - dtype: string (e.g., bf16/fp16)

- StageTiming
  - stage: enum (prefill, decode)
  - total_ms: float
  - calls: int
  - mean_ms: float (derived = total_ms / max(calls,1))

- OperatorRecord
  - op_name: string
  - total_ms: float
  - calls: int
  - mean_ms: float (derived)

- KernelRecord
  - kernel_name: string
  - device: string (GPU index/name)
  - total_ms: float
  - calls: int
  - mean_ms: float (derived)

## Relationships

- ProfilingSession 1 — 1 ModelTarget
- ProfilingSession 1 — N StageTiming
- ProfilingSession 1 — N OperatorRecord
- ProfilingSession 1 — N KernelRecord

## Validation Rules

- total_ms ≥ 0; calls ≥ 0; mean_ms recomputed, not user-provided
- stage ∈ {prefill, decode}; “vision (sam+clip+projector)” recorded as a note, not a stage
- artifacts paths must exist at write-time; non-existent paths are omitted

