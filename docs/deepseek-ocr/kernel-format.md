DeepSeek‑OCR Kernel YAML Format
===============================

Purpose
- Document the schema and meaning of each keyword used in `docs/deepseek-ocr/kernels.yaml`.
- Provide a consistent, human‑readable way to describe GPU kernels observed during profiling (operator type, source library, and shapes).

File Location
- Path: `docs/deepseek-ocr/kernels.yaml`

Schema Overview
- Top level is a YAML sequence (list). Each item describes one kernel.
- Keys per item:
  - `raw_name` (string): Full kernel or function symbol as reported by profiling tools (may include template parameters and decorations). Used for exact identification.
  - `friendly_name` (string): Concise human label for quick scanning (e.g., "CUTLASS GEMM (Tensor Core, BF16)").
  - `source_lib` (string): Primary library or component where the kernel originates. See “Source Libraries”.
  - `data_shape` (mapping): Canonical I/O shape description using symbolic notation. Contains:
    - `generic` (list): Abstract operator signature using symbols (e.g., `[(M, K), (K, N), "->", (M, N)]`).
    - `fixed` (list|null): Concrete specialization when known (e.g., `D=64` in attention heads). Use the same notation as `generic`. Set to `null` if not applicable.
    - `tiled` (list|null): Tile‑level view when kernel names encode tile sizes (e.g., CUTLASS). Use the same notation as `generic`. Set to `null` if not applicable.
  - `description` (string): One‑line context or classification note (e.g., how the entry was inferred, or notable traits).

Shape Notation
- Tuples denote tensor extents, e.g., `(M, K)`, `(K, N)`, `(M, N)`, `(N,)`.
- The arrow token `->` separates inputs from outputs within the list.
- Named tensors use `Name:(dims)` form, e.g., `Q:(B,H,T,D)`, `O:(B,H,T,D)`.
- Common symbols:
  - `B`: batch size
  - `H`: attention heads
  - `T`: target sequence length
  - `S`: source sequence length
  - `D`: head/channel dimension
  - `M, N, K`: matrix multiply dimensions
- Tile notation expresses one micro‑kernel tile, mirroring `(M×K)·(K×N)→(M×N)`, e.g., `[(256, 32), (32, 128), "->", (256, 128)]`.

Source Libraries
- Use a short, canonical value for `source_lib`:
  - `CUTLASS`: NVIDIA CUTLASS kernels (often Tensor Core GEMM; names may encode tile sizes).
  - `cuBLAS`: NVIDIA cuBLAS BLAS routines (e.g., GEMM, GEMV).
  - `cuDNN`: NVIDIA cuDNN convolution/NN primitives.
  - `PyTorch ATen`: PyTorch TensorIterator or CUDA ATen kernels (elementwise, reductions, copies, etc.).
  - `PyTorch MemEffAttention`: PyTorch memory‑efficient attention implementations (e.g., CUTLASS‑based FMHA).
  - `FlashAttention`: FlashAttention kernels (e.g., split‑KV forward/backward).
  - `CUDA`: Generic CUDA kernels not clearly attributed to a higher‑level library.

Authoring Guidelines
- Keep `friendly_name` short and descriptive; include key traits like data type or algorithm variant only when useful (e.g., “BF16”, “vectorized”).
- Always populate `generic` in `data_shape` when the operator class is known.
- Use `fixed` for known specializations (e.g., `D=64`) when visible in kernel traits or configuration; otherwise use `null`.
- Use `tiled` only when tile sizes are explicit or confidently inferred from the kernel name; otherwise use `null`.
- Keep `description` to a single line noting the operation or inference method.

Curation Notes
- `kernels.yaml` may be auto‑seeded by heuristics, but entries should be human‑reviewed for clarity and correctness.
- Avoid embedding implementation details; this file documents what kernels are and how to read their shapes, not how to generate or run them.

