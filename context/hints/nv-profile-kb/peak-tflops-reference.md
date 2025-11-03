# Peak TFLOPs Reference (Selected GPUs)

Purpose: Centralize theoretical peak throughput values used for MFU denominators.

Notes
- FP32 here refers to non‑tensor FP32 shader throughput (typical for classic graphics math).
- FP16 (Tensor Core, dense) denotes 3rd‑gen tensor core dense throughput. Values in parentheses for “sparsity” are higher and not used as MFU denominators.
- Prefer env override `MFU_PEAK_TFLOPS` for unlisted devices.

## NVIDIA GeForce RTX 3090 (GA102)
- FP32 (non‑tensor): ~35.6 TFLOPS
- FP16 (Tensor Core, dense): ~142 TFLOPS (≈ 285 TFLOPS with 2:4 sparsity)

Sources
- NVIDIA Ampere GA102 Whitepaper (Appendix A: RTX 3090): Peak FP32 ≈ 35.6 TFLOPS
  - https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
- Tom’s Hardware (spec table): “Tensor TFLOPS FP16 (Sparsity) 142 (285)”
  - https://www.tomshardware.com/reviews/nvidia-geforce-rtx-3090-review
- TechPowerUp (DB): FP32 ≈ 35.58 TFLOPS; FP16 (non‑tensor) shown 1:1 with FP32 for GA10x
  - https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622

MFU usage
- For consumer Ampere (GA10x) MFU in FP16/BF16 regimes, we use the dense Tensor Core FP16 number (~142 TFLOPS) as the denominator to represent practical mixed‑precision compute paths. Structured sparsity peak (~285) is not used for MFU denominators.

## NVIDIA GeForce RTX 4090 (AD102)
- FP16 (Tensor Core, dense): ~330 TFLOPS (used in code mapping)
- Source: multiple vendor and review aggregates; update if official whitepaper table is available.

## NVIDIA A100 / H100 (data center)
- A100 (SXM, FP16 dense): ~312 TFLOPS
- H100 (SXM, FP16 dense): ~990 TFLOPS
- Source: NVIDIA product briefs/whitepapers.

---

Implementation
- Code mapping: `src/llm_perf_opt/profiling/hw.py` contains a small lookup; prefer env var override for unlisted devices.
- If the reported GPU name includes “NVIDIA GeForce RTX 3090”, code uses 142.0 TFLOPS as default denominator.
