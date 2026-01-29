# About NVIDIA H20 GPU Specifications

This document provides a summary of the NVIDIA H20 GPU specifications, a model designed for specific markets (e.g., China) to comply with export controls while maintaining high memory bandwidth for LLM inference.

## Key Specifications

| Specification | Value | Notes |
| :--- | :--- | :--- |
| **Architecture** | Hopper | Based on H100 architecture |
| **Memory Capacity** | 96 GB HBM3 | Matches H100 SXM5 capacity |
| **Memory Bandwidth** | 4.0 TB/s | Excellent for memory-bound tasks |
| **Interconnect (NVLink)** | 900 GB/s | Bidirectional bandwidth |
| **TDP (Power)** | 400W | Configurable (down to ~350W) |
| **Interface** | SXM5 | Compatible with HGX H20 systems |

## Compute Performance

The H20 is significantly restricted in raw compute performance compared to the H100 to meet export control regulations (TPP < 4800).

| Precision | Performance (Dense) |
| :--- | :--- |
| **FP64** | 1 TFLOPS (approx) |
| **FP32** | 60 TFLOPS (approx) |
| **TF32** | 74 TFLOPS |
| **FP16 / BF16** | 148 TFLOPS |
| **FP8** | 296 TFLOPS |
| **INT8** | 296 TOPS |

*Note: Compute figures are derived from export control limits and available datasheets. The "Performance Density" cap limits the peak TFLOPS. Unlike the H100, the H20's strength lies in its memory bandwidth (4.0 TB/s) rather than raw FLOPS, making it suitable for inference-heavy workloads where memory bandwidth is the bottleneck.*

## Usage in LLM Profiling

When profiling on H20:

1.  **Memory Bound:** Expect kernels to be memory-bound more often than on H100/A100 due to the high bandwidth-to-compute ratio.
2.  **Roofline Model:** The arithmetic intensity ceiling will be much lower. Adjust your roofline analysis accordingly.
3.  **Cluster Scale:** With 900 GB/s NVLink, multi-GPU scaling performance should remain high for communication-heavy patterns.

## Sources
*   [NVIDIA H20 Datasheet (via 3rd party distributors)](https://www.nvidia.com/)
*   [Tom's Hardware Analysis](https://www.tomshardware.com/)
