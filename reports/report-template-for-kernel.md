# Report of Deepseek-OCR Kernel-level Profiling

## Overview

The intention of this report is to present the findings from kernel-level profiling of the Deepseek-OCR model using nsys and ncu, to identify the performance characteristics and potential bottlenecks during inference, to guide future NPU design (inference oriented).

This report shows:
- top kernels used by Deepseek-OCR during inference
- detailed performance metrics collected via NVIDIA Nsight Compute (NCU)
- histograms of kernel execution metrics
- roofline analysis for all profiled kernels
- classification of kernels by type (compute-bound, memory-bound, etc.)
- implementation details of profiling setup and methodology

## Experiment

### Setup

[what model, hardware, software versions, profiling configuration, etc.]
[hardware specs, including GPU model, CPU, memory, etc.]
[model specs, including precision, batch size, input dimensions, etc.]

### Top Kernels by Total Time

[table of top kernels by total execution time percentage]
[give the top kernels human-friendly names instead of function names if possible]
[in following tables/texts, use human-friendly names consistently, for the same kind of kernels that different by tiling or other minor variations, use postfixes to include that info]

### Histograms of Kernel Execution Metrics

### Conclusions

[summary of findings, performance bottlenecks identified, recommendations for optimization, etc.]

[recommendations for future NPU design based on profiling results]
[covering aspects like tensor-core/cuda-core balance, memory bandwidth needs, L1/L2 cache ratio, etc.]