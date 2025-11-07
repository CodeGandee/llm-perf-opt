# Report Version History

## Overview

This document tracks the differences between V1 and V2 of the DeepSeek-OCR kernel profiling reports.

## Version 1 (final-report-v1.md)

**Data Source:** `ncu/analysis/`
**Kernel Coverage:** 18/20 (90%)
**Missing Kernels:** 
- kernel_0009 (unknown)
- kernel_0016 (unknown)

**Key Statistics:**
- SM Throughput: 10.35%
- Memory Throughput: 27.35%
- Achieved Occupancy: 30.13%
- L1 Hit Rate: 8.85%
- L2 Hit Rate: 53.48%
- Longest Kernel: 165.09 μs (CUTLASS GEMM)

**Classification:**
- Memory-bound: 8 (44.4%)
- Balanced: 7 (38.9%)
- Compute-bound: 3 (16.7%)
- Unknown: 2 (10.0%)

## Version 2 (final-report-v2.md)

**Data Source:** `ncu-v2/analysis/`
**Kernel Coverage:** 20/20 (100%) ✓
**Missing Kernels:** None

**Key Statistics:**
- SM Throughput: 15.75% (+52% improvement)
- Memory Throughput: 34.62% (+27% improvement)
- Achieved Occupancy: 39.51% (+31% improvement)
- L1 Hit Rate: 8.96% (similar)
- L2 Hit Rate: 39.95% (-25% degradation)
- Longest Kernel: 404.70 μs (+145% vs V1)

**Classification:**
- Memory-bound: 9 (45.0%)
- Balanced: 7 (35.0%)
- Compute-bound: 4 (20.0%)
- Unknown: 0 (0%) ✓

**Newly Profiled Kernels:**
1. **kernel_0009** - Memory-Efficient Attention (CUTLASS BF16)
   - Duration: 160.74 μs
   - Classification: Balanced
   - 2nd longest-running kernel
   - Shows poor compute and memory utilization
   
2. **kernel_0016** - CUTLASS GEMM
   - Classification: Compute-bound
   - Joins the compute-bound category

## Key Findings from V2

1. **Complete Coverage**: All 20 kernels successfully profiled
2. **Attention Insights**: kernel_0009 (memory-efficient attention) takes 160.74 μs vs FlashAttention's 10.34 μs (15.5x difference)
3. **Higher Resource Utilization**: Improved metrics across SM, memory, and occupancy
4. **L2 Cache Degradation**: Newly profiled kernels have worse cache behavior (39.95% vs 53.48% hit rate)
5. **Compute-Bound Growth**: 4 kernels (20%) vs 3 in V1 (16.7%)

## Recommendations Updated

V2 recommendations now include:
- Consider replacing memory-efficient attention with FlashAttention variants
- Address L2 cache behavior in newly profiled kernels
- Optimize kernel_0009 (balanced, 160.74 μs) for better resource utilization

## Files

- `final-report-v1.md` - Original report with 90% coverage
- `final-report-v2.md` - Complete report with 100% coverage
- `ncu/analysis/` - V1 profiling data
- `ncu-v2/analysis/` - V2 profiling data (complete)
