# How to Identify LLM Inference Bottlenecks: A Systematic Approach

This guide provides a step-by-step methodology for identifying performance bottlenecks in LLM inference workloads on NVIDIA GPUs. Following this sequence helps you accurately diagnose whether your workload is compute-bound, memory-bound, or limited by synchronization/system overhead.

> **Critical principle**: Being right about the bottleneck type is essential. Each issue requires different solutions, and misdiagnosis wastes time on ineffective optimizations. ([Medium - Aruna Kolluru, 2025](https://medium.com/@aruna.kolluru/understanding-bottlenecks-in-llm-workloads-compute-memory-and-bandwidth-cdcef2fde252))

---

## Overview: Bottleneck Categories in LLM Inference

LLM inference performance is constrained by one of three primary bottlenecks:

1. **Compute-bound**: GPU compute units (SMs, Tensor Cores) are saturated
2. **Memory-bound**: DRAM bandwidth is the limiting factor
3. **System/sync-bound**: CPU overhead, synchronization latencies, or OS bottlenecks dominate

Knowing which resource constrains your workload helps determine optimal hardware configuration, batch sizing, and optimization strategies. ([Databricks Blog, 2025](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices))

---

## Step-by-Step Bottleneck Identification Workflow

### Phase 1: Understand Your Operational Regime

**Goal**: Determine if your inference workload is theoretically compute-bound or memory-bound.

**Method**: Calculate and compare arithmetic intensity:

```python
# Arithmetic intensity = FLOPs / Bytes accessed
# For attention: AI = O(seq_len) for prefill, O(1) for decode
# Compare against GPU's compute:bandwidth ratio

# Example for A100:
peak_tflops = 312  # FP16 with Tensor Cores
peak_bandwidth_tbs = 1.935  # TB/s HBM2
compute_to_bandwidth_ratio = peak_tflops / peak_bandwidth_tbs  # ~161 FLOP/byte

# If your kernel's AI < 161, it's memory-bound on A100
# If AI > 161, it's compute-bound
```

**Key insight**: Prefill operations (processing prompt) have higher arithmetic intensity due to quadratic attention scaling, making them more compute-intensive. Decode operations (generating tokens) access large KV caches with minimal computation per byte, making them memory-intensive. ([Medium - Dev Patel, 2025](https://medium.com/@devsp0703/inside-real-time-llm-inference-from-prefill-to-decode-explained-72a1c9b1d85a))

**References**:
- [Baseten LLM Inference Guide](https://www.baseten.co/blog/llm-transformer-inference-guide/)
- [arXiv: Mind the Memory Gap](https://arxiv.org/html/2503.08311v2)

---

### Phase 2: Distinguish Between Inference Phases

**Goal**: Separately analyze prefill and decode performance characteristics.

**Method**: Use NVTX ranges to mark each phase:

```python
from torch.cuda import nvtx

# Prefill phase: prompt processing
with nvtx.range("prefill"):
    logits = model(prompt_ids, use_cache=True, ...)

# Decode phase: autoregressive generation
with nvtx.range("decode"):
    for step in range(max_new_tokens):
        next_token = sample(logits)
        logits = model(next_token, past_key_values=cache, ...)
```

**Key characteristics**:
- **Prefill**: Compute-heavy, benefits from parallelism, quadratic attention O(n²)
- **Decode**: Memory-heavy (KV cache reads dominate), linear attention O(n)
- **TTFT** (Time To First Token) is determined by prefill performance
- **TPOT** (Time Per Output Token) is determined by decode performance

([Hao AI Lab - DistServe](https://hao-ai-lab.github.io/blogs/distserve/))

**Tool**: Nsight Systems will show prefill/decode phases clearly labeled in timeline view.

---

### Phase 3: Initial Timeline Profiling (Nsight Systems)

**Goal**: Get a holistic view of execution, identify hot kernels, CPU-GPU interaction patterns, and concurrency issues.

**Command**:
```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=nvtx --capture-range-end=nvtx \
  --sample=none \
  -o nsys_llm_profile \
  python your_inference_script.py
```

**What to look for**:
1. **Kernel execution patterns**: Are kernels back-to-back or with gaps?
2. **CPU-GPU bubbles**: Are there idle periods indicating launch overhead?
3. **Memory transfers**: Unexpected H2D/D2H copies?
4. **Phase durations**: Relative time spent in prefill vs decode
5. **Hot kernels**: Which kernels consume most GPU time?

**Key metrics from timeline**:
- Kernel concurrency (overlapping operations)
- CPU launch overhead
- Memory copy patterns
- NVTX range durations

([NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html))

**Pro tip**: Use `--capture-range=nvtx` to capture only the inference phase, reducing trace size and overhead.

---

### Phase 4: Kernel-Level Deep Dive (Nsight Compute)

**Goal**: Understand per-kernel bottlenecks: SM utilization, memory bandwidth, cache behavior, warp stalls.

**Command**:
```bash
ncu --target-processes all \
    --set full \
    --metrics \
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_sector_hit_rate.pct,\
smsp__warp_stall_memory_dependency_per_warp_active.pct \
    -o ncu_kernel_analysis \
    python your_inference_script.py
```

**Analysis checklist**:

1. **Check Tensor Core utilization** (`sm__pipe_tensor_op_hmma_cycles_active`):
   - Low (<50%): Check if matmul shapes are TC-eligible, ensure FP16/BF16 dtypes
   - High (>80%): Compute resources well utilized

2. **Check DRAM throughput** (`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`):
   - High (>70%): Memory bandwidth saturated → memory-bound
   - Low (<30%): Not memory-limited

3. **Check L2 hit rate** (`lts__t_sector_hit_rate.pct`):
   - Low hit rate with high DRAM traffic: Poor cache locality (common in KV cache access)
   - High hit rate: Good data reuse

4. **Check warp stalls** (Memory Workload Analysis section):
   - `stall_memory_dependency` high: Waiting on memory accesses
   - `stall_not_selected` high: Low occupancy, increase parallelism

([NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html))

**Critical finding from research**: In large-batch LLM inference, DRAM bandwidth is the primary bottleneck, with over 50% of attention kernel cycles stalled due to memory access delays. ([arXiv 2503.08311](https://arxiv.org/html/2503.08311v2))

---

### Phase 5: Calculate Arithmetic Intensity and Roofline Position

**Goal**: Quantitatively determine if each hot kernel is compute-bound or memory-bound.

**Method**:
```python
# From Nsight Compute report, extract per-kernel:
# - FLOPs performed (or calculate from shapes: 2*M*N*K for GEMM)
# - Bytes accessed (dram__bytes_read + dram__bytes_write)
# - Execution time

arithmetic_intensity = flops / bytes_accessed  # FLOP/byte
achieved_performance = flops / time_seconds    # FLOP/s

# Plot on roofline:
# - X-axis: arithmetic intensity
# - Y-axis: achieved FLOP/s
# - Compare against two ceilings:
#   1. Peak compute (horizontal line)
#   2. Peak bandwidth * AI (diagonal line)
```

**Interpretation**:
- **Below bandwidth ceiling**: Memory-bound (optimize data movement)
- **Below compute ceiling but above bandwidth**: Compute-bound (optimize FLOP efficiency)
- **At ceiling intersection**: Balanced workload

**Example**:
```python
# A100 ceilings:
peak_fp16_tflops = 312
peak_bandwidth_tbs = 1.935
ridge_point = peak_fp16_tflops / peak_bandwidth_tbs  # ~161 FLOP/byte

# If attention kernel has AI = 20 FLOP/byte:
# 20 < 161 → memory-bound
# Max achievable = 20 * 1.935 = 38.7 TFLOP/s (not 312)
```

([NERSC Roofline on NVIDIA GPUs](https://gitlab.com/NERSC/roofline-on-nvidia-gpus))

---

### Phase 6: Measure Memory Bandwidth Utilization (MBU)

**Goal**: Quantify how efficiently memory bandwidth is utilized.

**Metric**: Model Bandwidth Utilization (MBU)

```python
# MBU = achieved_bandwidth / peak_bandwidth
# where achieved_bandwidth = (model_params + kv_cache_size) / TPOT

model_size_gb = 70  # e.g., 70B model in FP16 = 140GB
kv_cache_size_gb = 8  # depends on batch size, seq length, layers
tpot_seconds = 0.025  # time per output token

achieved_bandwidth_gbs = (model_size_gb + kv_cache_size_gb) / tpot_seconds
peak_bandwidth_gbs = 1935  # A100 = 1935 GB/s

mbu = achieved_bandwidth_gbs / peak_bandwidth_gbs
print(f"MBU: {mbu:.1%}")  # Target: >70% for decode-bound workloads
```

**Typical findings**:
- **Decode phase**: DRAM read throughput remains <65% across models, indicating room for optimization ([arXiv research](https://arxiv.org/html/2503.08311v2))
- **Prefill phase**: Usually compute-limited (lower MBU expected)

**Key insight**: High memory bandwidth is critical for per-user throughput in serving scenarios. Synchronization latencies must be ~1µs or less, otherwise they nullify bandwidth advantages. ([SemiEngineering - NVIDIA LLM Bottlenecks](https://semiengineering.com/llm-inference-core-bottlenecks-imposed-by-memory-compute-capacity-synchronization-overheads-nvidia/))

---

### Phase 7: Identify Synchronization Overheads

**Goal**: Detect CPU overhead, kernel launch latencies, and multi-GPU communication bottlenecks.

**Method A: CPU overhead from Nsight Systems timeline**

Look for:
- Gaps between kernel launches
- Long CPU-side operation durations
- Python/framework overhead

**Research finding**: CPU overhead grows with batch size, reaching up to 30% of total execution time in some large-batch scenarios. ([arXiv 2503.08311](https://arxiv.org/html/2503.08311v2))

**Method B: Multi-GPU communication (if using Tensor/Pipeline Parallelism)**

```bash
# Enable NCCL logging
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL python your_script.py 2>&1 | tee nccl.log

# Look for:
# - AllReduce/AllGather times
# - Bus bandwidth utilization
# - Synchronization points
```

**Method C: System-level bottlenecks**

```bash
# Trace system calls during inference
strace -c -p <pid>  # summary of syscall counts/times

# Monitor page faults
perf stat -e page-faults python your_script.py
```

([OS-Level Challenges in LLM Inference](https://eunomia.dev/blog/2025/02/18/os-level-challenges-in-llm-inference-and-optimizations/))

---

### Phase 8: Operator-Level Performance Breakdown

**Goal**: Understand which model operations (attention, MLP, LayerNorm, etc.) dominate runtime and their bottleneck characteristics.

**Method**: PyTorch Profiler with NVTX correlation

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler("./profile_logs")
) as prof:
    with nvtx.range("inference"):
        output = model(input_ids)
    prof.step()

# View results
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))
```

**Look for**:
- `aten::matmul`, `aten::linear`: Should be Tensor Core-accelerated (check NCU)
- `aten::softmax`, `aten::layer_norm`: Memory-bound, check bandwidth
- Attention operations: Prefill should be compute-bound, decode memory-bound

**Advanced**: Use operator-level performance modeling to capture shifting bottlenecks (e.g., decode attention flipping between compute and memory-bound as KV cache grows). ([arXiv 2508.06133](https://arxiv.org/abs/2508.06133))

---

## Decision Tree: Interpreting Your Findings

```
Start → Profile with Nsight Systems (Phase 3)
    ↓
Identify hot kernels and phases
    ↓
    ├─→ [Prefill dominates runtime]
    │   ├─→ NCU shows low Tensor Core util → Fix: Check shapes, dtypes, fusion
    │   ├─→ NCU shows high SM util → Compute-bound, consider batching or faster GPU
    │   └─→ High CPU gaps → Reduce launch overhead, kernel fusion
    │
    └─→ [Decode dominates runtime]
        ├─→ NCU shows high DRAM%, low L2 hit rate → Memory-bound
        │   └─→ Solutions: KV cache quantization, PagedAttention, better layout
        ├─→ NCU shows memory stalls → Improve cache locality, prefetching
        ├─→ Low MBU (<50%) → CPU overhead or sync issues
        │   └─→ Check: kernel launch gaps, NCCL times, system calls
        └─→ Balanced (MBU ~70-80%) → Optimized! Consider scaling strategies
```

---

## Common Findings and Solutions

### Finding 1: Memory-Bound Decode (Most Common)

**Symptoms**:
- High `gpu__dram_throughput` (>70% of peak)
- Low L2 hit rate (<50%)
- Low Tensor Core utilization in attention
- MBU 50-70%

**Root cause**: KV cache reads dominate; decode attention is O(seq_len) memory accesses with O(1) compute per token.

**Solutions**:
- KV cache quantization (INT8/INT4)
- PagedAttention (vLLM) for better memory layout
- Continuous batching to amortize KV cache reads
- Flash Attention (reduce memory footprint)

### Finding 2: Compute-Bound Prefill

**Symptoms**:
- High Tensor Core utilization (>80%)
- Low DRAM throughput (<30%)
- Kernel durations scale with O(seq_len²)

**Root cause**: Attention computation in prefill is O(n²), compute-intensive.

**Solutions**:
- Already well-optimized if at hardware limit
- Use Flash Attention for better arithmetic intensity
- Consider larger batches (if memory allows)
- Scale to more/faster GPUs

### Finding 3: CPU/Synchronization Overhead

**Symptoms**:
- Large gaps between GPU kernels in Nsight Systems
- CPU time >20% of total
- Low GPU utilization despite low kernel times

**Root cause**: Framework overhead, Python GIL, kernel launch latency, insufficient pipelining.

**Solutions**:
- Kernel fusion (reduce launch count)
- Async kernel launches
- C++ inference engines (TensorRT-LLM, TGI)
- Prefill-decode disaggregation ([DistServe approach](https://hao-ai-lab.github.io/blogs/distserve/))

### Finding 4: Low Tensor Core Utilization on GEMMs

**Symptoms**:
- `sm__pipe_tensor_op_hmma_cycles_active` <40%
- Matrix multiplications don't reach expected TFLOP/s

**Root cause**: Non-optimal matrix shapes, wrong dtypes, or poor tiling.

**Solutions**:
- Ensure FP16/BF16 dtypes (not FP32)
- Pad dimensions to multiples of 8 (or 16 for better TC alignment)
- Check cuBLASLt algorithm selection logs
- Review custom kernel implementations

([Stack Overflow - Tensor Core Occupancy](https://stackoverflow.com/questions/78948612/how-to-check-my-tensor-core-occupancy-and-utilization-by-nsight-compute))

---

## Summary Checklist

**Before optimizing, complete this analysis sequence**:

- [ ] Phase 1: Calculate theoretical arithmetic intensity for your workload
- [ ] Phase 2: Separate prefill vs decode in traces (NVTX ranges)
- [ ] Phase 3: Profile with Nsight Systems, identify hot kernels and phases
- [ ] Phase 4: Deep-dive hot kernels with Nsight Compute (SM, DRAM, cache metrics)
- [ ] Phase 5: Compute arithmetic intensity and position on roofline model
- [ ] Phase 6: Calculate MBU (Memory Bandwidth Utilization)
- [ ] Phase 7: Check for CPU overhead, sync latencies, multi-GPU comm bottlenecks
- [ ] Phase 8: Operator-level breakdown (PyTorch Profiler)
- [ ] Classify bottleneck: Compute-bound / Memory-bound / Sync-bound
- [ ] Apply targeted optimizations based on bottleneck type

**Remember**: Accurate bottleneck identification is more valuable than premature optimization. Invest time in profiling before making changes.

---

## References

### Research Papers & Articles
- [Mind the Memory Gap: GPU Bottlenecks in Large-Batch LLM Inference (arXiv 2025)](https://arxiv.org/html/2503.08311v2)
- [Understanding Bottlenecks in LLM Workloads - Compute, Memory, and Bandwidth (Medium, 2025)](https://medium.com/@aruna.kolluru/understanding-bottlenecks-in-llm-workloads-compute-memory-and-bandwidth-cdcef2fde252)
- [Inside Real-Time LLM Inference: From Prefill to Decode (Medium, 2025)](https://medium.com/@devsp0703/inside-real-time-llm-inference-from-prefill-to-decode-explained-72a1c9b1d85a)
- [LLM Inference Performance Engineering: Best Practices (Databricks, 2025)](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [Throughput is Not All You Need: DistServe (UCSD Hao AI Lab)](https://hao-ai-lab.github.io/blogs/distserve/)
- [LLM Inference Series: Dissecting Model Performance (Medium, 2025)](https://medium.com/@plienhar/llm-inference-series-5-dissecting-model-performance-6144aa93168f)

### NVIDIA Documentation
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Profiling LLM Workflows on Grace Hopper (NVIDIA Blog)](https://developer.nvidia.com/blog/profiling-llm-training-workflows-on-nvidia-grace-hopper/)
- [LLM Inference Benchmarking with GenAI-Perf (NVIDIA Blog)](https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/)

### Tools & Libraries
- [NERSC Roofline on NVIDIA GPUs (GitLab)](https://gitlab.com/NERSC/roofline-on-nvidia-gpus)
- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [vLLM: PagedAttention and Continuous Batching](https://github.com/vllm-project/vllm)

### Additional Reading
- [A Guide to LLM Inference and Performance (Baseten)](https://www.baseten.co/blog/llm-transformer-inference-guide/)
- [OS-Level Challenges in LLM Inference (Eunomia, 2025)](https://eunomia.dev/blog/2025/02/18/os-level-challenges-in-llm-inference-and-optimizations/)
- [Scaling LLM Inference at Meta (Engineering Blog, 2025)](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)
