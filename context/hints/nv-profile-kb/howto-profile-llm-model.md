Question:
- we are benchmarking LLM inference on nvidia GPU, our goal is find out the performance bottleneck, computation resource usage pattern, resource utilization, dnn operations count, theoretical flops, actual flops achieved, cache hit/miss patterns, etc. In the end, we will alter the design of the LLM to fully utilize certain kind of hardware, including Nvidia/AMD/Apple/Snapdragon/Custom NPU processors, so we want to collect as much performance stats as possible, and relate those stats to the model design. First, according to the current best practices, what stats will be useful to find out (just in nvidia GPU, no need to consider others), list them, and then tell me what tools are needed to get those stats, and how. We focus on python.

Response:
Here’s a NVIDIA-only, Python-first checklist of the **metrics worth collecting for LLM inference**—and exactly **which tools** surface them (and how).

# What to measure (LLM inference on NVIDIA GPUs)

**End-to-end**

* Throughput & latency: TTFT (first-token), TPOT (per-token), tokens/s; request queueing/warmup time. Use timers + Nsight Systems for the whole timeline. ([NVIDIA Docs][1])
* GPU clocks, power, thermals, perf state (throttle reasons), per-process GPU util. Use NVML (pynvml) / DCGM. ([NVIDIA Developer][2])

**Kernel / SM (streaming multiprocessor)**

* **Achieved vs theoretical occupancy**, warp execution/issue efficiency. Nsight Compute “Occupancy” & “Warp State” sections. ([NVIDIA Docs][3])
* **Warp stall breakdown** (e.g., memory dependency, barrier, not selected). Nsight Compute’s sampled **stall_*** metrics. ([NVIDIA Docs][4])
* **Tensor Core & FP/INT pipeline utilization** (e.g., `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`, `sm__throughput.*`). Nsight Compute. ([NVIDIA Developer Forums][5])

**Memory & caches**

* **DRAM bandwidth / bytes moved** (`dram__throughput.avg.pct_of_peak_*`, `dram__bytes_[read|write].sum`). Nsight Compute. ([NVIDIA Developer Forums][6])
* **L2 hit rate** (`lts__t_sector_hit_rate.pct`) and L1TEX transactions/coalescing (`l1tex__t_sectors_*`, `l1tex__t_requests_*`). Nsight Compute Memory Workload Analysis. ([NVIDIA Docs][4])
* **Shared-memory bank conflicts** & local/texture traffic (L1TEX counters). Nsight Compute. ([NVIDIA Developer][7])
* **Arithmetic intensity (FLOPs/byte)** per hot kernel for a roofline view. Use Nsight Compute FLOP counters + DRAM bytes. ([about.gitlab.com][8])

**FLOPs & MFU**

* **Achieved FLOPs** per kernel/op (from shapes + time) and **MFU** (achieved / peak). Use Nsight Compute counters and/or cuBLAS logs for GEMM shapes; compute 2MNK / time. (Roofline/MFU workflow references below.) ([about.gitlab.com][8])
* **Theoretical peak** from device specs; compare to `sm__throughput.avg.pct_of_peak_*`. Nsight Compute has the %-of-peak views. ([Stack Overflow][9])

**Framework/op level**

* **Op-level time & kernel correlation**, tensor shapes, stacks (PyTorch). `torch.profiler` (Kineto) with CUDA activities; export Chrome trace. Nsight Systems consumes NVTX ranges for clean timelines. ([PyTorch Docs][10])
* **Allocator behavior** (fragmentation, active/peak, snapshots). `torch.cuda.memory_stats()` / memory snapshots. ([PyTorch][11])

**Serving/runtime specifics**

* **TensorRT engine layer times** (`trtexec --dumpProfile`, or Python IProfiler). ([NVIDIA Docs][12])
* **Triton Inference Server request/GPU metrics** (Prometheus endpoint). ([NVIDIA Docs][13])
* **NCCL comm** (multi-GPU TP/PP) debug timings via env logs. ([NVIDIA Docs][14])

---

# How to collect (Python-first workflow)

## 1) Put NVTX ranges around meaningful stages

NVTX makes Nsight Systems/Compute views readable.

```python
import torch, time
from torch.cuda import nvtx

def run_step(model, *args, **kw):
    with nvtx.range("prefill"):
        out = model(*args, **kw)  # e.g., prompt/prefill
    with nvtx.range("decode"):
        # your token-by-token loop
        pass
```

Nsight Systems/Compute record these ranges; use them to slice traces. ([NVIDIA GitHub][15])

## 2) PyTorch operator + CUDA timeline

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True, with_stack=True, profile_memory=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb")
) as prof:
    # run your inference loop here
    pass
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

Gives per-op times, shapes, and a Chrome trace (TensorBoard viewer). ([PyTorch Docs][10])

## 3) Whole-program timeline + concurrency (Nsight Systems)

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=nvtx --capture-range-end=nvtx \
  -o nsys_llm python serve.py --args
```

Open `.qdrep` in GUI to see CPU↔GPU, kernel overlaps, memcpy, launches, etc. NVTX `range()` calls gate capture. ([NVIDIA Docs][1])

## 4) Kernel deep-dive (Nsight Compute CLI)

Profile just the hot kernels from the NSYS run:

```bash
# Example: collect tensor-core, DRAM, L2, occupancy & stalls
ncu --target-processes all \
    --set full \
    --kernels ::matmul* \
    --metrics \
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_sector_hit_rate.pct \
    -o ncu_llm python serve.py --args
```

Use Memory Workload + Warp State + Occupancy sections for cache hits, bytes, stalls, and achieved occupancy. ([NVIDIA Docs][4])

## 5) FLOPs / MFU / roofline

* From NCU: take per-kernel time & bytes (`dram__bytes_*`) and compute arithmetic intensity; place points on a roofline (DRAM vs SM ceilings). Scripts in NERSC “roofline-on-nvidia-gpus” list the exact NCU metrics to pull. ([about.gitlab.com][8])
* For GEMMs, enable cuBLAS logging to capture M,N,K:

```bash
CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=cublas.log \
python your_infer.py
```

Then compute `2*M*N*K / kernel_time` for achieved FLOPs (sum across kernels) and divide by peak to get MFU. ([NVIDIA Docs][16])

## 6) System counters in Python (NVML / DCGM)

```python
import pynvml as nvml, time
nvml.nvmlInit()
h = nvml.nvmlDeviceGetHandleByIndex(0)
for _ in range(10):
    util = nvml.nvmlDeviceGetUtilizationRates(h)
    power = nvml.nvmlDeviceGetPowerUsage(h)  # mW
    clocks = nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM)
    print(util.gpu, util.memory, power, clocks)
    time.sleep(1)
```

For fleet monitoring, run **DCGM Exporter** (Prometheus `/metrics`) to scrape SM active/occupancy, clocks, power, XID errors. ([PyPI][17])

## 7) Framework/runtime specifics

* **TensorRT**: quick per-layer times with:

```bash
trtexec --loadEngine=llm.plan --dumpProfile --profilingVerbosity=detailed
```

Or in Python, attach `tensorrt.IProfiler` to an `IExecutionContext`. ([NVIDIA Docs][12])

* **Triton Server**: `curl localhost:8002/metrics` → end-to-end latency, queue time, GPU util, memory. ([NVIDIA Docs][13])
* **PyTorch allocator**: `torch.cuda.memory_stats()`, memory snapshots (drag-and-drop viewer) for fragmentation & peak tracking. ([PyTorch][11])

---

# Quick “starter” commands you can run

1. **Timeline first**

```bash
nsys profile --trace=cuda,nvtx,osrt -o nsys_llm python your_script.py
```

Open in Nsight Systems; identify hot ranges/kernels. ([NVIDIA Docs][1])

2. **Deep-dive the hot kernels**

```bash
ncu --target-processes all --set full \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_sector_hit_rate.pct \
    -o ncu_hot python your_script.py
```

Check Occupancy, Warp Stalls, Memory Analysis pages. ([NVIDIA Docs][4])

3. **cuBLAS shape logs (optional, helps FLOPs calc)**

```bash
CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=cublas.log python your_script.py
```

Parse M/N/K from `cublas.log`; compute `2MNK/time` per GEMM. ([NVIDIA Docs][16])

4. **NVML sampling (headless)**

```python
# see Python snippet above
```

Use for long-running jobs & to confirm throttling/clocks. ([NVIDIA Developer][2])

---

## Notes that help you map stats → model design

* If **SM% is low but DRAM% high** and `lts__t_sector_hit_rate` is low → memory-bound (KV-cache heavy). Consider KV-cache layout/paging (e.g., vLLM PagedAttention), quantized KV, or reducing reads via attention optimizations. ([NVIDIA Developer Forums][18])
* If **Tensor Core utilization** is low on GEMMs → check matmul shapes (batching, head dims) and ensure TC-eligible dtypes (BF16/FP16/INT8) & strides; cuBLASLt logs show selected algorithms. ([Stack Overflow][19])
* High **stall_memory_dependency / stall_not_selected** with low occupancy → adjust launch parameters/fusion, or increase parallelism to hide latency. ([NVIDIA Docs][4])

If you want, I can turn this into a minimal **profiling harness** that wraps your inference function with NVTX + `torch.profiler`, spits out an `nsys` capture, an `ncu` report, and a small CSV for FLOPs/byte—ready to compare variants.

[1]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html?utm_source=chatgpt.com "User Guide — nsight-systems"
[2]: https://developer.nvidia.com/management-library-nvml?utm_source=chatgpt.com "NVIDIA Management Library (NVML)"
[3]: https://docs.nvidia.com/nsight-compute/2023.2/ProfilingGuide/index.html?utm_source=chatgpt.com "Kernel Profiling Guide :: Nsight Compute Documentation"
[4]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html?utm_source=chatgpt.com "2. Profiling Guide — NsightCompute 12.9 documentation"
[5]: https://forums.developer.nvidia.com/t/tensor-metrics-in-nsightcompute/82939?utm_source=chatgpt.com "Tensor metrics in NsightCompute - Nsight Compute"
[6]: https://forums.developer.nvidia.com/t/how-to-get-dram-throughput-in-nsight-system/310253?utm_source=chatgpt.com "How to get dram throughput in Nsight system？"
[7]: https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/?utm_source=chatgpt.com "Using Nsight Compute to Inspect your Kernels"
[8]: https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/tree/cuda11.0.2-ncu?utm_source=chatgpt.com "roofline-on-nvidia-gpus - NERSC"
[9]: https://stackoverflow.com/questions/73679977/why-is-the-compute-throughput-s-value-different-from-the-actual-performance-pe?utm_source=chatgpt.com "Why is the Compute Throughput's value different from ..."
[10]: https://docs.pytorch.org/docs/stable/profiler.html?utm_source=chatgpt.com "torch.profiler — PyTorch 2.9 documentation"
[11]: https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html?utm_source=chatgpt.com "torch.cuda.memory_stats"
[12]: https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html?utm_source=chatgpt.com "Command-Line Programs — NVIDIA TensorRT ..."
[13]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html?utm_source=chatgpt.com "Metrics — NVIDIA Triton Inference Server"
[14]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html?utm_source=chatgpt.com "Environment Variables — NCCL 2.28.6 documentation"
[15]: https://nvidia.github.io/NVTX/?utm_source=chatgpt.com "NVTX - NVIDIA Tools Extension Library"
[16]: https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf?utm_source=chatgpt.com "cuBLAS"
[17]: https://pypi.org/project/nvidia-ml-py/?utm_source=chatgpt.com "nvidia-ml-py 13.580.82"
[18]: https://forums.developer.nvidia.com/t/what-does-memory-throughput-in-nsight-compute-mean/342631?utm_source=chatgpt.com "What does \"Memory Throughput\" in Nsight Compute mean?"
[19]: https://stackoverflow.com/questions/78948612/how-to-check-my-tensor-core-occupancy-and-utilization-by-nsight-compute?utm_source=chatgpt.com "How to check my tensor core occupancy and utilization by ..."

---

# Practical Example

Awesome — here’s a compact, production-ready **Python profiling harness** you can drop into any LLM inference script. It wraps **PyTorch Profiler** for op-level timing & memory, adds **NVTX** ranges so **Nsight Systems** timelines are readable, spawns an **NVML** sampler (power/clocks/util), and gives you one-liners to collect **Nsight Compute** kernel metrics for roofline/MFU work.

Citations at the end point to the exact docs for each tool/flag.

---

# 📦 `llmprof_harness.py`

```python
"""
LLM Profiling Harness (NVIDIA, Python-first)

Features
- PyTorch Profiler (CPU/CUDA, op-level timing, memory, shapes, stacks)
- NVTX ranges for clean Nsight Systems timelines
- NVML sampler thread -> CSV (power, clocks, utilization, memory)
- Helper to parse PyTorch Profiler into CSV
- One-liners to run Nsight Systems / Nsight Compute around your script
- Optional cuBLAS logging enablement for GEMM shape-based FLOPs

Usage (basic)
------------
from llmprof_harness import (
    NVMLSampler, nvtx_range, run_torch_profiler,
    advise_nsys_cmd, advise_ncu_cmd, enable_cublas_logging
)

# 1) NVML system counters while you run:
sampler = NVMLSampler(device_index=0, interval_s=0.2, out_csv="nvml.csv")
sampler.start()

with nvtx_range("e2e"):
    # your inference loop here (add more nvtx_range('prefill'), nvtx_range('decode'))
    do_inference()

sampler.stop(); sampler.join()

# 2) Wrap your inference step(s) with PyTorch Profiler:
def step(i):
    # run one end-to-end inference "step" (your code)
    do_inference()

prof = run_torch_profiler(step, steps=50, warmup=5, out_dir="prof_out")
prof_to_csv(prof, "ops.csv")

# 3) Outside Python: run Nsight Systems / Compute
print("NSYS:", advise_nsys_cmd(script="serve.py", out="nsys_llm"))
print("NCU :", advise_ncu_cmd(script="serve.py", out="ncu_llm"))

# (Optional) cuBLAS logs for GEMM shapes/timings -> FLOPs:
enable_cublas_logging("cublas.log")
"""

from __future__ import annotations
import csv
import os
import sys
import time
import threading
import subprocess
from contextlib import contextmanager

# ---- NVTX (PyTorch binding) -------------------------------------------------
try:
    import torch
    import torch.cuda.nvtx as nvtx
    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
except Exception as e:
    raise RuntimeError("This harness requires PyTorch with CUDA. pip install torch --index-url per your CUDA build") from e

@contextmanager
def nvtx_range(name: str):
    """NVTX range for Nsight Systems capture gating & readable timelines."""
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()

# ---- PyTorch Profiler wrapper -----------------------------------------------
def run_torch_profiler(
    step_fn,
    steps: int,
    warmup: int = 5,
    repeat: int = 1,
    out_dir: str = "prof_out",
    record_shapes: bool = True,
    with_stack: bool = True,
    profile_memory: bool = True,
):
    """
    Run an arbitrary step_fn(i) inside torch.profiler with CUDA & CPU activities,
    emitting TensorBoard traces in out_dir and returning the Profile object.

    On large models, keep repeat small; tune steps & warmup for your workload.
    """
    os.makedirs(out_dir, exist_ok=True)
    sch = schedule(wait=0, warmup=warmup, active=steps, repeat=repeat)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        schedule=sch,
        record_shapes=record_shapes,
        with_stack=with_stack,
        profile_memory=profile_memory,
        on_trace_ready=tensorboard_trace_handler(out_dir),
    ) as prof:
        total_iters = warmup + steps
        for i in range(total_iters):
            with nvtx_range(f"step_{i}"):
                step_fn(i)
            prof.step()
    return prof

def prof_to_csv(prof, csv_path: str):
    """
    Dump a compact CSV of op-level aggregates:
      key, cuda_time_total_us, cpu_time_total_us, self_cuda_time_total_us, calls, bytes (if available)
    """
    fields = ["key", "cuda_time_total_us", "cpu_time_total_us", "self_cuda_time_total_us", "calls", "input_shapes", "self_cpu_memory_usage_bytes"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for evt in prof.key_averages():
            row = [
                evt.key,
                getattr(evt, "cuda_time_total", 0.0),
                getattr(evt, "cpu_time_total", 0.0),
                getattr(evt, "self_cuda_time_total", 0.0),
                getattr(evt, "count", 0),
                getattr(evt, "input_shape", None) or getattr(evt, "input_shapes", None),
                getattr(evt, "self_cpu_memory_usage", None),
            ]
            w.writerow(row)

# ---- NVML sampler (util, mem, power, clocks) --------------------------------
class NVMLSampler(threading.Thread):
    """
    Periodically sample NVML counters to CSV.
    Columns: ts, gpu_util_%, mem_util_%, sm_clock_MHz, mem_clock_MHz, power_W, temp_C, mem_used_MiB, mem_total_MiB
    """
    def __init__(self, device_index=0, interval_s=0.5, out_csv="nvml.csv"):
        super().__init__(daemon=True)
        self.device_index = device_index
        self.interval_s = interval_s
        self.out_csv = out_csv
        self._running = threading.Event()
        self._running.set()
        self._init_nvml()

    def _init_nvml(self):
        import pynvml as nvml
        self.nvml = nvml
        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByIndex(self.device_index)
        # open file
        self.f = open(self.out_csv, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["ts", "gpu_util_pct", "mem_util_pct", "sm_clock_MHz", "mem_clock_MHz", "power_W", "temp_C", "mem_used_MiB", "mem_total_MiB"])

    def run(self):
        nvml = self.nvml
        try:
            while self._running.is_set():
                t = time.time()
                util = nvml.nvmlDeviceGetUtilizationRates(self.handle)
                sm_clk = nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_SM)
                mem_clk = nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_MEM)
                power = nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # W
                temp = nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)
                mem = nvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.w.writerow([
                    f"{t:.6f}", util.gpu, util.memory, sm_clk, mem_clk, f"{power:.3f}", temp,
                    int(mem.used / (1024*1024)), int(mem.total / (1024*1024))
                ])
                self.f.flush()
                time.sleep(self.interval_s)
        finally:
            self.f.close()
            # don't nvmlShutdown here (other samplers may run)

    def stop(self):
        self._running.clear()

# ---- Nsight Systems / Nsight Compute helpers --------------------------------
def advise_nsys_cmd(script: str, out: str = "nsys_llm", extra_args: list[str] | None = None) -> str:
    """
    Returns a ready-to-run Nsight Systems command that uses NVTX capture gating and CUDA tracing.
    Run it in your shell to collect a .qdrep you can open in the GUI.
    """
    extra_args = extra_args or []
    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx,osrt",
        "--capture-range=nvtx", "--capture-range-end=nvtx",
        "--sample=none",              # timeline only (reduce overhead)
        "-o", out,
        "python", script, *extra_args
    ]
    return " ".join(cmd)

def advise_ncu_cmd(script: str, out: str = "ncu_llm", extra_args: list[str] | None = None) -> str:
    """
    Returns an Nsight Compute CLI command capturing a solid default metric set for LLM kernels.
    Adjust kernels filter to your hotspots (e.g., ::matmul*, ::attention*).
    """
    extra_args = extra_args or []
    metrics = ",".join([
        # Tensor Core / SM throughput
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        # DRAM bytes & percentage of peak
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum","dram__bytes_write.sum",
        # L2 hit rate
        "lts__t_sector_hit_rate.pct",
        # Warp stall summary (available via sections; keep small here)
        "smsp__warp_stall_memory_dependency_per_warp_active.pct",
        "smsp__warp_stall_not_selected_per_warp_active.pct"
    ])
    cmd = [
        "ncu", "--target-processes", "all",
        "--set", "full",  # collect rich sections; use --set basic for lower overhead
        "--metrics", metrics,
        "-o", out,
        "python", script, *extra_args
    ]
    return " ".join(cmd)

# ---- cuBLAS logging (optional, for GEMM shapes/time -> FLOPs) ---------------
def enable_cublas_logging(log_path: str = "cublas.log"):
    """
    Enable cuBLAS logger via env vars (set before your process starts).
    In shell:   CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=cublas.log python serve.py
    """
    os.environ["CUBLAS_LOGINFO_DBG"] = "1"
    os.environ["CUBLAS_LOGDEST_DBG"] = log_path

def parse_cublas_log_for_gemm_flops(log_path: str) -> list[dict]:
    """
    Very lightweight parser (best-effort). Returns a list of dicts with m,n,k,algo,t_ms,flops.
    You’ll likely adapt this to your cuBLAS version/log format.
    """
    import re
    rows = []
    gemm_re = re.compile(r".*cublas.*gemm.*m=(\d+).*n=(\d+).*k=(\d+).*time=(\d+\.\d+) ms", re.IGNORECASE)
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = gemm_re.search(line)
            if m:
                M,N,K = map(int, m.groups()[:3])
                t_ms = float(m.group(4))
                flops = 2.0 * M * N * K / (t_ms * 1e-3)  # FLOPs/s
                rows.append({"M":M, "N":N, "K":K, "t_ms":t_ms, "flops_per_s":flops})
    return rows

# ---- tiny self-test ----------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic example: a tiny matmul-based "decode" step
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L, D = 2048, 4096
    W = torch.randn(D, D, device=device, dtype=torch.float16)
    x = torch.randn(L, D, device=device, dtype=torch.float16)

    def one_step(i):
        with nvtx_range("prefill" if i == 0 else "decode"):
            y = x @ W
            # simulate KV lookup / memory traffic
            y = y + torch.roll(y, shifts=1, dims=0)
            torch.cuda.synchronize()

    sampler = NVMLSampler(interval_s=0.2)
    sampler.start()
    prof = run_torch_profiler(one_step, steps=20, warmup=5, out_dir="prof_out")
    prof_to_csv(prof, "ops.csv")
    sampler.stop(); sampler.join()

    print("Try running these in your shell:")
    print("  ", advise_nsys_cmd(script=os.path.basename(__file__)))
    print("  ", advise_ncu_cmd(script=os.path.basename(__file__)))
```

---

## How to use it in your LLM script

1. **Instrument with NVTX** around your phases so Nsight Systems shows a clean, labeled timeline:

```python
from llmprof_harness import nvtx_range

with nvtx_range("e2e"):
    with nvtx_range("prefill"):
        logits = model(prompt_ids, use_cache=True)
    with nvtx_range("decode"):
        for _ in range(num_new_tokens):
            logits = model(next_token, use_cache=True)  # your decode loop
```

2. **Op-level timing & memory**:

```python
from llmprof_harness import run_torch_profiler, prof_to_csv
def step(_):
    run_one_full_request()  # your function
prof = run_torch_profiler(step, steps=50, warmup=5, out_dir="prof_out")
prof_to_csv(prof, "ops.csv")  # quick CSV summary
```

3. **System counters while you run**:

```python
from llmprof_harness import NVMLSampler
sampler = NVMLSampler(device_index=0, interval_s=0.2, out_csv="nvml.csv")
sampler.start()
run_many_requests()
sampler.stop(); sampler.join()
```

4. **Whole-program timeline (Nsight Systems)** — run the suggested command:

```
nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --capture-range-end=nvtx \
  -o nsys_llm python your_script.py
```

Open `nsys_llm.qdrep` in the GUI to see CPU↔GPU, kernel overlaps, memcpys, launch bottlenecks. ([NVIDIA Docs][1])

5. **Kernel deep-dive (Nsight Compute)** — start with a broad set, then narrow:

```
ncu --target-processes all --set full \
  --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,\
smsp__warp_stall_memory_dependency_per_warp_active.pct,smsp__warp_stall_not_selected_per_warp_active.pct \
  -o ncu_llm python your_script.py
```

Use `--set basic` for lower overhead, or pick sections/metrics explicitly once you know your hotspots. ([NVIDIA Docs][2])

6. **Optional: cuBLAS GEMM shapes → achieved FLOPs / MFU**

Set env vars before starting Python so cuBLAS logs M/N/K, times, and algorithms:

```
CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=cublas.log python your_script.py
```

Then parse `cublas.log` with `parse_cublas_log_for_gemm_flops()` and aggregate `2*M*N*K / time`. Compare to device peak (from specs or NCU %-of-peak) to estimate MFU. ([NVIDIA Docs][3])

7. **Fleet/serving metrics (optional)**

If you’re serving through Triton or have a cluster, you can also run **DCGM Exporter** to scrape per-GPU counters (Prometheus `/metrics`) alongside this harness. ([NVIDIA Docs][4])

---

## Why these pieces (and where they come from)

* **PyTorch Profiler** gives op-level timing/memory with shapes/stacks and exports a Chrome/TensorBoard trace. It’s the fastest way to find slow ops and correlate with your Python code. ([PyTorch Docs][5])
* **NVTX** ranges let **Nsight Systems** display readable E2E/prefill/decode bands and (with `--capture-range=nvtx`) *gate* collection to just the interesting window. ([PyTorch Docs][6])
* **Nsight Compute** collects kernel-level metrics (SM/Tensor Core utilization, stall breakdowns, DRAM bytes, L2 hit rate) and supports section sets like `--set full`/`basic` to balance detail vs. overhead. ([NVIDIA Docs][2])
* **NVML** (via `pynvml`) gives per-process/system **power, clocks, thermals, utilization**, essential to understand throttling and perf-state effects during your run. ([NVIDIA Developer][7])
* **cuBLAS logging** is the sanctioned way to capture GEMM shapes/algorithms to compute achieved FLOPs from first principles for MFU/roofline analysis. ([NVIDIA Docs][3])
* For cluster dashboards, **DCGM Exporter** exposes GPU telemetry on `/metrics`, which you can scrape alongside your per-run CSVs from this harness. ([NVIDIA Docs][4])

---

### Install notes

This section covers installation of profiling-specific tools on Ubuntu (tested on 22.04 and 24.04). Assumes you already have NVIDIA drivers and CUDA Toolkit installed.

#### Prerequisites Check

```bash
# Verify NVIDIA driver and CUDA are installed
nvidia-smi          # Should show GPU info
nvcc --version      # Should show CUDA version
```

#### 1. Install Nsight Systems

**Option A: Download from NVIDIA Developer (Latest version, recommended)**

```bash
# Visit https://developer.nvidia.com/nsight-systems/get-started
# Download the .deb file for your Ubuntu version

# Example installation (replace with current version):
wget https://developer.download.nvidia.com/devtools/nsight-systems/2024_4/nsight-systems-2024.4.1_2024.4.1.76-1_amd64.deb
sudo dpkg -i nsight-systems-2024.4.1_2024.4.1.76-1_amd64.deb
sudo apt --fix-broken install  # if there are dependency issues
```

**Option B: Included with CUDA Toolkit**

If you installed the full CUDA package, `nsys` should already be available:
```bash
nsys --version
```

**Option C: Ubuntu Repository (may be older)**

```bash
sudo apt install nsight-systems
```

**System requirement**: Nsight Systems requires write permission to `/var/lock`:
```bash
ls -ld /var/lock          # Check current permissions
sudo chmod 755 /var/lock  # Fix if needed
```

#### 2. Install Nsight Compute

**Option A: Download from NVIDIA Developer (Latest version, recommended)**

```bash
# Visit https://developer.nvidia.com/tools-overview/nsight-compute/get-started
# Download the .deb file for your Ubuntu version

# Example installation (replace with current version):
wget https://developer.download.nvidia.com/devtools/nsight-compute/2024_3/ncu_2024.3.0.13_amd64.deb
sudo dpkg -i ncu_2024.3.0.13_amd64.deb
sudo apt --fix-broken install  # if there are dependency issues
```

**Option B: Included with CUDA Toolkit**

If you installed the full CUDA package, `ncu` should already be available:
```bash
ncu --version
```

**Option C: Ubuntu Repository (may be older)**

```bash
sudo apt install nsight-compute
```

**Option D: Via Conda**

```bash
conda install nvidia::nsight-compute
```

#### 3. Install Python Dependencies

**PyTorch with CUDA support (includes NVTX)**

NVTX (NVIDIA Tools Extension) for profiling is included in PyTorch's CUDA builds via `torch.cuda.nvtx`. Install PyTorch with CUDA support:

```bash
# Visit https://pytorch.org/get-started/locally/ for the latest command
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify NVTX is available:
```bash
python -c "import torch; from torch.cuda import nvtx; print('NVTX available')"
```

**NVIDIA ML Python bindings (NVML/pynvml)**

For GPU monitoring (utilization, power, clocks, memory):

```bash
# Install official NVIDIA ML Python bindings
pip install nvidia-ml-py

# Alternative (older, now wraps nvidia-ml-py):
pip install nvidia-ml-py3
```

Note: `pynvml` package is now deprecated. Use `nvidia-ml-py` (official) or `nvidia-ml-py3` (maintained fork).

Verify NVML is available:
```bash
python -c "import pynvml; pynvml.nvmlInit(); print('NVML initialized successfully')"
```

**Optional: TensorBoard for viewing PyTorch Profiler traces**

```bash
pip install tensorboard
```

#### 4. Verify Installation

```bash
# Check profiling tools
nsys --version
ncu --version

# Test Python environment
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}')"
python -c "import pynvml; pynvml.nvmlInit(); print('NVML initialized successfully')"
```

#### Troubleshooting

**If `nsys` or `ncu` are not found:**
```bash
# Check if tools are in your CUDA installation
ls /usr/local/cuda/bin/{nsys,ncu}

# If present, ensure CUDA bin is in PATH:
export PATH=/usr/local/cuda/bin:$PATH
# Add to ~/.bashrc to make permanent
```

**If PyTorch doesn't detect CUDA:**
```bash
# Check CUDA driver vs PyTorch CUDA version compatibility
nvidia-smi  # Check driver version (e.g., CUDA 12.2)
python -c "import torch; print(torch.version.cuda)"  # Must be compatible
# If incompatible, reinstall PyTorch with matching CUDA version
```

**For GUI tools on remote servers:**
- Use X forwarding: `ssh -X user@server`
- Or run CLI-only and copy `.qdrep`/`.ncu-rep` files to local machine for GUI viewing

#### References
- Nsight Systems: https://developer.nvidia.com/nsight-systems/get-started
- Nsight Compute: https://developer.nvidia.com/tools-overview/nsight-compute/get-started
- PyTorch Installation: https://pytorch.org/get-started/locally/

---

If you want, I can tailor this to your exact stack (e.g., vLLM/TensorRT-LLM/Triton) and add prefill/decode hooks plus an NCU CSV parser that computes arithmetic intensity per hot kernel for an automated roofline CSV.

[1]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html?utm_source=chatgpt.com "User Guide — nsight-systems"
[2]: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html?utm_source=chatgpt.com "4. Nsight Compute CLI"
[3]: https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf?utm_source=chatgpt.com "cuBLAS"
[4]: https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html?utm_source=chatgpt.com "DCGM Exporter — NVIDIA GPU Telemetry 1.0.0 ..."
[5]: https://docs.pytorch.org/docs/stable/profiler.html?utm_source=chatgpt.com "torch.profiler — PyTorch 2.9 documentation"
[6]: https://docs.pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_push.html?utm_source=chatgpt.com "torch.cuda.nvtx.range_push"
[7]: https://developer.nvidia.com/management-library-nvml?utm_source=chatgpt.com "NVIDIA Management Library (NVML)"
