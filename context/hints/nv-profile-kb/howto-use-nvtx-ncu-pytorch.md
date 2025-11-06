# How to Use NVTX Ranges with NCU for PyTorch Models

**Last Updated**: 2025-11-07
**Target Audience**: Engineers profiling PyTorch models with Nsight Compute range replay
**Prerequisites**: CUDA-enabled GPU, `nvtx` Python package, Nsight Compute 2025.x+

---

## Overview

NVTX (NVIDIA Tools Extension) ranges allow you to selectively profile specific regions of your PyTorch model with Nsight Compute, avoiding the overhead of profiling the entire application. However, **PyTorch's asynchronous CUDA execution and NCU's range replay requirements create several pitfalls** that cause the infamous "No ranges were profiled" warning.

This guide provides battle-tested patterns for successful NVTX+NCU profiling with PyTorch.

---

## Quick Start: The Correct Pattern

```python
import torch
import torch.nn as nn
import nvtx

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ CORRECT: Use nvtx.annotate() + torch.cuda.synchronize()
        with nvtx.annotate("layer1"):
            x = self.layer1(x)
            torch.cuda.synchronize()  # 🔑 CRITICAL for NCU range replay

        with nvtx.annotate("layer2"):
            x = self.layer2(x)
            torch.cuda.synchronize()  # 🔑 Ensures kernels finish before range ends

        return x

# Initialize model and warm up CUDA context
model = MyModel().cuda()
x = torch.randn(32, 512).cuda()
_ = model(x)  # 🔑 Warm-up run BEFORE NCU profiling
torch.cuda.synchronize()

# Now ready for NCU profiling
```

**NCU Command**:
```bash
# Note the trailing slash "/" for push/pop ranges!
ncu --nvtx --nvtx-include "layer1/" \
    --nvtx-include "layer2/" \
    --replay-mode app-range \
    --set roofline \
    --csv --log-file ncu_output.csv \
    -o ncu_report \
    python your_script.py
```

---

## The Three Critical Requirements

### 1. **Synchronization After Each Range** ⚡

**Why**: PyTorch CUDA operations are **asynchronous**. When your Python code exits an NVTX range, the CUDA kernels may still be queued or executing. NCU requires all work to be **launched and synchronizable within range boundaries**.

```python
# ❌ WRONG: Kernels may launch AFTER range ends
with nvtx.annotate("my_layer"):
    x = self.conv(x)  # Returns immediately, kernels still queued
# Range ends here, but kernels are still running → NCU sees empty range

# ✅ CORRECT: Force synchronization before range ends
with nvtx.annotate("my_layer"):
    x = self.conv(x)
    torch.cuda.synchronize()  # Wait for all kernels to finish
```

**Official Requirement**:
> "It must be possible to synchronize all active CUDA contexts at the start of the range." — Nsight Compute Profiling Guide

---

### 2. **Trailing Slash for Push/Pop Ranges** 📝

**Why**: `nvtx.annotate()` and `nvtx.push_range/pop_range` use the **push/pop** API internally. NCU distinguishes between:
- **Push/Pop ranges**: Require trailing `/` → `--nvtx-include "name/"`
- **Start/End ranges**: No trailing slash → `--nvtx-include "name"`

```bash
# ❌ WRONG: Missing trailing slash for push/pop
ncu --nvtx --nvtx-include "layer1" python script.py
# NCU looks for start/end range, but nvtx.annotate() uses push/pop → no match

# ✅ CORRECT: Add trailing slash
ncu --nvtx --nvtx-include "layer1/" python script.py
```

**Reference**: [NVIDIA Developer Forums #265420](https://forums.developer.nvidia.com/t/nsight-compute-failed-to-profile-with-nvtx-ranges-in-pytorch/265420)

---

### 3. **CUDA Context Warm-Up** 🔥

**Why**: Range replay fails if the CUDA context is initialized **inside** the NVTX range. PyTorch lazily initializes contexts on first CUDA operation.

```python
# ❌ WRONG: First CUDA operation inside NVTX range
model = MyModel()  # Still on CPU
with nvtx.annotate("inference"):
    model = model.cuda()  # Context initialization happens HERE → unsupported API
    x = torch.randn(32, 512).cuda()
    y = model(x)

# ✅ CORRECT: Initialize context BEFORE first NVTX range
model = MyModel().cuda()  # Initialize context
x = torch.randn(1, 512).cuda()
_ = model(x)  # Warm-up pass
torch.cuda.synchronize()

# Now safe to profile with NVTX
with nvtx.annotate("inference"):
    y = model(x)
    torch.cuda.synchronize()
```

**Root Cause**: NCU Range Replay rejects ranges containing initialization APIs like `cuInit`, context creation, or memory pool setup.

---

## Multiple Ranges: Repeated Flags

For profiling multiple regions, **repeat** the `--nvtx-include` flag:

```bash
# ✅ CORRECT: Repeated flags for multiple ranges
ncu --nvtx \
    --nvtx-include "layer1/" \
    --nvtx-include "layer2/" \
    --nvtx-include "layer3/" \
    python script.py

# ❌ WRONG: Single flag with semicolons (may not parse correctly)
ncu --nvtx --nvtx-include "layer1;layer2;layer3" python script.py
```

**Alternative**: Use glob patterns if ranges follow a naming convention:
```bash
# Profile all ranges starting with "layer"
ncu --nvtx --nvtx-include "layer*/" python script.py
```

---

## Replay Modes: Which to Use?

NCU offers three replay modes with different tradeoffs:

| Mode | Description | When to Use | Restrictions |
|------|-------------|-------------|--------------|
| `kernel` | Profile all kernels (default) | Most reliable, no NVTX needed | No selective profiling |
| `range` | Profile NVTX ranges (strict) | When ranges are simple, no framework overhead | Fails with unsupported APIs |
| `app-range` | Profile app ranges (permissive) | **Recommended for PyTorch/frameworks** | More overhead, less strict |

**Recommendation for PyTorch**: Start with `app-range`:
```bash
ncu --nvtx --nvtx-include "layer1/" \
    --replay-mode app-range \
    python script.py
```

If that works, try `range` mode for less overhead. If `range` fails with "No ranges were profiled", fallback to `app-range`.

---

## Controlling Replay Behavior and Count

NCU uses **replay** to collect different metric sets. Each section or metric may require replaying kernels multiple times. Understanding how to control replay behavior is critical for managing profiling time and overhead.

### Understanding NCU Replay

**Key Concept**: NCU doesn't profile everything in a single pass. Instead, it:
1. Captures the application execution
2. **Replays** specific kernels/ranges multiple times
3. Collects different metrics on each replay pass
4. Combines results into a final report

**Impact**:
- More metrics → More replays → Longer profiling time
- Fewer metrics → Fewer replays → Faster profiling
- Some sections require 10+ replays to collect all metrics

---

### Limiting Profiling Scope

#### 1. Limit Kernels with `--kernel-id` / `--kernel-name`

**Problem**: By default, NCU profiles **all kernels**. For large models, this can mean hundreds of kernels.

**Solution**: Profile only specific kernels.

```bash
# Profile only kernels matching a regex pattern
ncu --nvtx --nvtx-include "layer1/" \
    --kernel-name-base demangled \
    --kernel-name regex:".*gemm.*" \
    --replay-mode app-range \
    python script.py
```

**Use Cases**:
- Profile only matrix multiplication kernels: `--kernel-name regex:".*gemm.*"`
- Profile only convolution kernels: `--kernel-name regex:".*conv.*"`
- Profile multiple kernel patterns: `--kernel-name regex:".*gemm.*|.*conv.*"`
- Profile specific kernel by ID: `--kernel-id 5::10::2` (skip 5, profile next 10, advance by 2)

**Important**: Always use the `regex:` prefix for pattern matching. Without it, NCU expects an exact kernel name match.

---

#### 2. Limit Replays with `--launch-count`

**Problem**: A single kernel may be launched hundreds of times. NCU will replay **every launch**.

**Solution**: Profile only a subset of kernel launches.

```bash
# Profile only the first 10 launches of each kernel
ncu --nvtx --nvtx-include "layer1/" \
    --launch-count 10 \
    --replay-mode app-range \
    python script.py
```

**Common Patterns**:
```bash
# Profile first 5 launches only (fastest)
--launch-count 5

# Profile launches 10-20 (skip first 10, profile next 10)
--launch-skip 10 --launch-count 10

# Profile every 10th launch (sampling)
--launch-skip 9 --launch-count 1
```

**Why This Matters**:
- Transformer decode step: Same kernel launched 100+ times (once per token)
- Profiling all launches is redundant and slow
- Profiling 5-10 launches gives representative data

---

#### 3. Control Number of Kernel Replays with `--target-processes`

**Use Case**: When profiling multiprocess applications, control which processes are profiled.

```bash
# Profile only the main process (skip child processes)
ncu --target-processes 1 \
    --nvtx --nvtx-include "layer1/" \
    python script.py

# Profile all processes (default)
ncu --target-processes all \
    --nvtx --nvtx-include "layer1/" \
    python script.py
```

---

#### 4. Limit Metrics with `--metrics` (Instead of `--set`)

**Problem**: `--set full` collects 100+ metrics, requiring 10-20 replays per kernel.

**Solution**: Request only specific metrics.

```bash
# Instead of --set full (many replays)
ncu --set full  # ❌ Slow: 15+ replays

# Request only essential metrics (fewer replays)
ncu --metrics gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed  # ✅ Fast: ~3 replays
```

**List Available Metrics**:
```bash
ncu --query-metrics
```

**Common Metrics**:
- `gpu__time_duration.sum` - Kernel execution time
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` - SM utilization
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` - Memory bandwidth
- `sm__sass_thread_inst_executed_op_*` - Instruction counts
- `l1tex__t_sectors_pipe_lsu_mem_*` - Memory access patterns

---

#### 5. Use `--kernel-replay-mode` for Fine-Grained Control

Controls how NCU replays kernels to collect metrics:

```bash
# application (default): Replay at application level
ncu --kernel-replay-mode application

# range: Replay only within NVTX ranges (faster for range replay)
ncu --kernel-replay-mode range
```

---

### Trade-Off Matrix

| Configuration | Replays Per Kernel | Profiling Time | Use Case |
|---------------|-------------------|----------------|----------|
| `--set brief --launch-count 5` | 2-3 | Fastest | Quick overview, CI/CD |
| `--set roofline --launch-count 10` | 5-7 | Fast | Standard profiling (recommended) |
| `--set detailed --launch-count 50` | 8-12 | Medium | Detailed analysis |
| `--set full` (all launches) | 15-20+ | Slowest | Comprehensive deep dive |

---

### Example: Fast Profiling for Iteration

**Goal**: Profile 3 layers quickly during development.

```bash
# Fast: Only essential metrics, limited launches
ncu --nvtx \
    --nvtx-include "layer1/" \
    --nvtx-include "layer2/" \
    --nvtx-include "layer3/" \
    --replay-mode app-range \
    --set brief \
    --launch-count 5 \
    --kernel-name regex:".*gemm.*|.*conv.*" \
    --csv --log-file quick_profile.csv \
    -o quick_report \
    python model.py

# Result: ~30 seconds for 3 layers (vs 5-10 minutes with full profiling)
```

---

### Example: Comprehensive Single-Layer Analysis

**Goal**: Deep dive into one critical layer.

```bash
# Comprehensive: All metrics, many launches
ncu --nvtx \
    --nvtx-include "attention/" \
    --replay-mode app-range \
    --set full \
    --launch-count 100 \
    --csv --log-file attention_full.csv \
    -o attention_detailed \
    python model.py

# Result: 10-15 minutes, but exhaustive data
```

---

### Example: Sampling Strategy for Long Sequences

**Goal**: Profile a 1000-token decode (same kernel launched 1000 times).

```bash
# Sample every 50th token (20 total samples)
ncu --nvtx \
    --nvtx-include "decode/" \
    --replay-mode app-range \
    --launch-skip 49 \
    --launch-count 1 \
    --set roofline \
    --csv --log-file decode_sampled.csv \
    python model.py

# Alternative: Profile first/middle/last tokens
# First 5 tokens
ncu ... --launch-count 5 ...
# Middle 5 tokens (skip 497, profile 5)
ncu ... --launch-skip 497 --launch-count 5 ...
# Last 5 tokens (skip 995, profile 5)
ncu ... --launch-skip 995 --launch-count 5 ...
```

---

### Determining Actual Replay Count

NCU logs show how many replays occurred:

```bash
# Run profiling
ncu --set detailed -o report python script.py

# Check NCU output/logs for replay information
grep -i "replay" ncu_stderr.txt

# Or import and check
ncu --import report.ncu-rep --print-summary | grep -i "replay"
```

**Interpreting Output**:
```
==PROF== Profiling "kernel_name" - 1/1
==PROF==   Replay 1/8: Collecting set 1
==PROF==   Replay 2/8: Collecting set 2
...
==PROF==   Replay 8/8: Collecting metrics
```

This shows 8 replays per kernel for the requested metric set.

---

### Best Practices for Replay Control

1. **Start minimal, expand as needed**:
   ```bash
   # Step 1: Quick overview with brief
   ncu --set brief --launch-count 5 ...

   # Step 2: If interesting, use roofline
   ncu --set roofline --launch-count 20 ...

   # Step 3: Deep dive with full (only if necessary)
   ncu --set full --launch-count 100 ...
   ```

2. **Use kernel filtering for large models**:
   ```bash
   # First, identify top kernels with nsys
   nsys profile -o nsys_report python model.py
   nsys stats --report cuda_gpu_kern_sum nsys_report.nsys-rep

   # Then profile only top 5 kernels with ncu (use actual kernel names from nsys)
   ncu --kernel-name regex:"kernel1|kernel2|kernel3|kernel4|kernel5" ...
   ```

3. **Limit launches for repetitive kernels**:
   ```bash
   # If kernel is called 500+ times, 10-20 launches is enough
   --launch-count 20
   ```

4. **Use `--set roofline` as default** (good balance):
   ```bash
   --set roofline  # ~5-7 replays, essential metrics
   ```

5. **Profile in stages** (not all layers at once):
   ```bash
   # Day 1: Profile encoder layers
   ncu --nvtx-include "encoder::*/" ...

   # Day 2: Profile decoder layers
   ncu --nvtx-include "decoder::*/" ...

   # Day 3: Profile attention
   ncu --nvtx-include "attention/" ...
   ```

---

### Project Integration: Configurable Replay Control

Add replay control to Hydra config:

```yaml
# conf/profiling/ncu/quick.yaml
ncu_cli:
  set: brief
  launch_count: 5
  kernel_name: null  # Profile all kernels
  replay_mode: app-range

# conf/profiling/ncu/standard.yaml
ncu_cli:
  set: roofline
  launch_count: 20
  kernel_name: regex:".*gemm.*|.*conv.*"  # Note: regex: prefix required for pattern matching
  replay_mode: app-range

# conf/profiling/ncu/comprehensive.yaml
ncu_cli:
  set: full
  launch_count: 100
  kernel_name: null
  replay_mode: range
```

Update `build_ncu_cmd()` to support these options:

```python
# src/llm_perf_opt/profiling/vendor/ncu.py
def build_ncu_cmd(
    # ... existing params
    launch_count: int | None = None,
    launch_skip: int | None = None,
    kernel_name: str | None = None,
    kernel_id: str | None = None,
) -> list[str]:
    cmd: list[str] = ["ncu"]

    # ... existing flags

    # Add launch filtering
    if launch_count is not None:
        cmd += ["--launch-count", str(launch_count)]
    if launch_skip is not None:
        cmd += ["--launch-skip", str(launch_skip)]

    # Add kernel filtering
    if kernel_name:
        cmd += ["--kernel-name-base", kernel_name_base or "demangled"]
        cmd += ["--kernel-name", kernel_name]
    if kernel_id:
        cmd += ["--kernel-id", kernel_id]

    # ... rest of command
    return cmd + list(work_argv)
```

---

## Saving Human-Readable Results with NCU

NCU provides multiple output formats beyond the binary `.ncu-rep` file. This section covers how to extract human-readable reports during capture or post-processing.

### Output Formats Overview

| Format | Flag | Use Case | Human-Readable |
|--------|------|----------|----------------|
| Binary report | `-o report` | Import into NCU GUI, post-processing | ❌ Binary |
| CSV log | `--csv --log-file out.csv` | Programmatic analysis, scripting | ✅ Structured text |
| Text sections | `--page details/raw --section <name>` | Quick terminal inspection | ✅ Plain text |
| Summary stdout | `--print-summary` | Quick overview | ✅ Plain text |

---

### Method 1: CSV Output (Most Common)

**Use Case**: Programmatic analysis, import into pandas/Excel, scripting.

```bash
# Capture profiling data + CSV log
ncu --nvtx --nvtx-include "layer1/" \
    --replay-mode app-range \
    --set roofline \
    --csv --log-file ncu_output.csv \
    -o ncu_report \
    python script.py

# View CSV in terminal
column -t -s',' ncu_output.csv | head -20

# Or use pandas
python -c "import pandas as pd; df = pd.read_csv('ncu_output.csv'); print(df.head())"
```

**CSV Columns Include**:
- `Kernel Name` - Demangled kernel name
- `NVTX Range Name` - Your NVTX label (if using range replay)
- `gpu__time_duration.sum` - Total GPU time (nanoseconds)
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` - SM utilization %
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` - Memory bandwidth %
- Section-specific metrics (depends on `--set` or `--section` flags)

---

### Method 2: Text Sections (Human-Readable Reports)

**Use Case**: Quick inspection in terminal, readable reports for documentation.

NCU "sections" are pre-defined metric groups (e.g., `SpeedOfLight`, `MemoryWorkloadAnalysis`, `LaunchStats`). You can print these as formatted text.

#### List Available Sections

```bash
# Show all available sections for your GPU
ncu --list-sections
```

**Common Useful Sections**:
- `SpeedOfLight` - SM/memory utilization, theoretical maximum
- `MemoryWorkloadAnalysis` - Cache hit rates, memory throughput breakdown
- `LaunchStats` - Grid/block dimensions, registers, shared memory
- `Occupancy` - Theoretical vs achieved occupancy
- `SourceCounters` - Per-source-line metrics (requires source correlation)
- `WarpStateStats` - Warp stall reasons

#### Capture + Extract Sections During Profiling

```bash
# Method A: Request specific sections during capture
ncu --nvtx --nvtx-include "layer1/" \
    --replay-mode app-range \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    -o ncu_report \
    python script.py

# Then import and print to text file
ncu --import ncu_report.ncu-rep \
    --page details \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    > ncu_readable_report.txt

# View in terminal
less ncu_readable_report.txt
```

**Note**: Requesting many sections increases profiling overhead. Use `--set` for predefined groups:

```bash
# --set bundles commonly-used sections
ncu --nvtx --nvtx-include "layer1/" \
    --replay-mode app-range \
    --set full \  # or: roofline, detailed, brief
    -o ncu_report \
    python script.py
```

**Available Sets**:
- `brief` - Minimal metrics (fastest)
- `roofline` - SM/memory utilization for roofline analysis (recommended default)
- `detailed` - More comprehensive metrics
- `full` - All available metrics (slowest, use sparingly)

---

### Method 3: Post-Processing Existing Reports

If you already have a `.ncu-rep` file, extract human-readable text without re-running profiling.

#### Print Summary to Terminal

```bash
# Quick summary of all kernels
ncu --import ncu_report.ncu-rep --print-summary

# Filter by NVTX range (if captured with range replay)
ncu --import ncu_report.ncu-rep \
    --nvtx-include "layer1/" \
    --print-summary
```

#### Extract Specific Sections

```bash
# Export specific sections to text file
ncu --import ncu_report.ncu-rep \
    --page details \
    --section SpeedOfLight \
    --section Occupancy \
    > layer1_details.txt

# Alternative: Use raw page for CSV-like output
ncu --import ncu_report.ncu-rep \
    --page raw \
    --section SpeedOfLight \
    > layer1_metrics.txt
```

**Page Types**:
- `details` - Human-readable formatted output (default)
- `raw` - Metric names and values (CSV-like, easier to parse)

#### Batch Extract All Sections

```bash
# Extract every captured section to separate files
for section in SpeedOfLight MemoryWorkloadAnalysis LaunchStats Occupancy; do
    ncu --import ncu_report.ncu-rep \
        --page details \
        --section "$section" \
        > "section_${section}.txt"
done

# Combine into single report
cat section_*.txt > full_report.txt
```

---

### Method 4: Combined Workflow (CSV + Sections)

**Best Practice**: Capture both CSV (for scripting) and sections (for human inspection).

```bash
# 1. Capture with sections + CSV
ncu --nvtx --nvtx-include "layer1/" \
    --replay-mode app-range \
    --set roofline \
    --csv --log-file ncu_output.csv \
    -o ncu_report \
    python script.py

# 2. Extract CSV metrics to stdout
head -20 ncu_output.csv

# 3. Extract human-readable sections to file
ncu --import ncu_report.ncu-rep \
    --page details \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    > ncu_readable.txt

# 4. Archive both for later analysis
mkdir -p profiling_results/layer1
mv ncu_output.csv ncu_report.ncu-rep ncu_readable.txt profiling_results/layer1/
```

---

### Method 5: Per-Region Reports (NVTX Range Replay)

When profiling multiple NVTX ranges, extract separate reports for each region.

```bash
# Capture multiple ranges
ncu --nvtx \
    --nvtx-include "layer1/" \
    --nvtx-include "layer2/" \
    --nvtx-include "layer3/" \
    --replay-mode app-range \
    --set roofline \
    --csv --log-file ncu_all_regions.csv \
    -o ncu_report \
    python script.py

# Extract per-region CSV rows
grep "layer1" ncu_all_regions.csv > layer1.csv
grep "layer2" ncu_all_regions.csv > layer2.csv
grep "layer3" ncu_all_regions.csv > layer3.csv

# Extract per-region sections (if .ncu-rep stores per-region data)
for region in layer1 layer2 layer3; do
    ncu --import ncu_report.ncu-rep \
        --nvtx-include "${region}/" \
        --page details \
        --section SpeedOfLight \
        > "${region}_details.txt"
done
```

---

### Example: Complete Readable Report

```bash
# Full workflow: Capture and generate comprehensive human-readable report
ncu --nvtx --nvtx-include "attention/" \
    --replay-mode app-range \
    --set full \
    --csv --log-file attention.csv \
    -o attention_profile \
    python model.py

# Generate multi-section text report
ncu --import attention_profile.ncu-rep \
    --page details \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --section WarpStateStats \
    > attention_full_report.txt

# View report
cat attention_full_report.txt
```

**Sample Output** (truncated):
```
==PROF== Connected to process 12345
==PROF== Profiling "Kernel: void at::native::vectorized_elementwise_kernel<...>"

Section: SpeedOfLight
---------------------------------------------------------------------------
Memory Throughput                                          74.23%   High
Compute (SM) Throughput                                    31.45%   Low

Max Theoretical Utilization:
  SM:    100.00%
  Memory: 100.00%

SOL Memory Throughput: 74.23% (DRAM read/write balanced)
SOL SM Throughput: 31.45% (Memory-bound)
---------------------------------------------------------------------------

Section: MemoryWorkloadAnalysis
---------------------------------------------------------------------------
L1 Cache Hit Rate:                                         82.3%
L2 Cache Hit Rate:                                         45.7%
DRAM Bandwidth Utilization:                                67.2%

Memory Workload: 2.34 GB
  L1:    1.92 GB (82.3% hit rate)
  L2:    1.07 GB (45.7% hit rate)
  DRAM:  0.58 GB
---------------------------------------------------------------------------

Section: LaunchStats
---------------------------------------------------------------------------
Grid Size:       1024
Block Size:      256
Registers/Thread: 48
Shared Memory:    16384 bytes
---------------------------------------------------------------------------
```

---

### Tips for Readable Output

1. **Use `--page details` for human-readable format** (not `raw` or `source`)
2. **Limit sections during capture** to reduce overhead:
   ```bash
   # Fast: Only essential metrics
   --section SpeedOfLight --section LaunchStats

   # Slower but comprehensive
   --set full
   ```
3. **Combine with grep/awk for filtering**:
   ```bash
   # Extract only memory-related metrics
   ncu --import report.ncu-rep --page raw --section SpeedOfLight | grep -i memory
   ```
4. **Redirect to file for large reports**:
   ```bash
   ncu --import report.ncu-rep --page details > report.txt
   ```
5. **Use `column` command for CSV alignment**:
   ```bash
   column -t -s',' ncu_output.csv | less -S
   ```

---

### Project Integration: Auto-Generate Readable Reports

Update the runner to automatically generate text reports alongside CSV:

```python
# src/llm_perf_opt/runners/deep_profile_runner.py (excerpt)

# After NCU capture
if Path(ncu_out).with_suffix(".ncu-rep").exists():
    # Generate human-readable sections report
    sections = ["SpeedOfLight", "MemoryWorkloadAnalysis", "LaunchStats"]
    sections_txt = artifacts.path("ncu/sections_report.txt")

    import_cmd = ["ncu", "--import", str(ncu_out) + ".ncu-rep", "--page", "details"]
    for sec in sections:
        import_cmd += ["--section", sec]

    with open(sections_txt, "w", encoding="utf-8") as f:
        subprocess.run(import_cmd, stdout=f, stderr=subprocess.STDOUT, check=False)

    print(f"Human-readable report: {sections_txt}")
```

This automatically creates `ncu/sections_report.txt` for quick inspection without opening the NCU GUI.

---

## Common Pitfalls and Troubleshooting

### Issue 1: "No ranges were profiled" Warning

**Symptoms**: NCU runs but CSV is empty or contains only warnings.

**Checklist**:
- [ ] Added `torch.cuda.synchronize()` after each NVTX range?
- [ ] Using trailing slash in `--nvtx-include`? (e.g., `"layer1/"`)
- [ ] Warm-up pass completed before profiling?
- [ ] Using `app-range` mode instead of `range`?
- [ ] NVTX range name matches exactly (case-sensitive)?

**Debug Command**:
```bash
# Test with kernel mode + NVTX to verify kernels exist
ncu --nvtx --nvtx-include "layer1/" \
    --replay-mode kernel \
    --csv --log-file debug.csv \
    python script.py

# Check CSV for kernels
grep -i "layer1" debug.csv
```

---

### Issue 2: Nested Ranges Not Profiled

**Problem**: Nested NVTX ranges may not appear in NCU output.

```python
with nvtx.annotate("outer"):
    with nvtx.annotate("inner"):  # May not be captured
        x = self.layer(x)
        torch.cuda.synchronize()
```

**Solution**: Use hierarchical naming with `::` separator:
```python
with nvtx.annotate("model::encoder"):
    x = self.encoder(x)
    torch.cuda.synchronize()

with nvtx.annotate("model::decoder"):
    x = self.decoder(x)
    torch.cuda.synchronize()

# NCU command
ncu --nvtx --nvtx-include "model::*/" python script.py
```

---

### Issue 3: Unsupported CUDA APIs

**Symptoms**: "No ranges were profiled" even with correct synchronization.

**Cause**: Range contains unsupported CUDA APIs (initialization, graph management, virtual memory).

**Solution**:
1. Move initialization **outside** ranges (warmup pass)
2. Use `app-range` mode (more permissive)
3. Avoid dynamic graph construction inside ranges

**Unsupported API Categories**:
- All initialization functions (`cuInit`, context creation)
- Virtual memory management (`cuMemMap*`, `cuMemAddressReserve`)
- CUDA graphs (`cuGraphInstantiate`, `cuGraphLaunch`)
- Memory pools (`cuMemPoolCreate`, `cuMemPoolSetAttribute`)

---

## Complete Working Example

```python
# profile_model.py
import torch
import torch.nn as nn
import nvtx

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with nvtx.annotate("fc1"):
            x = torch.relu(self.fc1(x))
            torch.cuda.synchronize()

        with nvtx.annotate("fc2"):
            x = torch.relu(self.fc2(x))
            torch.cuda.synchronize()

        with nvtx.annotate("fc3"):
            x = self.fc3(x)
            torch.cuda.synchronize()

        return x

def main():
    device = "cuda:0"

    # 1. Initialize model and move to GPU (context creation)
    model = SimpleModel().to(device)

    # 2. Warm-up pass (initialize kernels, caches)
    x_warmup = torch.randn(32, 1024, device=device)
    _ = model(x_warmup)
    torch.cuda.synchronize()
    print("Warm-up complete")

    # 3. Actual profiling run
    x = torch.randn(64, 1024, device=device)
    y = model(x)
    print(f"Output shape: {y.shape}")

if __name__ == "__main__":
    main()
```

**Profile it**:
```bash
# 1. Run NCU with NVTX range replay + CSV output
ncu --nvtx \
    --nvtx-include "fc1/" \
    --nvtx-include "fc2/" \
    --nvtx-include "fc3/" \
    --replay-mode app-range \
    --set roofline \
    --csv --log-file ncu_output.csv \
    -o ncu_report \
    python profile_model.py

# 2. Verify CSV output (structured data)
head -20 ncu_output.csv
# Should contain rows with "NVTX Range Name" column showing "fc1", "fc2", "fc3"

# 3. Generate human-readable text report
ncu --import ncu_report.ncu-rep \
    --page details \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    > ncu_readable_report.txt

# 4. View the readable report
cat ncu_readable_report.txt

# 5. Per-region analysis (optional)
for region in fc1 fc2 fc3; do
    echo "=== Analysis for ${region} ===" > "${region}_report.txt"
    grep "${region}" ncu_output.csv >> "${region}_report.txt"
    ncu --import ncu_report.ncu-rep \
        --nvtx-include "${region}/" \
        --page details \
        --section SpeedOfLight \
        >> "${region}_report.txt"
done
```

---

## Best Practices

### ✅ DO

1. **Always synchronize after NVTX ranges**:
   ```python
   with nvtx.annotate("layer"):
       x = layer(x)
       torch.cuda.synchronize()  # Non-negotiable
   ```

2. **Use `nvtx.annotate()` in Python** (recommended over `push/pop`):
   ```python
   # Preferred
   with nvtx.annotate("name"):
       pass

   # Avoid (more error-prone)
   nvtx.push_range("name")
   # code
   nvtx.pop_range()
   ```

3. **Profile narrow regions** (single operation or small sequence):
   ```python
   # Good: Focused region
   with nvtx.annotate("attention"):
       attn = self.attention(q, k, v)
       torch.cuda.synchronize()
   ```

4. **Start with `app-range` mode for PyTorch**

5. **Capture NCU stderr for debugging**:
   ```bash
   ncu ... python script.py 2> ncu_stderr.txt
   grep -i "warning\|error" ncu_stderr.txt
   ```

### ❌ DON'T

1. **Don't forget trailing slash for push/pop ranges**
2. **Don't profile huge regions** (entire model forward pass)
3. **Don't initialize CUDA inside NVTX ranges**
4. **Don't use semicolon-separated ranges in single flag**
5. **Don't mix multiple NVTX APIs** (`annotate` + `push/pop`) unnecessarily

---

## Integration with Project Codebase

### Updating `build_ncu_cmd()` to Auto-Fix Syntax

```python
# src/llm_perf_opt/profiling/vendor/ncu.py
def build_ncu_cmd(
    out_base: Path,
    work_argv: Sequence[str],
    *,
    nvtx_expr: str | list[str] | None,  # ✅ Accept list
    # ... other params
) -> list[str]:
    cmd: list[str] = ["ncu"]
    # ... set, replay-mode, metrics, sections

    if use_nvtx and nvtx_expr:
        cmd += ["--nvtx"]

        # Handle single string or list of expressions
        exprs = [nvtx_expr] if isinstance(nvtx_expr, str) else nvtx_expr

        for expr in exprs:
            # Auto-add trailing slash for push/pop ranges
            # Skip if already has wildcard or trailing slash
            if not expr.endswith(("*", "/")):
                expr = f"{expr}/"
            cmd += ["--nvtx-include", expr]

    # ... rest of command building
    return cmd + list(work_argv)
```

### Model Pattern Template

```python
# src/llm_perf_opt/dnn_models/base.py
import torch
import torch.nn as nn
import nvtx
from typing import Protocol

class NVTXProfileable(Protocol):
    """Protocol for models that support NVTX profiling."""

    def forward_with_nvtx(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with NVTX ranges and synchronization."""
        ...

class ProfileableModel(nn.Module):
    """Base class for models with built-in NVTX profiling support."""

    def forward_with_nvtx(self, x: torch.Tensor) -> torch.Tensor:
        """Override this to add NVTX ranges to your model's forward pass."""
        return self.forward(x)

    @staticmethod
    def sync_after_range():
        """Helper to synchronize after NVTX range."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
```

---

## Quick Reference: Common NCU Commands

### Capture with NVTX + CSV
```bash
ncu --nvtx --nvtx-include "layer/" \
    --replay-mode app-range \
    --set roofline \
    --csv --log-file output.csv \
    -o report \
    python script.py
```

### Generate Human-Readable Report (Post-Process)
```bash
ncu --import report.ncu-rep \
    --page details \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    > readable_report.txt
```

### List Available Sections
```bash
ncu --list-sections
```

### Print Summary
```bash
ncu --import report.ncu-rep --print-summary
```

### Extract Specific NVTX Range
```bash
ncu --import report.ncu-rep \
    --nvtx-include "layer/" \
    --page details \
    > layer_report.txt
```

### Batch Process Multiple Ranges
```bash
for region in layer1 layer2 layer3; do
    grep "${region}" output.csv > "${region}.csv"
    ncu --import report.ncu-rep \
        --nvtx-include "${region}/" \
        --page details --section SpeedOfLight \
        > "${region}_report.txt"
done
```

### Fast Profiling (Limited Launches and Metrics)
```bash
ncu --nvtx --nvtx-include "layer/" \
    --replay-mode app-range \
    --set brief \
    --launch-count 5 \
    --csv --log-file quick.csv \
    -o report \
    python script.py
```

### Sample Every Nth Launch (e.g., Every 50th Token)
```bash
ncu --nvtx --nvtx-include "decode/" \
    --replay-mode app-range \
    --launch-skip 49 \
    --launch-count 1 \
    --set roofline \
    python script.py
```

### Profile Specific Kernels Only
```bash
ncu --nvtx --nvtx-include "layer/" \
    --kernel-name-base demangled \
    --kernel-name regex:".*gemm.*|.*conv.*" \
    --launch-count 10 \
    python script.py
```

### Query Available Metrics
```bash
ncu --query-metrics
```

### Profile with Custom Metrics (Fewer Replays)
```bash
ncu --metrics gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv --log-file output.csv \
    python script.py
```

---

## References

### Official Documentation
- [Nsight Compute 2025.3 Profiling Guide](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html)
- [NVTX Python API Documentation](https://nvtx.readthedocs.io/en/latest/)
- [NVIDIA NVTX GitHub Repository](https://github.com/NVIDIA/NVTX)

### Community Resources
- [NVIDIA Developer Forums - Range Profiling Issues](https://forums.developer.nvidia.com/t/range-profiling-no-ranges-were-profiled/244691)
- [PyTorch NVTX Profiling Thread](https://forums.developer.nvidia.com/t/nsight-compute-failed-to-profile-with-nvtx-ranges-in-pytorch/265420)
- [NVIDIA Technical Blog - NVTX Annotations](https://developer.nvidia.com/blog/nvidia-tools-extension-api-nvtx-annotation-tool-for-profiling-code-in-python-and-c-c/)

### Project-Specific
- Code Review: `context/logs/code-review/20251107-001627-phase3-us1-nvtx-range-replay-issue.md`
- Implementation Guide: `context/tasks/003-nvtx-ncu-profiling/impl-phase-3-us1.md`

---

## Changelog

- **2025-11-07**:
  - Initial version based on Phase 3 US1 code review findings
  - **IMPORTANT FIX**: Corrected `--kernel-name` syntax to use `regex:` prefix for pattern matching
    - All examples now use `--kernel-name regex:"pattern"` instead of `--kernel-name "pattern"`
    - Added explicit note: "Always use the `regex:` prefix for pattern matching"
    - Fixed 6 instances throughout the guide
  - Added comprehensive section on **controlling replay behavior and count**:
    - How to limit profiling scope with `--kernel-name`, `--kernel-id`
    - How to control launch replays with `--launch-count`, `--launch-skip`
    - Limiting metrics with `--metrics` vs `--set` (trade-off analysis)
    - Fine-grained replay control with `--kernel-replay-mode`
    - Trade-off matrix showing replays per kernel for different configurations
    - Examples: fast iteration, comprehensive analysis, sampling strategies
    - Best practices for replay control in production workflows
    - Project integration patterns with Hydra config presets
  - Added comprehensive section on **saving human-readable results with NCU**:
    - 5 methods for extracting readable output (CSV, text sections, post-processing, combined, per-region)
    - Complete section reference (SpeedOfLight, MemoryWorkloadAnalysis, LaunchStats, etc.)
    - Sample output showing actual NCU section formatting
    - Tips for readable output and project integration code
  - Added **Quick Reference** section with common NCU commands including:
    - Fast profiling with limited launches
    - Sampling strategies for repetitive kernels
    - Kernel filtering patterns (with correct regex: syntax)
    - Custom metrics queries
  - Incorporates lessons from NCU "No ranges were profiled" debugging
  - Validated against Nsight Compute 2025.3 and PyTorch 2.5.1+
