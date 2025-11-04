# How to Get Kernel Call Stacks with NVIDIA Nsight Compute (NCU)

## Overview

When profiling CUDA kernels with `ncu`, you may want to understand the call context of kernel launches. NCU provides **CPU call stack backtraces** that show the host-side (CPU) code path leading to a kernel launch, rather than a traditional "call tree" of GPU kernel-to-kernel calls.

**Important Distinction:**
- NCU captures **CPU-side call stacks** (host code calling kernels)
- It does NOT capture GPU-side kernel call trees (device-side kernel-to-kernel calls)
- For device-side kernel launches (CUDA Dynamic Parallelism), use Nsight Systems (`nsys`) with `--cudabacktrace=all`

## CPU Call Stack Collection

### Basic Usage

To enable CPU call stack collection during profiling:

```bash
# Enable native (C/C++) call stack collection
ncu --call-stack your_application

# Or explicitly specify the type
ncu --call-stack-type native your_application
```

### Call Stack Types

NCU supports two types of CPU call stacks:

#### 1. Native Call Stack
Captures C/C++ function calls leading to kernel launches:

```bash
ncu --call-stack-type native -o profile.ncu-rep ./your_app
```

#### 2. Python Call Stack  
Captures Python function calls (requires CPython 3.9+):

```bash
ncu --call-stack-type python -o profile.ncu-rep python your_script.py
```

#### 3. Combined (Both Types)
```bash
ncu --call-stack-type native --call-stack-type python -o profile.ncu-rep python your_app.py
```

## Filtering Kernels by Call Stack

### Native Call Stack Filtering

Include or exclude kernels based on the native (C/C++) call stack:

```bash
# Include only kernels launched via specific function
ncu --call-stack --native-include "FunctionName" ./app

# Exclude kernels from specific call path
ncu --call-stack --native-exclude "DebugFunction" ./app

# Filter by file and function
ncu --call-stack --native-include "MyFile.cpp@MyFunction" ./app

# Filter by module, file, and function
ncu --call-stack --native-include "libmyapp.so@MyFile.cpp@MyFunction" ./app
```

### Python Call Stack Filtering

```bash
# Include kernels from specific Python function
ncu --call-stack-type python --python-include "train_model" python train.py

# Exclude kernels from data loading
ncu --call-stack-type python --python-exclude "DataLoader.load" python train.py
```

### Advanced Filtering Syntax

The filter format is: `[@@]`

**Quantifiers:**
- `/` - Delimiter between function names (must appear in order)
- `[` - Function at bottom of stack
- `]` - Function at top of stack  
- `+` - Exactly one function between two others
- `*` - Zero or more functions between two others
- `@` - Specify file name
- `@@` - Specify module name

**Examples:**

```bash
# Function A must precede function B in call stack
--native-include "FuncA/FuncB"

# Function A at bottom, FuncB at top
--native-include "[FuncA/FuncB]"

# Match functions with regex
--native-include "regex:Func[0-9]"

# Filter by file path
--native-include "/path/to/file.cpp@FuncA"

# Exactly one function between A and C
--native-include "FuncA/+/FuncC"

# Zero or more functions between A and C  
--native-include "FuncA/*/FuncC"
```

## Viewing Call Stacks in Reports

### In NCU GUI

1. Open the profile report: `ncu-ui profile.ncu-rep`
2. Navigate to a profiled kernel
3. In the Details page, look for the "Launch" section
4. The CPU call stack backtrace will be displayed

### In NCU CLI

When importing reports, call stacks are displayed inline:

```bash
ncu --import profile.ncu-rep
```

### Filtering During Import

You can filter results by call stack when importing:

```bash
# Show only kernels from specific call path
ncu --import profile.ncu-rep --native-include "MyFunction"
```

## Integration with NVTX

Combine call stack profiling with NVTX for richer context:

```bash
# Profile with both call stacks and NVTX ranges
ncu --call-stack --nvtx --nvtx-include "TrainingLoop/" -o report.ncu-rep ./app
```

## Use Cases

### 1. Identifying Kernel Launch Origins

Find where in your codebase specific kernels are being launched:

```bash
ncu --call-stack --kernel-name "myKernel" -o report.ncu-rep ./app
```

### 2. Debugging Performance Issues

Trace performance problems back to specific code paths:

```bash
# Profile only kernels from problematic module
ncu --call-stack --native-include "libslow.so@@" ./app
```

### 3. Profiling Multi-Library Applications

Isolate kernels from different libraries:

```bash
# Profile only PyTorch kernels
ncu --call-stack-type python --python-include "torch.*" python model.py
```

## Limitations and Notes

1. **Overhead**: Call stack collection adds runtime overhead. Use filtering to minimize impact.

2. **Requires Debug Symbols**: For meaningful native call stacks, compile with `-g` flag.

3. **Python Requirements**: Python call stacks require CPython 3.9 or later.

4. **No GPU-side Call Trees**: NCU does not capture device-side kernel call chains (CUDA Dynamic Parallelism).

5. **Stack Depth**: Very deep call stacks may be truncated. Check NCU documentation for limits.

6. **Sampling vs Profiling**: Call stacks show launch context, not execution flow within kernels.

## Nsight Systems Alternative

For device-side kernel backtraces (e.g., CUDA Dynamic Parallelism):

```bash
# Use nsys for GPU kernel backtraces
nsys profile --cudabacktrace=all:500 ./app

# Combined with CPU sampling
nsys profile --sample=cpu --cudabacktrace=kernel:1000 ./app
```

See: [Nsight Systems Documentation - cudabacktrace](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cudabacktrace)

## References

- [Nsight Compute CLI Documentation - CPU Call Stack Filtering](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#cpu-call-stack-filtering)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA Developer Forums - Kernel Call Stack](https://forums.developer.nvidia.com/t/kernel-call-stack/245882)

## Example Workflow

```bash
# 1. Profile with call stack collection
ncu --call-stack-type native \
    --set full \
    -o myapp_profile.ncu-rep \
    ./myapp

# 2. View in GUI
ncu-ui myapp_profile.ncu-rep

# 3. Or filter and analyze specific call paths
ncu --import myapp_profile.ncu-rep \
    --native-include "ProcessData/*/LaunchKernel" \
    --print-summary per-kernel
```

## Common Issues

### Why Do I See Other Kernels When I Specify a Kernel Name Filter?

**Problem**: Even when using `-k myKernel` or `--kernel-name myKernel`, you may still see other kernels in the report.

**Reasons**:

1. **Launch Metrics Are Always Collected**: NCU collects `launch__*` metrics (like grid size, block size, occupancy) for ALL kernel launches, regardless of filtering. These metrics don't require kernel replay and are collected automatically.

2. **Exact Match Required by Default**: Without `regex:` prefix, `-k` expects an exact match:
   ```bash
   # This WON'T match "myKernel_v2"
   ncu -k myKernel ./app
   
   # Use regex for partial matching
   ncu -k regex:myKernel ./app
   ```

3. **Kernel Name Mangling**: The actual kernel name may be mangled. Check with:
   ```bash
   # First, see what kernels are actually running
   nsys profile -o trace ./app
   nsys stats trace.nsys-rep
   
   # Or use --kernel-name-base option
   ncu --kernel-name-base demangled -k "myKernel" ./app
   ```

4. **Multiple Kernel Variants**: Template instantiations create different kernels:
   ```bash
   # You might have myKernel<int>, myKernel<float>, etc.
   ncu -k "regex:myKernel<" ./app
   ```

**Solutions**:

```bash
# Option 1: Use regex for flexible matching
ncu -k "regex:^myKernel$" -o report.ncu-rep ./app

# Option 2: Check kernel-name-base options
ncu --kernel-name-base function -k myKernel ./app  # Function name only
ncu --kernel-name-base demangled -k "myKernel(int*)" ./app  # Full signature
ncu --kernel-name-base mangled -k "_Z8myKernelPi" ./app  # Mangled name

# Option 3: Use --kernel-id for more precise filtering
ncu --kernel-id "::myKernel:1" ./app  # Only first invocation

# Option 4: Limit profiled kernels with launch-count
ncu -k regex:myKernel -c 1 ./app  # Profile only 1 matching kernel

# Option 5: Import and filter after profiling
ncu -o full_report.ncu-rep ./app  # Profile everything
ncu --import full_report.ncu-rep -k regex:myKernel --export filtered.ncu-rep
```

**Verification**:

```bash
# List all kernels that would be profiled
ncu -k regex:myKernel --print-summary per-kernel ./app

# Check what's actually in a report
ncu --import report.ncu-rep --print-summary per-kernel
```

## Pro Tips

1. **Start Simple**: Begin with `--call-stack` without filters, then refine.

2. **Use Regex**: Regular expressions make filtering flexible:
   ```bash
   --native-include "regex:^(?!debug)"  # Exclude debug functions
   ```

3. **Combine Filters**: Mix native and Python filters for hybrid apps.

4. **Check Overhead**: Monitor runtime impact and adjust filtering accordingly.

5. **Symbol Resolution**: Ensure binaries have debug symbols for readable stack traces.

6. **Verify Kernel Names**: Use `nsys` first to identify exact kernel names before profiling with `ncu`.

7. **Use Import/Export**: Profile once with full data, then filter multiple times:
   ```bash
   ncu -o full.ncu-rep ./app
   ncu --import full.ncu-rep -k kernel1 --export kernel1.ncu-rep
   ncu --import full.ncu-rep -k kernel2 --export kernel2.ncu-rep
   ```
