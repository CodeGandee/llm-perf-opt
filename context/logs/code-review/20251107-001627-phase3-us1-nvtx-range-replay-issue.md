# Code Review: Phase 3 US1 - NVTX Range Replay "No Ranges Were Profiled" Issue

**Date**: 2025-11-07
**Reviewer**: Claude Code
**Scope**: Phase 3 US1 implementation (NVTX range replay flow)
**Focus**: Root cause analysis of NCU "No ranges were profiled" warning

---

## Executive Summary

The implementation of NVTX range replay in Phase 3 US1 is architecturally sound but contains several critical issues that prevent Nsight Compute from successfully profiling NVTX ranges. The root causes span **API usage**, **synchronization**, **command-line syntax**, and **CUDA context initialization**. This review identifies **7 high-priority bugs** and **3 architectural improvements** based on official NVIDIA documentation and community best practices.

---

## Implementation Overview

### Files Reviewed

1. `src/llm_perf_opt/profiling/regions.py` - Region assembly from NCU CSV
2. `src/llm_perf_opt/profiling/export_regions.py` - Report exporters
3. `src/llm_perf_opt/runners/deep_profile_runner.py` - Stage 2 orchestration
4. `src/llm_perf_opt/profiling/vendor/ncu.py` - NCU command builder
5. `src/llm_perf_opt/profiling/artifacts.py` - Filesystem helpers
6. `src/llm_perf_opt/dnn_models/shallow_resnet.py` - Dummy workload with NVTX
7. `src/llm_perf_opt/profiling/harness.py` - NVTX context managers

### Third-Party Libraries Used

- **nvtx** (`nvtx>=0.2.13,<0.3` in `pyproject.toml`) - NVIDIA Tools Extension for Python
- **PyTorch** (`torch>=2.5.1`) - Deep learning framework with CUDA support
- **Hydra** (`hydra-core>=1.3.2,<2`) - Configuration management
- **OmegaConf** (`omegaconf>=2.3.0,<3`) - Configuration utilities

---

## Critical Issues Identified

### 🔴 **Issue 1: Missing CUDA Synchronization After NVTX Ranges**

**Location**: `src/llm_perf_opt/dnn_models/shallow_resnet.py:70-76`

**Problem**:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    with nvtx.annotate("stem"):
        x = self.stem(x)              # ❌ Kernels may launch asynchronously
    with nvtx.annotate("residual"):
        x = self.blocks(x)            # ❌ No synchronization between ranges
    with nvtx.annotate("head"):
        x = self.head(x)              # ❌ Kernels may execute AFTER range ends
    return x
```

**Root Cause**: PyTorch CUDA operations are **asynchronous by default**. When `self.stem(x)` returns, the NVTX range ends, but the actual CUDA kernels may still be queued or executing. NCU's range replay requires that **all work must be launched and synchronizable within the range boundaries**.

**Evidence from Official Docs**:
> "Ranges must fulfill several requirements [...] It must be possible to **synchronize all active CUDA contexts at the start of the range**." — [Nsight Compute Profiling Guide 2025.3](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html)

**Impact**: NCU sees empty ranges with no replayable kernels → `"No ranges were profiled"` warning.

**Recommendation**:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    with nvtx.annotate("stem"):
        x = self.stem(x)
        torch.cuda.synchronize()  # ✅ Ensure kernels complete before range ends
    with nvtx.annotate("residual"):
        x = self.blocks(x)
        torch.cuda.synchronize()  # ✅ Force boundary synchronization
    with nvtx.annotate("head"):
        x = self.head(x)
        torch.cuda.synchronize()  # ✅ Final sync
    return x
```

**Alternative**: Add synchronization in the profiling harness wrapper (less intrusive but less precise).

---

### 🔴 **Issue 2: Incorrect NVTX Include Syntax for Push/Pop Ranges**

**Location**: `src/llm_perf_opt/profiling/vendor/ncu.py:141`

**Problem**:
```python
if use_nvtx and nvtx_expr:
    cmd += ["--nvtx", "--nvtx-include", nvtx_expr]  # ❌ Missing trailing slash
```

**Root Cause**: Python's `nvtx.annotate()` uses **push/pop** internally. NCU distinguishes between:
- **Push/Pop ranges**: Require trailing `/` → `--nvtx-include "stem/"`
- **Start/End ranges**: No trailing slash → `--nvtx-include "stem"`

**Evidence from Forums**:
> "For push/pop ranges (rather than start/stop ranges), a forward slash suffix is required. The corrected syntax should be: `--nvtx-include 'NCU/'`" — [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/nsight-compute-failed-to-profile-with-nvtx-ranges-in-pytorch/265420)

**Current Behavior**:
```bash
ncu --nvtx --nvtx-include "stem" ...  # ❌ Won't match push/pop range
```

**Expected Behavior**:
```bash
ncu --nvtx --nvtx-include "stem/" ...  # ✅ Matches push/pop range
```

**Recommendation**: Auto-detect or document the requirement for trailing slash when using `nvtx.annotate()` or `push_range/pop_range`.

---

### 🔴 **Issue 3: Single `--nvtx-include` Flag for Multiple Ranges**

**Location**: `src/llm_perf_opt/profiling/vendor/ncu.py:140-141`

**Problem**: When user provides `'stem;residual;head'`, the code passes it as a single string:
```python
cmd += ["--nvtx", "--nvtx-include", "stem;residual;head"]  # ❌ May not parse correctly
```

**Root Cause**: NCU's `--nvtx-include` accepts **glob patterns** or requires **repeated flags** for multiple exact labels:
```bash
# Option 1: Glob pattern
ncu --nvtx --nvtx-include "*/"

# Option 2: Repeated flags (recommended)
ncu --nvtx --nvtx-include "stem/" --nvtx-include "residual/" --nvtx-include "head/"
```

**Current Implementation** assumes NCU will split semicolon-separated values, but NCU may treat the entire string as a single glob expression.

**Evidence**: The implementation guide (lines 257) notes:
> "Enhance the NCU builder to accept multiple include labels by emitting repeated `--nvtx-include` flags when a list is provided."

**Recommendation**: Update `build_ncu_cmd()` to accept `nvtx_expr` as `str | list[str]`:
```python
def build_ncu_cmd(
    out_base: Path,
    work_argv: Sequence[str],
    *,
    nvtx_expr: str | list[str] | None,  # ✅ Accept list
    # ...
) -> list[str]:
    # ...
    if use_nvtx and nvtx_expr:
        cmd += ["--nvtx"]
        exprs = [nvtx_expr] if isinstance(nvtx_expr, str) else nvtx_expr
        for expr in exprs:
            expr_normalized = expr if expr.endswith("/") else f"{expr}/"  # ✅ Auto-add slash
            cmd += ["--nvtx-include", expr_normalized]
```

---

### 🟡 **Issue 4: No CUDA Context Warm-up Before Range Replay**

**Location**: `src/llm_perf_opt/dnn_models/shallow_resnet.py:63-67`

**Problem**: The warmup method exists but may not be invoked before NCU profiling:
```python
@torch.no_grad()
def warmup(self, device: torch.device | str = "cpu") -> None:
    """Run a quick forward pass to initialize kernels and caches."""
    x = torch.randn(2, 3, 64, 64, device=device)
    _ = self.forward(x)
```

**Root Cause**: From forum discussions, NCU range replay fails when **no CUDA context is active** on the thread at range start:
> "The primary issue is that **no CUDA context is active on the thread when `cudaProfilerStart` is called**." — [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/range-profiling-no-ranges-were-profiled/244691)

PyTorch lazily initializes CUDA contexts. If the first CUDA operation happens **inside** an NVTX range, NCU may reject the range due to unsupported initialization API calls.

**Recommendation**: Ensure explicit warmup before first NVTX range:
```python
# In runner or workload setup
model = ShallowResNet(...).to(device)
model.warmup(device=device)  # ✅ Initialize context BEFORE NCU profiling starts
```

---

### 🟡 **Issue 5: Potential Unsupported CUDA APIs Inside Ranges**

**Location**: `src/llm_perf_opt/dnn_models/shallow_resnet.py:70-76`

**Problem**: PyTorch operations may internally call CUDA APIs not supported by Range Replay:

**Unsupported Categories** (from [NCU Docs](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html)):
- Initialization functions (e.g., `cuInit`, context creation)
- Virtual memory management
- Graph management
- Memory pool operations (e.g., `cuMemPoolCreate`)

**Impact**: If `nn.Conv2d`, `nn.BatchNorm2d`, or `nn.Linear` trigger any of these APIs during first execution, the range becomes non-replayable.

**Mitigation**:
1. Use `model.warmup()` to trigger initialization **outside** NVTX ranges
2. Consider `replay_mode=app-range` (more permissive) instead of `range`
3. Test with `replay_mode=kernel` + NVTX gating to verify kernels exist

---

### 🟡 **Issue 6: Mixed NVTX API Usage**

**Location**: Multiple files

**Observation**: The codebase uses two different NVTX APIs:
- `nvtx.annotate()` in `shallow_resnet.py` (context manager)
- `nvtx.push_range/pop_range` in `harness.py` (manual stack)

**Consistency Issue**: While both are valid, mixing APIs can cause confusion:
```python
# shallow_resnet.py
with nvtx.annotate("stem"):  # Uses push/pop internally
    pass

# harness.py
@contextmanager
def nvtx_range(name: str):
    nvtx.push_range(name)      # Direct push/pop
    try:
        yield
    finally:
        nvtx.pop_range()
```

**Best Practice**: Official docs recommend `annotate()` for Python:
> "When applicable, prefer to use annotate." — [NVTX Python Docs](https://nvtx.readthedocs.io/en/latest/)

**Recommendation**: Standardize on `nvtx.annotate()` unless performance-critical sections require `push/pop`.

---

### 🟢 **Issue 7: CSV Parsing Fallback May Hide Profiling Failures**

**Location**: `src/llm_perf_opt/runners/deep_profile_runner.py:368-382`

**Problem**: When CSV contains no valid rows, the code falls back to synthetic region reports:
```python
if replay_mode in {"range", "app-range"}:
    if ncu_csv_path.exists():
        rows = parse_ncu_csv(ncu_csv_path)
        reps = assemble_region_reports(rows, device=device_sel)
        # If assembly yields only unlabeled/empty, fall back to NVTX include labels
        if only_unlabeled:
            labels_from_expr = discover_region_labels(nvtx_expr) or ["NVTX@range"]
            reps = build_region_reports(labels_from_expr, device=device_sel, process=None)
            # ❌ User sees "success" even though no actual profiling occurred
```

**Impact**: Creates false positives - user believes profiling succeeded when NCU actually profiled nothing.

**Recommendation**: Add explicit warning when falling back:
```python
if only_unlabeled:
    import logging
    logging.warning(
        "NCU range replay produced no labeled regions. "
        "This likely means NCU failed to profile any NVTX ranges. "
        "Check for synchronization, API support, and NVTX syntax issues."
    )
    labels_from_expr = discover_region_labels(nvtx_expr) or ["NVTX@range"]
    reps = build_region_reports(labels_from_expr, device=device_sel, process=None)
```

---

## Architectural Observations

### ✅ **Strength 1: Best-Effort CSV Parsing**

`_range_key()` in `regions.py:100-110` uses multiple fallback column names:
```python
for k in ("NVTX Range Name", "Range Name", "NVTX Range", "Range"):
    v = row.get(k)
    if v:
        return str(v)
```

**Assessment**: Excellent defensive programming for NCU version compatibility.

---

### ✅ **Strength 2: Filesystem-Safe Label Sanitization**

`sanitize_region_label()` in `artifacts.py:254-284` properly handles special characters:
```python
s = str(label).strip()
s = s.replace("::", "__").replace("/", "_").replace("\\", "_")
```

**Assessment**: Robust cross-platform path handling.

---

### ⚠️ **Improvement 1: Region Discovery Heuristic**

`discover_region_labels()` in `regions.py:45-80` splits on multiple separators but doesn't validate against NCU syntax rules:
```python
for sep in [";", "|", ",", " "]:
    if sep in raw:
        toks = [t for t in (p.strip() for p in raw.split(sep)) if t]
```

**Recommendation**: Add validation for glob patterns and auto-append trailing `/` when needed:
```python
def discover_region_labels(nvtx_include_expr: str | None) -> list[str]:
    # ... existing parsing ...

    # Auto-fix for push/pop ranges
    out: list[str] = []
    for t in toks:
        if not t.endswith(("*", "/")):  # Skip globs and already-suffixed
            t = f"{t}/"  # ✅ Add trailing slash for push/pop
        out.append(t)
    return out
```

---

### ⚠️ **Improvement 2: Explicit Replay Mode Documentation**

The code supports `replay_mode` but doesn't document the differences:
- `kernel` - Profile all kernels (default, most reliable)
- `range` - Profile NVTX ranges (strict requirements)
- `app-range` - Profile app ranges (more permissive, better for frameworks)

**Recommendation**: Add config comments in Hydra presets explaining when to use each mode.

---

### ⚠️ **Improvement 3: Diagnostic Artifacts**

The code writes `cmd.txt` for reproducibility (lines 360-363) but doesn't capture NCU stderr warnings.

**Recommendation**: Capture NCU stderr to a file for post-mortem analysis:
```python
ncu_stderr = artifacts.out_dir("ncu") / "stderr.txt"
with open(ncu_stderr, "w") as f:
    subprocess.run(ncu_cmd, check=False, stderr=f)
```

---

## Online Resources Referenced

### `online-examples`

1. **Working NCU NVTX Command** (GitHub - ekrim/kernel-profiling)
   ```bash
   ncu --set full -o report --nvtx --nvtx-include "BENCHMARK/" --range-filter ::3 python script.py
   ```
   - Uses trailing slash for push/pop ranges
   - Combines NVTX with range filters

2. **PyTorch NVTX Best Practices** (PyTorch Dev Discuss)
   - Recommends `torch.cuda.synchronize()` between NVTX ranges
   - Warns about async kernel launch timing issues

### `api-doc`

1. **Nsight Compute 2025.3 Profiling Guide**
   - URL: https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html
   - Key Sections:
     - Range Replay Requirements (synchronization, API support)
     - Unsupported CUDA API list
     - Replay mode differences

2. **NVTX Python Documentation**
   - URL: https://nvtx.readthedocs.io/en/latest/
   - API: `nvtx.annotate()` vs `push_range/pop_range`
   - Recommendation: Prefer `annotate()` in Python

3. **NVIDIA Developer Forums**
   - Thread 1: "Range profiling: 'No ranges were profiled.'" (#244691)
     - Root cause: No CUDA context active at range start
   - Thread 2: "Nsight compute failed to profile with nvtx ranges in pytorch" (#265420)
     - Solution: Add trailing `/` for push/pop ranges

---

## Priority Fixes

### **High Priority** (Blocks functionality)

1. ✅ Add `torch.cuda.synchronize()` after each NVTX range in `shallow_resnet.py`
2. ✅ Update `build_ncu_cmd()` to auto-append trailing `/` for push/pop ranges
3. ✅ Support multiple `--nvtx-include` flags (repeated) for multiple regions

### **Medium Priority** (Improves reliability)

4. ✅ Enforce model warmup before NCU profiling starts
5. ✅ Add warning when CSV parsing falls back to synthetic regions
6. ✅ Capture NCU stderr to file for diagnostics

### **Low Priority** (Code quality)

7. ✅ Standardize on `nvtx.annotate()` across codebase
8. ✅ Document replay mode tradeoffs in config comments
9. ✅ Add validation/auto-fix in `discover_region_labels()`

---

## Testing Recommendations

### Validation Steps

1. **Test with minimal workload**:
   ```bash
   pixi run -e rtx5090 python -c "
   import torch, nvtx
   model = torch.nn.Linear(10, 10).cuda()
   x = torch.randn(2, 10).cuda()
   torch.cuda.synchronize()  # Warm up

   with nvtx.annotate('test'):
       y = model(x)
       torch.cuda.synchronize()  # Critical!
   "

   ncu --nvtx --nvtx-include "test/" python script.py
   ```

2. **Compare replay modes**:
   - `replay_mode=kernel` with `--nvtx --nvtx-include "stem/"` → Should profile kernels
   - `replay_mode=range` → May fail without synchronization
   - `replay_mode=app-range` → More permissive, better for PyTorch

3. **Verify CSV output**:
   ```bash
   # After profiling
   head -20 tmp/profile-output/*/ncu/raw.csv
   # Should contain "NVTX Range Name" column with actual labels
   ```

4. **Check stderr for NCU warnings**:
   ```bash
   grep -i "warning\|error\|no ranges" tmp/profile-output/*/ncu/stderr.txt
   ```

---

## Conclusion

The Phase 3 US1 implementation provides a solid architectural foundation for NVTX range replay, but **critical synchronization and syntax issues** prevent it from functioning correctly with Nsight Compute. The primary fixes (synchronization, trailing slash, multiple flags) are straightforward to implement and align with official NVIDIA documentation and community best practices.

**Estimated Effort**: 2-4 hours to implement all high-priority fixes and validate with manual tests.

---

## References

- [Nsight Compute 2025.3 Profiling Guide](https://docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html)
- [NVTX Python Documentation](https://nvtx.readthedocs.io/en/latest/)
- [NVIDIA Developer Forums - Range Profiling Issues](https://forums.developer.nvidia.com/t/range-profiling-no-ranges-were-profiled/244691)
- [PyTorch NVTX Integration](https://forums.developer.nvidia.com/t/nsight-compute-failed-to-profile-with-nvtx-ranges-in-pytorch/265420)
- [GitHub - ekrim/kernel-profiling](https://github.com/ekrim/kernel-profiling) (Working example)

---

**Report Generated**: 2025-11-07 00:16:27 UTC
**Next Steps**: Prioritize High Priority fixes (1-3) for immediate impact.
