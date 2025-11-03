# Code Review: Phase 3 NVTX Gating Issue with Nsight Systems

**Date**: 2025-10-30 06:12:11
**Reviewer**: Claude Code (Droid)
**Focus**: Why nsys has no output when using NVTX gating
**Scope**: Phase 3 (User Story 1) - Stage 2 Deep Profiling

---

## Executive Summary

**Problem Statement**: Nsight Systems profiling produces reports but fails to properly gate capture using NVTX ranges, resulting in either full captures or missing expected filtered data.

**Root Causes Identified**:

1. **CRITICAL**: NVTX range name mismatch - Configuration specifies `range@LLM` but code emits `prefill`, `decode`, `sam`, `clip`, `projector`
2. **CRITICAL**: Invalid nsys report name - Code uses `--report summary` which doesn't exist; should be `cuda_gpu_kern_sum`
3. **HIGH**: Missing `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` environment variable for non-registered NVTX strings
4. **HIGH**: NVTX capture format syntax error - Uses `range` instead of `range@DomainName` or specific range name
5. **MEDIUM**: Silent subprocess failures with `check=False` prevent error detection

**Impact**:
- Nsight Systems captures full timeline instead of gated NVTX ranges
- Top-N kernel selection fails silently because CSV file is never generated
- Nsight Compute runs without kernel filtering, increasing runtime
- No visibility into failures due to suppressed subprocess errors

**Status**: Phase 3 implementation is functionally broken for NVTX-gated capture. Profiling succeeds but without the intended optimization benefits.

---

## Code Intent Analysis

### Overall Architecture

Phase 3 implements a two-stage profiling workflow:

1. **Nsight Systems** (timeline profiler):
   - Capture GPU timeline with NVTX range gating to focus on specific model phases
   - Export statistics to identify top-N kernels by time
   - Generate SQLite database for post-processing

2. **Nsight Compute** (kernel profiler):
   - Profile only top-N kernels identified from Nsight Systems
   - Use NVTX expressions to filter decode regions
   - Collect roofline metrics for performance analysis

### NVTX Gating Strategy

The implementation intends to use NVTX ranges to:
- Gate Nsight Systems capture to reduce overhead and report size
- Filter Nsight Compute to specific decode kernels
- Segment timeline by model stages (prefill, decode, vision components)

### Key Files

| File | Role | NVTX Configuration |
|------|------|-------------------|
| `deep_profile_runner.py` | Orchestrator | Reads `nsys.nvtx_capture` config |
| `nsys.py` | Command builder | Accepts `nvtx_capture` parameter |
| `dsocr_session.py` | NVTX emission | Emits `prefill`, `decode`, `sam`, `clip`, `projector` |
| `nvtx_utils.py` | Helper ranges | Defines context managers for ranges |
| `stage2.yaml` | Configuration | Sets `nvtx_capture: range` |
| `nsys.default.yaml` | Preset | Sets `nvtx_capture: range` |

---

## Third-Party APIs Used

### NVIDIA Nsight Systems CLI

**Command**: `nsys profile`

**Key Options**:
- `--trace`: Trace sources (cuda, nvtx, osrt)
- `--sample`: CPU sampling mode
- `--capture-range`: Capture range selector (none, cudaProfilerApi, nvtx)
- `--nvtx-capture`: NVTX capture expression (range@Domain, Message@Domain, Message)

**Command**: `nsys stats`

**Key Options**:
- `--report`: Report type (cuda_gpu_kern_sum, nvtx_pushpop_sum, NOT "summary")
- `--format`: Output format (csv, column, json, table)
- `-o`: Output file base path

### NVIDIA Nsight Compute CLI

**Command**: `ncu`

**Key Options**:
- `--nvtx-include`: NVTX expression to filter kernels
- `--kernel-regex`: Regex to match kernel names
- `--set`: Metric set (roofline)
- `--target-processes`: Target processes mode

### Python NVTX Library

**Module**: `nvtx` (nvidia-nvtx package)

**Key Functions**:
- `nvtx.push_range(label)`: Start named range
- `nvtx.pop_range()`: End range
- Context managers for safe range management

---

## API Documentation & Best Practices

### 1. NVIDIA Nsight Systems NVTX Capture

**Source**: [NVIDIA Developer Forums - Using capture-range=nvtx](https://forums.developer.nvidia.com/t/using-capture-range-nvtx/254091)

**Key Documentation**:

#### NVTX Capture Expression Format

The `--nvtx-capture` option supports three formats:

1. **Message@Domain**: All ranges with given message in given domain
2. **Message@***: All ranges with given message in all domains
3. **Message**: All ranges with given message in default domain

**Example**:
```bash
# Capture ranges named "decode" in default domain
nsys profile --capture-range=nvtx --nvtx-capture=decode ./app

# Capture ranges named "LLM" with domain "profiling"
nsys profile --capture-range=nvtx --nvtx-capture=LLM@profiling ./app

# Capture all ranges named "compute" regardless of domain
nsys profile --capture-range=nvtx --nvtx-capture=compute@* ./app
```

#### Critical Limitation

**IMPORTANT**: The Nsight Systems CLI only triggers profiling for the **first** capture range encountered. Subsequent ranges with the same name are ignored.

#### String Registration Requirement

**Default Behavior**: Only NVTX registered strings are recognized to avoid overhead.

**Problem**: Python `nvtx.push_range("string")` uses non-registered strings by default.

**Solution**: Set environment variable before launching:
```bash
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0
nsys profile --capture-range=nvtx --nvtx-capture=decode ./app
```

Or inline:
```bash
nsys profile --capture-range=nvtx --nvtx-capture=decode \
    --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 ./app
```

**Reference**: Stack Overflow - [Nsys Does not show the CUDA kernels profiling output](https://stackoverflow.com/questions/73917993/nsys-does-not-show-the-cuda-kernels-profiling-output)

### 2. NVIDIA Nsight Systems Report Types

**Source**: `nsys stats --help-reports`

**Available Reports** (excerpt):
- `cuda_gpu_kern_sum` - CUDA GPU Kernel Summary ✓
- `cuda_api_sum` - CUDA API Summary ✓
- `nvtx_pushpop_sum` - NVTX Push/Pop Range Summary ✓
- `nvtx_sum` - NVTX Range Summary ✓
- **NO** `summary` report ✗

**Correct Usage**:
```bash
# WRONG - "summary" report doesn't exist
nsys stats --report summary --format csv -o output.csv report.nsys-rep

# CORRECT - Use cuda_gpu_kern_sum for kernel data
nsys stats --report cuda_gpu_kern_sum --format csv -o output.csv report.nsys-rep
```

**CSV Output Format** for `cuda_gpu_kern_sum`:
```csv
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
9.1,65295462,9263,7049.1,6911.0,6015,9472,939.2,"kernel_name(...)"
```

### 3. Python NVTX Best Practices

**Source**: PyTorch Developer Mailing List - [Using Nsight Systems to profile GPU workload](https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59)

**Best Practice**: Use context managers for safe range management:

```python
import nvtx

# GOOD - Context manager ensures pop even on exception
with nvtx.annotate("operation", color="blue"):
    do_work()

# ACCEPTABLE - Manual push/pop with try/finally
nvtx.push_range("operation")
try:
    do_work()
finally:
    nvtx.pop_range()

# BAD - No exception safety
nvtx.push_range("operation")
do_work()  # If this throws, pop_range() never called
nvtx.pop_range()
```

**Domain Labels**: NVTX supports domains for organizing ranges:

```python
# Default domain (no label)
nvtx.push_range("prefill")

# Named domain (requires NVTX 3.0+)
# Note: Python nvtx package may not expose domain API
nvtx.push_range("prefill", domain="LLM")
```

### 4. Subprocess Error Handling in Python

**Source**: [Python subprocess documentation](https://docs.python.org/3/library/subprocess.html)

**Best Practice**: Always use `check=True` or explicit error handling:

```python
# BAD - Silent failures
subprocess.run(cmd, check=False)

# GOOD - Raises CalledProcessError on failure
subprocess.run(cmd, check=True)

# GOOD - Explicit error handling
result = subprocess.run(cmd, check=False)
if result.returncode != 0:
    logger.error(f"Command failed: {result.returncode}")
    raise RuntimeError("Profiling failed")
```

**Rationale**: Profiling commands can fail for many reasons:
- Tools not in PATH
- Insufficient permissions
- CUDA driver errors
- Disk space issues
- Invalid arguments

Silent failures lead to incomplete artifacts and wasted time debugging.

---

## Detailed File-by-File Review

### File: `src/llm_perf_opt/runners/deep_profile_runner.py`

**Lines 115-123**: NVTX Capture Configuration

```python
115    gating_nvtx_nsys = bool(getattr(getattr(cfg, "nsys", {}), "gating_nvtx", True))
116    nsys_cmd = build_nsys_cmd(
117        nsys_out,
118        work,
119        nvtx_capture=str(getattr(getattr(cfg, "nsys", {}), "nvtx_capture", "range")) if gating_nvtx_nsys else "none",
120        trace=str(getattr(getattr(cfg, "nsys", {}), "trace", "cuda,nvtx,osrt")),
121        sample=str(getattr(getattr(cfg, "nsys", {}), "sample", "none")),
122        capture=str(getattr(getattr(cfg, "nsys", {}), "capture", "nvtx")) if gating_nvtx_nsys else "none",
123    )
124    subprocess.run(nsys_cmd, check=False)
```

**Issues**:

1. **CRITICAL** (Line 119): `nvtx_capture="range"` is invalid syntax
   - **Expected**: Specific range name like `"prefill"` or `"decode"` or `"prefill@LLM"`
   - **Actual**: Generic word `"range"` which matches nothing
   - **Impact**: No NVTX gating occurs; full capture runs

2. **CRITICAL** (Line 124): `check=False` suppresses errors
   - **Problem**: If nsys fails (wrong arguments, tool missing, etc.), code continues silently
   - **Impact**: Subsequent steps try to parse non-existent outputs

**Lines 127-134**: Stats Export

```python
127    from llm_perf_opt.profiling.vendor.nsys import resolve_nsys_report_path
128    report_path = resolve_nsys_report_path(nsys_out)
129    if report_path is not None:
130        nsys_summary_base = artifacts.path("nsys/summary")
131        nsys_stats_cmd = build_nsys_stats_cmd(report_path, nsys_summary_base)
132        subprocess.run(nsys_stats_cmd, check=False)
133        nsys_sqlite_cmd = build_nsys_export_sqlite_cmd(report_path)
134        subprocess.run(nsys_sqlite_cmd, check=False)
```

**Issues**:

1. **CRITICAL** (Line 131): `build_nsys_stats_cmd()` generates invalid command
   - **Problem**: Uses `--report summary` which doesn't exist in nsys
   - **Impact**: Command fails silently (line 132 `check=False`), no CSV generated

2. **HIGH** (Lines 132, 134): Silent failures prevent error detection
   - **Problem**: If stats or sqlite export fail, code continues
   - **Impact**: Line 149 tries to parse non-existent CSV

**Lines 145-155**: Top-N Kernel Selection

```python
145    # Select top-N kernels from nsys summary and focus on decode region
146    kernel_regex = None
147    try:
148        top_n = int(getattr(getattr(cfg, "run", {}), "top_n_kernels", 30))
149        names = top_kernels_from_nsys_summary(Path(str(nsys_summary_base) + ".csv"), top_n=top_n)
150        if names:
151            import re as _re
152            pats = ["(" + _re.escape(n) + ")" for n in names]
153            kernel_regex = "|".join(pats)
154    except Exception:
155        kernel_regex = None
```

**Issues**:

1. **HIGH** (Line 149): Fails silently when CSV doesn't exist
   - **Problem**: Broad `except Exception` catches FileNotFoundError
   - **Impact**: `kernel_regex` remains None, ncu profiles ALL kernels (very slow)

2. **MEDIUM** (Line 154): Overly broad exception handling
   - **Problem**: Catches and hides programming errors (typos, logic bugs)
   - **Best Practice**: Catch specific exceptions or let them propagate

**Recommendations**:

```python
# Fix NVTX capture to match actual range names
nvtx_capture = "prefill" if gating_nvtx_nsys else "none"
# Or configure dynamically from list of target ranges

# Add error handling for subprocess calls
try:
    subprocess.run(nsys_cmd, check=True)
except subprocess.CalledProcessError as e:
    logger.error(f"Nsight Systems failed: {e.returncode}")
    raise RuntimeError("Profiling failed") from e

# Add environment variable for NVTX string registration
import os
os.environ["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"
# Or add --env-var to nsys command

# Narrow exception handling
try:
    names = top_kernels_from_nsys_summary(csv_path, top_n=top_n)
except FileNotFoundError:
    logger.warning(f"Stats CSV not found: {csv_path}")
    names = []
except Exception as e:
    logger.error(f"Failed to parse kernel names: {e}")
    raise
```

---

### File: `src/llm_perf_opt/profiling/vendor/nsys.py`

**Lines 11-65**: NVTX Command Builder

```python
11  def build_nsys_cmd(
12      out_base: Path,
13      work_argv: Sequence[str],
14      *,
15      trace: str = "cuda,nvtx,osrt",
16      sample: str = "none",
17      capture: str = "nvtx",
18      nvtx_capture: str = "range@LLM",
19  ) -> list[str]:
...
52      cmd = [
53          "nsys",
54          "profile",
55          f"--trace={trace}",
56          f"--sample={sample}",
57      ]
58      # Only add capture-range/NVTX capture when not explicitly disabled
59      if str(capture).lower() != "none":
60          cmd += [f"--capture-range={capture}"]
61      if str(nvtx_capture).lower() != "none":
62          cmd += [f"--nvtx-capture={nvtx_capture}"]
63      cmd += ["-o", str(out_base)]
64      return cmd + list(work_argv)
```

**Issues**:

1. **CRITICAL** (Line 18): Default `nvtx_capture="range@LLM"` doesn't match emitted ranges
   - **Problem**: Code emits `prefill`, `decode`, `sam`, `clip`, `projector` (no `LLM` range)
   - **Expected**: Should be `"prefill"` or `"decode"` or configurable list
   - **Impact**: NVTX gating never triggers; full capture occurs

2. **MEDIUM** (Lines 59-62): No validation of NVTX capture expression format
   - **Problem**: Accepts any string; nsys fails silently on invalid syntax
   - **Best Practice**: Validate format (Message, Message@Domain, Message@*)

3. **MISSING**: No `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` environment variable
   - **Problem**: Python nvtx uses non-registered strings by default
   - **Impact**: Even with correct range name, capture may not trigger

**Lines 68-89**: Stats Command Builder

```python
68  def build_nsys_stats_cmd(report_path: Path, out_csv_base: Path) -> list[str]:
69      """Return `nsys stats` argv for summary CSV export.
...
78      return [
79          "nsys",
80          "stats",
81          "--report",
82          "summary",
83          "--format",
84          "csv",
85          "-o",
86          str(out_csv_base),
87          str(report_path),
88      ]
```

**Issues**:

1. **CRITICAL** (Line 82): Report name `"summary"` doesn't exist
   - **Correct**: Should be `"cuda_gpu_kern_sum"` for kernel summary
   - **Impact**: Command fails with "Report 'summary' could not be found"
   - **Evidence**:
     ```bash
     $ nsys stats --help-reports
     # Lists: cuda_gpu_kern_sum, nvtx_pushpop_sum, etc.
     # NO "summary" report
     ```

**Recommendations**:

```python
def build_nsys_cmd(
    out_base: Path,
    work_argv: Sequence[str],
    *,
    trace: str = "cuda,nvtx,osrt",
    sample: str = "none",
    capture: str = "nvtx",
    nvtx_capture: str = "prefill",  # FIX: Match actual range name
    env_vars: dict[str, str] | None = None,  # NEW: Support env vars
) -> list[str]:
    """Build an argv list for `nsys profile` with NVTX gating."""

    cmd = ["nsys", "profile"]

    # Add environment variables (e.g., NSYS_NVTX_PROFILER_REGISTER_ONLY)
    if env_vars:
        for k, v in env_vars.items():
            cmd += [f"--env-var={k}={v}"]

    cmd += [
        f"--trace={trace}",
        f"--sample={sample}",
    ]

    # Validate and add NVTX capture
    if str(capture).lower() != "none":
        cmd += [f"--capture-range={capture}"]
    if str(nvtx_capture).lower() not in ("none", ""):
        # Validate format: Message, Message@Domain, or Message@*
        if "@" not in nvtx_capture and nvtx_capture not in ("range", ""):
            # Valid bare message name
            pass
        elif nvtx_capture.count("@") == 1:
            msg, domain = nvtx_capture.split("@")
            if not msg or (not domain and domain != "*"):
                raise ValueError(f"Invalid nvtx_capture format: {nvtx_capture}")
        else:
            raise ValueError(f"Invalid nvtx_capture format: {nvtx_capture}")
        cmd += [f"--nvtx-capture={nvtx_capture}"]

    cmd += ["-o", str(out_base)]
    return cmd + list(work_argv)

def build_nsys_stats_cmd(report_path: Path, out_csv_base: Path) -> list[str]:
    """Return `nsys stats` argv for kernel summary CSV export."""
    return [
        "nsys",
        "stats",
        "--report",
        "cuda_gpu_kern_sum",  # FIX: Use correct report name
        "--format",
        "csv",
        "-o",
        str(out_csv_base),
        str(report_path),
    ]
```

---

### File: `src/llm_perf_opt/runners/dsocr_session.py`

**Lines 310-326**: NVTX Range Emission

```python
310        with prefill_range():
311            if use_pre:
312                inputs = {"input_ids": input_ids}
313            else:
314                nvtx.push_range("tokenizer")
315                try:
316                    inputs = self.m_tokenizer(prompt, return_tensors="pt").to(self.m_device)
317                finally:
318                    nvtx.pop_range()
319            _ = self.m_model(
320                **inputs,
321                images=images,
322                images_seq_mask=images_seq_mask,
323                images_spatial_crop=images_spatial_crop,
324            )
325        prefill_ms = (perf_counter() - t0) * 1000.0
326        logger.info("Prefill done | image=%s prefill_ms=%.3f", img_abs, prefill_ms)
...
330        with decode_range():
...
352        decode_ms = (perf_counter() - t1) * 1000.0
```

**Analysis**:

- **Line 310**: Emits NVTX range named `"prefill"` (via `nvtx_utils.prefill_range()`)
- **Line 330**: Emits NVTX range named `"decode"` (via `nvtx_utils.decode_range()`)

**Lines 119-146**: Stage-Level NVTX Hooks

```python
119        def _attach(mod: Any, label: str) -> None:
...
124                def _pre(_m, _inp):
125                    nvtx.push_range(label)
126                    _m.__dict__["__nvtx_t0"] = time.perf_counter()
127
128                def _post(_m, _inp, _out):
129                    nvtx.pop_range()
...
144        _attach(getattr(core, "sam_model", None), "sam")
145        _attach(getattr(core, "vision_model", None), "clip")
146        _attach(getattr(core, "projector", None), "projector")
```

**Analysis**:

- **Line 144-146**: Emits NVTX ranges `"sam"`, `"clip"`, `"projector"`
- **IMPORTANT**: All ranges use **default domain** (no domain label)

**Summary of Emitted Ranges**:

| Range Name | Domain | Source |
|------------|--------|--------|
| `prefill` | default | `nvtx_utils.prefill_range()` |
| `decode` | default | `nvtx_utils.decode_range()` |
| `sam` | default | Stage hook |
| `clip` | default | Stage hook |
| `projector` | default | Stage hook |
| `tokenizer` | default | Manual push/pop (conditional) |

**Evidence from Actual Profile**:

```csv
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Range
65.9,2782360479,1,2782360479.0,2782360479.0,2782360479,2782360479,0.0,:decode
20.7,874354275,1,874354275.0,874354275.0,874354275,874354275,0.0,:prefill
9.5,401811376,4,100452844.0,22405077.0,17208883,339792339,159609076.8,:sam
3.9,164658307,4,41164576.8,43661193.5,10149343,67186577,29123575.6,:clip
0.0,416739,4,104184.8,100113.5,71229,145283,35978.8,:projector
```

**Observations**:

1. ✓ All expected ranges are present
2. ✓ Ranges are in default domain (`:` prefix indicates default domain)
3. ✗ Configuration looks for `"range@LLM"` which doesn't exist
4. ✗ Configuration uses bare `"range"` which is not a valid range name

**Recommendation**:

No changes needed in `dsocr_session.py` - NVTX emission is correct. The bug is in the **configuration mismatch**.

---

### File: `src/llm_perf_opt/profiling/nvtx_utils.py`

**Lines 15-34**: Context Managers

```python
@contextmanager
def prefill_range() -> Iterator[None]:
    """NVTX range labeled 'prefill'."""
    nvtx.push_range("prefill")
    try:
        yield
    finally:
        nvtx.pop_range()

@contextmanager
def decode_range() -> Iterator[None]:
    """NVTX range labeled 'decode'."""
    nvtx.push_range("decode")
    try:
        yield
    finally:
        nvtx.pop_range()
```

**Analysis**:

✓ **GOOD**: Context managers ensure `pop_range()` is called even on exception
✓ **GOOD**: Clear naming matches profiling intent
✗ **ISSUE**: No domain labels - all ranges go to default domain

**Lines 37-56**: LLM-Prefixed Ranges (Unused)

```python
@contextmanager
def llm_prefill() -> Iterator[None]:
    """NVTX range labeled 'LLM@prefill' (Stage 2)."""
    nvtx.push_range("LLM@prefill")
    try:
        yield
    finally:
        nvtx.pop_range()

@contextmanager
def llm_decode_all() -> Iterator[None]:
    """NVTX range labeled 'LLM@decode_all' (Stage 2)."""
    nvtx.push_range("LLM@decode_all")
    try:
        yield
    finally:
        nvtx.pop_range()
```

**Analysis**:

- ✗ **UNUSED**: These context managers are defined but never imported/used
- ⚠️ **MISLEADING**: Label `"LLM@prefill"` looks like domain syntax but is actually a literal string with `@` character
- ⚠️ **CONFUSION**: Configuration references `"range@LLM"` but no code emits `"LLM"` range

**NVTX Domain Syntax Clarification**:

```python
# This is a LITERAL STRING "LLM@prefill" in the default domain
nvtx.push_range("LLM@prefill")

# To use actual domains, need NVTX 3.0+ API (Python package may not support):
nvtx.push_range("prefill", domain="LLM")
```

**Recommendation**:

Either:
1. Remove unused `llm_prefill()` and `llm_decode_all()` to reduce confusion
2. Switch `dsocr_session.py` to use these instead of bare `prefill_range()` / `decode_range()`
3. Update configuration to match actual range names (`"prefill"`, not `"range@LLM"`)

---

### File: `conf/runner/stage2.yaml`

**Lines 37-40**: NVTX Gating Switches

```yaml
nsys:
  gating_nvtx: true
ncu:
  gating_nvtx: true
```

**Analysis**:

✓ **GOOD**: Boolean switches allow disabling NVTX gating for testing
✗ **ISSUE**: No `nvtx_capture` override; relies on defaults from `nsys.default.yaml`

**Recommendation**:

Add explicit configuration with correct range names:

```yaml
nsys:
  gating_nvtx: true
  nvtx_capture: "prefill"  # or "decode" or comma-separated list
  # Alternative: capture first range (prefill), but profile entire run
  # nvtx_capture: "none"  # Disable gating for full capture
ncu:
  gating_nvtx: true
  nvtx_include: "decode*"  # Existing config is correct
```

---

### File: `conf/profiling/nsys/nsys.default.yaml`

**Lines 1-8**: Nsight Systems Preset

```yaml
# Nsight Systems defaults (Stage 2)

trace: cuda,nvtx,osrt    # sources to trace
sample: none             # CPU sampling: 'none' for low overhead
capture: nvtx            # capture-range selector
nvtx_capture: range      # gate capture by NVTX ranges (no label filter)
```

**Issues**:

1. **CRITICAL** (Line 6): `nvtx_capture: range` is invalid
   - **Problem**: `"range"` is not a valid NVTX range name
   - **Expected**: Specific range name like `"prefill"`, `"decode"`, etc.
   - **Impact**: Nsys command is malformed; no gating occurs

2. **MISLEADING** (Line 6 comment): "no label filter" is incorrect
   - **Actual**: Specifying `nvtx_capture` IS a label filter
   - **Correct comment**: "gate capture by NVTX range named 'XXX'"

**Recommendation**:

```yaml
# Nsight Systems defaults (Stage 2)

trace: cuda,nvtx,osrt    # sources to trace
sample: none             # CPU sampling: 'none' for low overhead
capture: nvtx            # capture-range selector (none, cudaProfilerApi, nvtx)
nvtx_capture: prefill    # NVTX range name to trigger capture (prefill, decode, etc.)
# Note: Only the first instance of the named range triggers capture
# Set to "none" to disable NVTX gating and capture full timeline
```

**Alternative** (if multiple ranges needed):

Since nsys only captures the first range, consider:
- Use `"prefill"` to capture initial forward pass
- Use `"decode"` to capture decode loop (typically longer)
- Use `"none"` to capture everything (no gating)

---

### File: `src/llm_perf_opt/profiling/nsys_stats.py`

**Lines 15-36**: Stats Command Builder (Duplicate)

```python
15  def build_nsys_stats_cmd(qdrep_path: Path, out_csv: Path) -> List[str]:
...
26      return [
27          "nsys",
28          "stats",
29          "--report",
30          "summary",
31          "--format",
32          "csv",
33          "-o",
34          str(out_csv),
35          str(qdrep_path),
36      ]
```

**Issues**:

1. **CRITICAL** (Line 30): Same bug as `nsys.py` - report `"summary"` doesn't exist
2. **DUPLICATION**: This function duplicates `build_nsys_stats_cmd()` in `nsys.py`
   - **Consequence**: Bug exists in two places; must fix both
   - **Best Practice**: Single source of truth; import from one module

**Lines 74-120**: CSV Parser

```python
74  def top_kernels_from_nsys_summary(csv_path: Path, top_n: int = 30) -> list[str]:
...
89      for i, row in enumerate(rows):
90          if row and row[0].strip().startswith("CUDA GPU Kernel Summary"):
91              # Next non-empty row is header
92              j = i + 1
93              while j < len(rows) and (not rows[j] or all(not c for c in rows[j])):
94                  j += 1
95              if j < len(rows):
96                  header = rows[j]
97                  start_idx = j + 1
98              break
```

**Issues**:

1. **CRITICAL** (Line 90): Looks for section named `"CUDA GPU Kernel Summary"`
   - **Problem**: `--report summary` output would have different sections
   - **Correct Report**: `cuda_gpu_kern_sum` has this section
   - **Impact**: Parser expects format that will never be generated

2. **FRAGILE** (Lines 92-94): Heuristic header detection
   - **Problem**: Assumes specific blank-line structure
   - **Risk**: Breaks if nsys changes CSV format

**Evidence**:

```bash
# Correct report (cuda_gpu_kern_sum) generates parseable CSV:
$ nsys stats --report cuda_gpu_kern_sum --format csv report.nsys-rep
Time (%),Total Time (ns),Instances,Avg (ns),...,Name
9.1,65295462,9263,7049.1,...,"kernel_name"

# No section headers, direct kernel data
```

**Observation**: Parser looks for `"CUDA GPU Kernel Summary"` section header, but `cuda_gpu_kern_sum` CSV has no section headers - it starts directly with column headers.

**Recommendation**:

```python
def top_kernels_from_nsys_summary(csv_path: Path, top_n: int = 30) -> list[str]:
    """Parse nsys cuda_gpu_kern_sum CSV and return top kernel names.

    Expects CSV format from: nsys stats --report cuda_gpu_kern_sum --format csv

    CSV Structure:
        Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
        9.1,65295462,9263,7049.1,6911.0,6015,9472,939.2,"kernel_name(...)"
        ...
    """

    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return []

    names: dict[str, float] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Verify expected columns
        if not reader.fieldnames or "Name" not in reader.fieldnames:
            logger.error(f"CSV missing 'Name' column: {csv_path}")
            return []

        # Find time column
        time_cols = ["Total Time (ns)", "Time (ns) Sum", "Total Time (ms)"]
        time_col = next((c for c in time_cols if c in reader.fieldnames), None)
        if not time_col:
            logger.error(f"CSV missing time column: {csv_path}")
            return []

        # Accumulate kernel times
        for row in reader:
            try:
                name = row["Name"].strip()
                time_val = float(row[time_col].replace(",", ""))
                names[name] = names.get(name, 0.0) + time_val
            except (KeyError, ValueError) as e:
                logger.debug(f"Skipping row: {e}")
                continue

    # Sort by total time descending
    sorted_names = sorted(names.items(), key=lambda kv: kv[1], reverse=True)
    return [n for n, _ in sorted_names[:top_n]]
```

---

## Root Cause Analysis

### Root Cause 1: NVTX Range Name Mismatch

**What**: Configuration specifies `nvtx_capture="range@LLM"` but code emits `"prefill"`, `"decode"`, etc.

**Why**:
1. Configuration was written based on hypothetical `"LLM"` range
2. Implementation uses descriptive range names (`"prefill"`, `"decode"`)
3. Unused helper functions (`llm_prefill()`, `llm_decode_all()`) suggest naming was changed
4. No integration test validates NVTX gating end-to-end

**Evidence**:
- `nsys.default.yaml` line 6: `nvtx_capture: range`
- `dsocr_session.py` lines 310, 330: Emits `"prefill"` and `"decode"`
- `nvtx_utils.py` lines 37-56: Unused `llm_prefill()` / `llm_decode_all()`
- Actual profile: NVTX ranges are `:prefill`, `:decode`, `:sam`, `:clip`, `:projector`

**Fix Priority**: CRITICAL

**Recommended Fix**:

1. **Option A** (minimal change): Update configuration to match existing code
   ```yaml
   # conf/profiling/nsys/nsys.default.yaml
   nvtx_capture: prefill  # or "decode"
   ```

2. **Option B** (semantic naming): Switch to LLM-prefixed ranges
   ```python
   # dsocr_session.py - replace imports
   from llm_perf_opt.profiling.nvtx_utils import llm_prefill, llm_decode_all

   # Line 310: with llm_prefill():
   # Line 330: with llm_decode_all():
   ```
   ```yaml
   # conf/profiling/nsys/nsys.default.yaml
   nvtx_capture: LLM@prefill  # literal string with @
   ```

3. **Option C** (no gating): Disable until proper configuration is determined
   ```yaml
   # conf/profiling/nsys/nsys.default.yaml
   nvtx_capture: none  # full capture
   ```

**Recommended Approach**: Option A (minimal change) to unblock Phase 3, then revisit naming in Polish phase.

---

### Root Cause 2: Invalid Nsys Report Name

**What**: Code uses `--report summary` but nsys doesn't have a report named "summary".

**Why**:
1. Documented nsys reports: `cuda_gpu_kern_sum`, `nvtx_pushpop_sum`, etc.
2. Code assumes generic "summary" report exists
3. No validation of report name before execution
4. Silent failure (`check=False`) hides the error

**Evidence**:
```bash
$ nsys stats --help-reports | grep summary
# No "summary" report listed

$ nsys stats --report summary report.nsys-rep
ERROR: Report 'summary' could not be found.
```

**Fix Priority**: CRITICAL

**Recommended Fix**:

```python
# nsys.py and nsys_stats.py - both locations
def build_nsys_stats_cmd(report_path: Path, out_csv_base: Path) -> list[str]:
    return [
        "nsys",
        "stats",
        "--report",
        "cuda_gpu_kern_sum",  # FIXED
        "--format",
        "csv",
        "-o",
        str(out_csv_base),
        str(report_path),
    ]
```

---

### Root Cause 3: Missing NSYS_NVTX_PROFILER_REGISTER_ONLY Environment Variable

**What**: Python nvtx uses non-registered strings, but nsys requires registered strings by default.

**Why**:
1. Nsys optimization: Only check registered strings to reduce overhead
2. Python nvtx.push_range() uses non-registered strings
3. No environment variable set to enable non-registered strings
4. NVTX ranges never recognized even with correct names

**Evidence**:
- NVIDIA documentation: "By default, only messages provided by NVTX registered strings are considered"
- Solution: Set `NSYS_NVTX_PROFILER_REGISTER_ONLY=0`

**Fix Priority**: HIGH

**Recommended Fix**:

```python
# deep_profile_runner.py - before nsys command
import os
os.environ["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"

# OR add to nsys command via --env-var
# nsys.py
def build_nsys_cmd(..., enable_nonregistered_nvtx: bool = True):
    cmd = ["nsys", "profile"]
    if enable_nonregistered_nvtx:
        cmd += ["--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0"]
    # ... rest of command
```

---

### Root Cause 4: Silent Subprocess Failures

**What**: All `subprocess.run()` calls use `check=False`, suppressing errors.

**Why**:
1. Convenience: Avoid handling exceptions
2. Assumption: Profiling is "best effort"
3. Consequence: Errors invisible; downstream steps fail mysteriously

**Evidence**:
- Line 124, 132, 134, 166: `subprocess.run(..., check=False)`
- No logging of return codes
- No validation of output artifacts before using them

**Fix Priority**: HIGH

**Recommended Fix**:

```python
# deep_profile_runner.py
import logging
logger = logging.getLogger(__name__)

# Pattern for all subprocess calls:
try:
    result = subprocess.run(nsys_cmd, check=True, capture_output=True, text=True)
    logger.info(f"Nsight Systems completed: {result.returncode}")
except subprocess.CalledProcessError as e:
    logger.error(f"Nsight Systems failed (exit {e.returncode})")
    logger.error(f"stderr: {e.stderr}")
    raise RuntimeError("Profiling failed") from e

# Alternative: Log and continue for optional steps
result = subprocess.run(nsys_stats_cmd, check=False)
if result.returncode != 0:
    logger.warning(f"Stats export failed: {result.returncode}")
else:
    logger.info("Stats export succeeded")
```

---

## Impact Assessment

### Current Behavior

When Stage 2 profiling runs:

1. ✓ Nsys executes successfully
2. ✗ NVTX gating doesn't trigger (wrong range name + missing env var)
3. ✓ Full timeline captured (fallback behavior)
4. ✓ Report file created (`.nsys-rep`, `.sqlite`)
5. ✗ Stats CSV generation fails silently (invalid report name)
6. ✗ Top-N kernel selection returns empty list
7. ✗ NCU profiles ALL kernels instead of top-N (very slow)
8. ✓ NCU report generated but contains excessive data

### Consequences

**Performance**:
- Full timeline capture increases:
  - Profiling overhead (~2-5x slowdown)
  - Report file size (100s of MB vs 10s of MB)
  - Analysis time (more data to filter)
- NCU profiling all kernels:
  - Extremely slow (minutes vs seconds)
  - May hit timeout/memory limits

**Correctness**:
- Missing top-N filtering → profile wrong kernels
- No validation → cannot detect failures
- Silent errors → waste hours debugging

**Usability**:
- Cannot use NVTX gating (key feature broken)
- Cannot optimize profiling runs
- Phase 3 acceptance criteria not met:
  - ✗ "capture kernel-level metrics" (yes, but wrong ones)
  - ✗ "top-N kernels strategy" (broken)

### Risk Level

**CRITICAL** - Core functionality broken:
- NVTX gating is advertised feature but doesn't work
- Performance optimization strategy (top-N kernels) is broken
- Phase 3 is not production-ready

---

## Recommended Fixes (Priority Order)

### 1. Fix NVTX Range Name Mismatch (CRITICAL)

**File**: `conf/profiling/nsys/nsys.default.yaml`

**Change**:
```yaml
# BEFORE
nvtx_capture: range      # WRONG

# AFTER
nvtx_capture: prefill    # Match actual range name
# Alternative: "decode" to capture decode loop
# Alternative: "none" to disable gating
```

**Rationale**: Minimal change; unblocks NVTX gating immediately.

---

### 2. Fix Invalid Nsys Report Name (CRITICAL)

**Files**:
- `src/llm_perf_opt/profiling/vendor/nsys.py` line 82
- `src/llm_perf_opt/profiling/nsys_stats.py` line 30

**Change**:
```python
# BEFORE
"summary",  # WRONG - doesn't exist

# AFTER
"cuda_gpu_kern_sum",  # CORRECT
```

**Rationale**: Fixes CSV generation; enables top-N kernel selection.

---

### 3. Add NSYS_NVTX_PROFILER_REGISTER_ONLY Environment Variable (HIGH)

**File**: `src/llm_perf_opt/profiling/vendor/nsys.py`

**Change**:
```python
def build_nsys_cmd(
    out_base: Path,
    work_argv: Sequence[str],
    *,
    trace: str = "cuda,nvtx,osrt",
    sample: str = "none",
    capture: str = "nvtx",
    nvtx_capture: str = "prefill",
) -> list[str]:
    cmd = [
        "nsys",
        "profile",
        # NEW: Enable non-registered NVTX strings (Python nvtx)
        "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
        f"--trace={trace}",
        f"--sample={sample}",
    ]
    # ... rest unchanged
```

**Rationale**: Ensures Python nvtx ranges are recognized by nsys.

---

### 4. Add Subprocess Error Handling (HIGH)

**File**: `src/llm_perf_opt/runners/deep_profile_runner.py`

**Change**:
```python
import logging
logger = logging.getLogger(__name__)

# Lines 124, 132, 134, 166 - replace ALL subprocess.run() calls

# BEFORE
subprocess.run(nsys_cmd, check=False)

# AFTER
try:
    subprocess.run(nsys_cmd, check=True)
except subprocess.CalledProcessError as e:
    logger.error(f"Nsight Systems failed: {e.returncode}")
    raise RuntimeError("Nsight Systems profiling failed") from e
```

**Rationale**: Makes failures visible; prevents cascading errors.

---

### 5. Fix CSV Parser for New Report Format (MEDIUM)

**File**: `src/llm_perf_opt/profiling/nsys_stats.py`

**Change**:
```python
def top_kernels_from_nsys_summary(csv_path: Path, top_n: int = 30) -> list[str]:
    """Parse cuda_gpu_kern_sum CSV and return top kernel names."""

    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return []

    names: dict[str, float] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # NEW: Direct column-based parsing (no section scanning)
        if not reader.fieldnames or "Name" not in reader.fieldnames:
            logger.error(f"Invalid CSV format: {csv_path}")
            return []

        time_col = "Total Time (ns)"
        if time_col not in reader.fieldnames:
            logger.error(f"Missing time column: {csv_path}")
            return []

        for row in reader:
            try:
                name = row["Name"].strip()
                time_val = float(row[time_col].replace(",", ""))
                names[name] = names.get(name, 0.0) + time_val
            except (KeyError, ValueError):
                continue

    sorted_names = sorted(names.items(), key=lambda kv: kv[1], reverse=True)
    return [n for n, _ in sorted_names[:top_n]]
```

**Rationale**: Aligns with actual `cuda_gpu_kern_sum` CSV structure.

---

### 6. Remove Duplicate Function (LOW)

**File**: `src/llm_perf_opt/profiling/nsys_stats.py`

**Change**: Remove `build_nsys_stats_cmd()` (lines 15-36), import from `vendor.nsys` instead.

**Rationale**: Single source of truth; reduce maintenance burden.

---

### 7. Validate NVTX Capture Expression Format (MEDIUM)

**File**: `src/llm_perf_opt/profiling/vendor/nsys.py`

**Change**: Add validation in `build_nsys_cmd()`:

```python
def _validate_nvtx_capture(expr: str) -> bool:
    """Validate NVTX capture expression format."""
    if expr in ("none", ""):
        return True
    # Format: Message, Message@Domain, or Message@*
    if "@" not in expr:
        return bool(expr)  # Bare message name
    parts = expr.split("@")
    if len(parts) != 2:
        return False
    msg, domain = parts
    return bool(msg) and (bool(domain) or domain == "*")

# In build_nsys_cmd():
if nvtx_capture.lower() not in ("none", ""):
    if not _validate_nvtx_capture(nvtx_capture):
        raise ValueError(f"Invalid NVTX capture expression: {nvtx_capture}")
    cmd += [f"--nvtx-capture={nvtx_capture}"]
```

**Rationale**: Fail fast on configuration errors.

---

## Testing Recommendations

### Integration Test for NVTX Gating

**File**: `tests/integration/test_nvtx_gating.py` (new)

```python
def test_nvtx_gating_produces_smaller_reports():
    """Verify NVTX gating reduces report size vs full capture."""

    # Run with NVTX gating
    run_with_gating = subprocess.run([
        "pixi", "run", "stage2-profile",
        "nsys.nvtx_capture=prefill",
        "nsys.gating_nvtx=true"
    ], check=True, capture_output=True)

    # Run without gating
    run_full = subprocess.run([
        "pixi", "run", "stage2-profile",
        "nsys.gating_nvtx=false"
    ], check=True, capture_output=True)

    # Parse report sizes
    gated_size = get_nsys_report_size(run_with_gating.stdout)
    full_size = get_nsys_report_size(run_full.stdout)

    # NVTX gating should produce smaller reports
    assert gated_size < full_size * 0.5, \
        "NVTX gating should reduce report size by at least 50%"
```

### Unit Test for CSV Parser

**File**: `tests/unit/test_nsys_stats.py` (new)

```python
def test_top_kernels_from_cuda_gpu_kern_sum():
    """Verify parser handles cuda_gpu_kern_sum CSV format."""

    csv_content = """Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
9.1,65295462,9263,7049.1,6911.0,6015,9472,939.2,"kernel_a"
7.4,53006414,8320,6371.0,6336.0,6175,7200,129.3,"kernel_b"
5.1,36789028,78,471654.2,393756.0,160376,6208071,940561.7,"kernel_c"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_path = Path(f.name)

    try:
        result = top_kernels_from_nsys_summary(csv_path, top_n=2)
        assert result == ["kernel_a", "kernel_b"], \
            f"Expected top 2 kernels by total time, got {result}"
    finally:
        csv_path.unlink()
```

### Manual Validation Steps

After applying fixes:

1. **Verify NVTX range names**:
   ```bash
   pixi run stage2-profile
   nsys stats --report nvtx_pushpop_sum --format csv \
       tmp/stage2/<run_id>/nsys/run.nsys-rep
   # Should show: prefill, decode, sam, clip, projector
   ```

2. **Verify stats CSV generation**:
   ```bash
   ls -lh tmp/stage2/<run_id>/nsys/summary.csv
   # Should exist and contain kernel data
   ```

3. **Verify top-N kernel selection**:
   ```bash
   grep "kernel_regex" tmp/stage2/<run_id>/config.yaml
   # Should show regex with top kernel names
   ```

4. **Verify NVTX gating reduces file size**:
   ```bash
   # With gating
   pixi run stage2-profile nsys.nvtx_capture=prefill
   ls -lh tmp/stage2/<run_id>/nsys/run.nsys-rep

   # Without gating
   pixi run stage2-profile nsys.gating_nvtx=false
   ls -lh tmp/stage2/<run_id>/nsys/run.nsys-rep

   # Gated report should be significantly smaller
   ```

---

## References

### Online Documentation

1. **NVIDIA Developer Forums** - Using capture-range=nvtx
   https://forums.developer.nvidia.com/t/using-capture-range-nvtx/254091
   - NVTX capture expression format
   - NSYS_NVTX_PROFILER_REGISTER_ONLY environment variable
   - First-range-only limitation

2. **Stack Overflow** - Nsys Does not show the CUDA kernels profiling output
   https://stackoverflow.com/questions/73917993/nsys-does-not-show-the-cuda-kernels-profiling-output
   - Non-registered NVTX strings issue
   - Environment variable solution

3. **NVIDIA Nsight Systems User Guide** (2025.3)
   https://docs.nvidia.com/nsight-systems/UserGuide/
   - Complete CLI reference
   - Report types and formats
   - NVTX integration best practices

4. **PyTorch Developer Mailing List** - Using Nsight Systems to profile GPU workload
   https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59
   - Python NVTX best practices
   - Context manager patterns

5. **Python subprocess documentation**
   https://docs.python.org/3/library/subprocess.html
   - Error handling with `check=True`
   - Best practices for subprocess invocation

### Context7 Queries

No Context7 queries performed (nvidia-nvtx not found in library index).

### Project Files Referenced

- `specs/002-nvidia-llm-profiling/tasks.md` - Phase 3 requirements
- `conf/runner/stage2.yaml` - Stage 2 configuration
- `conf/profiling/nsys/nsys.default.yaml` - Nsys preset
- `src/llm_perf_opt/runners/deep_profile_runner.py` - Main orchestrator
- `src/llm_perf_opt/profiling/vendor/nsys.py` - Nsys command builder
- `src/llm_perf_opt/profiling/nsys_stats.py` - Stats parser
- `src/llm_perf_opt/runners/dsocr_session.py` - NVTX emission
- `src/llm_perf_opt/profiling/nvtx_utils.py` - NVTX helpers

### Validation Evidence

- `nsys stats --help-reports` output (available report types)
- `nsys stats --report cuda_gpu_kern_sum` output (actual CSV format)
- `nsys stats --report nvtx_pushpop_sum` output (actual NVTX ranges)
- Actual profile data from `/data2/huangzhe/code/llm-perf-opt/tmp/stage2/20251030-060458/`

---

## Conclusion

Phase 3 NVTX gating implementation has **5 critical/high-priority bugs** that prevent the intended optimization features from working:

1. ✗ NVTX range name mismatch (`"range@LLM"` vs actual `"prefill"`, `"decode"`)
2. ✗ Invalid nsys report name (`"summary"` doesn't exist)
3. ✗ Missing NSYS_NVTX_PROFILER_REGISTER_ONLY environment variable
4. ✗ Silent subprocess failures hide errors
5. ✗ CSV parser expects wrong format

**Current Status**: Profiling runs but without NVTX gating benefits. Top-N kernel selection is broken.

**Recommended Action**:
1. Apply critical fixes #1-#3 immediately (< 30 minutes work)
2. Add error handling (#4) to prevent future issues
3. Update CSV parser (#5) to match new report format
4. Add integration tests to validate NVTX gating end-to-end

**Estimated Effort**: 2-4 hours to implement all fixes and validate.

**Risk**: MEDIUM - Fixes are localized to configuration and command builders; low risk of breaking existing functionality.

---

**Report Generated**: 2025-10-30 06:12:11
**Review Tool**: Claude Code (Droid) via Sonnet 4.5
**Guidelines**: `magic-context/instructions/review-code-by-search.md`
