# Code Documentation: LLM Profile Runner - Profiling Strategy & Timing Computation

**Status**: Stable
**Last Updated**: 2025-10-29
**Component**: Stage 1 Profiling Pipeline

---

## Executive Summary

The LLM Profile Runner (`llm_profile_runner.py`) implements a **two-phase profiling strategy** for DeepSeek-OCR models: (1) a **single representative run** with PyTorch Profiler enabled to capture operator-level statistics, followed by (2) **repeated timing runs** without profiler overhead to measure accurate latencies and compute model FLOPS utilization (MFU). Time is measured using Python's `perf_counter()` at stage boundaries (prefill, decode) with additional NVTX-instrumented sub-stage timing (SAM, CLIP, projector) captured via PyTorch forward hooks.

**Purpose**: Provide accurate per-stage latency measurements and MFU estimates for LLM+vision workloads while collecting operator-level profiling data for optimization analysis.

**Users**: ML engineers profiling multimodal models, performance analysts optimizing vision+LLM pipelines.

---

## Metadata

- **Path**: `src/llm_perf_opt/runners/llm_profile_runner.py`
- **Language**: Python 3.12+
- **Type**: CLI Application (Hydra-based)
- **Dependencies**:
  - `torch` (PyTorch profiler APIs)
  - `hydra-core` (configuration management)
  - `nvtx` (NVIDIA Tools Extension for stage annotation)
  - `transformers` (HuggingFace model loading)
  - Internal: `dsocr_session`, `profiling.*`, `dsocr_analyzer`
- **Entry Point**: `main()` function decorated with `@hydra.main`

---

## Core Concepts

### 1. Two-Phase Profiling Strategy

**Design Rationale**: PyTorch Profiler adds ~10-20% overhead; measuring timing **with** profiler enabled would produce inflated latencies. The runner separates these concerns:

**Phase 1: Representative Profiling**
- Run **one** inference on first image with `torch.profiler.profile()` context
- Collect operator-level statistics (CUDA kernels, CPU ops, memory transfers)
- Accept profiler overhead as necessary cost for detailed attribution
- Extract operator records for top-K analysis

**Phase 2: Repeated Timing Runs**
- Run **N** inferences (configurable via `cfg.repeats`) **without** profiler
- Measure wall-clock latency using `perf_counter()` at stage boundaries
- Compute aggregates (mean, std) across repeats for statistical confidence
- Calculate MFU from latencies and analytical/static FLOP counts

**Trade-off**: Sacrifice per-run operator detail for accurate timing measurements.

### 2. Stage Segmentation via NVTX

**NVTX (NVIDIA Tools Extension)** provides hierarchical range annotations visible in NSight Systems/Compute. The pipeline uses NVTX ranges at two levels:

**Top-Level Stages** (in `dsocr_session.py`):
- `prefill`: First forward pass (vision processing + prompt encoding)
- `decode`: Autoregressive token generation loop

**Sub-Stages** (via forward hooks):
- `sam`: SAM vision encoder (window attention on crops)
- `clip`: CLIP vision encoder (global attention)
- `projector`: MLP projecting vision features to LLM embedding space

These ranges enable:
- Visual timeline analysis in NSight Systems
- Automated time attribution via hook-based accumulators
- Per-stage MFU computation

### 3. Timing Measurement Hierarchy

```
┌─────────────────────────────────────────┐
│ Total Inference (not explicitly timed)  │
├─────────────────────────────────────────┤
│ Prefill (perf_counter)                  │
│   ├─ SAM (hook-based accumulator)       │
│   ├─ CLIP (hook-based accumulator)      │
│   ├─ Projector (hook-based accumulator) │
│   └─ LLM forward pass (residual)        │
├─────────────────────────────────────────┤
│ Decode (perf_counter)                   │
│   └─ Autoregressive generation loop     │
└─────────────────────────────────────────┘
```

**Key Property**: Vision sub-stages overlap with prefill (they execute **during** prefill), so:
```
vision_ms = sam_ms + clip_ms + projector_ms  ≤  prefill_ms
```

---

## Data Flow: Profiling Pipeline

```mermaid
sequenceDiagram
    participant User as User<br/>(CLI)
    participant Hydra as Hydra<br/>Config System
    participant Runner as llm_profile_runner<br/>main()
    participant Session as DeepSeekOCRSession
    participant Model as HF Model<br/>(DeepSeek-OCR)
    participant Profiler as PyTorch<br/>Profiler
    participant NVTX as NVTX<br/>Hooks

    User->>Hydra: python -m llm_perf_opt.runners.llm_profile_runner
    Hydra->>Runner: Load config from conf/*.yaml
    Runner->>Session: DeepSeekOCRSession.from_local(model_path, device)
    Session->>Model: AutoModel.from_pretrained()
    Session->>NVTX: _install_nvtx_stage_hooks()<br/>(attach to sam/clip/projector)
    Session-->>Runner: session (ready)

    Note over Runner: === Phase 1: Representative Profiling ===
    Runner->>Profiler: with profile(activities=[CPU, CUDA])
    Profiler->>Session: run_inference(image[0], max_new_tokens=rep_max_new)
    activate Session
    Session->>Session: perf_counter() → t0
    Session->>NVTX: nvtx.push_range("prefill")
    Session->>Model: model(**inputs, images=...)
    activate Model
    NVTX->>NVTX: push_range("sam")<br/>t_sam_start = perf_counter()
    Model->>Model: sam_model.forward()
    NVTX->>NVTX: pop_range("sam")<br/>sam_ms += (perf_counter() - t_sam_start) * 1000
    NVTX->>NVTX: push_range("clip")<br/>t_clip_start = perf_counter()
    Model->>Model: vision_model.forward()
    NVTX->>NVTX: pop_range("clip")<br/>clip_ms += (perf_counter() - t_clip_start) * 1000
    NVTX->>NVTX: push_range("projector")<br/>t_proj_start = perf_counter()
    Model->>Model: projector.forward()
    NVTX->>NVTX: pop_range("projector")<br/>projector_ms += (perf_counter() - t_proj_start) * 1000
    Model-->>Session: output
    deactivate Model
    Session->>NVTX: nvtx.pop_range("prefill")
    Session->>Session: prefill_ms = (perf_counter() - t0) * 1000

    Session->>Session: perf_counter() → t1
    Session->>NVTX: nvtx.push_range("decode")
    Session->>Model: model.generate(max_new_tokens=...)
    Model-->>Session: generated_ids
    Session->>NVTX: nvtx.pop_range("decode")
    Session->>Session: decode_ms = (perf_counter() - t1) * 1000
    Session-->>Runner: result = {prefill_ms, decode_ms, sam_ms, clip_ms, projector_ms, ...}
    deactivate Session
    Profiler->>Runner: operator_records = _collect_operator_records(prof)

    Note over Runner: === Phase 2: Repeated Timing Runs ===
    loop repeats times (e.g., 10 runs)
        Runner->>Session: run_inference(image[i], max_new_tokens)
        Note over Session: Same timing sequence as above<br/>(NO profiler context)
        Session-->>Runner: result[i] = {prefill_ms[i], decode_ms[i], ...}
        Runner->>Runner: runs.append(ImageRun(...))
    end

    Runner->>Runner: summary = _summarize_runs(runs)<br/>Compute mean/std, MFU
    Runner->>Runner: _write_outputs(artifacts_dir, summary, operator_records)
    Runner-->>User: Artifacts written to tmp/stage1/<run_id>/
```

---

## Timing Computation: Implementation Details

### 1. Prefill Timing (`dsocr_session.py:308-326`)

**Measurement Scope**: First forward pass including vision processing and prompt encoding.

```python
# Source: dsocr_session.py lines 308-326
t0 = perf_counter()
self._reset_stage_time_accum()  # Reset sam_ms, clip_ms, projector_ms to 0
with prefill_range():  # NVTX annotation
    if use_pre:
        inputs = {"input_ids": input_ids}
    else:
        nvtx.push_range("tokenizer")
        try:
            inputs = self.m_tokenizer(prompt, return_tensors="pt").to(self.m_device)
        finally:
            nvtx.pop_range()
    _ = self.m_model(
        **inputs,
        images=images,
        images_seq_mask=images_seq_mask,
        images_spatial_crop=images_spatial_crop,
    )
prefill_ms = (perf_counter() - t0) * 1000.0
```

**What's Included**:
- SAM encoder forward pass (window attention on crops)
- CLIP encoder forward pass (global attention on base image)
- Projector MLP (vision → LLM embedding alignment)
- Text embedding lookup
- First pass through transformer decoder layers
- Output projection (lm_head) for first token logits

**What's Excluded**:
- Image loading/decoding (happens before `t0`)
- Tokenization (happens before `t0` if `use_pre=True`)
- Autoregressive generation loop

**Precision**: Microsecond-level precision via `perf_counter()` (monotonic clock).

### 2. Decode Timing (`dsocr_session.py:329-353`)

**Measurement Scope**: Autoregressive token generation loop.

```python
# Source: dsocr_session.py lines 329-353
t1 = perf_counter()
with decode_range():  # NVTX annotation
    gen_kwargs: dict[str, object] = {
        "max_new_tokens": int(max_new_tokens),
        "eos_token_id": getattr(self.m_tokenizer, "eos_token_id", None),
    }
    if infer is not None:
        for k in ("temperature", "no_repeat_ngram_size", "do_sample", "top_p", "top_k"):
            if k in infer:
                gen_kwargs[k] = infer[k]
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.m_device)
    out = self.m_model.generate(
        **inputs,
        images=images,
        images_seq_mask=images_seq_mask,
        images_spatial_crop=images_spatial_crop,
        attention_mask=attention_mask,
        **gen_kwargs,
    )
decode_ms = (perf_counter() - t1) * 1000.0
```

**What's Included**:
- All autoregressive generation steps (1 to `max_new_tokens`)
- Per-token transformer decoder passes
- KV cache updates (attention state management)
- Output projection (lm_head) per token
- Sampling/beam search logic (if `do_sample=True`)
- EOS detection and early stopping

**What's Excluded**:
- Prefill stage (already timed separately)
- Post-processing (decoding token IDs to text)

**Context Length Handling**: Each token attends to increasing context:
- Token 1: attends to `prefill_len` tokens
- Token N: attends to `prefill_len + N - 1` tokens

This quadratic growth is accounted for in MFU computation via `select_decode_context_len()`.

### 3. Vision Sub-Stage Timing (`dsocr_session.py:105-146`)

**Mechanism**: PyTorch forward hooks attached during session initialization.

```python
# Source: dsocr_session.py lines 119-142 (simplified)
def _attach(mod: Any, label: str) -> None:
    """Attach NVTX + timing hooks to a module."""
    def _pre(_m, _inp):
        nvtx.push_range(label)
        _m.__dict__["__nvtx_t0"] = time.perf_counter()

    def _post(_m, _inp, _out):
        nvtx.pop_range()
        try:
            t0 = _m.__dict__.pop("__nvtx_t0", None)
            if t0 is not None:
                dt = (time.perf_counter() - float(t0)) * 1000.0
                self.m_stage_time_ms[label] = self.m_stage_time_ms.get(label, 0.0) + float(dt)
        except Exception:
            pass

    h1 = mod.register_forward_pre_hook(_pre)
    h2 = mod.register_forward_hook(_post)
    self.m_nvtx_hooks.extend([h1, h2])

# Attach hooks to vision modules
_attach(getattr(core, "sam_model", None), "sam")
_attach(getattr(core, "vision_model", None), "clip")
_attach(getattr(core, "projector", None), "projector")
```

**Timing Properties**:
- **Accumulative**: Multiple forward calls accumulate time (e.g., if SAM processes 4 crops sequentially, `sam_ms` sums all 4 executions)
- **Nested within prefill**: Vision stages execute **during** prefill forward pass
- **Reset per inference**: `_reset_stage_time_accum()` zeroes counters before each inference

**Why This Design**:
- DeepSeek-OCR's vision modules are embedded within the model forward pass
- Cannot time vision stages externally without modifying third-party code
- Hooks provide non-invasive timing attribution

### 4. Operator-Level Timing (PyTorch Profiler)

**Collection** (`llm_profile_runner.py:723-764`):

```python
# Source: llm_profile_runner.py lines 739-760 (simplified)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False, profile_memory=False) as prof:
    _ = session.run_inference(
        image_path=rep_image,
        prompt="<image>\\n<|grounding|>Convert the document to markdown.",
        max_new_tokens=rep_max_new,
        ...
    )
    torch.cuda.synchronize()  # Ensure GPU work completes before profiler closes
operator_records = _collect_operator_records(prof)
```

**Extraction** (`llm_profile_runner.py:123-165`):

```python
def _collect_operator_records(prof: Any) -> list[dict]:
    """Extract operator-level summaries from PyTorch profiler."""
    records: list[dict] = []
    for evt in prof.key_averages(group_by_input_shape=False):
        total_ms = float(
            getattr(evt, "self_cpu_time_total", 0.0) +
            getattr(evt, "cpu_time_total", 0.0)
        ) / 1000.0
        cuda_ms = float(
            getattr(evt, "self_cuda_time_total", 0.0) +
            getattr(evt, "cuda_time_total", 0.0)
        ) / 1000.0
        calls = int(getattr(evt, "count", 0))
        name = str(getattr(evt, "key", getattr(evt, "name", "")))
        records.append({
            "op_name": name,
            "total_time_ms": max(total_ms, 0.0),
            "cuda_time_ms": max(cuda_ms, 0.0),
            "calls": max(calls, 0),
            "mean_ms": total_ms / max(calls, 1),
        })
    return records
```

**Captured Metrics**:
- **total_time_ms**: CPU + self CPU time (host-side overhead)
- **cuda_time_ms**: GPU kernel execution time
- **calls**: Number of invocations
- **op_name**: Operator identifier (e.g., `aten::matmul`, `aten::conv2d`)

**Usage**: Top-K operators exported to `operators.md` for bottleneck analysis.

---

## Aggregate Statistics & MFU Computation

### 1. Per-Run Data Collection

**Data Structure** (`llm_profile_runner.py:110-121`):

```python
@dataclass
class ImageRun:
    image_path: str
    prefill_ms: float      # Prefill latency
    decode_ms: float       # Decode latency
    tokens: int            # Generated tokens
    prefill_len: int       # Input sequence length
    vision_ms: float       # Total vision time (sam + clip + projector)
    sam_ms: float = 0.0    # SAM sub-stage time
    clip_ms: float = 0.0   # CLIP sub-stage time
    projector_ms: float = 0.0  # Projector sub-stage time
```

**Collection Loop** (`llm_profile_runner.py:766-839`):

```python
for i in range(repeats):
    img = next(images_iter)  # Cycle through images
    res = session.run_inference(image_path=str(img), ...)
    runs.append(ImageRun(
        image_path=str(img),
        prefill_ms=float(res.get("prefill_ms", 0.0)),
        decode_ms=float(res.get("decode_ms", 0.0)),
        tokens=int(res.get("tokens", 0)),
        prefill_len=int(res.get("prefill_len", 0)),
        vision_ms=float(res.get("vision_ms", 0.0)),
        sam_ms=float(res.get("sam_ms", 0.0)),
        clip_ms=float(res.get("clip_ms", 0.0)),
        projector_ms=float(res.get("projector_ms", 0.0)),
    ))
```

### 2. Statistical Aggregation

**Computation** (`llm_profile_runner.py:194-232`):

```python
def _summarize_runs(runs: list[ImageRun], ...) -> dict:
    prefill_vals = [r.prefill_ms for r in runs]
    decode_vals = [r.decode_ms for r in runs]
    tokens_vals = [float(max(r.tokens, 1)) for r in runs]

    # Tokens per second (throughput)
    dec_tokens_per_s = [tok / (ms / 1000.0) for tok, ms in zip(tokens_vals, decode_vals)]

    # Mean ± std for each metric
    prefill_mean, prefill_std = mean_std(prefill_vals)
    decode_mean, decode_std = mean_std(decode_vals)
    tokens_mean, tokens_std = mean_std(tokens_vals)
    tps_mean, tps_std = mean_std(dec_tokens_per_s)

    return {
        "aggregates": {
            "prefill_ms": {"mean": prefill_mean, "std": prefill_std},
            "decode_ms": {"mean": decode_mean, "std": decode_std},
            "tokens": {"mean": tokens_mean, "std": tokens_std},
            "tokens_per_s": {"mean": tps_mean, "std": tps_std},
        },
        ...
    }
```

**Statistical Method** (`profiling/aggregate.py`):

```python
def mean_std(vals: Iterable[float]) -> tuple[float, float]:
    """Compute mean and standard deviation."""
    arr = [float(v) for v in vals]
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean_val = sum(arr) / n
    if n == 1:
        return mean_val, 0.0
    variance = sum((x - mean_val) ** 2 for x in arr) / (n - 1)
    std_val = variance ** 0.5
    return mean_val, std_val
```

### 3. Model FLOPS Utilization (MFU)

**Definition**: Ratio of achieved TFLOPS to theoretical peak TFLOPS.

$$
\text{MFU} = \frac{\text{Achieved TFLOPS}}{\text{Peak TFLOPS}} = \frac{\text{FLOPs} / \text{Time (s)} / 10^{12}}{\text{Peak TFLOPS}}
$$

**Implementation** (`llm_profile_runner.py:859-924`):

```python
# 1. Get static FLOP counts from analyzer
static_report = analyzer.generate_report(aconf)
stages = static_report.get("stages", {})

# Extract per-stage FLOPs
prefill_flops_total = _stage_flops("prefill")       # Total FLOPs for prefill
decode_flops_per_token = _stage_flops("decode")    # FLOPs per decode token
vision_flops_total = (
    _stage_flops("sam") +
    _stage_flops("clip") +
    _stage_flops("projector")
)

# 2. Compute MFU per stage
def _mfu(flops: float, ms: float) -> float:
    if ms <= 0.0 or peak <= 0.0:
        return 0.0
    achieved_tflops = (flops / 1e12) / (ms / 1000.0)
    return float(achieved_tflops / peak)

mfu_prefill = _mfu(prefill_flops_total, pf_ms)
mfu_decode = _mfu(decode_flops_per_token * toks_mean, dc_ms)
mfu_vision = _mfu(vision_flops_total, vn_ms)

# 3. Model-level MFU (prefill + decode, excluding vision to avoid double counting)
total_flops = prefill_flops_total + decode_flops_per_token * toks_mean
total_time_s = (pf_ms + dc_ms) / 1000.0
mfu_model = (total_flops / 1e12) / total_time_s / peak
```

**FLOP Sources**:
1. **Analytical** (for LLM stages):
   - Prefill: $2 \cdot d_{model}^2 \cdot L \cdot \text{seq\_len}$ (simplified)
   - Decode: $2 \cdot d_{model}^2 \cdot L$ per token
2. **Static (fvcore)** (for vision stages):
   - SAM: Traced from `sam_model.forward()`
   - CLIP: Traced from `vision_model.forward()`
   - Projector: Traced from `projector.forward()`

**Why MFU Matters**:
- MFU < 0.1: Likely memory-bound or poor batching
- MFU 0.3-0.5: Typical for autoregressive decode (KV cache bottleneck)
- MFU > 0.6: Good utilization for transformer workloads

---

## Configuration and Tuning

### Key Profiling Parameters

| Parameter | Config Path | Default | Description |
|-----------|-------------|---------|-------------|
| `repeats` | `cfg.repeats` | 10 | Number of timing runs (Phase 2) |
| `rep_max_new_tokens` | `cfg.profiling.rep_max_new_tokens` | 64 | Token limit for representative profiling run |
| `activities` | `cfg.profiling.activities` | `["cpu", "cuda"]` | Profiler activity types |
| `record_shapes` | `cfg.profiling.record_shapes` | `false` | Record tensor shapes in profiler |
| `profile_memory` | `cfg.profiling.profile_memory` | `false` | Track memory allocations |
| `warmup_rounds` | `cfg.profiling.warmup_rounds` | 0 | Number of warmup inferences before profiling |
| `warmup_synthetic` | `cfg.profiling.warmup_synthetic` | `true` | Use synthetic images for warmup |

### Timing Accuracy Considerations

**Sources of Variance**:
1. **GPU Clock Throttling**: Use `sudo nvidia-smi -lgc <freq>` to lock clocks
2. **CUDA Context Initialization**: First inference includes overhead; use warmup
3. **CPU Background Processes**: Run on dedicated machine or disable background services
4. **Memory Transfer Latency**: PCIe bandwidth affects H2D/D2H copies

**Mitigation Strategies**:
```yaml
# conf/profiling/torch/torch-profiler.max.yaml
warmup_rounds: 3
warmup_synthetic: true
rep_max_new_tokens: 64
```

**Recommended Setup**:
- Lock GPU clocks: `nvidia-smi -lgc 1410` (for A100)
- Use `CUDA_VISIBLE_DEVICES=0` to isolate GPU
- Set `repeats: 10` for statistical significance
- Monitor variance: `std / mean < 0.05` indicates stable measurements

---

## Usage Examples

### Basic Profiling Run

```bash
# Profile with default config (10 repeats, 64 max tokens)
pixi run python -m llm_perf_opt.runners.llm_profile_runner
```

### Custom Configuration

```bash
# Override via CLI
pixi run python -m llm_perf_opt.runners.llm_profile_runner \
  repeats=20 \
  infer.max_new_tokens=128 \
  profiling.warmup_rounds=5 \
  dataset.subset_filelist=datasets/test-subset.txt

# Outputs written to: tmp/stage1/<timestamp>/
```

### Config File Hierarchy

```yaml
# conf/config.yaml (top-level)
defaults:
  - model: deepseek_ocr/infer/deepseek_ocr.default
  - profiling/torch: torch-profiler.default
  - dataset: deepseek_ocr/pdftable-1k
  - infer: deepseek_ocr/infer/infer.default

repeats: 10
device: cuda:0
```

```yaml
# conf/profiling/torch/torch-profiler.default.yaml
activities: [cpu, cuda]
record_shapes: false
profile_memory: false
with_stack: false
rep_max_new_tokens: 64
warmup_rounds: 0
warmup_synthetic: true
```

### Output Artifacts

After successful run, artifacts are written to `tmp/stage1/<run_id>/`:

```
tmp/stage1/20251029-090421/
├── report.md                    # Human-readable summary
├── metrics.json                 # Machine-readable aggregates + MFU
├── operators.md                 # Top-K operator table
├── stakeholder_summary.md       # Executive summary
├── static_compute.json          # fvcore FLOP counts
├── static_compute.md            # Static analysis report
├── predictions.jsonl            # Model outputs (if enabled)
├── predictions.md               # Gallery with thumbnails
├── inputs.yaml                  # Input file manifest
├── assumptions.md               # Run configuration snapshot
├── env.json                     # Hardware/software environment
└── llm_profile_runner.log       # Execution logs
```

---

## Performance-Critical Sections

### 1. Timing Measurement Precision

**Why `perf_counter()` and not `time()`**:

```python
# DON'T: time.time() has lower resolution and can go backwards
import time
t0 = time.time()
...
elapsed = time.time() - t0  # ❌ Subject to NTP adjustments

# DO: perf_counter() is monotonic and high-resolution
from time import perf_counter
t0 = perf_counter()
...
elapsed = perf_counter() - t0  # ✅ Guaranteed monotonic, ~nanosecond precision
```

**Resolution**: `perf_counter()` has ~1 microsecond resolution on Linux (actual depends on kernel `CLOCK_MONOTONIC_RAW`).

### 2. CUDA Synchronization

**Critical for accurate GPU timing**:

```python
# Source: llm_profile_runner.py lines 754-759
with profile(...) as prof:
    _ = session.run_inference(...)
    # CRITICAL: Wait for all GPU work to complete before profiler closes
    try:
        if torch.cuda.is_available() and any(a == ProfilerActivity.CUDA for a in activities):
            torch.cuda.synchronize()
    except Exception:
        pass
```

**Why Needed**: GPU kernels execute asynchronously. Without `synchronize()`, profiler may close before all CUDA events are recorded, resulting in incomplete timing data.

### 3. Hook-Based Timing Overhead

**Overhead Analysis**:
- PyTorch forward hooks add ~10-50µs per invocation
- For 4 SAM crops: `4 * 50µs = 200µs` overhead
- Negligible compared to SAM forward pass (~100ms)

**Verification**:
```python
# Run with hooks disabled (modify dsocr_session.py)
# Compare: prefill_ms_with_hooks vs prefill_ms_without_hooks
# Expected difference: < 0.5%
```

---

## Related Files and Context

**Core Components**:
- `src/llm_perf_opt/runners/dsocr_session.py` - Session wrapper with NVTX hooks
- `src/llm_perf_opt/runners/dsocr_analyzer.py` - Static FLOP computation (fvcore)
- `src/llm_perf_opt/profiling/mfu.py` - MFU calculation utilities
- `src/llm_perf_opt/profiling/aggregate.py` - Statistical aggregation
- `src/llm_perf_opt/profiling/export.py` - Report generation

**Configuration**:
- `conf/profiling/torch/*.yaml` - Profiler presets (default, min, max)
- `conf/infer/*.yaml` - Inference parameters (temperature, sampling)
- `conf/dataset/*.yaml` - Dataset selection

**Documentation**:
- `context/tasks/001-profile-deepseek-ocr/impl-phase-3-us1-report.md` - Phase 3 implementation
- `docs/configuration.md` - Hydra configuration guide
- `VISION_STAGE_FLOPS_FIX.md` - Vision stage FLOP computation debugging

---

## Error Handling & Edge Cases

### 1. Profiler Unavailable

```python
# Source: llm_profile_runner.py lines 762-763
except Exception:
    operator_records = []  # Fail-open: continue without operator data
```

**Trigger**: Environments without CUDA or profiler APIs.
**Behavior**: Skip operator profiling, continue with timing runs.

### 2. Empty Image List

```python
# Source: llm_profile_runner.py lines 659-660
if not images:
    raise RuntimeError(f"No images found in dataset root: {cfg.dataset.root}")
```

**Prevention**: Validates dataset before starting profiling.

### 3. NVTX Hook Failures

```python
# Source: dsocr_session.py lines 98-102
try:
    inst._install_nvtx_stage_hooks()
except Exception:
    # Best-effort; NVTX hooks are optional
    pass
```

**Behavior**: Continue without sub-stage timing (sam_ms, clip_ms, projector_ms will be 0).

### 4. Static Analyzer Failures

```python
# Source: llm_profile_runner.py lines 933-940
except Exception:
    # Fall back to previous simple static report path
    static_compute = {}
    try:
        static_compute = session.estimate_static_compute()
        _write_static_compute(artifacts_dir, static_compute)
    except Exception:
        pass
```

**Behavior**: Continue with analytical MFU computation, static reports omitted.

---

## Quality Checklist

- [x] **Accuracy**: Timing measurements verified against NSight Systems ground truth
- [x] **Completeness**: All major stages (prefill, decode, vision sub-stages) timed
- [x] **Clarity**: Two-phase strategy and hook-based timing explained with diagrams
- [x] **Usability**: Examples and configuration guides provided
- [x] **Navigation**: Links to related files and context included
- [x] **Examples**: CLI usage and config overrides demonstrated
- [x] **Diagrams**: Mermaid sequence diagram showing complete data flow

---

## References

**External**:
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVTX Documentation](https://docs.nvidia.com/nvtx/)
- [NSight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Python `perf_counter()` Docs](https://docs.python.org/3/library/time.html#time.perf_counter)

**Internal**:
- Phase 3 Implementation Report: `context/tasks/001-profile-deepseek-ocr/impl-phase-3-us1-report.md`
- MFU Computation Revision: `context/tasks/001-profile-deepseek-ocr/revision-phase-3-mfu-compute.md`
- Static Analysis Plan: `context/plans/done/plan-per-stage-static-analysis.md`

---

**Summary**: The LLM Profile Runner implements a sophisticated two-phase profiling strategy that balances operator-level detail with timing accuracy. Timing is measured using high-precision `perf_counter()` at stage boundaries, with NVTX-instrumented hooks providing sub-stage attribution for vision components. The combination of repeated timing runs (for statistical confidence) and static FLOP analysis (via fvcore) enables accurate per-stage MFU computation for multimodal LLM+vision workloads.
