You’ve got three good paths depending on how “high-level” you want the view:

---

## 1) PyTorch (nn.Module-level) — hooks + profiler (works in eager mode)

* Use global **forward pre/post hooks** to bracket every module, and wrap the body in `torch.autograd.profiler.record_function("module_path")`. That gives you one event per `nn.Module`, which the PyTorch profiler can aggregate (and correlate to CUDA kernels on the timeline). Global hooks exist specifically for debugging/profiling. ([PyTorch Docs][1])
* In the table output, you can further **group by callsite** with `key_averages(group_by_stack_n=...)` to approximate per-module rollups if you prefer a code-path view. ([PyTorch Docs][2])
* If you like system timelines, add **NVTX ranges** (either manually via `torch.cuda.nvtx.range_push/pop`, or auto-insert with NVIDIA’s DLProf NVTX shim) so Nsight Systems shows clear module bands. ([PyTorch Docs][3])

> Heads-up: the old TensorBoard plugin works, but the PyTorch docs now recommend viewing the generated `trace.json` in **Perfetto**/Chrome tracing instead of TensorBoard. ([PyTorch Docs][4])

---

## 2) PyTorch (automatic “Module view”) — only if you can run TorchScript

* PyTorch’s profiler has `with_modules=True` to record **module hierarchy** and render a Module view (in TB/Perfetto). **Caveat:** today it’s supported for **TorchScript** models, not eager. If you can script/trace your model, this is the shortest path to a built-in module-level report. ([PyTorch Docs][2])
* Some folks note you must also set `with_stack=True` for the Module view to appear; device (CUDA) time may be limited in that view. (Community issue threads reflect this behavior.) ([GitHub][5])

---

## 3) ONNX-operator level — ONNX Runtime profiling (exact per-node timings)

* Export to ONNX and run with **ONNX Runtime** profiling enabled:

  ```python
  import onnxruntime as ort
  so = ort.SessionOptions(); so.enable_profiling = True
  sess = ort.InferenceSession("model.onnx", so, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
  sess.run(None, inputs)
  path = sess.end_profiling()  # JSON trace file
  ```

  You get a JSON trace with **each ONNX node’s latency** (and provider). View in Perfetto/Chrome tracing or parse to aggregate by `op_type`/node. ([ONNX Runtime][6])
* There’s a small overhead while profiling; don’t compare overall latency with profiling on vs off. (This is expected and discussed in ORT issues.) ([GitHub][7])
* Helpful extras:

  * `onnxruntime_perf_test -p profile.json` to profile from CLI. ([ONNX Runtime][6])
  * **onnxruntime-tools** includes a profiler script that summarizes costs by node/subgraph (nice starter if you don’t want to write your own parser). ([PyPI][8])

---

### Bonus (if you also target TensorRT)

If you deploy ONNX via TensorRT, `trtexec --dumpProfile --profilingVerbosity=detailed` prints a **layer-wise** timing table that’s roughly the same granularity as ONNX nodes post-fusion. ([NVIDIA Docs][9])

---

## What I’d pick for you

* **Stay in PyTorch (eager):** add global forward hooks + `record_function` (and optionally NVTX). Then use `torch.profiler` with CPU+CUDA activities and `key_averages()` (or view the trace in Perfetto) to get a clean **per-Module** breakdown. ([PyTorch Docs][1])
* **When you need ONNX-op granularity:** run the same model through **ONNX Runtime profiling** and aggregate the JSON by `op_type` / node name. ([ONNX Runtime][6])

These two together give you the high-level view you want, without drowning in `aten::` minutiae.

[1]: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_pre_hook.html?utm_source=chatgpt.com "torch.nn.modules.module."
[2]: https://docs.pytorch.org/docs/stable/profiler.html "torch.profiler — PyTorch 2.9 documentation"
[3]: https://docs.pytorch.org/docs/stable/generated/torch.cuda.nvtx.range_push.html?utm_source=chatgpt.com "torch.cuda.nvtx.range_push"
[4]: https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html?utm_source=chatgpt.com "PyTorch Profiler With TensorBoard"
[5]: https://github.com/pytorch/kineto/issues/733?utm_source=chatgpt.com "Module view dose not show device time · Issue #733"
[6]: https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html?utm_source=chatgpt.com "Profiling tools | onnxruntime"
[7]: https://github.com/microsoft/onnxruntime/issues/20238?utm_source=chatgpt.com "[Performance] InferenceSession with enable_profiling runs ..."
[8]: https://pypi.org/project/onnxruntime-tools/?utm_source=chatgpt.com "onnxruntime-tools"
[9]: https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html?utm_source=chatgpt.com "Command-Line Programs — NVIDIA TensorRT ..."

---


Here’s a single, end-to-end guide to get **module-level timings in PyTorch** and **line them up** with NVIDIA’s tools (**Nsight Systems / nsys** for the big picture, **Nsight Compute / ncu** for deep dives). It uses **NVTX ranges as the bridge** so the same labels appear in both PyTorch and NVIDIA tools.

---

# 0) What you’ll get

* A **per-`nn.Module` timeline** in PyTorch (and an exportable trace).
* The **same module names** shown as NVTX bands in **Nsight Systems**.
* Ability to **filter NCU** so it profiles **only kernels inside a chosen module**.

Refs: PyTorch `record_function`/profiler, NVTX APIs, Nsight Systems capture-range, Nsight Compute NVTX filtering. ([PyTorch Docs][1])

---

# 1) Instrument modules once (NVTX + PyTorch labels)

Drop this helper into your code and wrap your model for the step you want to profile:

```python
import torch
from contextlib import contextmanager

@contextmanager
def nvtxify_modules(model, *, min_depth=1, max_depth=2, prefix="mod", sync_inside=False):
    """
    Wrap each nn.Module.forward with:
      - torch.autograd.profiler.record_function(label)  (PyTorch profiler view)
      - torch.cuda.nvtx.range(label)                    (NVIDIA tools view)

    Depth filtering reduces noise (e.g., only block/layer modules).
    If sync_inside=True, we add a torch.cuda.synchronize() before closing the range
    to strictly bound async GPU work (higher overhead).
    """
    originals = []
    for name, m in model.named_modules():
        if not name:
            continue
        depth = name.count(".") + 1
        if depth < min_depth or depth > max_depth:
            continue

        label = f"{prefix}:{name}[{m.__class__.__name__}]"
        orig = m.forward

        def make_wrapped(orig_f, lab):
            def wrapped(*args, **kwargs):
                with torch.autograd.profiler.record_function(lab):  # shows up in PyTorch tables/traces
                    with torch.cuda.nvtx.range(lab):                # shows up in nsys/ncu
                        out = orig_f(*args, **kwargs)
                        if sync_inside and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        return out
            return wrapped

        m.forward = make_wrapped(orig, label)
        originals.append((m, orig))

    try:
        yield model
    finally:
        for m, orig in originals:
            m.forward = orig
```

* `record_function` creates named regions in PyTorch’s profiler table and trace. ([PyTorch Docs][1])
* `torch.cuda.nvtx.range` emits matching **NVTX ranges** (what Nsight tools understand). ([PyTorch Docs][2])
* NVTX is **CPU-side annotation**; Nsight projects those ranges onto GPU lanes by mapping the kernels launched inside them. (So host range boundaries may not perfectly align with device execution unless you synchronize.) ([NVIDIA Developer Forums][3])

> Alternative (no code changes): global forward hooks also exist for debugging/profiling, but the wrapper above is safer/easier to undo. ([PyTorch Docs][4])

---

# 2) Collect a PyTorch profiler trace (module-level)

Use the official profiler and export a Chrome/Perfetto trace:

```python
import torch
from torch.profiler import profile, ProfilerActivity

def run_one_step(model, batch):
    return model(batch)  # your step/inference

with nvtxify_modules(model, min_depth=2, max_depth=2, prefix="M"):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        _ = run_one_step(model, batch)

    # Console table (grouping is optional but handy)
    print(prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cuda_time_total", row_limit=200))

    # Export to trace viewer (Chrome/Perfetto)
    prof.export_chrome_trace("pt_trace.json")
```

* `torch.profiler` collects CPU + CUDA activity; you can group/aggregate with `key_averages` and stack grouping. ([PyTorch Docs][5])
* Exported traces are viewable in Chrome’s tracing or Perfetto UI. ([PyTorch Docs][6])

> If you want **every autograd op** to emit NVTX (very noisy, but sometimes useful), wrap your step in `with torch.autograd.profiler.emit_nvtx(): ...`. ([Caffe2][7])

---

# 3) Align with **Nsight Systems** (timeline & summaries)

### Capture

Either profile the full run:

```bash
nsys profile -o nsys_run \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=none \
  python your_script.py
```

…or **gate** collection to a small window using the NVTX range “CAPTURE”:

```python
with torch.cuda.nvtx.range("CAPTURE"):
    _ = run_one_step(model, batch)
```

```bash
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
nsys profile -o nsys_run \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --capture-range=nvtx --nvtx-capture "CAPTURE@*" \
  --sample=none \
  python your_script.py
```

* `--capture-range=nvtx` starts/stops profiling on your NVTX range. `--nvtx-capture` picks the message/domain. Env var shown makes Nsight consider **all** NVTX strings (not only pre-registered ones). ([NVIDIA Docs][8])

Open `nsys_run.nsys-rep` in the GUI: you’ll see **NVTX bands named like your modules** aligned above CUDA streams/kernels.

### Summaries (CLI)

Dump text/CSV summaries, including an **NVTX summary**:

```bash
nsys stats nsys_run.nsys-rep \
  --report nvtxsum,gpukernsum,cudaapisum \
  --format csv \
  --output nsys_stats
```

* `nvtxsum` is part of the default report set and prints time per NVTX range (i.e., your modules). ([NVIDIA Docs][9])

---

# 4) Deep-dive a hot module with **Nsight Compute**

From the nsys timeline (or your PT table), pick a hot module (e.g., `M:transformer.blocks.12[Block]`). Now profile **only its kernels**:

```bash
ncu --target-processes all \
    --set full \
    --nvtx \
    --nvtx-include "M:transformer.blocks.12[Block]/" \
    --launch-skip 5 --launch-count 1 \
    -o ncu_blocks12 \
    python your_script.py
```

* `--nvtx` enables NVTX-aware filtering; `--nvtx-include "NAME/"` matches **push/pop** ranges of that name (the trailing slash matters). Regex is supported, e.g. `--nvtx-include 'regex:M:transformer\\.blocks\\.[0-9]+\\[Block\\]/'`.
* `--launch-skip`/`--launch-count` limit overhead to a specific iteration.
* `--target-processes all` is handy if Python spawns CUDA work in subprocesses. ([NVIDIA Docs][10])

This gives you Tensor Core utilization, warp stalls, memory BW, etc., **scoped to that module’s kernels**.

---

# 5) Practical tips & pitfalls

* **Async reality:** NVTX is a **host** annotation. GPU work launched inside a range may finish after the range closes. If you need exact bounds for a measurement slice, set `sync_inside=True` in the wrapper (adds `torch.cuda.synchronize()`), but expect overhead/distortion—use only for spot checks. ([NVIDIA Developer Forums][3])
* **Keep noise down:** choose `min_depth/max_depth` for meaningful layers (e.g., Transformer block level).
* **Warmups:** do a few warmup steps before profiling to stabilize allocations/autotuning. (PyTorch docs/tuts recommend warmups; you can also use profiler schedules.) ([PyTorch Docs][5])
* **Export & view:** `prof.export_chrome_trace("pt_trace.json")` → open in Chrome/Perfetto; compare side-by-side with `nsys` timeline using your shared **module labels**. ([PyTorch Docs][6])
* **Alternative labeling:** instead of wrapping `forward`, you can register global forward pre/post hooks to push/pop NVTX, just remember to pop even on exceptions (context manager is simpler). ([PyTorch Docs][4])

---

## Minimal runnable sketch

```python
import torch, torch.nn as nn
from torch.profiler import profile, ProfilerActivity

model = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.GELU(),
    nn.Linear(4096, 4096),
).cuda().eval()

x = torch.randn(32, 4096, device="cuda")

with nvtxify_modules(model, min_depth=2, max_depth=2, prefix="M"):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True) as prof:
        with torch.no_grad():
            _ = model(x)

    print(prof.key_averages(group_by_stack_n=3).table(
        sort_by="self_cuda_time_total", row_limit=100))
    prof.export_chrome_trace("pt_trace.json")
```

Now run one of:

```bash
# Big-picture timeline with your module bands:
nsys profile -o nsys_run --trace=cuda,nvtx,osrt,cublas,cudnn --sample=none python script.py

# Or capture only the "CAPTURE" range:
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
nsys profile -o nsys_run --trace=cuda,nvtx,osrt,cublas,cudnn \
  --capture-range=nvtx --nvtx-capture "CAPTURE@*" --sample=none \
  python script.py

# Deep dive only the kernels inside a specific module range:
ncu --target-processes all --set full --nvtx \
    --nvtx-include "M:0[Sequential]/" \
    --launch-skip 5 --launch-count 1 \
    -o ncu_seq python script.py
```

Key options & behavior are covered in the official docs for **PyTorch profiler & record_function**, **NVTX**, **Nsight Systems capture-range/CLI**, **nsys stats** reports, and **Nsight Compute NVTX filtering**. ([PyTorch Docs][5])

---

If you want, I can turn this into a tiny utility package (`nvtxify`) with tests and a couple of presets (e.g., “transformer block level”), but the snippet above is enough to start correlating **module-level** timings with **GPU kernel behavior** quickly.

[1]: https://docs.pytorch.org/docs/stable/generated/torch.autograd.profiler.record_function.html?utm_source=chatgpt.com "record_function — PyTorch 2.9 documentation"
[2]: https://docs.pytorch.org/docs/stable/generated/torch.cuda.nvtx.range.html?utm_source=chatgpt.com "torch.cuda.nvtx.range"
[3]: https://forums.developer.nvidia.com/t/nvtx-with-gpu-timing/268356?utm_source=chatgpt.com "NVTX with GPU timing? - Profiling Linux Targets"
[4]: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html?utm_source=chatgpt.com "torch.nn.modules.module.register_module_forward_hook"
[5]: https://docs.pytorch.org/docs/stable/profiler.html?utm_source=chatgpt.com "torch.profiler — PyTorch 2.9 documentation"
[6]: https://docs.pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.export_chrome_trace.html?utm_source=chatgpt.com "torch.autograd.profiler.profile.export_chrome_trace"
[7]: https://caffe2.ai/doxygen-python/html/classtorch_1_1autograd_1_1profiler_1_1emit__nvtx.html?utm_source=chatgpt.com "torch.autograd.profiler.emit_nvtx Class Reference"
[8]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html "User Guide — nsight-systems"
[9]: https://docs.nvidia.com/nsight-systems/2022.4/UserGuide/index.html?utm_source=chatgpt.com "User Guide :: Nsight Systems Documentation"
[10]: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html "4. Nsight Compute CLI — NsightCompute 13.0 documentation"
