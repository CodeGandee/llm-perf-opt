# Issue Summary: PyTorch Profiler Shows 0 CUDA Time for Many Operators

Context: Stage 1 profiling for DeepSeek‑OCR (PyTorch 2.6.0+cu124, CUDA 12.4, FlashAttention 2.7.4). The operator summary (key_averages) frequently reports `cuda_time_ms = 0.000` for high‑level ops (e.g., `aten::mm`, `aten::linear`) even though GPU kernels ran.

## Symptoms
- In operator tables (key_averages), the `cuda_time_ms` column is 0 for many `aten::*` operators.
- Most device activity aggregates under entries like `cudaLaunchKernel`, kernel‑specific symbols (e.g., `flash_attn::_flash_attn_forward`), or PythonOp wrappers (e.g., `prim::PythonOp.FlashAttnFunc`).
- Adding a global `torch.cuda.synchronize()` improves completeness, but `cuda_time_ms` for many high‑level ops remains 0.

## Why This Happens
- Operator → kernel attribution gaps
  - PyTorch Profiler attributes GPU time to GPU kernels. High‑level operators may dispatch work that is recorded at kernel symbols, not credited back to the high‑level op row in `key_averages`.
  - Fused/custom ops (FlashAttention, vendor extensions) often execute via PythonOp or C++/CUDA extensions; kernel time ends up under those entries, not under `aten::mm/linear`.
- Asynchrony and flushing
  - Without an explicit device sync before ending the profiler region, some CUDA work can remain in flight. We added an explicit `torch.cuda.synchronize()` at the end of the profiled block to reduce this risk, but attribution behavior still applies.
- Grouping/aggregation semantics
  - `key_averages(group_by_input_shape=False)` aggregates by operator key; kernel‑level device time isn’t guaranteed to be redistributed to those aggregated rows.

Net effect: GPU time is present in the trace, but not credited to many high‑level operator rows; it appears under kernel entries and `cudaLaunchKernel`.

## Practical Implications
- Sorting by `total_time_ms` (CPU self+total) remains useful to find hotspots of orchestration/dispatch overhead.
- `cuda_time_ms` on high‑level rows under‑represents GPU cost; rely on kernel rows or stage‑level timing for device attribution.

## Mitigations and Options
- Immediate (already applied)
  - Ensure CUDA activity is enabled in profiler and call `torch.cuda.synchronize()` before exiting the profiling context to flush timings.
  - Add `mean_ms = total_time_ms / calls` to operator tables to provide per‑call guidance while still sorting by total time.
- Kernel‑level aggregation (optional enhancement)
  - Aggregate `evt.device_duration` by kernel symbol (e.g., via `prof.events()`) and render a separate “Top GPU Kernels” table. This gives an honest GPU time ranking independent of high‑level ops.
- Improve attribution visibility
  - Add NVTX ranges around logical blocks (already used for stages) to get clear per‑stage device timing in both PyTorch Profiler and Nsight.
  - Enable `with_stack=True` for deeper attribution (more overhead); export traces for analysis in TensorBoard Profiler.
- Tooling for accuracy
  - Nsight Systems/Compute: use for authoritative kernel‑level timing and occupancy, with NVTX to correlate back to model phases.
  - Keep PyTorch Profiler for lightweight, scriptable summaries and operator counts.
- Vendor/custom ops
  - Where feasible, instrument custom CUDA kernels and PythonOps with NVTX ranges and/or register op schemas to improve attribution.

## Recommended Approach for This Project
- Retain the current operator table (sorted by total time) with `mean_ms` per call for quick signal.
- Optionally add a “Top GPU Kernels” table using kernel device durations to complement the operator view.
- Continue using NVTX stage ranges + analyzer‑based FLOPs for MFU; use Nsight when kernel‑level accuracy is required.

## Notes
- This behavior is a known characteristic of PyTorch’s kineto‑based profiler: high‑level operator rows do not always receive kernel device time, especially for fused/custom kernels.
- Our change to call `torch.cuda.synchronize()` ensures trace completeness at region end but does not force redistribution of device time to `aten::*` rows.
