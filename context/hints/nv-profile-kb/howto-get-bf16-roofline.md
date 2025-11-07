Here’s the fast way to get **BF16-specific** numbers out of Nsight Compute (ncu) so you can make an **absolute roofline** (AI = FLOPs / bytes; y-axis = FLOP/s):

---

### 1) Make sure perf counters are enabled (or ncu will return N/A)

On Linux/WSL/Docker you must allow access to GPU performance counters; otherwise the BF16/Tensor metrics won’t collect. NVIDIA documents the fix here (admin toggle / kernel module setting). ([NVIDIA Developer][1])

---

### 2) Collect BF16 “math ops” (Tensor Core) + memory bytes + duration

**Why these?**

* *Work* (numerator) = # of BF16 math ops actually executed.
* *Traffic* (denominator) = bytes moved at the memory level you care about (DRAM or L2).
* *Time* = kernel duration to convert ops → FLOP/s.

**BF16 work metrics (Tensor Core path)**
Use the `sm__ops_path_tensor_src_bf16_*` family. The baseline counter already aggregates sparse/non-sparse variants; if you want them split, use the `_sparsity_on/off` forms. Typical ones to pull:

* `sm__ops_path_tensor_src_bf16_dst_fp32.sum`
* (Optionally) `sm__ops_path_tensor_src_bf16_dst_bf16.sum` (if your kernel accumulates to BF16)
* If you need sparsity breakdowns, add:
  `sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_on.sum`,
  `sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum`
  These counters are the recommended way to count Tensor Core FLOPs by datatype. ([NVIDIA Developer Forums][2])

**Memory traffic**
Pick the level you want your roofline against:

* DRAM bytes: `dram__bytes_read.sum, dram__bytes_write.sum` (or `dram__bytes.sum` if available) ([NVIDIA Developer Forums][3])
* L2 bytes: `lts__t_bytes.sum` (for an L2 roofline). ([Stack Overflow][4])

**Duration**

* Kernel time: `gpu__time_duration.sum`. ([NVIDIA Developer Forums][5])

> Tip: to discover the exact suffixes your ncu build wants (`.sum`, `.avg`, etc.), query with suffix mode:
> `ncu --query-metrics-mode suffix --query-metrics sm__ops_path_tensor_src_bf16_dst_fp32` ([Indico][6])

**One-shot CLI example**

```bash
ncu -f --target-processes all \
  --metrics \
sm__ops_path_tensor_src_bf16_dst_fp32.sum,\
sm__ops_path_tensor_src_bf16_dst_bf16.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,\
gpu__time_duration.sum \
  -o bf16_roofline \
  ./your_app <args>
```

---

### 3) Compute the absolute roofline coordinates

Let

* `OPS_bf16 = sm__ops_path_tensor_src_bf16_dst_fp32.sum (+ dst_bf16.sum if present)`
  *(If you included `_sparsity_on/off`, sum those two instead of the baseline to avoid double-counting.)* ([NVIDIA Developer Forums][2])
* `BYTES_dram = dram__bytes_read.sum + dram__bytes_write.sum` (or use `lts__t_bytes.sum` for an L2-roofline). ([NVIDIA Developer Forums][3])
* `T = gpu__time_duration.sum` (seconds)

Then:

* **Arithmetic intensity (BF16, DRAM)**: `AI_bf16_dram = OPS_bf16 / BYTES_dram`
* **Achieved BF16 performance**: `P_bf16 = OPS_bf16 / T` (convert to TFLOP/s)

These are exactly the quantities the ncu roofline concept uses (work, bandwidth, and peaks). You can plot them yourself, or customize ncu’s Speed-of-Light roofline section if you want different peak lines/bandwidths. ([NVIDIA Docs][7])

> FYI: Nsight Compute’s built-in “Tensor Core Roofline” has had format limitations historically; using the `sm__ops_path_tensor_src_*` counters is the officially recommended path to get datatype-accurate FLOP counts (including BF16) today. ([NVIDIA Developer Forums][2])

---

### 4) (Optional) Automate from the `.ncu-rep`

If you export a report, the **Python Report Interface** lets you read metrics and compute AI / TFLOP/s programmatically for plotting. ([NVIDIA Developer Forums][8])

---

### 5) Sanity checks & gotchas

* If a BF16 kernel shows zeros in `smsp__sass_thread_inst_executed_*` FP counters, that’s expected—Tensor Core ops aren’t counted there. Use the `sm__ops_path_tensor_src_*` family. ([NVIDIA Developer Forums][9])
* DRAM byte counters include all DRAM clients; they’re still the right denominator for a DRAM roofline, but be aware of what’s included. ([NVIDIA Developer Forums][10])
* You can edit the **SpeedOfLight_RooflineChart.section** to change the memory slope (bandwidth) or compute peak lines if you want a custom “peak BF16” roof. ([NVIDIA Docs][7])

---

If you want, I can drop a tiny Python snippet that ingests the `.ncu-rep`, computes `AI_bf16` and `P_bf16`, and spits out a CSV ready for plotting.

[1]: https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute?utm_source=chatgpt.com "ERR_NVGPUCTRPERM: Nsight Compute Permission ..."
[2]: https://forums.developer.nvidia.com/t/counting-flops-using-ncu/298104 "Counting FLOPs using ncu - Nsight Compute - NVIDIA Developer Forums"
[3]: https://forums.developer.nvidia.com/t/how-to-compute-dram-bytes-read-sum-dram-bytes-read-sum/307220?utm_source=chatgpt.com "How to compute dram__bytes_read.sum & ..."
[4]: https://stackoverflow.com/questions/73679977/why-is-the-compute-throughput-s-value-different-from-the-actual-performance-pe?utm_source=chatgpt.com "Why is the Compute Throughput's value different from ..."
[5]: https://forums.developer.nvidia.com/t/profiling-overhead/201115?utm_source=chatgpt.com "Profiling overhead - Nsight Compute"
[6]: https://indico.cern.ch/event/962112/contributions/4110591/attachments/2159863/3643851/CERN_Nsight_Compute.pdf?utm_source=chatgpt.com "Nsight Compute"
[7]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html "2. Profiling Guide — NsightCompute 13.0 documentation"
[8]: https://forums.developer.nvidia.com/t/extract-data-from-roofline-plot/322219?utm_source=chatgpt.com "Extract data from roofline plot - Nsight Compute"
[9]: https://forums.developer.nvidia.com/t/confusion-about-the-d-f-h-mul-add-fma-count-in-the-nsight-compute/265702?utm_source=chatgpt.com "Confusion about the (d/f/h)(mul/add/fma) count in the nsight ..."
[10]: https://forums.developer.nvidia.com/t/dram-metrics-at-sm-or-device-level/168994?utm_source=chatgpt.com "DRAM metrics at SM or device level? - Nsight Compute"
