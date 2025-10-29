Question:
given an LLM model in pytorch, we want to use profiling to find out its MFU in each of its stages, how to do this via Nvidia tools?

Answer:
Here’s a reliable way to measure **MFU (Model FLOPs Utilization)** for a PyTorch LLM, split by **prefill** and **decode**, using only NVIDIA tools.

> **MFU recap.** MFU is “the ratio of observed throughput to the theoretical maximum throughput if the hardware ran at peak FLOPs,” introduced in Google’s PaLM work. In practice you can compute it either from hardware counters (Nsight Compute) or from model FLOP accounting × tokens/s. We’ll use the NVIDIA-tools route below. ([arXiv][1])

---

# 1) Add stage markers (NVTX) in your PyTorch code

Mark the two stages so NVIDIA profilers can filter and aggregate exactly those regions.

```python
import torch
import nvtx  # pip install nvtx

def run_generation(model, tokenizer, prompt_ids, max_new_tokens=128):
    model.eval()
    with torch.inference_mode():
        # ----- Prefill -----
        with nvtx.annotate("LLM@prefill", domain="LLM"):
            out = model(input_ids=prompt_ids, use_cache=True)
            kv = out.past_key_values  # cache to reuse in decode

        # ----- Decode (all steps) -----
        with nvtx.annotate("LLM@decode_all", domain="LLM"):
            next_token = torch.argmax(out.logits[:, -1], dim=-1, keepdim=True)
            generated = [next_token]
            for _ in range(max_new_tokens - 1):
                # Optional per-step range if you want micro-level stats
                with nvtx.annotate("LLM@decode_step", domain="LLM"):
                    out = model(input_ids=next_token, past_key_values=kv, use_cache=True)
                    kv = out.past_key_values
                    next_token = torch.argmax(out.logits[:, -1], dim=-1, keepdim=True)
                    generated.append(next_token)
    return torch.cat(generated, dim=1)
```

NVTX ranges are picked up by Nsight Systems timelines and can be used by Nsight Compute to **profile only kernels inside a range** (we’ll use `--nvtx --nvtx-include`). You can also turn on PyTorch’s built-in autograd→NVTX bridge if desired (`with torch.autograd.profiler.emit_nvtx(): ...`). ([GitHub][2])

---

# 2) Sanity-check timing and stage boundaries with Nsight Systems

Grab a low-overhead end-to-end trace that shows CPU↔GPU overlap and your two NVTX ranges:

```bash
nsys profile -t cuda,nvtx,cublas,cudnn,osrt \
  --sample=none \
  --pytorch \
  -o nsys_llm_run \
  python run_llm.py
```

Open the `.nsys-rep` in the GUI to confirm stage boundaries (prefill vs decode). You can also restrict capture to just your NVTX range with `--capture-range=nvtx --nvtx-capture="LLM@decode_all"` (set `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` if you use non-registered strings). ([NVIDIA Docs][3])

> Tip: `nsys stats --export=csv -o out.csv nsys_llm_run.nsys-rep` gives CSV summaries if you want to script checks. ([RCAC][4])

---

# 3) Collect FLOP and “% of peak” counters per stage with Nsight Compute

Use NVTX filtering so **only kernels inside the stage** are profiled. Start with the Roofline/SOL (Speed-of-Light) sections, which give you FLOPs, time, bandwidth, and “x% of peak sustained” metrics.

**Prefill:**

```bash
ncu --target-processes all \
    --nvtx --nvtx-include "LLM@prefill/" \
    --set roofline --section ".*SpeedOfLight.*" \
    --metrics flop_count_hp,flop_count_sp,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    -o ncu_prefill \
    python run_llm.py
```

**Decode (aggregate all steps):**

```bash
ncu --target-processes all \
    --nvtx --nvtx-include "LLM@decode_all/" \
    --set roofline --section ".*SpeedOfLight.*" \
    --metrics flop_count_hp,flop_count_sp,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    -o ncu_decode \
    python run_llm.py
```

*Why these pieces:*

* `--nvtx --nvtx-include ...` filters collection to your range (regex is also supported). ([NVIDIA Docs][5])
* `roofline` / `SpeedOfLight` sections include duration, achieved FLOPs, and %-of-peak compute/memory indicators. ([Oak Ridge Leadership Computing Facility][6])
* `flop_count_*` give **actual operations executed** (separate FP16/FP32/FP64). On modern GPUs they’re appropriate inputs for Roofline/MFU calculations. ([Wiley Online Library][7])

Export to CSV for scripting:

```bash
ncu --import ncu_prefill.ncu-rep --page raw --csv --log-file prefill_raw.csv
ncu --import ncu_decode.ncu-rep --page raw --csv --log-file decode_raw.csv
```

The raw page includes the metrics you requested (one row per kernel). ([NVIDIA Developer Forums][8])

---

# 4) Compute MFU for each stage from Nsight Compute data

Let:

* `F_stage` = sum of FLOPs executed by kernels in that stage (choose the dtype column that matches your run, e.g., `flop_count_hp` for FP16/BF16 or `flop_count_sp` for TF32/FP32 paths).
* `T_stage` = total GPU time of those kernels (from `gpu__time_duration.sum`, same report).
* `Peak_stage` = **theoretical peak FLOPs/s** for that dtype on your GPU. Nsight Compute’s Speed-of-Light/Roofline sections compute these peaks using the **actual clock** during profiling; you can read the peak from the section or export the numbers. ([NVIDIA Developer Forums][9])

Then:

```
Achieved_GFLOP/s_stage = F_stage / T_stage
MFU_stage = Achieved_GFLOP/s_stage / Peak_stage
```

As sanity proxies, also look at:

* **SM % of peak**: `sm__throughput.avg.pct_of_peak_sustained_elapsed` (high ⇒ compute-bound).
* **DRAM % of peak**: `dram__throughput.avg.pct_of_peak_sustained_elapsed` (high ⇒ memory-bound).
  These are not MFU per se, but they help interpret prefill (often higher SM %) versus decode (often higher DRAM %). ([NVIDIA Docs][10])

> Practical note: lock clocks (and Tensor Core boost) for apples-to-apples runs. Nsight Compute exposes Tensor Core boost state control; set it to `stable` for reproducible results. ([NVIDIA Docs][11])

---

# 5) Optional: MFU via tokens/s (cross-check)

If you already know your model’s **FLOPs/token** (architecture math), you can compute:

```
Achieved_GFLOP/s_stage = (FLOPs_per_token_stage × tokens_per_second_stage)
MFU_stage = Achieved_GFLOP/s_stage / Peak_stage
```

This matches the PaLM definition directly and is handy when you don’t want to collect FLOP counters every run. Use Nsight Systems to measure the stage wall-time precisely and your code to count tokens. ([arXiv][1])

---

## What you should expect

* **Prefill**: larger matmuls, typically **higher SM % of peak** and **higher MFU** than decode (compute-heavier).
* **Decode**: small-batch, KV reads/writes dominate; expect **lower MFU**, **higher DRAM % of peak**, more kernel-launch overhead visible in Nsight Systems. (Use your actual profiles to verify; this is the common pattern.)

---

## Common gotchas & fixes

* **Too much overhead in `ncu`**: start with `--set roofline` and NVTX filtering; avoid `--set full` on full apps. Export CSV and aggregate offline. ([NVIDIA Docs][12])
* **No NVTX markers showing**: ensure you use the Python `nvtx` package (or `torch.cuda.nvtx`) and that Nsight is tracing `nvtx`. The `--pytorch` switch in Nsight Systems can also auto-annotate PyTorch ops. ([NVIDIA Docs][3])
* **Can’t isolate a stage**: use `--capture-range=nvtx` in Nsight Systems and `--nvtx-include` in Nsight Compute to bound exactly the region you want. ([NVIDIA Docs][3])

---

## Minimal checklist (copy/paste)

1. Add NVTX ranges (`LLM@prefill`, `LLM@decode_all`). ([NVIDIA GitHub][13])
2. `nsys profile -t cuda,nvtx,cublas,cudnn,osrt --pytorch -o nsys_llm_run ...` to verify boundaries/overlap. ([NVIDIA Docs][3])
3. `ncu --nvtx --nvtx-include "LLM@prefill/" --set roofline ... -o ncu_prefill ...` (repeat for decode). ([NVIDIA Docs][5])
4. Export CSV (`--import ... --page raw --csv`) and compute:

   * `F_stage` from `flop_count_*`
   * `T_stage` from `gpu__time_duration.sum`
   * `Peak_stage` from Speed-of-Light/Roofline peak numbers
   * `MFU_stage = (F_stage/T_stage) / Peak_stage` ([Oak Ridge Leadership Computing Facility][6])

If you want, I can drop in a tiny Python post-processor that reads the two CSVs and prints `MFU_prefill` and `MFU_decode`, plus SM%/DRAM% summaries.

[1]: https://arxiv.org/pdf/2204.02311?utm_source=chatgpt.com "PaLM: Scaling Language Modeling with Pathways"
[2]: https://github.com/NVIDIA/NVTX?utm_source=chatgpt.com "The NVIDIA® Tools Extension SDK (NVTX) is a C-based ..."
[3]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html?utm_source=chatgpt.com "User Guide — nsight-systems"
[4]: https://www.rcac.purdue.edu/knowledge/profilers?all=true&utm_source=chatgpt.com "Knowledge Base: Profilers - Purdue's RCAC"
[5]: https://docs.nvidia.com/nsight-compute/2023.1/pdf/NsightComputeCli.pdf?utm_source=chatgpt.com "Nsight Compute Command Line Interface"
[6]: https://www.olcf.ornl.gov/wp-content/uploads/2020/02/OLCF-Webinar-Nsight-Compute.pdf?utm_source=chatgpt.com "Nsight Compute"
[7]: https://onlinelibrary.wiley.com/doi/am-pdf/10.1002/cpe.5547?utm_source=chatgpt.com "Hierarchical Roofline Analysis for GPUs: Accelerating ..."
[8]: https://forums.developer.nvidia.com/t/converting-nsys-rep-file-into-a-csv-file-with-formatting-like-the-summary-page-in-ncu-gui/231717?utm_source=chatgpt.com "Converting nsys-rep file into a CSV file with formatting like ..."
[9]: https://forums.developer.nvidia.com/t/about-the-flops-in-ncu-report/215350?utm_source=chatgpt.com "About the flops in ncu report - Nsight Compute"
[10]: https://docs.nvidia.com/nsight-compute/2024.1/NsightComputeCli/index.html?utm_source=chatgpt.com "4. Nsight Compute CLI"
[11]: https://docs.nvidia.com/nsight-compute/2025.1/NsightComputeCli/index.html?utm_source=chatgpt.com "4. Nsight Compute CLI"
[12]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html?utm_source=chatgpt.com "2. Profiling Guide — NsightCompute 13.0 documentation"
[13]: https://nvidia.github.io/NVTX/?utm_source=chatgpt.com "NVTX - NVIDIA Tools Extension Library"
