Great list. Here’s what each **Nsight Compute (ncu) section** gives you, plus a pragmatic shortlist of what to turn on when you’re chasing **compute** vs **memory** bottlenecks in LLM kernels.

---

### What each section tells you (and why it matters)

| Section                                                      | What it tells you / Why it’s useful                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SpeedOfLight**                                             | High-level “% of peak” throughput for SM, Tensor pipe, DRAM (“frame buffer”), etc. A quick litmus test for **compute vs memory**: low SM% with high DRAM% ⇒ memory-bound; high SM% (or Tensor pipe %) ⇒ compute-bound. SOL is defined as theoretical max vs achieved. ([NVIDIA][1])                                       |
| **ComputeWorkloadAnalysis**                                  | Breaks down which compute pipelines did work (FP32/64, INT, Tensor, etc.). If Tensor utilization is near zero in GEMM/FMHA kernels, you’re leaving Tensor Cores idle. Pairs well with SOL to confirm compute-boundedness. ([Indico][2])                                                                                   |
| **MemoryWorkloadAnalysis** / **…_Tables** / **…_Chart**      | Detailed memory subsystem analysis (L1/L2/DRAM, texture/surface, request/throughput patterns). This is your go-to for **memory bandwidth and cache behavior**; the tables give precise numbers, the chart visualizes flow. NVIDIA’s memory analysis training recommends starting here for memory work. ([NVIDIA Docs][3]) |
| **WorkloadDistribution**                                     | Where cycles were spent across SM/SMSP, **L1/L2**, **DRAM** (active vs elapsed). Fast way to see whether your time is predominantly compute pipelines or memory hierarchy activity. (Section file defines L1/L2/DRAM active/elapsed cycle metrics.) ([Hugging Face][4])                                                   |
| **Occupancy**                                                | Theoretical vs **Achieved Occupancy** (ratio of active warps to max). Low achieved vs theoretical points to launch config/resource limits or load imbalance. Achieved occupancy definition and causes are documented. ([NVIDIA Docs][5])                                                                                  |
| **SchedulerStats**                                           | How many warps were **eligible/issued** per scheduler; low issue rate hints at latency (e.g., waiting on memory). Use it to corroborate “latency-bound” diagnoses from SOL/MWA. ([Lumetta][6])                                                                                                                            |
| **WarpStateStats**                                           | **Stall reasons** summary (long scoreboard/memory dep, barriers, inst fetch, exec dep). This tells you *why* warps weren’t issuing—great for deciding between cache/latency fixes vs ILP/occupancy fixes. ([NVIDIA Docs][3])                                                                                              |
| **InstructionStats**                                         | Instruction mix and counts (e.g., how many math vs memory ops per warp). Useful to see if a kernel is dominated by loads/stores or if math intensity is there but Tensor ops are missing. ([Stack Overflow][7])                                                                                                           |
| **PmSampling** / **PmSampling_WarpStates**                   | **Timeline** sampling of single-pass metrics across the kernel’s lifetime (e.g., see Tensor pipe or DRAM use rise/fall across prefill vs decode). Great for phase behavior in LLMs. ([NVIDIA Developer Forums][8])                                                                                                        |
| **LaunchStats**                                              | Launch config & resource facts: grid/block dims, registers, shared memory (including “Shared Memory Configuration Size”). Pairs with Occupancy to explain limits. ([NVIDIA Docs][3])                                                                                                                                      |
| **SourceCounters**                                           | Hotspots mapped to **source/PTX/SASS** with per-line metrics and (when compiled with `-lineinfo`) guidance to fix uncoalesced access, etc. Handy when MWA/Scheduler say “it’s memory/latency”—this shows *where*. ([NVIDIA Docs][3])                                                                                      |
| **SpeedOfLight_RooflineChart** & **Hierarchical * Roofline** | Classic **roofline** plots to classify **compute- vs memory-bound** at a glance (including Tensor-Core rooflines). Expensive to collect but very informative once you’re close to peak. ([NVIDIA Developer Forums][9])                                                                                                    |
| **Nvlink** / **Nvlink_Tables** / **Nvlink_Topology**         | For multi-GPU or CPU-GPU fabrics: link utilization, per-link tables, and topology graph with measured TX/RX. Check when model/tensors span GPUs or you stream over NVLink. ([YouTube][10])                                                                                                                                |
| **NumaAffinity**                                             | Shows NUMA proximity/affinity of GPUs vs CPU memory—relevant for host-device traffic and Grace Hopper configs. ([NVIDIA Docs][3])                                                                                                                                                                                         |
| **C2CLink**                                                  | NVLink-**C2C** (CPU↔GPU cache-coherent) link info on Grace Hopper/Blackwell systems; useful if you spill/stream to CPU RAM. (Recent Nsight Compute versions fixed detection and added C2C collection.) ([NVIDIA Developer][11])                                                                                           |

*(The other “*_Tables/Chart/Topology” variants are alternate visualizations of the same underlying metrics.)*

---

## Which sections to enable for LLM **compute** vs **memory** bottlenecks

### If you suspect a **compute bottleneck**

Turn on:

* **SpeedOfLight** (verify SM/Tensor % of peak),
* **ComputeWorkloadAnalysis** (confirm FP/Tensor pipelines actually used),
* **Occupancy** (theoretical vs achieved),
* **SchedulerStats** + **WarpStateStats** (identify latency vs issue limits),
* *(Optional)* **Tensor Roofline** if you want a roofline verdict.
  These are the most “signal-dense” for confirming you’re limited by math throughput or by not using the right math pipes (e.g., missing Tensor Cores). ([NVIDIA][1])

### If you suspect a **memory bottleneck**

Turn on:

* **MemoryWorkloadAnalysis_Tables** (numbers) + **…_Chart** (shape),
* **WorkloadDistribution** (L1/L2/DRAM active vs elapsed cycles),
* **SpeedOfLight** (DRAM vs SM % of peak),
* **WarpStateStats** (e.g., Long Scoreboard stalls), **SchedulerStats**,
* **SourceCounters** (pinpoint uncoalesced/strided lines),
* *(Multi-GPU/Grace Hopper)* **Nvlink / …_Topology** and **C2CLink**, **NumaAffinity**.
  This combo tells you if you’re saturating memory bandwidth, stuck on latency, or bouncing in caches/links—plus exactly where to fix it in code. ([NVIDIA Docs][3])

---

## Quick presets you can copy

**Compute-focused preset**

```bash
ncu --section SpeedOfLight \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats \
    -o ncu_compute ./run_llm.sh
```

Check **SOL: SM/Tensor % of peak**, **ComputeWorkloadAnalysis** pipeline mix, **Achieved Occupancy**, and **stall reasons** to decide if you’re math-limited or latency-limited. ([NVIDIA][1])

**Memory-focused preset**

```bash
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis_Tables \
    --section MemoryWorkloadAnalysis_Chart \
    --section WorkloadDistribution \
    --section WarpStateStats \
    --section SchedulerStats \
    --section SourceCounters \
    -o ncu_memory ./run_llm.sh
```

Look for **high DRAM % of peak** with **low SM%**, heavy **L2/DRAM active cycles**, **LG/Long Scoreboard stalls**, and SourceCounters hotspots (uncoalesced/strided). Add **Nvlink/NumaAffinity/C2CLink** on multi-GPU/GH systems. ([NVIDIA Docs][3])

---

### Small reading map (optional)

* What SOL actually means and why it’s often the first thing to read. ([NVIDIA][1])
* NVIDIA’s Memory Workload Analysis tutorials (how to reason about L1/L2/DRAM). ([NVIDIA][12])
* Stall reasons & warp scheduler states (why warps don’t issue). ([NVIDIA Docs][3])
* WorkloadDistribution section (exact metrics it includes). ([Hugging Face][4])
* NVLink/NUMA/C2C sections (when your model crosses devices/CPU). ([YouTube][10])

If you want, tell me your exact GPU(s) and LLM stack (PyTorch + cuBLASLt, TRT-LLM, vLLM…), and I’ll tailor a *minimal* metric set by architecture (Hopper/Blackwell vs Ampere) so you get almost all of the above in just 1–2 passes.

[1]: https://www.nvidia.com/en-us/on-demand/session/other2024-sol/?utm_source=chatgpt.com "CUDA Developer Tools | SOL Analysis with NVIDIA Nsight ..."
[2]: https://indico.cern.ch/event/962112/contributions/4110591/attachments/2159863/3643851/CERN_Nsight_Compute.pdf?utm_source=chatgpt.com "Nsight Compute"
[3]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html?utm_source=chatgpt.com "2. Profiling Guide — NsightCompute 13.0 documentation"
[4]: https://huggingface.co/datasets/yyyzzzzyyy/envss/resolve/ea72e051ac77a10399e18d5f00271fd56436f43f/vila-u/nsight-compute-2024.3.2/sections/WorkloadDistribution.section?download=true&utm_source=chatgpt.com "3.52 kB"
[5]: https://docs.nvidia.com/nsight-visual-studio-edition/4.6/Content/Analysis/Report/CudaExperiments/KernelLevel/AchievedOccupancy.htm?utm_source=chatgpt.com "Achieved Occupancy"
[6]: https://lumetta.web.engr.illinois.edu/408-S20/slide-copies/20200416_ece408.pdf?utm_source=chatgpt.com "Using Nsight Compute and Nsight Systems"
[7]: https://stackoverflow.com/questions/79104080/what-do-the-instruction-statistics-fields-in-nsight-compute-mean-how-do-they-re?utm_source=chatgpt.com "What do the Instruction Statistics fields in Nsight Compute ..."
[8]: https://forums.developer.nvidia.com/t/how-to-utilize-pm-sampling/287170?utm_source=chatgpt.com "How to utilize PM sampling? - Nsight Compute"
[9]: https://forums.developer.nvidia.com/t/i-cant-see-roofline-tensor-core/316521?utm_source=chatgpt.com "I cant see roofline tensor core - Nsight Compute"
[10]: https://www.youtube.com/watch?v=51K5EqGqzCM&utm_source=chatgpt.com "Profiling GPU codes with Nsight"
[11]: https://developer.nvidia.com/nsight-compute-2023_2-new-features?utm_source=chatgpt.com "Nsight Compute 2023.2 - New Features"
[12]: https://www.nvidia.com/en-us/on-demand/session/other2024-memory/?utm_source=chatgpt.com "CUDA Developer Tools | Memory Analysis with ..."
