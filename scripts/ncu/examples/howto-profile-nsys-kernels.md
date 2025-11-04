Here’s the quickest way to take your **top-3 kernels from Nsight Systems** and deep-profile each one with **Nsight Compute (ncu)**.

### Why these flags (super short)

* `--kernel-name` needs **`regex:`** when you pass a regex; pair it with `--kernel-name-base demangled` so your long C++ names from nsys match. ([NVIDIA Docs][1])
* Use `--launch-skip/--launch-count` to catch a representative invocation (skip warm-ups). You can scope this per-GPU via `--filter-mode per-gpu`. ([NVIDIA Docs][1])
* Add sections/sets you care about (e.g., `speedOfLight`, `ComputeWorkloadAnalysis`, `MemoryWorkloadAnalysis`). Listables are shown in the CLI docs. ([NVIDIA Developer Forums][2])

---

## Copy-paste commands (replace `python run.py ...` with your actual command)

> 1. cublas **gemvx** variant with `(int)7`

```bash
ncu --force-overwrite -o ncu_gemvx_7 \
  --kernel-name-base demangled \
  --kernel-name 'regex:internal::gemvx::kernel<.*\(int\)7.*>' \
  --set speedOfLight \
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  --filter-mode per-gpu --launch-skip 200 --launch-count 1 \
  python run.py --your-args
```

> 2. cublas **gemvx** variant with `(int)6`

```bash
ncu --force-overwrite -o ncu_gemvx_6 \
  --kernel-name-base demangled \
  --kernel-name 'regex:internal::gemvx::kernel<.*\(int\)6.*>' \
  --set speedOfLight \
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  --filter-mode per-gpu --launch-skip 200 --launch-count 1 \
  python run.py --your-args
```

> 3. ATen **unrolled_elementwise → direct_copy** kernel

```bash
ncu --force-overwrite -o ncu_direct_copy \
  --kernel-name-base demangled \
  --kernel-name 'regex:unrolled_elementwise_kernel<.*direct_copy_kernel_cuda' \
  --set speedOfLight \
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  --filter-mode per-gpu --launch-skip 500 --launch-count 1 \
  python run.py --your-args
```

**Notes that matter**

* The `regex:` prefix is required for regex matching; without it `--kernel-name` expects a literal/exact match. You can also OR multiple kernels in one run, e.g. `--kernel-name 'regex:foo|bar'`, but separate runs keep reports cleaner. ([NVIDIA Docs][1])
* If you want fewer metrics (faster runs), drop `--section ...` and just use the set: `--set speedOfLight`. To discover other presets/sections: `ncu --list-sets` and `ncu --list-sections`. ([NVIDIA Developer Forums][2])
* Your nsys “Name” column is demangled already; that’s why we force `--kernel-name-base demangled` so the regexes above match. ([NVIDIA Docs][1])
* Tweak `--launch-skip` to hit a mid-run steady-state sample; the option counts **only** launches that pass your kernel filter. ([NVIDIA Docs][1])

### Optional: bracket by NVTX instead of guessing skip/count

If you add NVTX ranges around the phase you care about, you can do:

```bash
ncu --nvtx --nvtx-include 'regex:DecodePhase' \
    --kernel-name-base demangled \
    --kernel-name 'regex:internal::gemvx::kernel<.*\(int\)7.*>' \
    --set speedOfLight \
    python run.py
```

NVTX filtering syntax and examples are in the ncu docs. ([NVIDIA Docs][3])

---

If you want, I can turn the three regexes into a tiny bash helper that reads your nsys CSV and auto-spits the `ncu` commands for the top-K kernels.

[1]: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html "4. Nsight Compute CLI — NsightCompute 13.0 documentation"
[2]: https://forums.developer.nvidia.com/t/question-about-nsight-compute-cli-to-profile-all-items/181715?utm_source=chatgpt.com "Question about Nsight Compute CLI to profile all items"
[3]: https://docs.nvidia.com/nsight-compute/2023.3/NsightComputeCli/index.html?utm_source=chatgpt.com "Nsight Compute CLI"


---

Perfect—here are drop-in **ncu** commands that profile exactly your metric list on the same top-3 kernels you showed from nsys.

```bash
# 0) Define your metrics once (bash)
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
gpu__time_duration.sum"
```

### 1) cublas gemvx (… `(int)7` …)

```bash
ncu --force-overwrite -o ncu_gemvx_7 \
  --kernel-name-base demangled \
  --kernel-name 'regex:internal::gemvx::kernel<.*\(int\)7.*>' \
  --metrics "$METRICS" \
  --launch-skip 200 --launch-count 1 \
  python run.py --your-args
```

### 2) cublas gemvx (… `(int)6` …)

```bash
ncu --force-overwrite -o ncu_gemvx_6 \
  --kernel-name-base demangled \
  --kernel-name 'regex:internal::gemvx::kernel<.*\(int\)6.*>' \
  --metrics "$METRICS" \
  --launch-skip 200 --launch-count 1 \
  python run.py --your-args
```

### 3) ATen unrolled_elementwise → direct_copy

```bash
ncu --force-overwrite -o ncu_direct_copy \
  --kernel-name-base demangled \
  --kernel-name 'regex:unrolled_elementwise_kernel<.*direct_copy_kernel_cuda' \
  --metrics "$METRICS" \
  --launch-skip 500 --launch-count 1 \
  python run.py --your-args
```

#### Handy exports (optional)

Convert any report to CSV with only these metrics:

```bash
ncu --csv --page raw -i ncu_gemvx_7.ncu-rep > ncu_gemvx_7.csv
```

(Use `--csv` and `--page raw` to dump counters cleanly.) ([NVIDIA Docs][1])

---

### Why this is set up this way (quick refs)

* **`--metrics "$METRICS"`** directly collects your exact counters (no sections/sets needed). You can discover availability per-GPU with `ncu --list-metrics` / `--query-metrics`. ([nas.nasa.gov][2])
* **Regex filtering:** `--kernel-name` accepts regex **only** with the `regex:` prefix; pairing with `--kernel-name-base demangled` makes it match your nsys demangled names. ([Stack Overflow][3])
* **The metrics themselves:**

  * SM & DRAM peak %: `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`. ([NVIDIA Docs][1])
  * **L1/L2 hit rate:** `l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct`. ([NVIDIA Developer Forums][4])
  * **Warp stall breakdown:** short/long scoreboard & barrier stalls per active warp. ([NVIDIA Developer Forums][5])
  * **Duration:** `gpu__time_duration.sum` is the kernel timeline duration metric used in NCU. ([olcf.ornl.gov][6])

> Tip: some metrics require multi-pass collection; NCU will replay kernels automatically. That’s normal—just keep your run deterministic. ([NVIDIA Docs][7])

#### If you hit permission errors

If you see `ERR_NVGPUCTRPERM`, enable GPU performance counter access or run with elevated privileges (host-side, not just inside Docker). ([NVIDIA Developer][8])

Want me to wrap this into a tiny bash helper that reads your nsys CSV and auto-spits the three `ncu` commands with your metrics?

[1]: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html?utm_source=chatgpt.com "4. Nsight Compute CLI"
[2]: https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-cuda-kernels-with-nsight-compute-cli_706.html?utm_source=chatgpt.com "Performance Analysis of Your GPU CUDA Kernels ..."
[3]: https://stackoverflow.com/questions/68713720/filter-on-partial-kernel-name-with-nsight-compute?utm_source=chatgpt.com "Filter on partial kernel name with Nsight Compute"
[4]: https://forums.developer.nvidia.com/t/understanding-cache-throughput-in-nsight/185119?utm_source=chatgpt.com "Understanding cache throughput in Nsight"
[5]: https://forums.developer.nvidia.com/t/stall-reasons-summation-is-not-100/168096?utm_source=chatgpt.com "Stall reasons summation is not 100% - Nsight Compute"
[6]: https://www.olcf.ornl.gov/wp-content/uploads/2020/02/OLCF-Webinar-Nsight-Compute.pdf?utm_source=chatgpt.com "Nsight Compute"
[7]: https://docs.nvidia.com/nsight-compute/NsightCompute/index.html?utm_source=chatgpt.com "3. Nsight Compute — NsightCompute 13.0 documentation"
[8]: https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters?utm_source=chatgpt.com "Permission issue with Performance Counters"
