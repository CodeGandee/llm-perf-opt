Title: How to use Nsight Compute (ncu) to decide compute-bound vs memory-bound, analyze cache misses, and bandwidth

Problem
- Given a hot CUDA kernel (from nsys or prior runs), quickly determine the bottleneck (compute vs memory), inspect cache hit/miss behavior, and measure memory bandwidth using Nsight Compute CLI (ncu).

Quick Rules of Thumb
- Start with SpeedOfLight: shows SM vs Memory throughput relative to peak. High memory%, low SM% → memory-bound. High SM%, low memory% → compute-bound. Both low → often latency/occupancy or divergence issues; check Scheduler/Occupancy.
- If memory-bound: go to Memory Workload Analysis for DRAM bandwidth and cache hit rates (L1/L2). If compute-bound: check Instruction/Compute Workload Analysis and Occupancy to see math pipe saturation and occupancy limits.
- Verify you are profiling the intended kernel instance. Use demangled names and regex; keep collection light (first instance only) for iteration.

Minimal, fast pass (recommended first)
```bash
# Profile first instance of matching kernel; collect SOL only (fast)
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name '.*your_kernel_substring.*' \
  --kernel-id :::1 \
  --section SpeedOfLight \
  --replay-mode kernel \
  -o tmp/ncu_sol_quick \
  <your-cmd>
```

Memory focus (cache + bandwidth)
```bash
# Add Memory Workload Analysis sections for cache hit rates & bandwidth
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name '.*your_kernel_substring.*' \
  --kernel-id :::1 \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section MemoryWorkloadAnalysis_Chart \
  --replay-mode kernel \
  -o tmp/ncu_mem \
  <your-cmd>
```

Deeper cache metrics (portable set)
```bash
# If sections aren’t available, or you want raw counters, query & pick metrics
ncu --list-sections            # discover section names on your ncu version
ncu --query-metrics | rg -i 'l1|l2|dram|throughput|hit'  # discover counters

# Example counters commonly used for cache/bandwidth insight:
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name '.*your_kernel_substring.*' \
  --kernel-id :::1 \
  --metrics \
    l1tex__t_sector_hit_rate.pct, \
    lts__t_sector_hit_rate.pct, \
    dram__throughput.avg.pct_of_peak_sustained_elapsed, \
    sm__throughput.avg.pct_of_peak_sustained_elapsed \
  --replay-mode kernel \
  -o tmp/ncu_cache_bw \
  <your-cmd>
```

Interpreting results
- Compute-bound vs Memory-bound (SpeedOfLight):
  - sm__throughput near peak, dram__throughput far from peak → compute-bound
  - dram__throughput near peak, sm__throughput far from peak → memory-bandwidth-bound
  - Both low → latency-bound: inspect Warp/Scheduler stall reasons and Occupancy
- Cache analysis (Memory Workload Analysis / metrics):
  - l1tex__t_sector_hit_rate.pct and lts__t_sector_hit_rate.pct indicate L1/L2 efficiency
  - Low L2 hit rate + high DRAM throughput → consider data layout, access patterns, blocking/tiling, and reuse
  - High L1 miss but good L2 hit → consider coalescing, alignment, and avoiding thrashing (stride-1 or warp-friendly layouts)
- Bandwidth:
  - Memory Workload Analysis shows achieved bandwidth vs peak; 80–90%+ of DRAM peak typically indicates you’re bandwidth-limited already
  - If bandwidth is low and misses are high, fix locality/coalescing; if bandwidth is low and stalls show dependencies, address latency (e.g., increase occupancy, use async copies, prefetch)

Reducing profiling overhead while iterating
- Profile only one instance: `--kernel-id :::1` or `--launch-skip N --launch-count 1`
- Narrow kernel selection: `--kernel-name-base demangled --kernel-name '<regex>'`
- Collect only what you need: start with `--section SpeedOfLight`; add memory sections later
- Use `--replay-mode kernel` for faster turnaround on stable kernels; avoid application replay for large Python apps unless required

Python/LLM pipelines tips
- Launch Python with small, deterministic workload (few samples, few tokens) to ensure kernel is hit quickly
- If collecting deep metrics causes excessive passes, scale down problem size and use `--kernel-id :::1`
- If sections don’t match your ncu version, list them and adjust; section names can vary by version

Troubleshooting
- No sections found: run `ncu --list-sections` and use exact names (e.g., `SpeedOfLight`, `MemoryWorkloadAnalysis`)
- Empty report / no kernels: ensure the kernel actually launches (small input, warmup disabled) and that you profiled the correct process (`--target-processes all` helps for Python/torch)
- Version mismatch: prefer newest ncu; some sections/metrics only exist on newer toolkits

References
- Nsight Compute Profiling Guide (NVIDIA Docs): https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- Using Nsight Compute & Systems (slides, section names/examples): https://lumetta.web.engr.illinois.edu/408-S20/slide-copies/20200416_ece408.pdf
- Intro to Kernel Performance Analysis with Nsight Compute (YouTube, SOL-first workflow): https://www.youtube.com/watch?v=fsC3QeZHM1U
- Memory Analysis with Nsight Compute (YouTube, cache/bandwidth focus): https://www.youtube.com/watch?v=GCkdiHk6fUY
- NVIDIA Forums (cache hit rate metrics examples): https://forums.developer.nvidia.com/t/average-of-all-kernels-l1-l2-cache-hit-rate/321515


Roofline plotting in Python (no GUI)
- Collect roofline data (section name may vary by version):
```bash
# Try the dedicated roofline section first; fall back to detailed set if missing
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name '.*your_kernel_substring.*' \
  --kernel-id :::1 \
  --section SpeedOfLight \
  --section SpeedOfLight_RooflineChart \
  --replay-mode kernel \
  -o tmp/ncu_roofline \
  <your-cmd> || \
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name '.*your_kernel_substring.*' \
  --kernel-id :::1 \
  --set detailed \
  --replay-mode kernel \
  -o tmp/ncu_roofline \
  <your-cmd>

# Export roofline + SOL tables to CSV
ncu --import tmp/ncu_roofline.ncu-rep --csv --section SpeedOfLight_RooflineChart > tmp/roofline.csv || true
ncu --import tmp/ncu_roofline.ncu-rep --csv --section SpeedOfLight > tmp/sol.csv
```

- Plot roofline with pandas/matplotlib (auto-detect columns):
```python
import re, io, pandas as pd, matplotlib.pyplot as plt

def read_section_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        # Skip commented header lines starting with '#'
        lines = [ln for ln in f.readlines() if not ln.startswith('#')]
    return pd.read_csv(io.StringIO(''.join(lines)))

roof = read_section_csv('tmp/roofline.csv') if os.path.exists('tmp/roofline.csv') else pd.DataFrame()
sol  = read_section_csv('tmp/sol.csv')

def pick(df, *substrs):
    if df.empty: return None
    cols = df.columns
    for c in cols:
        low = c.lower()
        if all(s in low for s in substrs):
            return c
    return None

# Try to find AI (FLOPs/byte) and achieved GFLOP/s from roofline section
ai_col   = pick(roof, 'arithmetic', 'intensity')
gflop_col= pick(roof, 'flop', '/s')
kernel_col = pick(roof, 'name') or pick(roof, 'kernel')

if roof.empty or ai_col is None or gflop_col is None:
    raise SystemExit('Roofline section not present; reprofile with --section SpeedOfLight_RooflineChart or --set detailed')

roof2 = roof[[kernel_col, ai_col, gflop_col]].dropna()
roof2.columns = ['kernel','ai','gflops']

# Extract peaks (compute & memory) from SOL table if available
peak_mem_col = pick(sol, 'peak', 'memory') or pick(sol, 'bandwidth')
peak_comp_col= pick(sol, 'peak', 'compute') or pick(sol, 'gflop')

peak_mem = None
peak_comp = None
for df in [sol]:
    for c in df.columns:
        low = c.lower()
        if 'device memory peak' in low and ('gb/s' in low or 'bandwidth' in low):
            try: peak_mem = float(df[c].iloc[0]); break
            except: pass
    for c in df.columns:
        low = c.lower()
        if ('device' in low and 'compute' in low and 'peak' in low) or ('peak gflop' in low):
            try: peak_comp = float(df[c].iloc[0]); break
            except: pass

if peak_mem is None or peak_comp is None:
    print('Could not auto-detect peaks from SOL; please set peak_mem/peak_comp manually')
    # Example manual inputs (adjust to your GPU)
    # peak_mem = 1000.0  # GB/s
    # peak_comp = 60000.0  # GFLOP/s

# Plot
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(roof2['ai'], roof2['gflops'], s=30)
for _,r in roof2.iterrows():
    ax.annotate(r['kernel'][:24], (r['ai'], r['gflops']))

xmax = max(roof2['ai'].max()*1.5, 1.0)
x = pd.Series([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
if peak_mem and peak_comp:
    y_mem = peak_mem * x  # y = BW_peak * AI
    y_comp = pd.Series([peak_comp]*len(x))
    ax.plot(x, y_mem, '--', label=f'Memory roofline ({peak_mem:.0f} GB/s)')
    ax.plot(x, y_comp, '--', label=f'Compute roofline ({peak_comp:.0f} GFLOP/s)')

ax.set_xscale('log')
ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)')
ax.set_ylabel('Achieved Performance (GFLOP/s)')
ax.set_title('Roofline (Nsight Compute)')
ax.legend()
plt.tight_layout(); plt.show()
```

Note
- Section identifiers differ across versions. If `SpeedOfLight_RooflineChart` is missing, list your sections and adapt accordingly: `ncu --list-sections`.
- The UI also computes a Roofline, but the steps above avoid the GUI entirely while producing an equivalent plot.


Python-only visualization (no GUI)
- Option A: Use Nsight Compute’s Python Report Interface (`ncu_report`) to load `.ncu-rep` and extract metrics directly in Python.
- Option B: Use CLI export/import to CSV and read with pandas.

Option A: ncu_report (preferred for programmatic analysis)
- The module ships with Nsight Compute at `extras/python/ncu_report.py`. Add that directory to `PYTHONPATH`.
- Docs: https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html

Example: load report and build a quick SOL/cache summary for top kernels
```python
import os, sys, glob, pathlib
import pandas as pd
import matplotlib.pyplot as plt

# 1) Locate ncu_report.py (adjust to your installation)
candidates = [
    os.environ.get("NSIGHT_COMPUTE_PYTHON"),
    "/opt/nvidia/nsight-compute/*/extras/python",
    "/usr/local/cuda*/nsight-compute*/extras/python",
    "/usr/local/cuda*/NsightCompute*/extras/python",
]
for pat in candidates:
    if not pat:
        continue
    for p in glob.glob(pat):
        if (pathlib.Path(p) / "ncu_report.py").exists():
            sys.path.insert(0, p)
            break

from ncu_report import load_report

rep_path = "tmp/ncu_top1_gemvx.ncu-rep"  # your report
ctx = load_report(rep_path)

rows = []
for ri in range(ctx.num_ranges()):
    r = ctx.range_by_idx(ri)
    for ai in range(r.num_actions()):
        a = r.action_by_idx(ai)
        name = a.name()
        def m(name):
            try:
                return a.metric_by_name(name).value()
            except Exception:
                return None
        rows.append({
            "kernel": name,
            "time_ms": (m("gpu__time_duration.sum") or 0.0) / 1e6,
            "sm_pct": m("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            "dram_pct": m("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            "l1_hit_pct": m("l1tex__t_sector_hit_rate.pct"),
            "l2_hit_pct": m("lts__t_sector_hit_rate.pct"),
        })

df = pd.DataFrame(rows).dropna(subset=["sm_pct","dram_pct"]).sort_values("time_ms", ascending=False)
top = df.head(10)

# 2) Bar chart: SOL SM vs DRAM for top kernels
ax = top.set_index("kernel")[ ["sm_pct","dram_pct"] ].plot(kind="bar", figsize=(10,4))
ax.set_ylabel("% of peak")
ax.set_title("Top kernels: SpeedOfLight SM vs DRAM")
plt.tight_layout(); plt.show()

# 3) Scatter: L2 hit vs DRAM throughput to spot memory-bound kernels
mem = top.dropna(subset=["l2_hit_pct","dram_pct"]) 
ax = mem.plot.scatter(x="l2_hit_pct", y="dram_pct", figsize=(5,4), title="L2 hit vs DRAM %peak")
for i,row in mem.iterrows():
    ax.annotate(row["kernel"].split("(")[0][:24], (row["l2_hit_pct"], row["dram_pct"]))
plt.tight_layout(); plt.show()
```

Notes
- Metric names must have been collected. If a value is `None`, re-run ncu with the corresponding sections:
  - `--section SpeedOfLight` (for `sm__throughput.*`, `dram__throughput.*`)
  - `--section MemoryWorkloadAnalysis` (for cache hit rates)
- Use `ncu --list-sections` and `ncu --query-metrics` to discover names on your version.

Option B: Import `.ncu-rep` as CSV via CLI, then analyze with pandas
```python
import io, subprocess, pandas as pd

rep = "tmp/ncu_top1_gemvx.ncu-rep"

# Export SpeedOfLight section table to CSV (works without GUI)
csv_bytes = subprocess.check_output([
    "ncu", "--import", rep, "--csv", "--section", "SpeedOfLight"
])
sol = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")), comment="#")
print(sol.head())

# Similarly for Memory Workload Analysis (if available)
try:
    csv_bytes = subprocess.check_output([
        "ncu", "--import", rep, "--csv", "--section", "MemoryWorkloadAnalysis"
    ])
    mem = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")), comment="#")
    print(mem.head())
except subprocess.CalledProcessError:
    print("MemoryWorkloadAnalysis section not present; reprofile with --section MemoryWorkloadAnalysis")
```

Tip: Export SQLite to enable custom SQL
- You can produce a SQLite DB instead of a `.ncu-rep` using `--export sqlite`.
```bash
# At profile time
ncu -o tmp/run --export sqlite --section SpeedOfLight --section MemoryWorkloadAnalysis <your-cmd>

# Or convert an existing report
ncu --import tmp/run.ncu-rep --export sqlite --output tmp/run.sqlite
```
```python
# Then use sqlite3 / pandas to query
import sqlite3, pandas as pd
con = sqlite3.connect("tmp/run.sqlite")
df = pd.read_sql_query("SELECT * FROM Metrics WHERE name IN ('sm__throughput.avg.pct_of_peak_sustained_elapsed','dram__throughput.avg.pct_of_peak_sustained_elapsed')", con)
print(df.head())
```
