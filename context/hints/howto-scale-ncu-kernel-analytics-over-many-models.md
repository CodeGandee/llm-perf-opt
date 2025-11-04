Title: How to scale Nsight Compute kernel analytics across many models

Problem
- You want a cross-model, cross-run view of CUDA kernels to answer:
  - Which kernels are most popular across many LLMs/vision models?
  - How do they perform (compute vs memory, cache, occupancy, stalls)?
  - How does performance relate to call patterns and launch parameters?
  - Where are systemic weaknesses that suggest GPU design or library gaps?

Approach (high level)
- Use Nsight Systems (nsys) to discover hot kernels per model, then Nsight Compute (ncu) to collect focused metrics for those kernels.
- Export reports to machine-friendly formats (CSV/SQLite) and aggregate with pandas/DuckDB/Polars.
- Normalize kernel names and tag metadata (model, dtype, shapes, batch size, tokens, GPU arch, driver, PyTorch/CUDA versions).
- Build views: popularity, roofline, SOL (SpeedOfLight), cache hit rates, occupancy, stall reasons, shape/launch correlates.

Toolchain overview
- nsys: find hot kernels and call patterns, then feed kernel regexes to ncu.
- ncu: collect sections — SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats, InstructionStats (or `--set detailed`).
- Export: `--export csv` or `--export sqlite`, or post-process `--import` to CSV.
- Python: `ncu_report` (official), pandas/Polars, DuckDB; optional Thicket for multi-run analyses.

References
- Nsight Compute CLI (sets/sections, export/import): https://docs.nvidia.com/nsight-compute/NsightComputeCli/
- Nsight Compute Profiling Guide (SOL, Memory Workload, Occupancy): https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- Python Report Interface (`ncu_report`): https://docs.nvidia.com/nsight-compute/PythonReportInterface/
- Thicket Nsight Compute reader: https://thicket.readthedocs.io/en/latest/nsight_compute.html
- NASA HECC Nsight Compute CLI guide: https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-cuda-kernels-with-nsight-compute-cli_706.html
- Example community CSV export with `ncu --import --csv`: https://github.com/Majestic-labs-AI/profiling/blob/main/auto_parser.py

Metrics to watch (aligned to context/design/metrics-to-watch.md)
- Compute unit utilization (SM/Tensor/FP32)
  - Sections: SpeedOfLight, ComputeWorkloadAnalysis
  - Metrics (typical names; confirm via `ncu --query-metrics`):
    - `sm__throughput.avg.pct_of_peak_sustained_elapsed` (overall SM %peak)
    - Tensor utilization (if exposed): `sm__pipe_tensor_throughput.avg.pct_of_peak_sustained_elapsed` or `smsp__inst_executed_pipe_tensor.sum`
    - FP32 mix (proxy): `smsp__inst_executed_pipe_fp32.sum` (or instruction breakdown in ComputeWorkloadAnalysis)
- Cache hit rate (L1/TEX/L2)
  - Sections: MemoryWorkloadAnalysis (+_Chart)
  - Metrics: `l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct`
- Memory bandwidth (DRAM %peak)
  - Sections: SpeedOfLight, MemoryWorkloadAnalysis
  - Metrics: `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- Warp efficiency (stall reasons)
  - Sections: SchedulerStats, Occupancy
  - Metrics (names vary by version; use `--query-metrics`):
    - `smsp__warp_issue_stalled_barrier_per_warp_active.pct`
    - `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct`
    - `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`
- Kernel time
  - Source: nsys `cuda_gpu_kern_sum` CSV; ncu also provides `gpu__time_duration.sum` per kernel
- Overall bottleneck classification
  - Use SOL (SM vs DRAM %peak), cache hit rates, and stall reasons to classify: Compute / Memory / Mixed / Latency-bound

Curated minimal metrics list (when sections vary)
```bash
# If sections differ across hosts, explicitly request a portable metric subset
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
gpu__time_duration.sum"

ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name ".*(gemm|gemvx|attention|layernorm).*" \
  --kernel-id :::1 \
  --metrics ${METRICS} \
  --replay-mode kernel \
  -o tmp/ncu_${MODEL}_portable \
  <your-python-cmd>
```

Programmatic classification example (pandas)
```python
import pandas as pd

def classify_row(r, sm_thr=60, dram_thr=60, l2_low=60, stall_lat=30):
    sm = r.get('sm_pct', 0) or 0
    dram = r.get('dram_pct', 0) or 0
    l2 = r.get('l2_hit_pct', None)
    long_sb = r.get('stall_long_sb_pct', 0) or 0
    short_sb = r.get('stall_short_sb_pct', 0) or 0
    barrier = r.get('stall_barrier_pct', 0) or 0
    # Heuristics
    if dram >= dram_thr and sm < sm_thr:
        return 'Memory-bound'
    if sm >= sm_thr and dram < dram_thr:
        return 'Compute-bound'
    if (long_sb+short_sb+barrier) >= stall_lat and sm < sm_thr and dram < dram_thr:
        return 'Latency/Occupancy-bound'
    if l2 is not None and l2 < l2_low and dram >= dram_thr:
        return 'Cache-inefficiency (memory-bound)'
    return 'Mixed/Unclear'

# df should have columns from ncu export: sm_pct, dram_pct, l2_hit_pct,
# stall_long_sb_pct, stall_short_sb_pct, stall_barrier_pct
df['bottleneck'] = df.apply(classify_row, axis=1)
```

Data collection pipeline
1) Per-model discovery with nsys (short run, deterministic input)
```bash
# Hot kernels per model (short, deterministic workload)
nsys profile -o tmp/nsys_${MODEL} --trace=cuda,nvtx --force-overwrite true --capture-range=none \
  <your-python-cmd --small-workload>

# Summaries for kernel time/popularity (CSV)
nsys stats --report cuda_gpu_kern_sum --format csv tmp/nsys_${MODEL}.qdrep > tmp/${MODEL}_kern_sum.csv
```

2) Kernel selection
- Parse `${MODEL}_kern_sum.csv` to get top N unique kernel base names (demangled), e.g., GEMM/attention/transpose/layernorm/reduction.
- Build a kernel regex per model or a global master regex list for “popular kernels”.

3) Targeted ncu collection (fast iteration first)
```bash
# Minimal set for quick bound classification + cache/bandwidth
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name ".*(gemm|gemvx|attention|layernorm).*" \
  --kernel-id :::1 \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --replay-mode kernel \
  -o tmp/ncu_${MODEL}_quick \
  <your-python-cmd --small-workload>

# For deep analysis (add more sections or use a set)
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name ".*(gemm|gemvx|attention|layernorm).*" \
  --kernel-id :::1 \
  --set detailed \
  --replay-mode kernel \
  -o tmp/ncu_${MODEL}_detailed \
  <your-python-cmd>

# Export machine-readable data (choose one)
ncu --import tmp/ncu_${MODEL}_detailed.ncu-rep --csv --section SpeedOfLight > tmp/${MODEL}_sol.csv
ncu --import tmp/ncu_${MODEL}_detailed.ncu-rep --csv --section MemoryWorkloadAnalysis > tmp/${MODEL}_mem.csv || true
ncu --import tmp/ncu_${MODEL}_detailed.ncu-rep --export sqlite --output tmp/${MODEL}.sqlite
```

What the ncu variants mean (cost vs insight)
- Minimal quick pass (SpeedOfLight only)
  - Command: `--section SpeedOfLight --kernel-id :::1 --replay-mode kernel`
  - Insight: bound classification (compute vs memory), high-level % of peak for SM and DRAM; very fast to collect.
  - Cost: low. Few metrics, minimal replay passes. Use for discovery and large-scale sweeps.
- Memory-focused (add MemoryWorkloadAnalysis)
  - Command: add `--section MemoryWorkloadAnalysis` (and optionally `_Chart`)
  - Insight: achieved memory bandwidth, L1/L2 hit rates, transaction/caching behavior; confirms memory-bound and points to locality issues.
  - Cost: low-to-moderate. More counters may add passes, but still manageable if scoping to one kernel instance.
- Occupancy/Scheduler/Instruction detail
  - Command: add `--section Occupancy --section SchedulerStats --section InstructionStats`
  - Insight: achieved vs theoretical occupancy, top warp stall reasons (e.g., memory dependency, barrier), instruction mix and tensor core usage.
  - Cost: moderate. More replays; keep workload small and limit instances.
- Detailed set
  - Command: `--set detailed`
  - Insight: broad coverage incl. ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, SpeedOfLight, Occupancy, SchedulerStats, SourceCounters; often enough to perform deep diagnosis and roofline.
  - Cost: high. Many metrics → many replay passes. Restrict to the top 1 kernel instance per run.
- Roofline chart
  - Command: add `--section SpeedOfLight_RooflineChart` (name varies by version)
  - Insight: arithmetic intensity vs achieved GFLOP/s; visualize whether compute- or bandwidth-limited and headroom relative to roofs.
  - Cost: moderate-to-high depending on version. Use with `--kernel-id :::1`.
- Raw metrics selection
  - Command: `--metrics <list>` instead of sections
  - Insight: precision control for research questions (e.g., pick specific l1tex/lts/dram/sm metrics);
  - Cost: variable. Selecting counters from different hardware groups increases pass count. Favor curated sections when possible.

Replay mode and kernel selection
- `--replay-mode kernel`: fastest for Python apps; replays just the targeted kernel launches.
- `--replay-mode application`: replays the entire application between metric groups; heavy and typically unnecessary for LLM inference.
- Reduce instances with `--kernel-id :::1` or `--launch-skip N --launch-count 1` to keep runtime bounded.
- Always narrow with `--kernel-name-base demangled --kernel-name '<regex>'` to avoid profiling uninteresting kernels.

Run metadata (record alongside metrics)
- Model id/version, task (OCR/LLM), dtype (fp16/bf16/fp8), batch size, tokens, image size.
- GPU model, SM arch, driver, CUDA version, PyTorch version; clocks/power mode.
- Workload seed; numbers of warmup/iterations; exact ncu/nsys version.

Kernel name normalization
- Demangle, strip template-unique suffixes and launch IDs.
- Classify by library (cuBLAS/CUTLASS/Triton/FlashAttention) via substrings.
- Collapse variants (e.g., `gemm_` with shapes/dtype → same base op, features kept as columns).

Python aggregation (SQLite with DuckDB)
```python
import duckdb, glob

con = duckdb.connect()
con.sql("CREATE VIEW runs AS SELECT * FROM read_sqlite('tmp/*.sqlite','Metrics')")

# Example: popularity by kernel base name (across all runs)
pop = con.sql('''
SELECT
  regexp_replace(name, '\\(.*', '') AS kernel,  -- trim call sig
  COUNT(*) AS occurrences,
  COUNT(DISTINCT run_id) AS models
FROM runs
WHERE name IS NOT NULL
GROUP BY 1
ORDER BY occurrences DESC
LIMIT 50
''').df()

# Example: performance summary per kernel (SpeedOfLight metrics must exist)
summary = con.sql('''
SELECT
  kernel,
  AVG(CASE WHEN metric='sm__throughput.avg.pct_of_peak_sustained_elapsed' THEN value END) AS sm_pct,
  AVG(CASE WHEN metric='dram__throughput.avg.pct_of_peak_sustained_elapsed' THEN value END) AS dram_pct,
  AVG(CASE WHEN metric='l1tex__t_sector_hit_rate.pct' THEN value END) AS l1_hit,
  AVG(CASE WHEN metric='lts__t_sector_hit_rate.pct' THEN value END) AS l2_hit,
  COUNT(*) AS samples
FROM (
  SELECT regexp_replace(name, '\\(.*', '') AS kernel, metric, value FROM runs
)
GROUP BY 1
ORDER BY samples DESC
''').df()
```

Call pattern/context analysis
- Without NVTX: use nsys timeline around hot kernels to infer preceding/following ops, stream IDs, and concurrency; export `cuda_gpu_kern_sum` plus timeline slices.
- With NVTX (recommended for deep context): tag PyTorch ops/layers; in Nsight Systems you can correlate ranges to kernels; in ncu you can filter via NVTX push ranges.
- Group context features: batch size, seq length, image tiling, dtype, tensor shapes, stream, graphs vs eager, library backend (cuBLASLt/Triton/CUTLASS).

Roofline and bound classification at scale
- Collect `SpeedOfLight` and `SpeedOfLight_RooflineChart` (name may vary). Aggregate AI (FLOPs/byte) and achieved GFLOP/s by kernel class and model.
- Identify clusters: kernels that are consistently memory-bound (high DRAM%, low SM%), compute-bound (high SM%, low DRAM%), or latency/occupancy limited (both low).

Stall reasons and occupancy
- Gather `SchedulerStats`, `Occupancy`, `InstructionStats` to understand:
  - top stalls (memory dependency, barrier, warp serialization)
  - achieved vs theoretical occupancy, register/shared-mem pressure
  - instruction mix (ld/st vs math), tensor core usage

Scaling tips
- Keep collection light for discovery: `--kernel-id :::1`, small workload, `--replay-mode kernel`, minimal sections first.
- Cache inputs and models; run with fixed seeds; pin clocks to reduce drift.
- Version-proof: list sections/metrics per ncu version (`--list-sections`, `--query-metrics`) and dynamically adapt.
- Use a run manifest (JSON) per profile with all metadata; store paths to `.ncu-rep`, CSV/SQLite, and nsys summaries.

From findings to GPU design signals
- If many models spend time in memory-bound kernels with low L2 hit: suggests pressure on L2/DRAM and opportunities for larger caches, better prefetch, or better swizzle/layout ops.
- If compute-bound but low tensor-core utilization on matmul-like ops: indicates datatype mismatches or library gaps for problem shapes (e.g., small-K GEMMs, ragged shapes).
- If occupancy-limited across kernel families due to register pressure: points to compiler or ISA register constraints; kernel redesign or compiler heuristics.
- If launch/config patterns (tiny grids, high launch counts) dominate: suggests kernel fusion or CUDA Graphs adoption opportunities.

Optional: Thicket for multi-run analytics
- Thicket can ingest ncu and call-tree profiles, enabling cross-run comparisons and structured analytics without bespoke glue.
- Docs: https://thicket.readthedocs.io/en/latest/nsight_compute.html

Appendix — small utilities
```bash
# List available sections/metrics on your ncu version
ncu --list-sections | sed -n '1,120p'
ncu --query-metrics | rg -i 'throughput|hit_rate|occupancy|stall|dram|l1|l2'
```

```python
# Minimal `.ncu-rep` reader with ncu_report (official)
import sys, glob
sys.path.insert(0, '/opt/nvidia/nsight-compute/*/extras/python')  # adjust
from ncu_report import load_report
ctx = load_report('tmp/ncu_model.ncu-rep')
for i in range(ctx.num_ranges()):
    r = ctx.range_by_idx(i)
    for j in range(r.num_actions()):
        a = r.action_by_idx(j)
        sm = a.metric_by_name('sm__throughput.avg.pct_of_peak_sustained_elapsed').value()
        dm = a.metric_by_name('dram__throughput.avg.pct_of_peak_sustained_elapsed').value()
        print(a.name(), sm, dm)
```
