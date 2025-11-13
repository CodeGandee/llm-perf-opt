# Nsight Compute Metrics Profiled

This document lists the metrics captured by Nsight Compute (ncu) for the run artifacts under `reports/20251107-dsocr/ncu-v2`.

It is organized by Nsight Compute “sections” and provides a brief, practical description of each metric.

## Sections Overview
- GPU Speed Of Light Roofline Chart: 18 metrics
- Occupancy: 13 metrics
- Command-Line Counters: 12 metrics
- GPU Speed Of Light Throughput: 10 metrics
- Memory Workload Analysis: 7 metrics
- Scheduler Statistics: 5 metrics
- Report Metadata: 2 metrics

## Contents
- Command-Line Counters
- GPU Speed Of Light Roofline
- GPU Speed Of Light Throughput
- Memory Workload Analysis
- Occupancy
- Scheduler Statistics
- Report Metadata

## Command-Line Counters
Purpose: low-level raw counters requested via CLI for quick checks on bandwidth, FLOPs, and timing.

- `dram__bytes_read.sum` — total DRAM bytes read (reported as Kbyte/Mbyte); gauge read traffic volume.
- `dram__bytes_write.sum` — total DRAM bytes written (byte/Kbyte/Mbyte); gauge write traffic volume.
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — achieved DRAM bandwidth vs sustained peak (%); detect memory‑bound behavior.
- `flop_count_hp.sum` — total half‑precision FLOPs executed; estimate arithmetic intensity/compute utilization.
- `flop_count_sp.sum` — total single‑precision FLOPs executed; estimate arithmetic intensity/compute utilization.
- `gpu__time_duration.sum` — kernel duration (us); baseline timing and comparisons.
- `sm__ops_path_tensor_src_bf16_dst_bf16.sum` — BF16→BF16 tensor core ops; quantify pure BF16 tensor‑core usage.
- `sm__ops_path_tensor_src_bf16_dst_fp32.sum` — BF16→FP32 tensor core ops; quantify mixed‑precision tensor‑core usage.
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — achieved SM compute throughput vs sustained peak (%); assess compute saturation.

## GPU Speed Of Light Roofline
Purpose: relate compute throughput to operational intensity and memory bandwidth via a roofline view.

### Double Precision (DP)
- Achieved — DRAM Bandwidth: measured off‑chip bandwidth during kernel; defines roofline x‑axis.
- Achieved — Predicated‑On DADD/DFMA/DMUL per Cycle: achieved DP add/FMA/mul per cycle (active threads); defines roofline y‑axis.
- Achieved — SM Frequency: SM clock during kernel; used to normalize time/cycles.
- Roofline — DRAM Frequency: assumed memory clock; used to compute theoretical memory roof.
- Roofline — SM Frequency: assumed SM clock; used to compute theoretical compute roof.
- Roofline — Theoretical DRAM Bytes Accessible: theoretical bytes accessible; draws memory roof.
- Roofline — Theoretical Predicated‑On DFMA Operations: theoretical DP FMA ops; draws compute roof.

### Single Precision (SP)
- Achieved — DRAM Bandwidth: measured off‑chip bandwidth during kernel; defines roofline x‑axis.
- Achieved — Predicated‑On FADD/FFMA/FMUL per Cycle: achieved SP add/FMA/mul per cycle (active threads); defines roofline y‑axis.
- Achieved — SM Frequency: SM clock during kernel; used to normalize time/cycles.
- Roofline — DRAM Frequency: assumed memory clock; used to compute theoretical memory roof.
- Roofline — SM Frequency: assumed SM clock; used to compute theoretical compute roof.
- Roofline — Theoretical DRAM Bytes Accessible: theoretical bytes accessible; draws memory roof.
- Roofline — Theoretical Predicated‑On FFMA Operations: theoretical SP FMA ops; draws compute roof.

## GPU Speed Of Light Throughput
Purpose: high‑level throughput indicators across SM and memory subsystems.

- Compute (SM) Throughput (%) — achieved SM pipeline throughput vs peak; spot compute saturation.
- DRAM Frequency (Ghz) — device memory clock; estimate theoretical bandwidth.
- DRAM Throughput (%) — achieved bandwidth vs peak; gauge memory pressure.
- Duration (us) — kernel runtime; baseline performance.
- Elapsed Cycles (cycle) — total SM cycles elapsed; cycle‑level normalization.
- L1/TEX Cache Throughput (%) — L1/TEX traffic vs peak; identify L1/TEX bottlenecks.
- L2 Cache Throughput (%) — L2 traffic vs peak; identify L2 saturation.
- Memory Throughput (%) — aggregate memory throughput vs peak; assess memory‑bound behavior.
- SM Active Cycles (cycle) — cycles with any active warp; compute utilization and IPC.
- SM Frequency (Ghz) — SM clock; translate cycles to time.

## Memory Workload Analysis
Purpose: cache effectiveness and memory subsystem utilization.

- L1/TEX Hit Rate (%) — fraction of memory requests served by L1/TEX; assess locality/coalescing.
- L2 Compression Ratio — effective L2 compression factor (>1 implies compression); understand off‑chip traffic reduction.
- L2 Hit Rate (%) — fraction served by L2; estimate DRAM traffic.
- Max Bandwidth (%) — peak fraction of theoretical bandwidth observed; closeness to memory roof.
- Mem Busy (%) — fraction of cycles with memory subsystem busy; assess latency hiding needs.
- Memory Throughput (Gbyte/s) — absolute off‑chip bandwidth; compare with device specs.
- Mem Pipes Busy (%) — cycles with memory pipelines occupied; detect pipeline saturation.

## Occupancy
Purpose: active‑warp capacity and resource‑limited residency on SMs.

- Achieved Active Warps Per SM (warp) — average resident warps active per SM; judge latency‑hiding capacity.
- Achieved Occupancy (%) — achieved active warps / theoretical maximum; spot imbalance/under‑utilization.
- Block Limit Barriers (block) — max resident blocks limited by barrier resources; observe barrier effects.
- Block Limit Registers (block) — max resident blocks limited by registers; tune launch shape/register usage.
- Block Limit Shared Mem (block) — max resident blocks limited by shared memory; tune smem size.
- Block Limit SM (block) — architectural cap on concurrent blocks per SM; understand concurrency limits.
- Block Limit Warps (block) — limits from warps/block vs SM capacity; adjust block size.
- Cluster Occupancy (%) — achieved occupancy with thread‑block clusters; evaluate cluster config (Hopper+).
- Max Active Clusters (cluster) — theoretical max clusters given resources; size cluster concurrency.
- Max Cluster Size (block) — blocks per cluster; understand cluster granularity.
- Overall GPU Occupancy (%) — time‑weighted occupancy across SMs; whole‑GPU view.
- Theoretical Active Warps per SM (warp) — max active warps from launch/resources; occupancy ceiling.
- Theoretical Occupancy (%) — theoretical occupancy; target/reference.

## Scheduler Statistics
Purpose: instruction issue efficiency and latency hiding at the warp scheduler.

- Active Warps Per Scheduler (warp) — resident warps in each scheduler’s pool; ensure sufficient warp supply.
- Eligible Warps Per Scheduler (warp) — warps ready to issue per cycle; low values indicate stalls and poor latency hiding.
- Issued Warp Per Scheduler — average warps issuing instructions per cycle; assess issue slot utilization.
- No Eligible (%) — cycles with no eligible warp; memory/dep stalls likely.
- One or More Eligible (%) — cycles with at least one eligible warp; steadier issuance is better.

## Report Metadata
Purpose: helper columns used for table rendering; not performance metrics.

- Body Item Label — Metric Name — metadata mapping for table rendering.
- Metric Name — Metric Unit — metadata mapping for table rendering.
