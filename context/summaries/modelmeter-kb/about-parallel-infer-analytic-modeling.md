# About parallel inference analytic modeling (ModelMeter)

## HEADER
- **Purpose**: Explain ModelMeter's multi-device inference cost model helpers (DP/EP/SP), what they compute, and the main correctness limits for multi-GPU and multi-host setups.
- **Status**: Active
- **Date**: 2026-01-22
- **Dependencies**: extern/modelmeter/models/multi_device_cost.py, extern/modelmeter/layers/com_all2all.py, extern/modelmeter/layers/com_sp.py, extern/modelmeter/devices/gpu.py, extern/modelmeter/models/deepseek_r1.py, extern/modelmeter/analysis/plot_fun.py, extern/modelmeter/examples/ana_deepseek_r1_multi_device.py
- **Target**: Developers using ModelMeter analytic layers to reason about distributed inference scaling and bottlenecks.

## Overview
ModelMeter's core analytic layers provide per-layer estimates for Tensor Core FLOPs (TFLOPs), CUDA-core FLOPs (TFLOPs), and I/O volume (Tb), plus optional communication volume (GB).
The recent "multi-device" additions build a simple cost model that converts those per-layer metrics into an estimated wall-time cost (seconds) under a chosen parallel strategy and a device/network throughput configuration.

The cost model is intentionally lightweight and should be treated as a first-order "bottleneck model" rather than a cycle-accurate simulator.
It is most useful for relative comparisons (e.g., which strategy becomes comm-bound first as sequence length grows) after calibrating bandwidth/throughput numbers to your system.

## Where the functionality lives
- Cost model core: `extern/modelmeter/models/multi_device_cost.py`.
- Communication-only analytic layers: `extern/modelmeter/layers/com_all2all.py` and `extern/modelmeter/layers/com_sp.py`.
- Device parameters (compute and bandwidth): `extern/modelmeter/devices/gpu.py`.
- Example model integration: `extern/modelmeter/models/deepseek_r1.py` (DP+EP and SP+EP variants).
- Plotting helpers for cost breakdown curves: `extern/modelmeter/analysis/plot_fun.py` and `extern/modelmeter/examples/README_cost_plots.md`.

## Core data model
`CostResult` (`extern/modelmeter/models/multi_device_cost.py`) aggregates a few coarse time components (seconds).
- `tensor_cost`: Tensor Core time estimate (tensor TFLOPs / tensor peak TFLOPs per second).
- `cuda_cost`: CUDA-core time estimate (cuda TFLOPs / cuda peak TFLOPs per second).
- `io_cost`: memory-I/O time estimate (I/O Tb / IO TB/s).
- `com_cost`: communication time estimate (communication GB / bandwidth GB/s).
- `total_cost`: a strategy-specific combination of the above (often `max(compute/io) + comm`).
- SP-specific extras: `sp_com_cost` and `flashatten_cost` for breaking down attention SP cases.

## How per-layer metrics are aggregated
`layers_stats(layers)` in `extern/modelmeter/models/multi_device_cost.py` sums these across a list of analytic layers.
- FLOPs: `sum(layer.forward_{tensor,cuda}_core_flops())` (TFLOPs).
- I/O: `sum(layer.forward_cal_io())` (Tb).
- Communication: `sum(layer.forward_communication_memory())` (GB).

This means a "layer group" can include a real compute layer plus a synthetic communication layer (for example, an MoE layer plus an All-to-All comm layer).

## Parallel strategies currently modeled
The dispatcher `multi_device_cost(parallel_type=...)` supports these string values.

### 1) `dp` (data parallel)
Implementation: `dp_cost(...)` in `extern/modelmeter/models/multi_device_cost.py`.
- Assumption: inference DP has no communication cost (each device serves independent batches).
- Effective device utilization is capped by batch size: `effective_devices = min(device_num, batch_size)`.
- Time model: `total_cost = max(tensor_cost, cuda_cost, io_cost)` where each component is divided by `effective_devices`.

Interpretation: DP is modeled as an ideal speedup until batch size becomes the limiter, and the slowest of compute-vs-IO dominates.

### 2) `ep` (expert parallel for MoE)
Implementation: `ep_cost(...)` in `extern/modelmeter/models/multi_device_cost.py`.
- Assumption: MoE expert compute scales with `device_num`, and token dispatch/gather uses an All-to-All-like exchange.
- Compute/IO time model: `max(tensor_cost, cuda_cost, io_cost)` with each component divided by `device_num`.
- Communication time model: `com_cost = comm_gb / bisection_bandwidth_gb_s`.
- Total time model: `total_cost = max(tensor_cost, cuda_cost, io_cost) + com_cost` (no overlap assumed between comm and compute).

Interpretation: EP is modeled as "compute/IO roofline time" plus a serialized All-to-All penalty.

### 3) `attention_sp` (sequence parallel for attention)
Implementation: `attention_sp_cost(...)` in `extern/modelmeter/models/multi_device_cost.py`.
- Assumption: attention has an SP communication step (modeled by a `ComSP` layer) that competes with flash-attention compute.
- Communication time model: `sp_com_cost = comm_gb / p2p_bandwidth_gb_s`.
- For layers whose class name contains `"Attention"`, the code splits attention into two parts:
  - A "flash attention" sublayer cost (via `layer.attn.*`) that is compared against SP comm.
  - A residual attention projection cost (total attention minus flash attention part).
- Total time model: `total_cost += max(sp_com_cost, flashatten_cost)` for the attention core, then adds `max(atten_proj_costs)` for the remaining projections.
- For non-attention layers in the list, it adds `max(layer_tensor_cost, layer_cuda_cost, layer_io_cost)` per layer.

Interpretation: SP attention tries to model overlap between attention-core compute and the SP communication (via `max`), while treating the projection parts as separate roofline-limited chunks.

### 4) `ffn_sp` (sequence parallel for non-attention parts)
Implementation: `ffn_sp_cost(...)` in `extern/modelmeter/models/multi_device_cost.py`.
- Assumption: for these layer groups, SP introduces no explicit communication bottleneck (or it is ignored).
- Time model is the same roofline max as DP/EP compute: `total_cost = max(tensor_cost, cuda_cost, io_cost)` with components divided by `device_num`.

Interpretation: this is effectively "perfect scaling without comm" for the provided layer list.

## Communication volume models (synthetic layers)
The comm layers do not contribute FLOPs or I/O; they only contribute `forward_communication_memory()` in GB.

### `ComAll2All` (MoE token dispatch/gather)
Implementation: `extern/modelmeter/layers/com_all2all.py`.
- Approximates bytes moved as `(cast_bytes + gather_bytes) * B * S * H * topk * route`.
- `route` includes a factor roughly equal to `(n_device - 1) / n_device` (scaled by a `capacity_factor`).

This is intended to approximate the fraction of tokens that must be sent to experts on other devices in a sharded-expert layout.

### `ComSP` (sequence-parallel attention exchange)
Implementation: `extern/modelmeter/layers/com_sp.py`.
- Approximates bytes moved as `(K_dim + V_dim) * prefill_len * B * (n_device - 1) / n_device * bits/8`.

This resembles an all-gather / exchange of K/V slices required by some SP attention variants.

## How the model is applied in DeepSeek-R1
DeepSeek-R1 wires these strategies into end-to-end model estimates in `extern/modelmeter/models/deepseek_r1.py`.
- `dp_ep_performance()`: uses DP for attention blocks, and EP for the MoE blocks (by calling `multi_device_cost(parallel_type='ep', layers=[moe, all2all], ...)`).
- `sp_ep_performance()`: uses SP for attention blocks (by including `ComSP` in the attention group) and EP for MoE blocks.

The example driver `extern/modelmeter/examples/ana_deepseek_r1_multi_device.py` sweeps long prefill lengths, uses chunking for very large sequence lengths, and generates breakdown plots.

## Is this "correct" for multi-GPU multi-host inference?
It is directionally useful but not fully correct as a general multi-host inference performance model.
The current functions are closer to a calibrated, first-order bottleneck estimator than a faithful model of distributed execution.

### What the current model gets reasonably right (if parameters are calibrated)
- It includes a communication-volume term for EP (All-to-All) and for SP (sequence exchange), which is often the dominant new bottleneck when moving off a single GPU.
- It separates compute into Tensor Core FLOPs and CUDA-core FLOPs, and compares both against I/O to pick a dominant bottleneck via `max(...)`.
- It provides a simple hook to express "comm overlaps with attention core compute" (via `max(sp_comm, flashattention)`).

### Major limitations for multi-host accuracy
- Topology is collapsed into a single `p2p_bandwidth` and a single `bisection_bandwidth`, but real systems have hierarchical links (NVLink/PCIe intra-node plus InfiniBand/RoCE inter-node) and non-uniform contention.
- Collective communication time is modeled as `bytes / bandwidth`, which omits latency terms and the scaling behavior of real collectives (for many collectives, time is closer to `alpha * f(p) + beta * bytes * g(p)`).
- The EP path assumes no overlap between communication and computation (`+ com_cost`), which can overestimate if pipelining/overlap is present, but can also underestimate if network contention or synchronization dominates.
- The SP attention overlap model is heuristic and depends on layer objects exposing `layer.attn` for the attention core, and on class-name matching (`"Attention"`), which is fragile across models.
- The compute-side uses peak throughput numbers and assumes perfect scaling with `device_num` inside a layer group, but real kernels have utilization effects, launch overheads, and scheduling constraints that reduce scaling.
- It does not model other common parallel inference schemes (tensor parallel all-reduce, pipeline parallel bubbles, context parallel, KV-cache sharding costs, interleaving, or request-level scheduling).

### Practical takeaway
- For a multi-host setup, treat the current results as relative indicators (which term dominates, and how costs scale with S/B/topk) rather than absolute latency predictions.
- If you want the model to be "correct enough" for a specific cluster, calibrate `p2p_bandwidth` and `bisection_bandwidth` using microbenchmarks of the exact collective patterns you expect (including inter-node cases), and accept that the model still omits latency and contention effects.

### Suggested next steps to improve correctness
- Add a topology-aware bandwidth model (separate intra-node and inter-node links, and a hierarchical collective cost).
- Introduce an `alpha + beta` collective model for All-to-All and All-Gather (and optionally All-Reduce) rather than `bytes/bw` only.
- Make overlap behavior explicit and configurable (for example, `overlap_comm_with_compute: none|partial|max` per strategy).
- Add first-class support for tensor parallel collectives (all-reduce / reduce-scatter / all-gather) and pipeline parallel bubble modeling.
- Validate the cost model against measured TTFT/TPOT on representative clusters and fit effective parameters per regime (small vs large messages, intra-node vs inter-node).
