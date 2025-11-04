# Metrics to Watch
- 算力单元利用率 (SM/Tensor/FP32)
- Cache命中率 (L1/TEX/L2)
- 内存带宽 (DRAM 吞吐占峰值)
- Warp 效率 (Barrier/Short/Long Scoreboard stalls)
- Kernel耗时 (来自 cuda_gpu_kern_sum)
- 整体推理瓶颈 (基于SM/DRAM/Cache的启发式:Compute/Mermory / Mixed / Memory/Latency)