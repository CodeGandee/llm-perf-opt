# How to model Tensor Core vs CUDA core FLOPs for PyTorch modules

This hint explains how to split analytic FLOPs between Tensor Cores and CUDA cores for PyTorch models, and how to validate that split against runtime profiling. It is written in terms of methods named `forward_tensor_core_flops()` and `forward_cuda_core_flops()`, but the ideas apply to any analytic interface that distinguishes Tensor-Core vs CUDA-core work.

Where useful, we reference concrete examples such as vision transformer attention blocks (`Attention`), MLP blocks (`MLPBlock` / `FeedForward`), LayerNorms (`LayerNorm2d`), and patch-embedding convolutions (`PatchEmbed`) as patterns rather than as the main focus.

---

## 1. Concept and definitions

In a typical analytic cost model, each layer exposes:

- `forward_tensor_core_flops()`: TFLOPs expected to run on Tensor Cores (matrix-multiply pipelines).
- `forward_cuda_core_flops()`: TFLOPs expected to run on “regular” CUDA cores (FMA / FP32/FP16/INT pipelines).

Splitting FLOPs this way is:

- An **analytic approximation** based on the math and typical implementation of the op.
- Intended to align with how vendor libraries (cuBLAS, cuDNN) and PyTorch map ops to hardware when using fp16/bf16/TF32 on Volta/Ampere+ GPUs.

You should:

- Treat the split as **by operation type** (matmul/conv vs elementwise/reduction), not by guessing individual kernels.
- Use profiling (PyTorch FLOP counters + Nsight Compute) to calibrate and sanity-check the split.

---

## 2. What goes to Tensor Cores vs CUDA cores?

### 2.1 High-level rule of thumb

- Put FLOPs in `forward_tensor_core_flops()` when they come from:
  - Dense GEMMs and batched matmuls (e.g., `nn.Linear`, `torch.matmul`, attention QKV and output projections).
  - Conv2d layers (`nn.Conv2d`, `F.conv2d`) and related conv-like kernels.
  - Scaled dot-product attention matmuls (`QK^T` and `AV`) when implemented via tensor-core GEMMs/SDPA.
- Put FLOPs in `forward_cuda_core_flops()` when they come from:
  - Elementwise ops: activations (GELU, ReLU, SiLU), bias adds, residual adds, masking, scaling.
  - Reductions: LayerNorm/RMSNorm/BatchNorm, softmax, log-softmax, sum/mean operations.
  - Cheap bookkeeping kernels (indexing, small reshapes) if you choose to model them at all.

This matches common practice:

- Tensor Cores: large matmuls and convs in fp16/bf16/TF32.
- CUDA cores: scalar work, elementwise transforms, reductions, normalizations.

### 2.2 Examples from a vision-transformer-style stack

The following archetypal modules (common in ViT-like or transformer architectures) illustrate the split. The definitions are given in torch-like pseudo code so you can recognize them in your own models.

- Patch embedding convolution
  - Pseudo code:

    ```python
    class PatchEmbed(nn.Module):
        def __init__(self, in_chans: int, embed_dim: int, patch_size: int) -> None:
            super().__init__()
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C_in, H, W) -> (B, C_out, H_out, W_out)
            return self.proj(x)
    ```

  - Conceptually: a single `Conv2d` that turns images `(B, C_in, H, W)` into patch embeddings `(B, C_out, H_out, W_out)` with kernel size and stride equal to the patch size.
  - Tensor Core FLOPs:
    - Use the standard conv formula  
      `2 * B * H_out * W_out * kernel_h * kernel_w * C_in * C_out`, converted to TFLOPs.
  - CUDA core FLOPs:
    - Often modeled as `0.0`, treating the conv as Tensor-Core-dominated when using fp16/bf16/TF32 and aligned shapes.

- Multi-head self-attention block
  - Pseudo code:

    ```python
    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
            self.proj = nn.Linear(embed_dim, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, S, C)
            B, S, C = x.shape
            qkv = self.qkv(x)                            # (B, S, 3C)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, S, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k.view(B, S, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = v.view(B, S, self.num_heads, C // self.num_heads).transpose(1, 2)
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(B, S, C)
            return self.proj(attn)                       # (B, S, C)
    ```

  - Conceptually: takes tokens of shape `(B, S, C)` and performs QKV projections, scaled dot-product attention, and an output projection.
  - Tensor Core FLOPs:
    - QKV projections: dense matmuls from `C` to `3C`.
    - SDPA matmuls (`QK^T` and `AV`): `2 * B * heads * S * S * head_dim` each.
    - Output projection: dense matmul from `C` to `C`.
  - CUDA core FLOPs:
- MLP / feed-forward block
  - Pseudo code:

    ```python
    class FeedForward(nn.Module):
        def __init__(self, dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, S, dim)
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            return x
    ```

  - Conceptually: two dense layers with a nonlinearity in between, mapping `(B, S, dim)` → `(B, S, hidden_dim)` → `(B, S, dim)`.
  - Tensor Core FLOPs:
    - Two matmuls (`dim → hidden_dim`, `hidden_dim → dim`) each using `2 * B * S * dim * hidden_dim` FLOPs.
  - CUDA core FLOPs:
    - Activations (e.g., GELU, QuickGELU) modeled as  
      `activation_flops_per_element * B * S * hidden_dim`.

- Layer normalization
  - Pseudo code:

    ```python
    class LayerNorm1d(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.ln = nn.LayerNorm(hidden_size, eps=eps)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, S, hidden_size)
            return self.ln(x)
    ```

  - Conceptually: normalizes activations along the channel/feature dimension (e.g., `nn.LayerNorm`, 1D LayerNorm, or 2D variants).
  - Tensor Core FLOPs:
    - Modeled as `0.0` (LayerNorm is dominated by scalar arithmetic, not matmuls).
  - CUDA core FLOPs:
- Aggregator modules (transformer blocks and stacks)
  - Pseudo code:

    ```python
    class TransformerBlock(nn.Module):
        def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.attn = MultiHeadSelfAttention(dim, num_heads)
            self.ln2 = nn.LayerNorm(dim)
            self.mlp = FeedForward(dim, int(dim * mlp_ratio))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x


    class TransformerEncoder(nn.Module):
        def __init__(self, block: nn.Module, depth: int) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([copy_of(block) for _ in range(depth)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for blk in self.blocks:
                x = blk(x)
            return x
    ```

  - Conceptually: compose attention + MLP + LayerNorm (and possibly window partition/unpartition) into a single block, and stack many such blocks.
  - Tensor Core FLOPs:
    - Simply sum Tensor Core FLOPs from submodules (attention, MLP, patch embedding, conv neck).
  - CUDA core FLOPs:
    - Sum CUDA-core FLOPs from submodules (activations, norms, softmax, elementwise ops).
    - Optionally add explicit CUDA-core terms for LayerNorm and any window-partition overhead if not modeled inside submodules.

These patterns show how to split by operation type; any PyTorch model can be decomposed into similar matmul/conv vs elementwise/reduction components.

---

## 3. Writing `forward_tensor_core_flops()` and `forward_cuda_core_flops()` for your own modules

### 3.1 General recipe

For a PyTorch module that you want to model analytically:

1. Identify the **matrix multiplies / convs / SDPA** inside the forward pass and derive their FLOP counts using shapes (batch, sequence length, channels, head dim).
2. Identify **elementwise / reduction ops** (norms, softmax, activations, residual adds).
3. Decide which parts you consider **Tensor-Core dominated** vs **CUDA-core dominated** based on:
   - Operation type (matmul/conv vs elementwise).
   - Data types (fp16, bf16, TF32, fp32).
   - Alignment and size rules (e.g., dimensions multiple-of-8 / multiple-of-16).
4. Encode:
   - Tensor Core FLOPs in `forward_tensor_core_flops()` (return TFLOPs).
   - CUDA-core FLOPs in `forward_cuda_core_flops()` (also TFLOPs).
5. For backward:
   - Either derive explicit formulas or approximate as `~2×` forward FLOPs for each path.

### 3.2 Example: MLP block (two linears + GELU)

Abstracting from `MLPBlock`:

```python
class MyMlpBlock(BaseLayer):
    def __init__(self, *, batch: int, seq_len: int, dim: int, hidden_dim: int, activation_flops_per_element: float = 8.0) -> None:
        ...
        self.m_batch = batch
        self.m_seq_len = seq_len
        self.m_dim = dim
        self.m_hidden_dim = hidden_dim
        self.m_activation_flops_per_element = activation_flops_per_element

    def _num_tokens(self) -> float:
        return float(self.m_batch) * float(self.m_seq_len)

    def forward_tensor_core_flops(self) -> float:
        \"\"\"Dense layers (dim→hidden_dim→dim) are modeled as Tensor-Core matmuls: 2 * B * S * dim * hidden_dim for each projection, returned in TFLOPs.\"\"\"
        tokens = self._num_tokens()
        d = float(self.m_dim)
        h = float(self.m_hidden_dim)
        flops_fc1 = 2.0 * tokens * d * h
        flops_fc2 = 2.0 * tokens * h * d
        return (flops_fc1 + flops_fc2) / 1.0e12

    def forward_cuda_core_flops(self) -> float:
        \"\"\"GELU is treated as CUDA-core elementwise work: activation_flops_per_element * B * S * hidden_dim, returned in TFLOPs.\"\"\"
        tokens = self._num_tokens()
        h = float(self.m_hidden_dim)
        flops = self.m_activation_flops_per_element * tokens * h
        return flops / 1.0e12
```

### 3.3 Example: Attention block

Abstracting from `Attention` / `NoTPAttention`:

```python
def forward_tensor_core_flops(self) -> float:
    \"\"\"QKV projections + SDPA matmuls + output projection treated as Tensor-Core matmuls (2 FLOPs per MAC, returned in TFLOPs).\"\"\"
    b = float(self.m_num_windows or self.m_batch_size)
    s = float(self._seq_len())
    c = float(self.m_dim)
    h = float(self.m_num_heads)
    d = c / h
    qkv_flops = 2.0 * b * s * c * (3.0 * c)
    sdpa_flops = 2.0 * b * h * s * s * d * 2.0  # QK^T + AV
    proj_flops = 2.0 * b * s * c * c
    return (qkv_flops + sdpa_flops + proj_flops) / 1.0e12

def forward_cuda_core_flops(self) -> float:
    \"\"\"Optionally account for softmax, scaling, masking as CUDA-core FLOPs if needed; many analytic models treat them as negligible and return 0.0.\"\"\"
    return 0.0
```

### 3.4 Example: LayerNorm

Abstracting from `LayerNorm2d` and final LayerNorms in `VitModel`:

```python
def forward_tensor_core_flops(self) -> float:
    \"\"\"LayerNorm is modeled as a CUDA-core-dominated operation, so Tensor Core FLOPs are taken to be zero.\"\"\"
    return 0.0

def forward_cuda_core_flops(self) -> float:
    \"\"\"Approximate LayerNorm as ~5 * B * C * H * W scalar FLOPs on CUDA cores, returned in TFLOPs.\"\"\"
    elems = float(batch) * float(channels) * float(height) * float(width)
    flops = 5.0 * elems
    return flops / 1.0e12
```

### 3.5 Example: Aggregator modules (transformer block, encoder stack, full model)

Aggregator modules (e.g., `Block`, `NoTPTransformerBlock`, `NoTPTransformer`, `VitModel`, `ImageEncoderViT`) follow a simple pattern:

- `forward_tensor_core_flops()`:
  - Sum Tensor Core FLOPs from submodules (attention, MLP, patch embedding, conv neck, etc.).
  - Do not re-attribute normalization or elementwise FLOPs here.
- `forward_cuda_core_flops()`:
  - Sum CUDA-core FLOPs from submodules.
  - Add any explicit LayerNorm/window/other overhead that is not modeled inside submodules.

This keeps each layer responsible for its own math, and higher-level aggregators just add things up.

---

## 4. Verifying total FLOPs (Tensor + CUDA cores) vs PyTorch

To check that your analytic FLOPs (Tensor Core + CUDA core) match what a PyTorch implementation does, you can use `torch.utils.flop_counter.FlopCounterMode` on the reference module with a representative input.

### 4.1 Generic verification pattern

```python
from typing import Any

import torch
from torch.utils.flop_counter import FlopCounterMode


def verify_layer_flops(
    analytic_layer: Any,
    reference_module: torch.nn.Module,
    example_input: torch.Tensor,
    *,
    device: str = "cuda:0",
    accept_rel_diff: float = 0.05,
    compensate_missing_sdpa: bool = False,
    sdpa_analytic_flops: float = 0.0,
) -> None:
    \"\"\"Compare analytic FLOPs to PyTorch FLOP counter for a layer or block.

    - Analytic FLOPs are taken as (forward_tensor_core_flops + forward_cuda_core_flops) * 1e12.
    - Measured FLOPs come from FlopCounterMode on the reference_module.
    - Optionally, you can compensate for missing SDPA FLOPs if the profiler does not see fused scaled_dot_product_attention kernels.
    \"\"\"

    reference_module = reference_module.to(device).eval()
    example_input = example_input.to(device)

    flop_counter = FlopCounterMode(mods=reference_module, display=False, depth=None)
    with flop_counter:
        _ = reference_module(example_input)
    measured = flop_counter.get_total_flops()

    if compensate_missing_sdpa:
        measured += sdpa_analytic_flops  # e.g., from an Attention analytic layer

    analytic_tflops = (analytic_layer.forward_tensor_core_flops() or 0.0) + (analytic_layer.forward_cuda_core_flops() or 0.0)
    analytic = analytic_tflops * 1.0e12

    rel_diff = abs(analytic - measured) / float(measured)
    if rel_diff > accept_rel_diff:
        raise AssertionError(f\"Analytic vs measured FLOPs differ by {rel_diff:.3%} (analytic={analytic:.3e}, measured={measured:.3e})\")
```

Notes:

- PyTorch’s flop counter may miss FLOPs for fused ops like `scaled_dot_product_attention`. In such cases you can:
  - Manually compute SDPA FLOPs from your analytic `Attention` layer.
  - Add them to the measured FLOPs (`compensate_missing_sdpa=True`).
- For convs and linears, FlopCounterMode generally matches analytic formulas, so the sum `(forward_tensor_core_flops + forward_cuda_core_flops)` should align with measured FLOPs within a modest tolerance for typical layers/blocks.

---

## 5. Verifying Tensor vs CUDA core usage with Nsight Compute (ncu)

Total FLOPs matching PyTorch does not guarantee that those FLOPs are executed on Tensor Cores vs CUDA cores as you modeled. To confirm the pipeline usage, use Nsight Compute.

### 5.1 High-level workflow

1. Run a small workload that exercises your model or layer (e.g., a few batches, minimal tokens or image size).
2. Profile with `ncu`, focusing on:
   - Tensor-pipe utilization (Tensor Cores).
   - SM/FMA utilization (CUDA cores).
   - FLOP metrics (`flop_count_hp`, `flop_count_sp`, etc.).
3. Check that:
   - Kernels you model as Tensor-Core dominated (GEMMs, convs, SDPA) show high Tensor-pipe utilization.
   - Kernels you model as CUDA-core dominated (LayerNorm, activations, softmax) have negligible Tensor-pipe utilization but non-zero FMA/FP32 utilization.

### 5.2 Generic `ncu` CLI examples

Minimal SpeedOfLight + ComputeWorkloadAnalysis run:

```bash
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name '.*(gemm|s168|sdpa|attention|conv).*' \
  --section SpeedOfLight \
  --section ComputeWorkloadAnalysis \
  --replay-mode kernel \
  -o tmp/ncu/model_tc_vs_cuda \
  <your-python-or-binary-command>
```

Metric-focused run to compare Tensor vs FP32 pipelines:

```bash
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name '.*gemm.*' \
  --metrics \
    flop_count_hp, \
    flop_count_sp, \
    sm__pipe_tensor_active.avg.pct_of_peak_sustained_elapsed, \
    sm__pipe_fp32_active.avg.pct_of_peak_sustained_elapsed \
  --replay-mode kernel \
  -o tmp/ncu/model_flops_pipelines \
  <your-python-or-binary-command>
```

Interpretation:

- For Tensor-Core-dominated kernels (matmuls/conv/SDPA):
  - Expect high `sm__pipe_tensor_active` and significant `flop_count_hp`/`flop_count_sp` associated with tensor-core instructions.
- For CUDA-core-dominated kernels (LayerNorm, activations):
  - Expect `sm__pipe_tensor_active` near zero and non-zero `sm__pipe_fp32_active` (or related FP/INT pipelines).

References:

- Nsight Compute CLI docs and metric definitions: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- Nsight Compute profiling guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- NVIDIA forum on Tensor Core utilization metrics: https://forums.developer.nvidia.com/t/am-i-using-tensor-core/185492

---

## 6. Which PyTorch ops typically run on Tensor Cores vs CUDA cores?

This section summarizes what is commonly known (from PyTorch docs and NVIDIA forums) about which layers/ops are executed on Tensor Cores vs the “regular” FP/INT pipelines. It is approximate and hardware-dependent; always validate on your target GPU.

### 6.1 Tensor-Core-friendly ops

Widely known to use Tensor Cores on Volta/Ampere+ when data types and shapes are suitable:

- `torch.matmul`, `torch.mm`, `torch.bmm`, `nn.Linear`, `F.linear`
  - For fp16 / bf16 on Volta and newer, and for fp32 on Ampere+ when TF32 is enabled.
  - PyTorch devs state that “fp16 should use tensor cores by default for common ops like matmul and conv”; for Ampere+ fp16/bf16 should use Tensor Cores for common ops and fp32 convs use Tensor Cores via TF32.
- `nn.Conv2d`, `F.conv2d`
  - On Ampere and newer, fp32 convolutions run in TF32 on Tensor Cores by default unless TF32 is disabled (see `torch.backends.cudnn.conv.fp32_precision` / `torch.backends.cudnn.allow_tf32`).
  - fp16/bf16 convs also map to tensor-core conv kernels when batch size and in/out channels are multiples of 8/16 (exact conditions are arch- and kernel-dependent).
- SDPA and attention
  - `torch.nn.functional.scaled_dot_product_attention` and transformer attention implementations (including FlashAttention-style kernels) are typically built on tensor-core matmuls internally, so the core QKᵀ / AV matmuls run on Tensor Cores for fp16/bf16/TF32.

Key configuration knobs:

- `torch.set_float32_matmul_precision("high" | "medium" | "highest")`
  - Controls whether float32 matmuls can use TF32 / bfloat16-based algorithms on Tensor Cores.
  - Docs: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
- `torch.backends.cuda.matmul.fp32_precision` and `torch.backends.cudnn.conv.fp32_precision`
  - Control TF32 usage for matmuls and convolutions.
  - See the “Numerical accuracy” note: https://pytorch.org/docs/stable/notes/numerical_accuracy.html

References (PyTorch forums summary):

- “For Volta: fp16 should use tensor cores by default for common ops like matmul and conv. For Ampere and newer, fp16, bf16 should use tensor cores for common ops and fp32 for convs (via TF32).”  
  https://discuss.pytorch.org/t/does-pytorch-use-tensor-cores-by-default/167676
- Conditions for fp16/bf16 GEMM/conv (multiples-of-8 sizes):  
  https://discuss.pytorch.org/t/how-do-i-know-that-tensor-cores-used-in-pytorch-for-fp16-bfloat16-int8/169763

### 6.2 Ops that usually stay on CUDA cores

Elementwise / reduction ops that are not implemented as Tensor Core matmuls generally run on the standard FP/INT pipelines (“CUDA cores”):

- Normalization: `nn.LayerNorm`, `nn.BatchNorm*`, project-specific `LayerNorm2d`.
- Activations: `nn.ReLU`, `nn.GELU`, `nn.SiLU`, `F.relu`, `F.gelu`, etc.
- Per-element arithmetic: add, mul, div, clamp, etc.
- Softmax and log-softmax reductions.
- Indexing / gather / scatter, some pooling variants, many small kernels.

Some frameworks may use Tensor Cores indirectly for certain fused ops, but the dominant FLOPs for these layers are modeled safely as CUDA-core FLOPs at this analytic level.

---

## 7. How do people know which cores a given op uses, and can you infer FLOPs from kernels alone?

### 7.1 How practitioners determine Tensor vs CUDA core usage

In practice, people do not rely only on high-level layer names; they combine:

- Profiler metrics (Nsight Compute / Nsight Systems)
  - Tensor Core utilization metrics, e.g.:
    - `sm__pipe_tensor_active.avg.pct_of_peak_sustained_elapsed` (Nsight Compute; older nvprof metric `tensor_precision_fu_utilization` is similar).
    - HMMA instruction counts: `smsp__sass_thread_inst_executed_op_hmma_pred_on.sum`.
  - FLOP counts (`flop_count_hp`, `flop_count_sp`, etc.) broken down by kernel.
  - See “Am I using Tensor Core?”:  
    https://forums.developer.nvidia.com/t/am-i-using-tensor-core/185492
- CUDA SASS / PTX inspection
  - Disassembling kernels with `nvdisasm` or `cuobjdump` and scanning for Tensor Core instructions (HMMA ops).
  - NVIDIA guidance on inspecting HMMA usage:  
    https://forums.developer.nvidia.com/t/is-there-a-way-to-see-if-cuda-api-execution-happened-on-tensor-cores-or-not/56403
- Library and framework docs
  - cuBLAS/cuDNN docs indicate which GEMM/conv paths use Tensor Cores for which data types and shapes (e.g., `CUBLAS_GEMM_DEFAULT_TENSOR_OP`).
  - NVIDIA Tensor Core programming blog for supported WMMA shapes:  
    https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/

These tools collectively answer “is this op actually using Tensor Cores on my workload and GPU?” better than static reasoning alone.

### 7.2 Is the mapping “layer → kernel → Tensor vs CUDA core FLOPs” exact?

Linking each PyTorch layer/op to its underlying CUDA kernels (e.g., via source code or profiler kernel names) is helpful, but there are important limitations:

- One high-level op ≠ one kernel
  - `nn.Linear`, attention, or an entire transformer block often fuse multiple logical operations into a single kernel (or split into several kernels) depending on backend and version.
  - Fused kernels may mix Tensor Core matmuls with CUDA-core elementwise work; Nsight metrics typically report aggregate utilization for the whole kernel.
- Autotuning and dynamic choices
  - cuBLAS/cuDNN and PyTorch choose kernels based on tensor shapes, strides, alignment, and heuristics; small shape changes may flip between Tensor-Core and non–Tensor-Core implementations.
  - Hitting or missing “multiple-of-8/16” conditions can determine whether GEMM/conv uses Tensor Cores at all.
- Instruction-level vs FLOP-level attribution
  - Even if you know a kernel uses Tensor Cores (via HMMA instructions), the hardware may still use FP32/INT ALUs for surrounding work.
  - Nsight’s `flop_count_*` and pipeline utilization metrics give a measured split of FLOPs across pipelines, but mapping that back to each analytic layer requires careful correlation of kernels to layers (often via NVTX ranges).

Implications for analytic modeling:

- You can approximate Tensor vs CUDA core FLOPs per layer by:
  - Modeling matmul/conv/SDPA math as Tensor-Core FLOPs.
  - Modeling normalization/activations and remaining elementwise work as CUDA-core FLOPs.
  - Validating at kernel granularity with Nsight (`SpeedOfLight`, `ComputeWorkloadAnalysis`, `flop_count_*`, Tensor-pipe metrics).
- You cannot, in general, derive an exact per-layer Tensor vs CUDA core FLOP breakdown purely from static kernel inspection, because:
  - Kernels can be reused across layers.
  - Kernel fusion and dynamic dispatch obscure a 1:1 correspondence.
  - Some FLOPs are encapsulated inside library kernels whose internal split isn’t exposed.

---

## 8. Is there a global catalog of Tensor-Core vs CUDA-core usage for all PyTorch ops?

Based on public docs and discussion, there is no widely published, exhaustive table that says “every PyTorch op X always uses Tensor Cores / CUDA cores.” Instead:

- Per-model tooling (DLProf)
  - NVIDIA’s Deep Learning Profiler (DLProf) integrates with Nsight Systems/Compute to analyze a given model run, and exposes:
    - An “All Ops” summary (aggregated over all operations).
    - A “Tensor Core Report” listing:
      - Ops using Tensor Core kernels.
      - Ops eligible for Tensor Cores but not using them.
      - Other ops.
  - DLProf infers Tensor Core usage from the actual kernels seen in the Nsight Systems trace (kernel names, cuDNN/cuBLAS patterns), so results are model- and workload-specific.
  - Docs: https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html
- Targeted microbenchmarks and papers
  - Some research systematically profiles specific kernel families (e.g., GEMM/conv microbenchmarks, a few end-to-end models) with Nsight Compute to measure Tensor Core FLOPs and utilization.
  - These works do not cover the full PyTorch op set; they focus on representative compute-heavy ops.
- Community practice
  - Practitioners typically:
    - Apply rules of thumb (sections 2 and 6) about GEMM/conv/SDPA vs norm/activation ops.
    - Use Nsight Compute / DLProf dynamically on their own models to see which ops are actually hitting Tensor Cores.
    - Adjust data types, shapes, and backend flags (TF32, AMP) to improve Tensor Core coverage.

Takeaway for analytic modeling:

- There is no static, authoritative “op → Tensor vs CUDA core” map to import.
- The best you can do is:
  - Use analytic formulas and operation-type heuristics at the layer level (as shown in the examples above).
  - Validate and refine them with runtime tools like PyTorch’s FLOP counters, Nsight Compute, and DLProf for the specific workloads and GPUs you care about.
