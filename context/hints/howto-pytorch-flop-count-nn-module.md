# How to count FLOPs for a PyTorch `nn.Module` using built‑in tools

This hint explains practical ways to measure floating point operations (FLOPs) for a PyTorch `nn.Module` using PyTorch’s own utilities (no third‑party FLOP libraries). It is meant as a quick reference for analytic modeling and sanity checks.

## Overview

PyTorch currently exposes two primary FLOP‑related mechanisms:

- **`torch.utils.flop_counter.FlopCounterMode`**  
  A built‑in FLOP counter that instruments tensor operations using `__torch_dispatch__` and can return a scalar FLOP count for a model and input.
- **`torch.profiler.profile(..., with_flops=True)`**  
  The PyTorch profiler can report FLOPs per operator alongside timing information, where supported.

These tools are version‑dependent and may be considered experimental; always check against your installed PyTorch version.

---

## Option 1 – `torch.utils.flop_counter.FlopCounterMode`

`FlopCounterMode` is the simplest way to get a scalar FLOP count for a forward or forward+backward pass of a model, using representative inputs.

### Minimal helper

```python
from typing import Union, Tuple

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def count_flops(
    model: nn.Module,
    inp: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    with_backward: bool = False,
) -> int:
    """Return total FLOPs for a model on a single input."""

    was_training = model.training
    model.eval()

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)

    # Normalize to a single Tensor input for simplicity
    if isinstance(inp, torch.Tensor):
        example = inp
    else:
        example = inp[0]

    with flop_counter:
        if with_backward:
            out = model(example)
            out.sum().backward()
        else:
            _ = model(example)

    total_flops: int = flop_counter.get_total_flops()

    if was_training:
        model.train()
    return total_flops
```

### Example: ResNet‑18 FLOPs

```python
from torchvision.models import resnet18
import torch

model = resnet18()
inp = torch.randn(1, 3, 224, 224)

fwd_flops = count_flops(model, inp, with_backward=False)
fwd_bwd_flops = count_flops(model, inp, with_backward=True)

print(f"ResNet-18 forward FLOPs: {fwd_flops}")
print(f"ResNet-18 forward+backward FLOPs: {fwd_bwd_flops}")
```

### Notes and limitations

- **Version dependence**: `torch.utils.flop_counter.FlopCounterMode` is only available in recent PyTorch versions and may change; treat it as an internal/experimental API.
- **Coverage**: FLOPs are computed from tensor operations seen by `__torch_dispatch__`. Custom C++ ops or unusual autograd patterns may not be fully captured.
- **Inputs matter**: FLOPs depend on tensor shapes (batch, sequence/image size). Always use inputs that match your target workload (e.g., 1×3×640×640 vs 1×3×224×224).
- **Forward vs backward**: Forward‑only counts are useful for analytic compare/contrast; forward+backward counts are better for training cost estimates.

---

## Option 2 – `torch.profiler` with FLOP reporting

If you already use `torch.profiler` for performance analysis, you can enable FLOP reporting per operator and aggregate results.

### Basic profiler pattern

```python
import torch
from torch import nn


def profile_with_flops(model: nn.Module, inp: torch.Tensor) -> None:
    model.eval()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_flops=True,  # request FLOPs if supported
    ) as prof:
        _ = model(inp)

    # Summarize top operators by FLOPs
    print(prof.key_averages().table(sort_by="flops", row_limit=20))
```

### When to use the profiler approach

- You want **per‑operator** FLOP breakdowns (e.g., how much is in `addmm` vs `bmm` vs `sdpa`).
- You are already capturing profiler traces and want FLOPs alongside latency.
- You need to calibrate your analytic layer models against actual kernel behavior (including fusions).

### Caveats

- `with_flops=True` support may depend on PyTorch and CUDA versions; in some setups FLOPs can be missing or zero.
- Profiler FLOPs are **implementation‑specific**: they reflect the actual backend kernels (e.g., fused SDPA) rather than purely mathematical FLOPs.
- Profiling incurs overhead; use small batches or reduced iterations when you only need FLOP counts.

---

## How to use these tools for analytic modeling

- **Validate analytic formulas**  
  Use `FlopCounterMode` on an isolated module (e.g., an attention block or MLP) with the same shapes your analytic model targets. Compare:
  - Absolute FLOP counts.
  - Scaling behavior vs. sequence length / image size / batch size.

- **Check scaling vs resolution**  
  For vision models, run FLOP counting at multiple image sizes or window configurations (e.g., global 64×64 vs 25 windows of 14×14) to confirm the expected `O(S²)` attention scaling and windowing benefits.

- **Align counting conventions**  
  Ensure your hand‑derived formulas use the same convention as PyTorch:
  - Typically **2 FLOPs per multiply‑add** (MAC = 2 FLOPs).
  - Some literature counts 1 FLOP per MAC; be explicit when comparing.

- **Use profiler data to refine models**  
  If profiler FLOPs consistently differ from your analytic numbers for certain ops (e.g., fused SDPA, fused MLP), consider:
  - Adding correction factors for those ops.
  - Modeling them as fused kernels instead of sums of simpler primitives.

---

## References

- PyTorch built‑in FLOPs counter (blog walkthrough with code)  
  **Alessio Devoto**, *“Flops with Pytorch built-in flops counter”*  
  https://alessiodevoto.github.io/Compute-Flops-with-Pytorch-built-in-flops-counter/

- PyTorch dev discussion on an “ideal” FLOP counter using `__torch_dispatch__`  
  https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505

