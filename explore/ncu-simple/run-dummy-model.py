"""Minimal CUDA workload for Nsight Compute testing.

This script runs the tiny ShallowResNet model directly (no runners),
initializes weights randomly, and executes a fixed number of forward
passes on cuda:0 with random inputs. The model itself emits NVTX ranges
inside its forward method (stem, residual, head), enabling range replay
tests with `ncu`.

Example usage:
  python explore/ncu-simple/run-dummy-model.py --iters 100 --batch-size 8

Example ncu (range replay over the 'stem' NVTX range):
  ncu --target-processes all --replay-mode range \
      --nvtx --nvtx-include stem \
      python explore/ncu-simple/run-dummy-model.py --iters 100 --batch-size 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Final

import torch
import torch.nn as nn
import logging


def _ensure_src_on_path() -> None:
    """Add the repository "src" directory to sys.path for imports."""

    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # <repo>/explore/ncu-simple/run-dummy-model.py -> <repo>
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

# Import after adjusting sys.path
from llm_perf_opt.dnn_models.shallow_resnet import ShallowResNet  # noqa: E402


def init_random_weights(m: nn.Module) -> None:
    """Initialize Conv/Linear weights with a standard random scheme."""

    import math

    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # type: ignore[name-defined]
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)  # type: ignore[attr-defined]
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100, help="Number of forward iterations (default: 100)")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for synthetic input (default: 8)")
    ap.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="How often to log progress in iterations (default: 10)",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("ncu-simple")

    if not torch.cuda.is_available():
        log.error("CUDA not available; require cuda:0 for this test")
        raise RuntimeError("CUDA is required for this test; cuda:0 not available")

    # Pin device to cuda:0 explicitly and build model there
    torch.cuda.set_device(0)
    device: Final = torch.device("cuda:0")

    log.info("Using device: %s | name: %s", device, torch.cuda.get_device_name(0))

    log.info("Constructing ShallowResNet and moving to device")
    model = ShallowResNet().to(device)

    log.info("Initializing random weights (Conv/Linear)")
    model.apply(init_random_weights)
    model.eval()

    # Warmup a few runs to initialize kernels/caches
    log.info("Warmup: 5 iterations (batch_size=%d)", args.batch_size)
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(args.batch_size, 3, 1024, 1024, device=device)
            _ = model(x)
        torch.cuda.synchronize()

    # Main loop: run N forward passes with random inputs on cuda:0
    total_iters = max(1, int(args.iters))
    log.info("Starting main loop: %d iterations (batch_size=%d)", total_iters, args.batch_size)
    with torch.no_grad():
        for i in range(total_iters):
            x = torch.randn(args.batch_size, 3, 1024, 1024, device=device)
            _ = model(x)
            if args.log_interval > 0 and ((i + 1) % args.log_interval == 0 or (i + 1) == total_iters):
                log.info("Progress: %d/%d iterations", i + 1, total_iters)
        torch.cuda.synchronize()

    # Small output to confirm completion without cluttering profiler logs
    log.info("Completed %d iterations on %s with batch_size=%d", total_iters, device, args.batch_size)


if __name__ == "__main__":
    main()
