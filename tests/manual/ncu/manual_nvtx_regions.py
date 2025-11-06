from __future__ import annotations

"""Manual test: generate NVTX ranges around a dummy model forward.

Ranges emitted: A, nested A::A1, and B. Use with Nsight Compute replay_mode=range
to verify per-region aggregation and outputs.
"""

import argparse

import torch
import nvtx  # type: ignore[import-untyped]

from llm_perf_opt.dnn_models import get_model


def _forward(model: torch.nn.Module, x: torch.Tensor) -> None:
    _ = model(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0", help="Device string, e.g., cuda:0 or cpu")
    ap.add_argument("--batch", type=int, default=4, help="Batch size for synthetic input")
    args = ap.parse_args()

    device = args.device
    model = get_model("dummy_shallow_resnet", device=device)
    x = torch.randn(args.batch, 3, 64, 64, device=device)

    # Range A
    nvtx.push_range("A")
    _forward(model, x)
    # Nested A::A1
    nvtx.push_range("A::A1")
    _forward(model, x)
    nvtx.pop_range()
    nvtx.pop_range()

    # Range B
    nvtx.push_range("B")
    _forward(model, x)
    nvtx.pop_range()


if __name__ == "__main__":
    main()

