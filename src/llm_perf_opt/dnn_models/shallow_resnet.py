from __future__ import annotations

"""A tiny residual CNN used as a dummy workload.

The model is intentionally small and emits NVTX ranges inside ``forward`` so
that NVTX‑based Nsight Compute range replay can be exercised without requiring
large datasets or vendor models.
"""

from typing import Final

import torch
import torch.nn as nn
import nvtx  # type: ignore[import-untyped]


class BasicBlock(nn.Module):
    """Minimal residual block with two 3x3 convs."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1: Final = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1: Final = nn.BatchNorm2d(channels)
        self.relu: Final = nn.ReLU(inplace=True)
        self.conv2: Final = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2: Final = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # Use nvtx.annotate as a context manager; nvtx.range is not provided by the
        # Python nvtx package. This ensures Nsight tools can gate on these labels.
        with nvtx.annotate("block.conv1"):
            out = self.relu(self.bn1(self.conv1(x)))
            # Ensure kernels complete inside the NVTX range for NCU range replay
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        with nvtx.annotate("block.conv2"):
            out = self.bn2(self.conv2(out))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return self.relu(out + identity)


class ShallowResNet(nn.Module):
    """A small residual network for synthetic profiling workload.

    Args:
        in_channels: Input channels, default 3.
        base_channels: Width of the network, default 16.
        num_blocks: Number of residual blocks, default 3.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 16, num_blocks: int = 3) -> None:
        super().__init__()
        self.stem: Final = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks: Final = nn.Sequential(*[BasicBlock(base_channels) for _ in range(num_blocks)])
        self.head: Final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 10),
        )

    @torch.no_grad()
    def warmup(self, device: torch.device | str = "cpu") -> None:
        """Run a quick forward pass to initialize kernels and caches."""

        x = torch.randn(2, 3, 64, 64, device=device)
        _ = self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with nvtx.annotate("stem"):
            x = self.stem(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        with nvtx.annotate("residual"):
            x = self.blocks(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        with nvtx.annotate("head"):
            x = self.head(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return x
