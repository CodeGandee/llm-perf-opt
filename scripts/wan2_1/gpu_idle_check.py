from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import pynvml  # type: ignore


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    name: str
    util_gpu_pct: float
    util_mem_pct: float
    mem_used_bytes: int
    mem_total_bytes: int

    @property
    def mem_used_pct(self) -> float:
        if self.mem_total_bytes <= 0:
            return 0.0
        return 100.0 * float(self.mem_used_bytes) / float(self.mem_total_bytes)


def _parse_cuda_visible_devices(value: str) -> list[int]:
    # CUDA_VISIBLE_DEVICES can also be UUIDs; we only accept integer indices here.
    out: list[int] = []
    for part in value.split(","):
        p = part.strip()
        if not p:
            continue
        if p.isdigit():
            out.append(int(p))
    return out


def resolve_physical_gpu_ids(default: Sequence[int] | None = None) -> list[int]:
    """Return physical GPU indices to check for idleness.

    Preference order:
    1) CUDA_VISIBLE_DEVICES (when it is a comma-separated list of integer indices).
    2) `default` (caller-provided physical indices).
    """

    env_val = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_val:
        ids = _parse_cuda_visible_devices(env_val)
        if ids:
            return ids
    return list(default or [])


def snapshot_gpus(indices: Iterable[int]) -> list[GpuSnapshot]:
    pynvml.nvmlInit()
    snaps: list[GpuSnapshot] = []
    for idx in indices:
        h = pynvml.nvmlDeviceGetHandleByIndex(int(idx))
        name = pynvml.nvmlDeviceGetName(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        snaps.append(
            GpuSnapshot(
                index=int(idx),
                name=name.decode("utf-8", errors="replace") if isinstance(name, (bytes, bytearray)) else str(name),
                util_gpu_pct=float(util.gpu),
                util_mem_pct=float(util.memory),
                mem_used_bytes=int(mem.used),
                mem_total_bytes=int(mem.total),
            )
        )
    return snaps


def is_idle(
    snaps: Sequence[GpuSnapshot],
    *,
    max_util_gpu_pct: float = 1.0,
    max_mem_used_pct: float = 1.0,
) -> bool:
    for s in snaps:
        if float(s.util_gpu_pct) > float(max_util_gpu_pct):
            return False
        if float(s.mem_used_pct) > float(max_mem_used_pct):
            return False
    return True


def format_snapshots(snaps: Sequence[GpuSnapshot]) -> str:
    parts: list[str] = []
    for s in snaps:
        used_gib = float(s.mem_used_bytes) / (1024.0**3)
        total_gib = float(s.mem_total_bytes) / (1024.0**3)
        parts.append(
            f"gpu={s.index} name={s.name!r} util_gpu={s.util_gpu_pct:.1f}% mem_used={used_gib:.2f}/{total_gib:.2f} GiB ({s.mem_used_pct:.2f}%)"
        )
    return " | ".join(parts)


def wait_for_idle(
    indices: Sequence[int],
    *,
    max_util_gpu_pct: float = 1.0,
    max_mem_used_pct: float = 1.0,
    timeout_s: float = 0.0,
    poll_s: float = 5.0,
) -> list[GpuSnapshot]:
    """Wait for given GPUs to become idle, or raise RuntimeError."""

    deadline = time.time() + float(timeout_s)
    last: list[GpuSnapshot] = []
    while True:
        last = snapshot_gpus(indices)
        if is_idle(last, max_util_gpu_pct=max_util_gpu_pct, max_mem_used_pct=max_mem_used_pct):
            return last
        if timeout_s <= 0 or time.time() >= deadline:
            raise RuntimeError(
                "GPUs not idle: "
                + format_snapshots(last)
                + f" (thresholds: util<={max_util_gpu_pct}%, mem_used<={max_mem_used_pct}%)"
            )
        time.sleep(float(poll_s))

