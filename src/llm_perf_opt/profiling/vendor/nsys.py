from pathlib import Path
from typing import Sequence


def build_nsys_cmd(
    out_base: Path,
    work_argv: Sequence[str],
    *,
    trace: str = "cuda,nvtx,osrt",
    sample: str = "none",
    capture: str = "nvtx",
    nvtx_capture: str = "range@LLM",
) -> list[str]:
    """Build an argv list for `nsys profile` with NVTX gating.

    Parameters
    ----------
    out_base
        Base output path (without extension) for Nsight Systems artifacts.
    work_argv
        The target command (and args) to execute under Nsight Systems.
    trace
        Trace sources, defaults to CUDA + NVTX + OS runtime.
    sample
        CPU sampling mode. "none" recommended for lower overhead in GPU focus.
    capture
        Capture range selector. "nvtx" to gate by NVTX ranges.
    nvtx_capture
        NVTX capture expression (e.g., "range@LLM").

    Returns
    -------
    list[str]
        Complete argv to invoke `nsys profile`.
    """

    return [
        "nsys",
        "profile",
        f"--trace={trace}",
        f"--sample={sample}",
        f"--capture-range={capture}",
        f"--nvtx-capture={nvtx_capture}",
        "-o",
        str(out_base),
    ] + list(work_argv)

