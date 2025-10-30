from pathlib import Path
from typing import Sequence


DEFAULT_METRICS = (
    "flop_count_hp,flop_count_sp,gpu__time_duration.sum,"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
)


def build_ncu_cmd(out_base: Path, work_argv: Sequence[str], *, nvtx_expr: str) -> list[str]:
    """Build an argv list for `ncu` focused on roofline metrics.

    Parameters
    ----------
    out_base
        Base output path (without extension) for Nsight Compute outputs.
    work_argv
        The target command (and args) to execute under Nsight Compute.
    nvtx_expr
        NVTX include expression (e.g., "LLM@*") to constrain kernel capture.

    Returns
    -------
    list[str]
        Complete argv to invoke `ncu` capturing roofline metrics.
    """

    return [
        "ncu",
        "--target-processes",
        "all",
        "--nvtx",
        "--nvtx-include",
        nvtx_expr,
        "--set",
        "roofline",
        "--section",
        ".*SpeedOfLight.*",
        "--metrics",
        DEFAULT_METRICS,
        "-o",
        str(out_base),
    ] + list(work_argv)

