"""Nsight Compute command builders.

Produces `ncu` command argv for kernel-level metric collection. These helpers
do not execute the profiler; they only construct lists for subprocess use.
"""

from pathlib import Path
from typing import Sequence


DEFAULT_METRICS = (
    "flop_count_hp,flop_count_sp,gpu__time_duration.sum,"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
)


def build_ncu_cmd(
    out_base: Path,
    work_argv: Sequence[str],
    *,
    nvtx_expr: str,
    kernel_regex: str | None = None,
    csv_log: Path | None = None,
    use_nvtx: bool = True,
) -> list[str]:
    """Build an argv list for `ncu` focused on roofline metrics.

    Parameters
    ----------
    out_base : pathlib.Path
        Base output path (without extension) for Nsight Compute outputs.
    work_argv : sequence of str
        The target command (and args) to execute under Nsight Compute.
    nvtx_expr : str
        NVTX include expression (e.g., "LLM@*") to constrain kernel capture.
    kernel_regex : str or None, optional
        If provided, inject ``--kernel-name=<regex>`` to restrict capture to top kernels.
    csv_log : pathlib.Path or None, optional
        If provided, inject ``--csv --log-file=<csv_log>`` to export raw CSV.

    Examples
    -------
    Profile kernels for the Stage 1 runner while disabling its static analyzer:
    >>> work = [
    ...     'python', '-m', 'llm_perf_opt.runners.llm_profile_runner',
    ...     'dataset.subset_filelist=/abs/dev-20.txt', 'device=cuda:0', 'repeats=1',
    ...     'infer.max_new_tokens=64', 'runner@stage1_runner=stage1.no-static'
    ... ]
    >>> cmd = build_ncu_cmd(Path('tmp/stage2/run/ncu/kernels'), work, nvtx_expr='LLM@*')

    Returns
    -------
    list[str]
        Complete argv to invoke `ncu` capturing roofline metrics.
    """

    cmd: list[str] = [
        "ncu",
        "--target-processes",
        "all",
        "--set",
        "roofline",
        "--metrics",
        DEFAULT_METRICS,
        "-o",
        str(out_base),
    ]
    if use_nvtx:
        cmd += ["--nvtx", "--nvtx-include", nvtx_expr]
    if kernel_regex:
        cmd += [
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            str(kernel_regex),
        ]
    if csv_log:
        cmd += [
            "--csv",
            "--log-file",
            str(csv_log),
        ]
    return cmd + list(work_argv)
