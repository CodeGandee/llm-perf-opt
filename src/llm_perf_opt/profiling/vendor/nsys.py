"""Nsight Systems command builders.

Build safe argv lists for profiling workloads with `nsys profile`. These helpers
do not execute commands; they only construct argument lists for subprocess use.
"""

from pathlib import Path
from typing import Sequence, Optional


def build_nsys_cmd(
    out_base: Path,
    work_argv: Sequence[str],
    *,
    trace: str = "cuda,nvtx,osrt",
    sample: Optional[str] = None,
    capture: Optional[str] = None,
    capture_end: Optional[str] = None,
    nvtx_capture: Optional[str] = None,
    enable_nonregistered_nvtx: bool = True,
) -> list[str]:
    """Build an argv list for `nsys profile` (no NVTX gating by default).

    Parameters
    ----------
    out_base : pathlib.Path
        Base output path (without extension) for Nsight Systems artifacts.
    work_argv : sequence of str
        The target command (and args) to execute under Nsight Systems.
    trace : str, optional
        Trace sources, defaults to CUDA + NVTX + OS runtime.
    sample : Optional[str], optional
        CPU sampling mode. When None/empty, the option is omitted (nsys default).
    capture : Optional[str], optional
        Capture range selector. When None/empty, the option is omitted (nsys default).
        Use "nvtx" to gate by NVTX ranges if you provide an `nvtx_capture` expression.
        to gate by NVTX ranges if you provide an `nvtx_capture` expression.
    nvtx_capture : Optional[str], optional
        NVTX capture expression (e.g., "decode", "prefill", or "decode@*").
        When None/empty and/or when capture != 'nvtx', no NVTX gating argument is passed.
    enable_nonregistered_nvtx : bool, optional
        When True, pass `--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0` to allow
        non-registered NVTX strings (common in Python). Recommended when using
        NVTX gating.

    Examples
    -------
    To profile the Stage 1 runner as a workload while avoiding extra overhead:
    >>> work = [
    ...     'python', '-m', 'llm_perf_opt.runners.llm_profile_runner',
    ...     'dataset.subset_filelist=/abs/dev-20.txt', 'device=cuda:0', 'repeats=1',
    ...     'infer.max_new_tokens=64', 'pipeline.static_analysis.enable=false', 'pipeline.torch_profiler.enable=false'
    ... ]
    >>> cmd = build_nsys_cmd(Path('tmp/profile-output/run/nsys'), work)

    Returns
    -------
    list[str]
        Complete argv to invoke `nsys profile`.
    """

    cmd = ["nsys", "profile", f"--trace={trace}"]
    # Add CPU sampling only if explicitly requested and not 'none'
    smp = (str(sample).strip().lower() if sample is not None else "")
    if smp and smp != "none":
        cmd += [f"--sample={sample}"]

    # Add capture-range controls only if explicitly requested and not 'none'
    cap = (str(capture).strip().lower() if capture is not None else "")
    nvx = (str(nvtx_capture).strip().lower() if nvtx_capture is not None else "")
    if cap and cap != "none":
        cmd += [f"--capture-range={capture}"]
        if capture_end and str(capture_end).strip():
            cmd += [f"--capture-range-end={capture_end}"]
        if enable_nonregistered_nvtx and cap == "nvtx":
            cmd += ["--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0"]
        # Only pass NVTX capture expression when capture-range uses NVTX gating
        if cap == "nvtx" and nvx:
            cmd += [f"--nvtx-capture={nvtx_capture}"]
    cmd += ["-o", str(out_base)]
    return cmd + list(work_argv)


def build_nsys_stats_cmd(report_path: Path, out_csv_base: Path) -> list[str]:
    """Return `nsys stats` argv for summary CSV export.

    Parameters
    ----------
    report_path : pathlib.Path
        Path to the report file (`.nsys-rep` or `.qdrep`).
    out_csv_base : pathlib.Path
        Output base path; Nsight Systems will append `.csv`.
    """

    return [
        "nsys",
        "stats",
        "--report",
        "cuda_gpu_kern_sum",
        "--format",
        "csv",
        "-o",
        str(out_csv_base),
        str(report_path),
    ]


def build_nsys_export_sqlite_cmd(report_path: Path, force_overwrite: bool = True) -> list[str]:
    """Return `nsys export --type sqlite` argv for a report file.

    Parameters
    ----------
    report_path : pathlib.Path
        Path to the report file (`.nsys-rep` or `.qdrep`).
    """

    cmd = [
        "nsys",
        "export",
        "--type",
        "sqlite",
    ]
    if force_overwrite:
        cmd += ["--force-overwrite", "true"]
    cmd += [str(report_path)]
    return cmd


def resolve_nsys_report_path(out_base: Path) -> Path | None:
    """Return the existing report path for a given output base, if any.

    Checks for `<out_base>.nsys-rep` and `<out_base>.qdrep` in that order.
    """

    for ext in (".nsys-rep", ".qdrep"):
        p = Path(str(out_base) + ext)
        if p.exists():
            return p
    return None
