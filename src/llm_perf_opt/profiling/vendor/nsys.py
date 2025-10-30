"""Nsight Systems command builders.

Build safe argv lists for profiling workloads with `nsys profile`. These helpers
do not execute commands; they only construct argument lists for subprocess use.
"""

from pathlib import Path
from typing import Sequence


def build_nsys_cmd(
    out_base: Path,
    work_argv: Sequence[str],
    *,
    trace: str = "cuda,nvtx,osrt",
    sample: str = "none",
    capture: str = "nvtx",
    nvtx_capture: str = "decode",
    enable_nonregistered_nvtx: bool = True,
) -> list[str]:
    """Build an argv list for `nsys profile` with NVTX gating.

    Parameters
    ----------
    out_base : pathlib.Path
        Base output path (without extension) for Nsight Systems artifacts.
    work_argv : sequence of str
        The target command (and args) to execute under Nsight Systems.
    trace : str, optional
        Trace sources, defaults to CUDA + NVTX + OS runtime.
    sample : str, optional
        CPU sampling mode. "none" recommended for lower overhead in GPU focus.
    capture : str, optional
        Capture range selector. "nvtx" to gate by NVTX ranges.
    nvtx_capture : str, optional
        NVTX capture expression (e.g., "decode", "prefill", or "decode@*").
    enable_nonregistered_nvtx : bool, optional
        When True, pass `--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0` to allow
        non-registered NVTX strings (common in Python). Recommended when using
        NVTX gating.

    Examples
    -------
    To profile the Stage 1 runner without static analyzer (recommended for Nsight runs):
    >>> work = [
    ...     'python', '-m', 'llm_perf_opt.runners.llm_profile_runner',
    ...     'dataset.subset_filelist=/abs/dev-20.txt', 'device=cuda:0', 'repeats=1',
    ...     'infer.max_new_tokens=64', 'runner@stage1_runner=stage1.no-static'
    ... ]
    >>> cmd = build_nsys_cmd(Path('tmp/stage2/run/nsys'), work)

    Returns
    -------
    list[str]
        Complete argv to invoke `nsys profile`.
    """

    cmd = [
        "nsys",
        "profile",
        f"--trace={trace}",
        f"--sample={sample}",
    ]
    # Only add capture-range/NVTX capture when not explicitly disabled
    cap = str(capture).lower()
    nvx = str(nvtx_capture).lower()
    if cap != "none":
        cmd += [f"--capture-range={capture}"]
        if enable_nonregistered_nvtx:
            cmd += ["--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0"]
    if nvx != "none":
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


def build_nsys_export_sqlite_cmd(report_path: Path) -> list[str]:
    """Return `nsys export --type sqlite` argv for a report file.

    Parameters
    ----------
    report_path : pathlib.Path
        Path to the report file (`.nsys-rep` or `.qdrep`).
    """

    return [
        "nsys",
        "export",
        "--type",
        "sqlite",
        str(report_path),
    ]


def resolve_nsys_report_path(out_base: Path) -> Path | None:
    """Return the existing report path for a given output base, if any.

    Checks for `<out_base>.nsys-rep` and `<out_base>.qdrep` in that order.
    """

    for ext in (".nsys-rep", ".qdrep"):
        p = Path(str(out_base) + ext)
        if p.exists():
            return p
    return None
