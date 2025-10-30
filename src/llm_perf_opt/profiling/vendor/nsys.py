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
    nvtx_capture: str = "range@LLM",
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
        NVTX capture expression (e.g., "range@LLM").

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
        "summary",
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
