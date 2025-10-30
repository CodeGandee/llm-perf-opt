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
    ...     'infer.max_new_tokens=64', 'runners=stage1.no-static'
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


def build_nsys_stats_cmd(qdrep_base: Path, out_csv_base: Path) -> list[str]:
    """Return `nsys stats` argv for summary CSV export.

    Notes
    -----
    Convenience re-export to keep vendor-facing helpers together.
    ``nsys stats -o <out_csv_base> <qdrep_base>.qdrep`` produces
    ``<out_csv_base>.csv``.
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
        str(qdrep_base) + ".qdrep",
    ]


def build_nsys_export_sqlite_cmd(qdrep_base: Path, out_sqlite: Path) -> list[str]:
    """Return `nsys export --sqlite` argv using base path for `.qdrep`.

    Parameters
    ----------
    qdrep_base : pathlib.Path
        Base path (without extension) for the `.qdrep` file.
    out_sqlite : pathlib.Path
        Destination SQLite path.
    """

    return [
        "nsys",
        "export",
        "--sqlite",
        str(out_sqlite),
        str(qdrep_base) + ".qdrep",
    ]
