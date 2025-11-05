"""Nsight Compute command builders.

Produces `ncu` command argv for kernel-level metric collection. These helpers
do not execute the profiler; they only construct lists for subprocess use.
"""

from pathlib import Path
from typing import Sequence, Optional, Sequence as _Seq, Union
import subprocess


DEFAULT_METRICS = (
    "gpu__time_duration.sum,"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
)


def _list_available_metrics() -> list[str]:
    """Return available metric names from `ncu --list-metrics`.

    Best-effort: on failure returns an empty list (caller should fallback).
    """
    try:
        proc = subprocess.run(["ncu", "--list-metrics"], check=False, capture_output=True, text=True)
        out = proc.stdout or ""
        names: list[str] = []
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("-") or line.startswith("Device "):
                continue
            # Metric name is the first token (non-space)
            part = line.split()[0]
            if part and all(c not in part for c in "|:"):
                names.append(part)
        return names
    except Exception:
        return []


def _filter_metrics(metrics_csv: Optional[str]) -> Optional[str]:
    """Filter a comma-separated metrics string to only those available.

    Returns the filtered CSV string, or None if input is None/empty after filtering.
    """
    if not metrics_csv:
        return None
    toks = [m.strip() for m in metrics_csv.split(",") if m.strip()]
    if not toks:
        return None
    avail = set(_list_available_metrics())
    if not avail:
        # Could not list metrics; return original to let NCU decide
        return ",".join(toks)
    filtered = [m for m in toks if m in avail]
    return ",".join(filtered) if filtered else None


def _normalize_metrics_csv(metrics: Union[str, Sequence[str], None]) -> Optional[str]:
    """Normalize metrics (str | sequence | None) to a CSV string or None."""
    if metrics is None:
        return None
    if isinstance(metrics, str):
        m = metrics.strip()
        return m if m else None
    try:
        # Accept any iterable of strings
        toks = [str(x).strip() for x in metrics if str(x).strip()]
    except TypeError:
        return None
    return ",".join(toks) if toks else None


def build_ncu_cmd(
    out_base: Path,
    work_argv: Sequence[str],
    *,
    nvtx_expr: Optional[str],
    kernel_regex: str | None = None,
    csv_log: Path | None = None,
    use_nvtx: bool = True,
    set_name: str | None = "roofline",
    metrics: Union[str, Sequence[str], None] = None,
    sections: Optional[_Seq[str]] = None,
    target_processes: Optional[str] = "all",
    force_overwrite: bool = False,
    kernel_name_base: str = "demangled",
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
    Profile kernels for the Stage 1 runner while avoiding extra workload overhead:
    >>> work = [
    ...     'python', '-m', 'llm_perf_opt.runners.llm_profile_runner',
    ...     'dataset.subset_filelist=/abs/dev-20.txt', 'device=cuda:0', 'repeats=1',
    ...     'infer.max_new_tokens=64', 'pipeline.static_analysis.enable=false', 'pipeline.torch_profiler.enable=false'
    ... ]
    >>> cmd = build_ncu_cmd(Path('tmp/profile-output/run/ncu/kernels'), work, nvtx_expr='LLM@*')

    Returns
    -------
    list[str]
        Complete argv to invoke `ncu` capturing roofline metrics.
    """

    cmd: list[str] = ["ncu"]
    # Target processes (default to 'all' if omitted)
    if target_processes:
        cmd += ["--target-processes", str(target_processes)]
    if set_name:
        cmd += ["--set", str(set_name)]
    # Normalize metrics (list or str) then filter for device compatibility; allow None to skip --metrics entirely
    met = _filter_metrics(_normalize_metrics_csv(metrics))
    if met:
        cmd += ["--metrics", met]
    # Add selected sections (limits profiling to these summaries)
    if sections:
        for sec in sections:
            if sec:
                cmd += ["--section", str(sec)]
    cmd += ["-o", str(out_base)]
    if use_nvtx and nvtx_expr:
        cmd += ["--nvtx", "--nvtx-include", nvtx_expr]
    if kernel_regex:
        cmd += [
            "--kernel-name-base",
            str(kernel_name_base or "demangled"),
            "--kernel-name",
            str(kernel_regex),
        ]
    if csv_log:
        cmd += [
            "--csv",
            "--log-file",
            str(csv_log),
        ]
    if force_overwrite:
        cmd += ["--force-overwrite"]
    return cmd + list(work_argv)


def build_ncu_import_sections_cmd(rep_path: Path, sections: _Seq[str], page: str = "raw") -> list[str]:
    """Build an argv list to import an existing `.ncu-rep` and print sections.

    The caller should capture stdout to a file. Multiple ``--section`` flags are
    added, one per entry in ``sections``.
    """

    cmd: list[str] = ["ncu", "--import", str(rep_path), "--page", str(page)]
    for sec in sections:
        if sec:
            cmd += ["--section", str(sec)]
    return cmd
