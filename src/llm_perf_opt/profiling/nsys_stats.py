"""Nsight Systems stats/export helpers (Stage 2).

Lightweight utilities for post-processing Nsight Systems recordings (e.g.,
summary CSV export or SQLite extraction). Implementations are stubs and can be
extended when US2/US3 require data ingestion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import csv


def build_nsys_stats_cmd(qdrep_path: Path, out_csv: Path) -> List[str]:
    """Return `nsys stats` argv for summary CSV export.

    Parameters
    ----------
    qdrep_path : Path
        Path to the `.qdrep` file.
    out_csv : Path
        Destination CSV path (no header control here; defaults applied).
    """

    return [
        "nsys",
        "stats",
        "--report",
        "cuda_gpu_kern_sum",
        "--format",
        "csv",
        "-o",
        str(out_csv),
        str(qdrep_path),
    ]


def build_nsys_export_sqlite_cmd(qdrep_path: Path, out_sqlite: Path) -> List[str]:
    """Return `nsys export` argv to produce a SQLite database from `.qdrep`.

    Parameters
    ----------
    qdrep_path : Path
        Path to the `.qdrep` input file.
    out_sqlite : Path
        Path to the destination SQLite file.
    """

    return [
        "nsys",
        "export",
        "--sqlite",
        str(out_sqlite),
        str(qdrep_path),
    ]


def _detect_time_columns(headers: Iterable[str]) -> list[str]:
    """Return columns likely to represent total duration/time."""

    cands = [
        "Time (ns) Sum",
        "Time (ms) Sum",
        "Total Time (ns)",
        "Total Time (ms)",
        "Duration",
        "Time",
    ]
    hs = list(headers)
    return [c for c in cands if c in hs]


def top_kernels_from_nsys_summary(csv_path: Path, top_n: int = 30) -> list[str]:
    """Parse `nsys stats --report summary --format csv` and return top kernel names.

    Strategy
    --------
    - Find the "CUDA GPU Kernel Summary" section by scanning for its header row.
    - Accumulate a map of kernel name → total time using the first matching time column.
    - Return top-N names sorted by total time descending.
    """

    names: dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # Find section header
    start_idx = -1
    header: list[str] | None = None
    for i, row in enumerate(rows):
        if row and row[0].strip().startswith("CUDA GPU Kernel Summary"):
            # Next non-empty row is header
            j = i + 1
            while j < len(rows) and (not rows[j] or all(not c for c in rows[j])):
                j += 1
            if j < len(rows):
                header = rows[j]
                start_idx = j + 1
            break
    if header is None or start_idx < 0:
        return []
    time_cols = _detect_time_columns(header)
    name_idx = header.index("Name") if "Name" in header else -1
    time_idx = header.index(time_cols[0]) if time_cols else -1
    if name_idx < 0 or time_idx < 0:
        return []
    # Read until blank line (end of section)
    i = start_idx
    while i < len(rows) and rows[i] and any(c for c in rows[i]):
        row = rows[i]
        try:
            name = row[name_idx].strip()
            tval = float(str(row[time_idx]).replace(",", ""))
            names[name] = names.get(name, 0.0) + float(tval)
        except Exception:
            pass
        i += 1
    sorted_names = sorted(names.items(), key=lambda kv: kv[1], reverse=True)
    return [n for n, _ in sorted_names[: max(0, int(top_n))]]
