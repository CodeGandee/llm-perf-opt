"""Nsight Systems stats/export helpers (Stage 2).

Lightweight utilities for post-processing Nsight Systems recordings (e.g.,
summary CSV export or SQLite extraction). Implementations are stubs and can be
extended when US2/US3 require data ingestion.
"""

from __future__ import annotations

from pathlib import Path
from typing import List


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
        "summary",
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

