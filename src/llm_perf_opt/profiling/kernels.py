from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

from llm_perf_opt.data.models import KernelRecord


def parse_ncu_csv(csv_path: str | Path) -> list[dict]:
    """Parse Nsight Compute CSV into a list of dictionaries (rows).

    Parameters
    ----------
    csv_path : str | Path
        Path to a CSV file exported by `ncu`.
    """

    rows: List[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def parse_ncu_json(json_path: str | Path) -> list[dict]:
    """Parse Nsight Compute JSON into a list of dictionaries (records).

    The JSON structure from `ncu --export` may vary by version; this function
    returns a flat list if a top-level list is present, otherwise wraps the
    loaded object.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    return [obj]


def kernels_from_ncu_rows(rows: Iterable[dict], device: str = "cuda:0") -> list[KernelRecord]:
    """Convert generic `ncu` rows into `KernelRecord` entries.

    This is a best-effort mapper that looks for common column headers and falls
    back to zeros when fields are missing.
    """

    out: list[KernelRecord] = []
    for r in rows:
        # Common column names seen in Nsight Compute CSVs
        name = (
            r.get("Kernel Name")
            or r.get("Name")
            or r.get("Kernel")
            or r.get("Function Name")
            or ""
        )
        # Durations: prefer total time if available; else approximate from Avg * Calls
        total_ms = 0.0
        for key in ("Duration", "Time", "Total Time", "gpu__time_duration.sum"):
            v = r.get(key)
            if v is not None:
                try:
                    total_ms = float(str(v).replace(",", ""))
                    break
                except Exception:
                    pass
        calls = 0
        for key in ("Calls", "Invocations", "Count"):
            v = r.get(key)
            if v is not None:
                try:
                    calls = int(float(str(v).replace(",", "")))
                    break
                except Exception:
                    pass
        mean_ms = 0.0
        for key in ("Avg Duration", "Average", "Mean"):
            v = r.get(key)
            if v is not None:
                try:
                    mean_ms = float(str(v).replace(",", ""))
                    break
                except Exception:
                    pass
        if mean_ms <= 0.0 and total_ms > 0 and calls > 0:
            mean_ms = total_ms / max(calls, 1)
        try:
            out.append(
                KernelRecord(
                    kernel_name=str(name),
                    device=str(device),
                    total_ms=float(max(total_ms, 0.0)),
                    calls=int(max(calls, 0)),
                    mean_ms=float(max(mean_ms, 0.0)),
                )
            )
        except Exception:
            # Skip malformed rows
            continue
    return out

