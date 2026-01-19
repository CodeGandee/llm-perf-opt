"""Exporters for NVTX region reports (Markdown + JSON)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from llm_perf_opt.data.ncu_regions import NCUProfileRegionReport
from llm_perf_opt.profiling.artifacts import sanitized_region_dir


def write_regions_json(reports: Sequence[NCUProfileRegionReport], path: str | Path, *, scope: str = "aggregate") -> Path:
    """Write a consolidated JSON payload for region reports.

    The structure mirrors the `NCUProfileRegionReport` contract bundle in
    `specs/003-nvtx-ncu-profiling/contracts/openapi.yaml`.
    """

    import json as _json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": scope,
        "regions": [
            {
                "name": r.region.name,
                "parent": r.region.parent,
                "depth": r.region.depth,
                "process": r.region.process,
                "device": r.region.device,
                "total_ms": r.total_ms,
                "kernel_count": r.kernel_count,
                "sections_path": r.sections_path,
                "csv_path": r.csv_path,
                "markdown_path": r.markdown_path,
                "json_path": r.json_path,
            }
            for r in reports
        ],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "totals": {},
        "config_fingerprint": "",
    }
    p.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    return p


def write_regions_markdown(reports: Sequence[NCUProfileRegionReport], path: str | Path) -> Path:
    """Write a concise Markdown summary for region reports."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Nsight Compute – NVTX Regions", ""]
    if reports:
        lines.append("## Regions")
        for r in reports:
            lines.append(f"- {r.region.name} (depth={r.region.depth})")
    else:
        lines.append("No regions discovered.")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def export_region_reports(artifacts, reports: Sequence[NCUProfileRegionReport]) -> list[Path]:
    """High-level export: consolidated JSON/MD and ensure per-region dirs exist.

    Returns the list of generated consolidated paths.
    """

    base = Path(artifacts.out_dir("ncu") / "regions")
    base.mkdir(parents=True, exist_ok=True)
    j = write_regions_json(reports, base / "report.json")
    m = write_regions_markdown(reports, base / "report.md")
    for r in reports:
        sanitized_region_dir(artifacts, r.region.name)
    return [m, j]
