"""NVTX region assembly helpers.

This module builds per-region reports for Nsight Compute range replay runs.
It intentionally remains best-effort and filesystem-driven so it can operate
from artifacts alone without requiring the `ncu` binary at read time.

Primary entry points:
- discover_region_labels: list region labels given an NVTX include expression
- build_region_reports: construct NCUProfileRegionReport objects for labels
- write_consolidated_reports: emit JSON and Markdown aggregates
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from attrs import define

from llm_perf_opt.data.ncu_regions import NCUProfileRegion, NCUProfileRegionReport
from llm_perf_opt.profiling.artifacts import sanitize_region_label
from collections import defaultdict
from typing import List


def _infer_parent(label: str) -> tuple[str | None, int]:
    """Return (parent, depth) from an NVTX label using ``::`` nesting.

    Examples
    --------
    >>> _infer_parent('A')
    (None, 0)
    >>> _infer_parent('A::B')
    ('A', 1)
    """

    parts = str(label).split("::") if label else [""]
    depth = max(len(parts) - 1, 0)
    parent = parts[-2] if len(parts) >= 2 else None
    return (parent, depth)


def discover_region_labels(nvtx_include_expr: str | None) -> list[str]:
    """Discover candidate region labels from an NVTX include expression.

    This is a conservative best-effort parser used when we don't have direct
    access to NCU's range list. Users may provide a semicolon/pipe/comma
    separated list of explicit labels; wildcards are passed through as-is.

    Examples
    --------
    >>> discover_region_labels('A;B;A::A1')
    ['A', 'B', 'A::A1']
    >>> discover_region_labels('LLM@*')
    ['LLM@*']
    """

    if not nvtx_include_expr:
        return []
    raw = str(nvtx_include_expr).strip()
    if not raw:
        return []
    # Accept common separators
    toks: list[str] = []
    for sep in [";", "|", ",", " "]:
        if sep in raw:
            toks = [t for t in (p.strip() for p in raw.split(sep)) if t]
            break
    if not toks:
        toks = [raw]
    # De-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def build_region_reports(labels: Sequence[str], *, device: str | None = None, process: str | None = None) -> list[NCUProfileRegionReport]:
    """Return a list of minimal NCUProfileRegionReport entries for labels.

    This function does not parse `.ncu-rep` files; it constructs metadata-only
    reports that downstream exporters can render, allowing manual tests to verify
    per-region artifact layout and consolidated outputs even on systems without
    Nsight installed.
    """

    reports: list[NCUProfileRegionReport] = []
    for label in labels:
        parent, depth = _infer_parent(label)
        reg = NCUProfileRegion(name=str(label), parent=parent, depth=depth, process=process, device=device)
        reports.append(NCUProfileRegionReport(region=reg))
    return reports


def _range_key(row: dict) -> str:
    """Extract a best-effort NVTX range name from an NCU CSV row.

    Tries multiple known headers across NCU versions.
    """

    for k in ("NVTX Range Name", "Range Name", "NVTX Range", "Range"):
        v = row.get(k)
        if v:
            return str(v)
    return "(unlabeled)"


def assemble_region_reports(rows: Iterable[dict], *, device: str = "cuda:0", process: str | None = None) -> List[NCUProfileRegionReport]:
    """Group NCU rows by NVTX range and build NCUProfileRegionReport entries."""

    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[_range_key(r)].append(r)

    out: List[NCUProfileRegionReport] = []
    for name, group in buckets.items():
        total_ms = 0.0
        kernel_count = 0
        for r in group:
            # Sum best-effort duration metrics
            val = r.get("gpu__time_duration.sum", r.get("Total Time"))
            if val is not None:
                try:
                    total_ms += float(str(val).replace(",", ""))
                except Exception:
                    pass
            # Approximate kernel invocations
            v_calls = r.get("Calls", r.get("Invocations", None))
            if v_calls is not None:
                try:
                    kernel_count += int(float(str(v_calls).replace(",", "")))
                except Exception:
                    kernel_count += 1
            else:
                kernel_count += 1
        parent, depth = _infer_parent(name)
        region = NCUProfileRegion(name=name, parent=parent, depth=depth, device=device, process=process)
        out.append(NCUProfileRegionReport(region=region, total_ms=max(total_ms, 0.0), kernel_count=max(kernel_count, 0)))
    return out

def write_consolidated_reports(
    reports: Sequence[NCUProfileRegionReport],
    out_root: Path,
    *,
    scope: str = "aggregate",
) -> tuple[Path, Path]:
    """Write consolidated JSON and Markdown reports under ``out_root``.

    Returns JSON and Markdown output paths.
    """

    out_root.mkdir(parents=True, exist_ok=True)
    # JSON
    import json as _json

    json_path = out_root / "report.json"
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
    json_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown (lightweight summary)
    md_path = out_root / "report.md"
    lines: list[str] = ["# Nsight Compute – NVTX Regions", ""]
    if reports:
        lines.append("## Regions")
        for r in reports:
            lines.append(f"- {r.region.name} (depth={r.region.depth})")
    else:
        lines.append("No regions discovered.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return (json_path, md_path)
