"""Export helpers for operator summaries and stakeholder outputs.

Functions
---------
top_n_operators
    Sort operator dicts by total time and return top-N.
write_operator_markdown
    Emit a Markdown table for a list of operator records.
"""

from __future__ import annotations

from typing import Iterable
from mdutils.mdutils import MdUtils  # type: ignore[import-untyped]


def top_n_operators(records: Iterable[dict], n: int = 20) -> list[dict]:
    """Return top-N operator dicts by ``total_time_ms``.

    Parameters
    ----------
    records : Iterable[dict]
        Operator dictionaries containing at least ``op_name``,
        ``total_time_ms``, ``cuda_time_ms``, and ``calls``.
    n : int, default=20
        Maximum number of records to return.

    Returns
    -------
    list[dict]
        Sorted records limited to the top-N by ``total_time_ms``.
    """

    sorted_recs = sorted(records, key=lambda r: float(r.get("total_time_ms", 0.0)), reverse=True)
    return sorted_recs[:n]


def write_operator_markdown(records: Iterable[dict], path: str, top_k: int = 20) -> None:
    """Write a top‑K operator summary as a Markdown table using mdutils.

    Parameters
    ----------
    records : Iterable[dict]
        Operator dictionaries to summarize.
    path : str
        Destination file path (created/overwritten). Accepts ``.md`` suffix; it is stripped
        to satisfy mdutils' file naming (which appends ``.md`` automatically).
    top_k : int, default=20
        Number of rows to include in the output table.
    """

    rows = top_n_operators(list(records), n=top_k)
    # mdutils expects a flattened list row-wise (including header)
    header = ["op_name", "total_time_ms", "cuda_time_ms", "calls"]
    table_data: list[str] = header.copy()
    for r in rows:
        table_data.extend(
            [
                str(r.get("op_name", "")),
                f"{float(r.get('total_time_ms', 0.0)):.3f}",
                f"{float(r.get('cuda_time_ms', 0.0)):.3f}",
                str(int(r.get("calls", 0))),
            ]
        )

    file_base = path[:-3] if path.endswith(".md") else path
    md = MdUtils(file_name=file_base)
    md.new_header(level=1, title="Operator Summary (Top‑K)")
    md.new_paragraph(f"Rows: {len(rows)} (k={int(top_k)})")
    md.new_table(columns=4, rows=len(rows) + 1, text=table_data, text_align="center")
    md.create_md_file()
