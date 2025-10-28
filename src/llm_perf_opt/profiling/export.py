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
    """Write a top-K operator summary as a Markdown table.

    Parameters
    ----------
    records : Iterable[dict]
        Operator dictionaries to summarize.
    path : str
        Destination file path (created/overwritten).
    top_k : int, default=20
        Number of rows to include in the output table.
    """

    rows = top_n_operators(list(records), n=top_k)
    lines = ["| op_name | total_time_ms | cuda_time_ms | calls |", "|---|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r.get('op_name','')} | {r.get('total_time_ms',0):.3f} | {r.get('cuda_time_ms',0):.3f} | {int(r.get('calls',0))} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
