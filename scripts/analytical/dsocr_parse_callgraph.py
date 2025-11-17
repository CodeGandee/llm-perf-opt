"""Parse and group DeepSeek-OCR TorchLens call graph JSON.

Usage
-----
    python scripts/analytical/dsocr_parse_callgraph.py \\
        tmp/dsocr-torchlens-callgraph/dsocr-call-graph-torchlens.json

This script reads the TorchLens-derived call graph JSON produced by
``dsocr_torchlens_callgraph.py``, derives ``for N`` (depth-wise) and
``parfor N`` (repeated sibling) groupings, and writes:

- ``dsocr-call-graph-grouped.json`` – machine-readable grouping.
- ``dsocr-call-graph-grouped.md`` – human-readable Markdown summary.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_perf_opt.utils.dsocr_callgraph_parse import (
    compute_grouped_callgraph,
    grouped_to_json_dict,
    grouped_to_dot,
    grouped_to_markdown,
    load_callgraph_json,
)


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Group DeepSeek-OCR TorchLens call graph into for/parfor constructs.",
    )
    parser.add_argument(
        "graph_json",
        help="Path to dsocr-call-graph-torchlens.json produced by TorchLens.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for grouped outputs "
        "(default: same directory as graph_json).",
    )
    parser.add_argument(
        "--min-family-size",
        type=int,
        default=2,
        help="Minimum size of indexed module family to emit a for-group "
        "(default: 2).",
    )
    parser.add_argument(
        "--min-edge-count",
        type=int,
        default=2,
        help="Minimum dynamic edge count to emit a parfor-group (default: 2).",
    )
    return parser


def main() -> None:
    """Entry point for call graph parsing and grouping."""

    parser = build_argparser()
    args = parser.parse_args()

    graph_path = Path(args.graph_json).resolve()
    if not graph_path.is_file():
        raise SystemExit(f"Input graph JSON not found: {graph_path}")

    callgraph = load_callgraph_json(graph_path)

    metadata: dict[str, Any] = {
        "source_json": str(graph_path),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "min_family_size": int(args.min_family_size),
        "min_edge_count": int(args.min_edge_count),
    }

    grouped = compute_grouped_callgraph(
        callgraph=callgraph,
        min_family_size=int(args.min_family_size),
        min_edge_count=int(args.min_edge_count),
        metadata=metadata,
    )

    out_dir = Path(args.output_dir).resolve() if args.output_dir else graph_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "dsocr-call-graph-grouped.json"
    md_path = out_dir / "dsocr-call-graph-grouped.md"
    dot_path = out_dir / "dsocr-call-graph-grouped.dot"
    svg_path = out_dir / "dsocr-call-graph-grouped.svg"

    json_data = grouped_to_json_dict(grouped)
    json_path.write_text(
        __import__("json").dumps(json_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    md_text = grouped_to_markdown(grouped)
    md_path.write_text(md_text, encoding="utf-8")

    print(f"[dsocr-parse-callgraph] Wrote grouped JSON: {json_path}")
    print(f"[dsocr-parse-callgraph] Wrote grouped Markdown: {md_path}")

    dot_text = grouped_to_dot(grouped)
    dot_path.write_text(dot_text, encoding="utf-8")
    print(f"[dsocr-parse-callgraph] Wrote grouped DOT: {dot_path}")

    # Best-effort SVG generation; dot may not be installed in all environments.
    try:
        import subprocess

        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[dsocr-parse-callgraph] Wrote grouped SVG: {svg_path}")
    except Exception:
        print("[dsocr-parse-callgraph] WARNING: failed to generate SVG via dot; is Graphviz installed?")


if __name__ == "__main__":
    main()
