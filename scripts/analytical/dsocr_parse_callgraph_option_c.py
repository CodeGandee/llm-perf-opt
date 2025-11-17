"""Post-process DeepSeek-OCR TorchLens call graph (Option C).

This script reads:
- tmp/dsocr-torchlens-callgraph/dsocr-call-graph-torchlens.json
- tmp/dsocr-torchlens-callgraph/dsocr-callgraph-runtime-metadata.json

and emits a summarized view of:
- Depth-wise stacks: `XYZModule for N` (distinct indices along depth).
- Same-level reuse: `parent -> XYZModule parfor N` (child used N times under parent).

It does not modify or re-run the model; it purely aggregates existing artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from llm_perf_opt.patches.dsocr_torchlens import (
    DsocrCallgraph,
    load_callgraph_json,
    load_runtime_metadata,
    summarize_parfor_edges,
    summarize_stacks,
    build_grouped_dot_from_option_c,
    build_grouped_mermaid_from_option_c,
)


def find_repo_root(start: Path) -> Path:
    """Locate the repository root by walking up to find pyproject.toml."""

    for current in (start, *start.parents):
        if (current / "pyproject.toml").exists():
            return current
    return start


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-process DeepSeek-OCR TorchLens call graph into for/parfor summary.",
    )
    parser.add_argument(
        "--callgraph-json",
        default=None,
        help="Path to dsocr-call-graph-torchlens.json "
        "(default: <repo_root>/tmp/dsocr-torchlens-callgraph/dsocr-call-graph-torchlens.json).",
    )
    parser.add_argument(
        "--runtime-json",
        default=None,
        help="Path to dsocr-callgraph-runtime-metadata.json "
        "(default: <repo_root>/tmp/dsocr-torchlens-callgraph/dsocr-callgraph-runtime-metadata.json).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write grouped summary JSON "
        "(default: <repo_root>/tmp/dsocr-torchlens-callgraph/dsocr-callgraph-grouped-option-c.json).",
    )
    parser.add_argument(
        "--output-dot",
        default=None,
        help="Optional path to write grouped DOT "
        "(default: <repo_root>/tmp/dsocr-torchlens-callgraph/dsocr-call-graph-grouped-option-c.dot).",
    )
    parser.add_argument(
        "--output-svg",
        default=None,
        help="Optional path to write grouped SVG "
        "(default: <repo_root>/tmp/dsocr-torchlens-callgraph/dsocr-call-graph-grouped-option-c.svg).",
    )
    parser.add_argument(
        "--output-mermaid",
        default=None,
        help="Optional path to write grouped Mermaid graph "
        "(default: <repo_root>/tmp/dsocr-torchlens-callgraph/dsocr-call-graph-grouped-option-c.mmd).",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional path to write a Markdown file embedding the Mermaid graph "
        "(default: <repo_root>/tmp/dsocr-torchlens-callgraph/dsocr-call-graph-grouped-option-c.md).",
    )
    return parser


def build_default_paths(repo_root: Path, args: argparse.Namespace) -> Dict[str, Path]:
    tmp_root = repo_root / "tmp" / "dsocr-torchlens-callgraph"
    callgraph = Path(args.callgraph_json).resolve() if args.callgraph_json else tmp_root / "dsocr-call-graph-torchlens.json"
    runtime = Path(args.runtime_json).resolve() if args.runtime_json else tmp_root / "dsocr-callgraph-runtime-metadata.json"
    output = Path(args.output_json).resolve() if args.output_json else tmp_root / "dsocr-callgraph-grouped-option-c.json"
    dot = Path(args.output_dot).resolve() if args.output_dot else tmp_root / "dsocr-call-graph-grouped-option-c.dot"
    svg = Path(args.output_svg).resolve() if args.output_svg else tmp_root / "dsocr-call-graph-grouped-option-c.svg"
    mermaid = (
        Path(args.output_mermaid).resolve()
        if args.output_mermaid
        else tmp_root / "dsocr-call-graph-grouped-option-c.mmd"
    )
    markdown = (
        Path(args.output_md).resolve()
        if args.output_md
        else tmp_root / "dsocr-call-graph-grouped-option-c.md"
    )
    return {
        "callgraph": callgraph,
        "runtime": runtime,
        "output": output,
        "dot": dot,
        "svg": svg,
        "mermaid": mermaid,
        "markdown": markdown,
    }


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    paths = build_default_paths(repo_root, args)

    cg: DsocrCallgraph = load_callgraph_json(paths["callgraph"])
    runtime_log = load_runtime_metadata(paths["runtime"])

    stacks = summarize_stacks(cg)
    parfor_edges = summarize_parfor_edges(cg, min_parfor=2)

    summary: Dict[str, Any] = {
        "stacks": stacks,
        "parfor_edges": parfor_edges,
        "runtime_events": runtime_log.events,
    }

    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also print a concise human-readable view
    print("=== Stacked modules (for N) ===")
    for s in stacks:
        fam = s["family"]
        indices = s["indices"]
        print(f"{fam} for {s['for']} (indices={indices})")

    print("\n=== Same-level reuse (parfor N) ===")
    for e in parfor_edges:
        print(f"{e['parent']} -> {e['child']} parfor {e['parfor']}")

    print("\n=== Runtime events ===")
    for ev in runtime_log.events:
        print(ev)

    # Build grouped DOT + SVG for Option C
    dot_text = build_grouped_dot_from_option_c(cg, stacks, parfor_edges)
    paths["dot"].write_text(dot_text, encoding="utf-8")

    # Build grouped Mermaid graph for Option C
    mermaid_text = build_grouped_mermaid_from_option_c(cg, stacks, parfor_edges)
    paths["mermaid"].parent.mkdir(parents=True, exist_ok=True)
    paths["mermaid"].write_text(mermaid_text, encoding="utf-8")

    # Build a small Markdown wrapper that embeds the Mermaid graph.
    md_lines = [
        "# DeepSeek-OCR Call Graph (Option C)",
        "",
        "```mermaid",
        mermaid_text,
        "```",
        "",
    ]
    paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["markdown"].write_text("\n".join(md_lines), encoding="utf-8")

    try:
        import subprocess

        subprocess.run(
            ["dot", "-Tsvg", str(paths["dot"]), "-o", str(paths["svg"])],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"\n[dsocr-parse-callgraph-option-c] Wrote grouped DOT: {paths['dot']}")
        print(f"[dsocr-parse-callgraph-option-c] Wrote grouped SVG: {paths['svg']}")
        print(f"[dsocr-parse-callgraph-option-c] Wrote grouped Mermaid: {paths['mermaid']}")
        print(f"[dsocr-parse-callgraph-option-c] Wrote Markdown with Mermaid: {paths['markdown']}")
    except Exception:
        print("[dsocr-parse-callgraph-option-c] WARNING: failed to generate SVG via dot; is Graphviz installed?")


if __name__ == "__main__":
    main()
