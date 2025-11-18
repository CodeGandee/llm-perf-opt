"""Transform TorchInfo static graph by pruning torch.* layers.

This script reads a TorchInfo layers JSON produced by
``dsocr_find_static_components.py`` (torchinfo-layers.json), removes
all layers whose fully-qualified class name starts with ``torch.``,
and aggregates their numeric statistics into the nearest non-torch
ancestor module.

Usage
-----
    pixi run -e rtx5090 python scripts/analytical/dsocr_static_transform_graph_no_torch.py \\
        tmp/op-analysis/static/<run_id>/torchinfo-layers.json \\
        --output tmp/op-analysis/static/<run_id>/torchinfo-layers-no-torch.json

If ``--output`` is omitted, the transformed JSON is written to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


NUMERIC_FIELDS: tuple[str, ...] = (
    "num_params",
    "trainable_params",
    "param_bytes",
    "output_bytes",
    "macs",
)


def _zero_stats() -> Dict[str, int]:
    """Return a zero-initialized stats mapping."""

    return {field: 0 for field in NUMERIC_FIELDS}


def _stats_from_node(node: Dict[str, Any]) -> Dict[str, int]:
    """Extract numeric stats from a layer node with sane defaults."""

    stats: Dict[str, int] = {}
    for field in NUMERIC_FIELDS:
        value = node.get(field, 0)
        # TorchInfo may store None; treat as zero.
        stats[field] = int(value or 0)
    return stats


def _add_stats(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    """Element-wise addition of two stats dictionaries."""

    out: Dict[str, int] = {}
    for field in NUMERIC_FIELDS:
        out[field] = int(a.get(field, 0)) + int(b.get(field, 0))
    return out


def _is_torch_builtin(node: Dict[str, Any]) -> bool:
    """Identify torch built-in layers using the exported flag."""

    return bool(node.get("is_torch_builtin"))


def _transform_node(
    node: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Recursively prune torch.* nodes and aggregate their stats.

    Returns
    -------
    kept_nodes : list of dict
        One or more transformed nodes that should be attached at the caller.
    agg_stats : dict
        Aggregated statistics from pruned (torch.*) nodes that still need
        to be added into the nearest non-torch ancestor.
    """

    children = node.get("children") or []
    kept_children: List[Dict[str, Any]] = []
    agg_from_children: Dict[str, int] = _zero_stats()

    for child in children:
        child_kept, child_agg = _transform_node(child)
        kept_children.extend(child_kept)
        agg_from_children = _add_stats(agg_from_children, child_agg)

    own_stats = _stats_from_node(node)

    if _is_torch_builtin(node):
        # This node is pruned. Its own stats and any aggregated stats
        # from pruned descendants propagate upward to the nearest
        # non-torch ancestor. Any non-torch descendants become direct
        # children of that ancestor.
        agg_total = _add_stats(own_stats, agg_from_children)
        return kept_children, agg_total

    # This node is kept. Aggregate stats from pruned descendants into it.
    total_stats = _add_stats(own_stats, agg_from_children)

    new_node: Dict[str, Any] = dict(node)
    new_node["children"] = kept_children
    for field in NUMERIC_FIELDS:
        new_node[field] = total_stats[field]

    # Aggregation has been absorbed at this level; do not propagate up.
    return [new_node], _zero_stats()


def _transform_hierarchy(hierarchy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply pruning/aggregation to a list of root nodes."""

    kept_roots: List[Dict[str, Any]] = []
    for root in hierarchy:
        root_kept, root_agg = _transform_node(root)
        kept_roots.extend(root_kept)
        # Any remaining aggregated stats would indicate a root that was
        # fully pruned; in practice, model roots are non-torch modules.
        if any(val != 0 for val in root_agg.values()):
            # Best-effort: attach to the first kept root if present.
            if kept_roots:
                for field in NUMERIC_FIELDS:
                    kept_roots[0][field] = kept_roots[0].get(field, 0) + root_agg[field]
    return kept_roots


def _flatten_hierarchy(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a flat list of nodes from a hierarchical tree."""

    flat: List[Dict[str, Any]] = []

    def visit(n: Dict[str, Any]) -> None:
        flat.append(n)
        for child in n.get("children") or []:
            visit(child)

    for root in nodes:
        visit(root)
    return flat


def _format_shape_field(value: Any) -> str:
    """Format an input/output shape field for text summary."""

    if value is None:
        return "--"
    # TorchInfo uses empty list for unspecified shapes.
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return "--"
    return str(value)


def _format_int_field(value: Any) -> str:
    """Format an integer statistic for text summary."""

    try:
        iv = int(value)
    except (TypeError, ValueError):
        iv = 0
    if iv == 0:
        return "--"
    return f"{iv:,}"


def _build_text_summary(hierarchy: List[Dict[str, Any]]) -> str:
    """Build a human-readable summary table similar to TorchInfo."""

    lines: List[str] = []

    header = (
        "Layer (type (var_name):depth-idx)"
        "                                      Input Shape"
        "               Output Shape"
        "              Param #"
        "                   Kernel Shape"
        "              Mult-Adds"
    )
    bar = "=" * len(header)

    lines.append(bar)
    lines.append(header)
    lines.append(bar)

    def visit(node: Dict[str, Any]) -> None:
        depth = int(node.get("depth", 0) or 0)
        depth_index = int(node.get("depth_index", 0) or 0)
        class_name = str(node.get("class_name") or "")
        var_name = str(node.get("var_name") or "")

        # Build prefix with a simple tree-like indentation.
        if depth == 0:
            layer_label = f"{class_name} ({class_name})"
        else:
            indent = "│   " * max(depth - 1, 0) + "├─"
            layer_label = f"{indent}{class_name} ({var_name})"

        col0 = f"{layer_label}: {depth}-{depth_index}"
        col1 = _format_shape_field(node.get("input_size"))
        col2 = _format_shape_field(node.get("output_size"))
        col3 = _format_int_field(node.get("num_params"))
        col4 = _format_shape_field(node.get("kernel_size"))
        col5 = _format_int_field(node.get("macs"))

        line = f"{col0:<70}{col1:<26}{col2:<26}{col3:<27}{col4:<26}{col5}"
        lines.append(line)

        for child in node.get("children") or []:
            visit(child)

    for root in hierarchy:
        visit(root)

    text = "Total output lines: " + str(len(lines)) + "\n\n" + "\n".join(lines)
    return text


def _build_unique_layers(hierarchy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect unique layers by qualified class name with parent/child types."""

    by_qualname: Dict[str, Dict[str, Any]] = {}

    def ensure_entry(qual: str) -> Dict[str, Any]:
        if qual not in by_qualname:
            by_qualname[qual] = {
                "class_name_qualified": qual,
                "count": 0,
                "parents": set(),
                "children": set(),
                "class_names": set(),
                "var_names": set(),
                "instance_names": set(),
                "filepaths": set(),
            }
        return by_qualname[qual]

    def visit(node: Dict[str, Any], parent_qual: str | None) -> None:
        qual = node.get("class_name_qualified")
        if isinstance(qual, str) and qual:
            entry = ensure_entry(qual)
            entry["count"] = int(entry.get("count", 0)) + 1

            node_class = node.get("class_name")
            node_var = node.get("var_name")
            node_module_name = node.get("instance_name")
            node_filepath = node.get("filepath")

            if isinstance(node_class, str) and node_class:
                entry["class_names"].add(node_class)
            if isinstance(node_var, str) and node_var:
                entry["var_names"].add(node_var)
            if isinstance(node_module_name, str):
                entry["instance_names"].add(node_module_name)
            if isinstance(node_filepath, str) and node_filepath:
                entry["filepaths"].add(node_filepath)

            if isinstance(parent_qual, str) and parent_qual:
                parent_entry = ensure_entry(parent_qual)
                entry["parents"].add(parent_qual)
                parent_entry["children"].add(qual)

            this_parent_qual = qual
        else:
            this_parent_qual = parent_qual

        for child in node.get("children") or []:
            visit(child, this_parent_qual)

    for root in hierarchy:
        visit(root, parent_qual=None)

    unique_layers: List[Dict[str, Any]] = []
    for qual, entry in by_qualname.items():
        class_names = sorted(entry.get("class_names", set()))
        var_names = sorted(entry.get("var_names", set()))
        instance_names = sorted(entry.get("instance_names", set()))
        filepaths = sorted(entry.get("filepaths", set()))

        unique_layers.append(
            {
                "class_name_qualified": qual,
                "class_name": class_names[0] if class_names else None,
                "var_name": var_names,
                "instance_name": instance_names,
                "filepaths": filepaths,
                "count": int(entry.get("count", 0)),
                "parents": sorted(entry.get("parents", set())),
                "children": sorted(entry.get("children", set())),
            },
        )

    unique_layers.sort(key=lambda x: x["class_name_qualified"])
    return unique_layers


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prune torch.* layers from TorchInfo hierarchy and aggregate "
            "their stats into the nearest non-torch ancestor."
        ),
    )
    parser.add_argument(
        "layers_json",
        help="Path to torchinfo-layers.json produced by dsocr_find_static_components.py.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output path (default: write JSON to stdout).",
    )
    args = parser.parse_args()

    layers_path = Path(args.layers_json).resolve()
    if not layers_path.is_file():
        raise SystemExit(f"Input JSON not found: {layers_path}")

    data = json.loads(layers_path.read_text(encoding="utf-8"))

    hierarchy = data.get("hierarchy")
    if not isinstance(hierarchy, list):
        raise SystemExit(f"Input JSON {layers_path} does not contain a 'hierarchy' list")

    transformed_hierarchy = _transform_hierarchy(hierarchy)
    transformed_flat = _flatten_hierarchy(transformed_hierarchy)

    # Preserve existing metadata but replace hierarchy / layers_flat
    output_payload: Dict[str, Any] = dict(data)
    output_payload["hierarchy"] = transformed_hierarchy
    output_payload["layers_flat"] = transformed_flat

    output_text = json.dumps(output_payload, indent=2)

    if args.output is not None:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")

        # Also emit a text summary similar to torchinfo-summary.txt.
        summary_text = _build_text_summary(transformed_hierarchy)
        summary_path = out_path.with_name("no-torch-summary.txt")
        summary_path.write_text(summary_text, encoding="utf-8")

        # Emit unique layer types by qualified name with parent/child relations.
        unique_layers = _build_unique_layers(transformed_hierarchy)
        unique_payload = {
            "generated_from": str(layers_path),
            "num_unique_layers": len(unique_layers),
            "layers": unique_layers,
        }
        unique_path = out_path.with_name("unique-layers.json")
        unique_path.write_text(json.dumps(unique_payload, indent=2), encoding="utf-8")
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
