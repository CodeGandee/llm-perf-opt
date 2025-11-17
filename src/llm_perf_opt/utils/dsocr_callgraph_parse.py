"""DeepSeek-OCR TorchLens call graph parsing and grouping utilities.

This module provides small data models and helpers to:

- Load the TorchLens-derived DeepSeek-OCR call graph JSON.
- Convert flattened edge keys into structured pairs.
- Detect indexed module families and derive ``for N`` depth-wise groupings.
- Derive ``parfor N`` groupings from repeated dynamic edges between parent
  and child modules at a single graph level.

The intent is to build a compact, model-aware representation of the dynamic
call graph suitable for downstream analytical models and documentation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


@dataclass
class CallGraphData:
    """Structured representation of the TorchLens call graph JSON."""

    call_counts: Dict[str, int]
    edges: Dict[Tuple[str, str], int]
    op_call_counts: Dict[str, int]
    module_children: Dict[str, List[str]]
    module_classes: Dict[str, str]


@dataclass
class GroupNode:
    """Single grouped construct derived from the call graph.

    Attributes
    ----------
    name
        Canonical module or module family name, e.g. ``sam_model.blocks`` or
        ``sam_model.blocks.0.mlp``.
    kind
        Either ``\"for\"`` (depth-wise stack of distinct modules) or
        ``\"parfor\"`` (repeated uses of the same module node under the same
        parent).
    count
        Integer multiplicity N for the grouping.
    indices
        Optional list of integer indices that participate in the group,
        typically used for ``for`` groupings over index-based module names.
    stage
        Optional coarse stage label (e.g., ``\"sam\"``, ``\"vision\"``,
        ``\"llm\"``).
    parent
        For ``parfor`` groupings, the parent module name whose dynamic edges
        repeatedly target ``name``.
    annotations
        Free-form comments or notes for human readers.
    """

    name: str
    kind: str
    count: int
    indices: Optional[List[int]] = None
    stage: Optional[str] = None
    parent: Optional[str] = None
    annotations: List[str] = field(default_factory=list)
    class_name: Optional[str] = None


@dataclass
class GroupedCallGraph:
    """Grouped view over a DeepSeek-OCR call graph."""

    groups: List[GroupNode]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _split_edge_key(key: str) -> Tuple[str, str]:
    """Split a flattened TorchLens edge key ``\"parent->child\"``."""

    parent, child = key.split("->", 1)
    return parent, child


def load_callgraph_json(path: Path) -> CallGraphData:
    """Load TorchLens-derived DeepSeek-OCR call graph from JSON.

    Parameters
    ----------
    path
        Path to ``dsocr-call-graph-torchlens.json``.
    """

    data = json.loads(path.read_text(encoding="utf-8"))

    raw_edges: Mapping[str, Any] = data.get("edges", {})
    edges: Dict[Tuple[str, str], int] = {}
    for key, value in raw_edges.items():
        parent, child = _split_edge_key(str(key))
        edges[(parent, child)] = int(value)

    call_counts: Dict[str, int] = {
        str(name): int(count) for name, count in data.get("call_counts", {}).items()
    }
    op_call_counts: Dict[str, int] = {
        str(name): int(count) for name, count in data.get("op_call_counts", {}).items()
    }

    module_children: Dict[str, List[str]] = {}
    for parent, children in data.get("module_children", {}).items():
        module_children[str(parent)] = [str(ch) for ch in children]

    raw_classes = data.get("module_classes", {})
    module_classes: Dict[str, str] = {
        str(name): str(cls_name) for name, cls_name in raw_classes.items()
    }

    return CallGraphData(
        call_counts=call_counts,
        edges=edges,
        op_call_counts=op_call_counts,
        module_children=module_children,
        module_classes=module_classes,
    )


def split_numeric_suffix(name: str) -> Tuple[str, Optional[int]]:
    """Split a module name into (base, numeric_suffix) if applicable.

    Examples
    --------
    >>> split_numeric_suffix(\"sam_model.blocks.10\")
    ('sam_model.blocks', 10)
    >>> split_numeric_suffix(\"layers.1.mlp.experts.3\")
    ('layers.1.mlp.experts', 3)
    >>> split_numeric_suffix(\"sam_model\")
    ('sam_model', None)
    """

    parts = name.split(".")
    if not parts:
        return name, None

    tail = parts[-1]
    if tail.isdigit():
        base = ".".join(parts[:-1]) if len(parts) > 1 else ""
        return base, int(tail)
    return name, None


def normalize_numeric_segments(name: str) -> str:
    """Replace numeric path segments with a wildcard ``*``.

    Examples
    --------
    >>> normalize_numeric_segments(\"sam_model.blocks.0.attn\")
    'sam_model.blocks.*.attn'
    >>> normalize_numeric_segments(\"layers.11.mlp.experts.7\")
    'layers.*.mlp.experts.*'
    """

    parts = name.split(".")
    if not parts:
        return name
    norm = ["*" if p.isdigit() else p for p in parts]
    return ".".join(norm)


def _pattern_matches(pattern: str, name: str) -> bool:
    """Return True if a dotted-name ``pattern`` matches ``name``.

    Pattern segments may contain ``*`` as a wildcard that matches any single
    path segment.
    """

    p_parts = pattern.split(".")
    n_parts = name.split(".")
    if len(p_parts) != len(n_parts):
        return False
    for p, n in zip(p_parts, n_parts):
        if p == "*":
            continue
        if p != n:
            return False
    return True


def assign_class_names(groups: Iterable[GroupNode], module_classes: Mapping[str, str]) -> None:
    """Populate ``class_name`` on each group using module_classes mapping."""

    for g in groups:
        cls: Optional[str] = None

        # Case 1: indexed family (for-groups) – prefer concrete instance like
        # ``base.idx`` over the base container.
        if g.indices and "*" not in g.name:
            candidate_name = f"{g.name}.{g.indices[0]}"
            cls = module_classes.get(candidate_name)
            if cls is None:
                cls = module_classes.get(g.name)

        # Case 2: exact module name (parfor or single instance).
        if cls is None and "*" not in g.name and not g.indices:
            cls = module_classes.get(g.name)

        # Case 3: wildcard pattern – scan for matching modules.
        if cls is None and "*" in g.name:
            for mod_name, mod_cls in module_classes.items():
                if _pattern_matches(g.name, mod_name):
                    cls = mod_cls
                    break

        g.class_name = cls


def infer_stage_from_name(name: str) -> Optional[str]:
    """Best-effort stage classification based on module name prefixes."""

    if name.startswith("sam_model"):
        return "sam"
    if name.startswith("vision_model") or name.startswith("clip_model"):
        return "vision"
    if name.startswith("layers.") or name.startswith("embed_tokens") or name.startswith("norm") or name.startswith(
        "lm_head",
    ):
        return "llm"
    return None


def find_indexed_families(
    module_names: Iterable[str],
    min_count: int = 2,
) -> Dict[str, List[int]]:
    """Detect module families that end with a numeric index.

    Families are keyed by the base prefix (e.g., ``sam_model.blocks``) and
    values are sorted unique indices observed in the call graph.
    """

    families: Dict[str, set[int]] = {}
    for name in module_names:
        base, idx = split_numeric_suffix(name)
        if idx is None:
            continue
        if base not in families:
            families[base] = set()
        families[base].add(idx)

    result: Dict[str, List[int]] = {}
    for base, indices in families.items():
        if len(indices) >= int(min_count):
            result[base] = sorted(indices)
    return result


def build_for_groups(
    callgraph: CallGraphData,
    min_family_size: int = 2,
    families: Optional[Mapping[str, List[int]]] = None,
) -> List[GroupNode]:
    """Build ``for N`` groups from indexed module families.

    Parameters
    ----------
    callgraph
        Parsed call graph data.
    min_family_size
        Minimum number of indexed siblings required to emit a ``for`` group.
    families
        Optional precomputed mapping ``base -> sorted indices``. If omitted,
        this will be derived from ``callgraph.call_counts``.
    """

    if families is None:
        families = find_indexed_families(callgraph.call_counts.keys(), min_count=min_family_size)
    groups: List[GroupNode] = []

    for base, indices in families.items():
        if not indices:
            continue

        # Form contiguous ranges of indices.
        start = indices[0]
        current_range: List[int] = [start]

        # Find a representative parent for this family, if any.
        parent_candidates: List[str] = []
        for parent, children in callgraph.module_children.items():
            for idx in indices:
                child_name = f"{base}.{idx}"
                if child_name in children:
                    parent_candidates.append(parent)
                    break
        parent_name: Optional[str] = None
        if parent_candidates:
            # Use a deterministic choice if multiple parents exist.
            parent_name = sorted(set(parent_candidates))[0]

        def flush_range(range_indices: List[int]) -> None:
            if not range_indices:
                return
            count = len(range_indices)
            group = GroupNode(
                name=base,
                kind="for",
                count=count,
                indices=list(range_indices),
                stage=infer_stage_from_name(base),
                parent=parent_name,
            )
            groups.append(group)

        for prev, idx in zip(indices, indices[1:]):
            if idx == prev + 1:
                current_range.append(idx)
            else:
                flush_range(current_range)
                current_range = [idx]

        flush_range(current_range)

    return groups


def build_parfor_groups(
    callgraph: CallGraphData,
    min_edge_count: int = 2,
    indexed_families: Optional[Mapping[str, List[int]]] = None,
) -> List[GroupNode]:
    """Build ``parfor N`` groups based on repeated parent->child edges.

    To keep semantics clean, modules that belong to an indexed family
    (i.e., stacked modules like ``sam_model.blocks.{i}``) are excluded
    from ``parfor`` groups so they are represented only via ``for N``
    at the family level.
    """

    indexed_bases: set[str] = set()
    if indexed_families is None:
        indexed_families = find_indexed_families(callgraph.call_counts.keys(), min_count=2)
    indexed_bases.update(indexed_families.keys())

    groups: List[GroupNode] = []
    for (parent, child), count in callgraph.edges.items():
        if count < int(min_edge_count):
            continue

        base, _ = split_numeric_suffix(child)
        if base in indexed_bases:
            # This child is part of a stacked family; represent it via `for N`
            # and avoid cluttering the grouped graph with per-index parfor.
            continue

        group = GroupNode(
            name=child,
            kind="parfor",
            count=int(count),
            indices=None,
            stage=infer_stage_from_name(child),
            parent=parent,
        )
        groups.append(group)
    return groups


def aggregate_parfor_groups(parfor_groups: Iterable[GroupNode]) -> List[GroupNode]:
    """Aggregate ``parfor`` groups across indexed instances.

    Groups like ``sam_model.blocks.0.attn`` and ``sam_model.blocks.1.attn``
    become a single logical module type ``sam_model.blocks.*.attn`` with
    counts summed across all occurrences (and similarly for parents).
    """

    buckets: Dict[Tuple[Optional[str], str], int] = {}

    for g in parfor_groups:
        if g.kind != "parfor":
            continue
        parent_norm = normalize_numeric_segments(g.parent) if g.parent else None
        child_norm = normalize_numeric_segments(g.name)
        key = (parent_norm, child_norm)
        buckets[key] = buckets.get(key, 0) + int(g.count)

    aggregated: List[GroupNode] = []
    for (parent_norm, child_norm), count in buckets.items():
        aggregated.append(
            GroupNode(
                name=child_norm,
                kind="parfor",
                count=int(count),
                indices=None,
                stage=infer_stage_from_name(child_norm),
                parent=parent_norm,
            ),
        )

    return aggregated


def compute_grouped_callgraph(
    callgraph: CallGraphData,
    min_family_size: int = 2,
    min_edge_count: int = 2,
    metadata: Optional[MutableMapping[str, Any]] = None,
) -> GroupedCallGraph:
    """Compute grouped view (``for`` and ``parfor``) over a call graph."""

    families = find_indexed_families(callgraph.call_counts.keys(), min_count=min_family_size)

    for_groups = build_for_groups(
        callgraph,
        min_family_size=min_family_size,
        families=families,
    )
    parfor_raw = build_parfor_groups(
        callgraph,
        min_edge_count=min_edge_count,
        indexed_families=families,
    )
    parfor_groups = aggregate_parfor_groups(parfor_raw)

    groups: List[GroupNode] = []
    groups.extend(for_groups)
    groups.extend(parfor_groups)

    md: Dict[str, Any] = {}
    if metadata is not None:
        md.update(metadata)

    # Attach best-effort class names to groups to support human-friendly
    # labels of the form ``ClassName @ runtime_pattern``.
    assign_class_names(groups, callgraph.module_classes)

    return GroupedCallGraph(groups=groups, metadata=md)


def grouped_to_json_dict(grouped: GroupedCallGraph) -> Dict[str, Any]:
    """Convert a grouped call graph into a JSON-serializable dict."""

    def serialize_group(g: GroupNode) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "name": g.name,
            "kind": g.kind,
            "count": int(g.count),
        }
        if g.indices is not None:
            data["indices"] = list(g.indices)
        if g.stage is not None:
            data["stage"] = g.stage
        if g.parent is not None:
            data["parent"] = g.parent
        if g.annotations:
            data["annotations"] = list(g.annotations)
        return data

    return {
        "groups": [serialize_group(g) for g in grouped.groups],
        "metadata": dict(grouped.metadata),
    }


def grouped_to_markdown(grouped: GroupedCallGraph) -> str:
    """Render a grouped call graph as a human-readable Markdown summary."""

    lines: List[str] = []
    lines.append("# DeepSeek-OCR Call Graph – Grouped View")
    if grouped.metadata:
        lines.append("")
        lines.append("## Metadata")
        for key, value in grouped.metadata.items():
            lines.append(f"- **{key}**: {value}")

    lines.append("")
    lines.append("## Groups")

    if not grouped.groups:
        lines.append("")
        lines.append("_No groups found with current thresholds._")
        return "\n".join(lines)

    for g in grouped.groups:
        indices_repr = ""
        if g.indices:
            if len(g.indices) == 1:
                indices_repr = f"[{g.indices[0]}]"
            else:
                indices_repr = f"[{g.indices[0]}..{g.indices[-1]}]"

        runtime_pattern = f"{g.name}{indices_repr}".strip()

        if g.class_name:
            header = f"{g.class_name} @ {runtime_pattern}"
        else:
            header = runtime_pattern

        parts: List[str] = []
        kind_label = "for" if g.kind == "for" else "parfor"
        parts.append(f"`{header}` {kind_label} {g.count}")

        if g.parent:
            parts.append(f"under `{g.parent}`")
        if g.stage:
            parts.append(f"(stage: {g.stage})")

        line = "- " + " ".join(parts)
        if g.annotations:
            note = " ".join(g.annotations)
            line += f"  # {note}"
        lines.append(line)

    return "\n".join(lines)


def grouped_to_dot(grouped: GroupedCallGraph) -> str:
    """Render a grouped call graph as a Graphviz DOT string.

    Nodes represent grouped modules (``for`` / ``parfor``). Directed edges
    encode the parent→child calling relationship implied by each group.
    """

    lines: List[str] = []
    lines.append("digraph dsocr_grouped {")
    lines.append("  rankdir=LR;")

    node_ids: Dict[str, str] = {}

    def register_node(key: str, label: str, shape_box: bool = True) -> str:
        """Register a node if not yet present and return its DOT id."""

        if key in node_ids:
            return node_ids[key]
        # Use the key (a human-friendly identifier such as
        # ``ClassName @ runtime_pattern`` or a module runtime name) as the DOT
        # node id so that edges are readable.
        node_id = f"\"{key}\""
        node_ids[key] = node_id
        esc_label = label.replace('"', '\\"')
        shape_attr = ", shape=box" if shape_box else ""
        lines.append(f'  {node_id} [label="{esc_label}"{shape_attr}];')
        return node_id

    group_node_ids: List[str] = []

    # First, create nodes for all groups.
    for idx, g in enumerate(grouped.groups):
        if g.indices:
            if len(g.indices) == 1:
                indices_repr = f"[{g.indices[0]}]"
            else:
                indices_repr = f"[{g.indices[0]}..{g.indices[-1]}]"
        else:
            indices_repr = ""

        runtime_pattern = f"{g.name}{indices_repr}".strip()
        if g.class_name:
            header = f"{g.class_name} @ {runtime_pattern}"
        else:
            header = runtime_pattern

        kind_part = f"{g.kind} {g.count}"

        label_parts = [header, kind_part]
        if g.stage:
            label_parts.append(f"stage={g.stage}")
        if g.parent:
            label_parts.append(f"parent={g.parent}")

        label = "\\n".join([part for part in label_parts if part])
        # Use the header (ClassName @ runtime-pattern) as the node id key.
        key = header
        node_id = register_node(key, label, shape_box=True)
        group_node_ids.append(node_id)

    # Then, add edges based on parent relationships.
    for idx, g in enumerate(grouped.groups):
        if not g.parent:
            continue

        parent_label = g.parent
        parent_key = parent_label
        parent_id = register_node(parent_key, parent_label, shape_box=False)
        child_id = group_node_ids[idx]

        edge_label = ""
        if g.kind == "parfor":
            edge_label = f' [label="parfor {g.count}"]'
        elif g.kind == "for":
            edge_label = f' [label="for {g.count}"]'

        lines.append(f"  {parent_id} -> {child_id}{edge_label};")

    lines.append("}")
    return "\n".join(lines)
