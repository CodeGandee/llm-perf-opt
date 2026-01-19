"""Human-readable summary rendering for Wan2.1 analytic reports."""

from __future__ import annotations

from collections import defaultdict

from llm_perf_opt.data.analytic_common import AnalyticModuleNode, ModuleMetricsSnapshot
from llm_perf_opt.data.wan2_1_analytic import Wan2_1AnalyticModelReport


def _index_modules(report: Wan2_1AnalyticModelReport) -> dict[str, AnalyticModuleNode]:
    return {m.module_id: m for m in report.modules}


def _index_metrics(report: Wan2_1AnalyticModelReport) -> dict[str, ModuleMetricsSnapshot]:
    return {m.module_id: m for m in report.module_metrics}


def _is_leaf(module: AnalyticModuleNode) -> bool:
    return not module.children


def iter_leaf_metrics(report: Wan2_1AnalyticModelReport) -> list[ModuleMetricsSnapshot]:
    """Return metrics snapshots for leaf modules (no children)."""

    modules = _index_modules(report)
    leaf_ids = {mid for mid, m in modules.items() if _is_leaf(m)}
    return [m for m in report.module_metrics if m.module_id in leaf_ids]


def total_flops_tflops(report: Wan2_1AnalyticModelReport) -> float:
    """Return total FLOPs (TFLOPs) for the report workload."""

    metrics = _index_metrics(report)
    root = metrics.get("diffusion/dit")
    if root is not None:
        return float(root.total_flops_tflops)
    # Fallback: sum leaf metrics (avoids double counting for hierarchical snapshots).
    return float(sum(m.total_flops_tflops for m in iter_leaf_metrics(report)))


def top_k_layers_by_flops(
    report: Wan2_1AnalyticModelReport,
    *,
    k: int = 10,
    leaf_only: bool = True,
) -> list[tuple[AnalyticModuleNode, ModuleMetricsSnapshot]]:
    """Return top-k modules by FLOPs with deterministic tie-breaking."""

    modules = _index_modules(report)
    metrics = _index_metrics(report)
    if leaf_only:
        module_ids = {mid for mid, m in modules.items() if _is_leaf(m)}
    else:
        module_ids = set(modules.keys())

    rows: list[tuple[AnalyticModuleNode, ModuleMetricsSnapshot]] = []
    for mid in module_ids:
        m = modules.get(mid)
        s = metrics.get(mid)
        if m is None or s is None:
            continue
        rows.append((m, s))

    rows.sort(key=lambda pair: (-float(pair[1].total_flops_tflops), pair[0].module_id))
    return rows[: max(int(k), 0)]


def aggregate_category_flops(
    report: Wan2_1AnalyticModelReport,
    *,
    leaf_only: bool = True,
) -> dict[str, float]:
    """Aggregate FLOPs by operator category.

    By default this aggregates only leaf modules to avoid double counting when parent modules also have snapshots.
    """

    if leaf_only:
        metrics = iter_leaf_metrics(report)
    else:
        metrics = list(report.module_metrics)

    totals: dict[str, float] = defaultdict(float)
    for snap in metrics:
        for op in snap.operator_breakdown:
            totals[str(op.category_id)] += float(op.flops_tflops)
    return dict(totals)


def top_k_categories_by_flops(
    report: Wan2_1AnalyticModelReport,
    *,
    k: int = 5,
    leaf_only: bool = True,
) -> list[tuple[str, float]]:
    """Return top-k categories by total FLOPs with deterministic tie-breaking."""

    totals = aggregate_category_flops(report, leaf_only=leaf_only)
    rows = sorted(totals.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    return rows[: max(int(k), 0)]


def render_wan2_1_summary_md(report: Wan2_1AnalyticModelReport) -> str:
    """Render a stakeholder-friendly markdown summary of a Wan2.1 analytic report."""

    total = total_flops_tflops(report)
    lines: list[str] = []
    lines.append("# Wan2.1 Analytic Report – Summary")
    lines.append("")
    lines.append(f"- Report ID: `{report.report_id}`")
    lines.append(f"- Model: `{report.model.model_id}` (`{report.model.model_variant}`)")
    lines.append(f"- Workload profile: `{report.workload.profile_id}`")
    lines.append(f"- Total FLOPs: `{total:.6f}` TFLOPs")
    lines.append("")
    lines.append("## Workload")
    lines.append("")
    lines.append(f"- batch_size: `{report.workload.batch_size}`")
    lines.append(f"- num_frames: `{report.workload.num_frames}`")
    lines.append(f"- height×width: `{report.workload.height}×{report.workload.width}`")
    lines.append(f"- num_inference_steps: `{report.workload.num_inference_steps}`")
    lines.append(f"- text_len: `{report.workload.text_len}`")
    lines.append("")
    lines.append("## Top layers by FLOPs")
    lines.append("")
    lines.append("| Rank | Module ID | Name | FLOPs (TFLOPs) | Share |")
    lines.append("|------|-----------|------|---------------|-------|")
    top_layers = top_k_layers_by_flops(report, k=10, leaf_only=True)
    for idx, (module, snap) in enumerate(top_layers, start=1):
        share = float(snap.total_flops_tflops) / total if total > 0.0 else 0.0
        lines.append(f"| {idx} | `{module.module_id}` | {module.name} | {float(snap.total_flops_tflops):.6f} | {share:.4f} |")
    lines.append("")
    lines.append("## Top categories by FLOPs")
    lines.append("")
    lines.append("| Rank | Category | FLOPs (TFLOPs) | Share |")
    lines.append("|------|----------|---------------|-------|")
    top_cats = top_k_categories_by_flops(report, k=5, leaf_only=True)
    for idx, (cat_id, flops) in enumerate(top_cats, start=1):
        share = float(flops) / total if total > 0.0 else 0.0
        lines.append(f"| {idx} | `{cat_id}` | {float(flops):.6f} | {share:.4f} |")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "aggregate_category_flops",
    "iter_leaf_metrics",
    "render_wan2_1_summary_md",
    "top_k_categories_by_flops",
    "top_k_layers_by_flops",
    "total_flops_tflops",
]
