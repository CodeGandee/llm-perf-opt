"""Markdown generation for DeepSeek-OCR analytic layer reports.

This module renders lightweight per-layer and summary Markdown
documentation from :class:`AnalyticModelReport` artifacts produced by
the analytic modeling pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from llm_perf_opt.data.deepseek_ocr_analytic import (
    AnalyticModelReport,
    AnalyticModuleNode,
    ModuleMetricsSnapshot,
)


def _index_modules(report: AnalyticModelReport) -> Dict[str, AnalyticModuleNode]:
    return {m.module_id: m for m in report.modules}


def _index_metrics(report: AnalyticModelReport) -> Dict[str, ModuleMetricsSnapshot]:
    return {m.module_id: m for m in report.module_metrics}


def _format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _write_summary(report: AnalyticModelReport, out_dir: Path) -> None:
    modules_by_id = _index_modules(report)
    metrics_by_id = _index_metrics(report)

    # Sort modules with metrics by descending share_of_model_time.
    rows: list[tuple[AnalyticModuleNode, ModuleMetricsSnapshot]] = []
    for module_id, metric in metrics_by_id.items():
        module = modules_by_id.get(module_id)
        if module is not None:
            rows.append((module, metric))
    rows.sort(key=lambda pair: pair[1].share_of_model_time, reverse=True)

    lines: list[str] = []
    lines.append("# DeepSeek-OCR Analytic Model – Summary")
    lines.append("")
    lines.append(f"- Report ID: `{report.report_id}`")
    lines.append(f"- Model variant: `{report.model.model_variant}`")
    lines.append(f"- Workload profile: `{report.workload.profile_id}`")
    lines.append("")
    lines.append("## Module Metrics")
    lines.append("")
    lines.append(
        "| Module | Stage | Calls | Time (ms) | FLOPs (TFLOPs) | "
        "Weights (GB) | Activations (GB) | KV cache (GB) | Time share |"
    )
    lines.append(
        "|--------|-------|-------|-----------|----------------|"
        "--------------|------------------|--------------|-----------|"
    )

    for module, metric in rows:
        lines.append(
            "| {name} | {stage} | {calls} | {time_ms} | {flops} | {w_gb} | "
            "{a_gb} | {kv_gb} | {share} |".format(
                name=module.name,
                stage=module.stage,
                calls=metric.calls,
                time_ms=_format_float(metric.total_time_ms, digits=3),
                flops=_format_float(metric.total_flops_tflops, digits=3),
                w_gb=_format_float(metric.memory_weights_gb, digits=4),
                a_gb=_format_float(metric.memory_activations_gb, digits=4),
                kv_gb=_format_float(metric.memory_kvcache_gb, digits=4),
                share=_format_float(metric.share_of_model_time, digits=3),
            ),
        )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_module_doc(
    out_dir: Path,
    module: AnalyticModuleNode,
    metric: Optional[ModuleMetricsSnapshot],
) -> None:
    safe_id = module.module_id.replace("/", "_")
    path = out_dir / f"{safe_id}.md"

    lines: list[str] = []
    lines.append(f"# {module.name} (`{module.module_id}`)")
    lines.append("")
    lines.append(f"- Stage: `{module.stage}`")
    lines.append(f"- Qualified class: `{module.qualified_class_name}`")
    lines.append(f"- Parent: `{module.parent_id or '-'}`")
    lines.append(f"- Children: {', '.join(module.children) if module.children else 'none'}")
    lines.append(f"- Repetition: `{module.repetition}`")
    if module.repetition_count is not None:
        lines.append(f"- Repetition count: `{module.repetition_count}`")
    lines.append("")

    if module.constructor_params:
        lines.append("## Constructor parameters")
        lines.append("")
        for key, value in module.constructor_params.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    lines.append("## Analytic metrics")
    lines.append("")
    if metric is None:
        lines.append("No metrics recorded for this module in the current report.")
    else:
        lines.append(f"- Calls: `{metric.calls}`")
        lines.append(f"- Total time (ms): `{_format_float(metric.total_time_ms, 3)}`")
        lines.append(f"- Total FLOPs (TFLOPs): `{_format_float(metric.total_flops_tflops, 3)}`")
        lines.append(f"- Total I/O (Tb): `{_format_float(metric.total_io_tb, 3)}`")
        lines.append(f"- Weight memory (GB): `{_format_float(metric.memory_weights_gb, 4)}`")
        lines.append(f"- Activation memory (GB): `{_format_float(metric.memory_activations_gb, 4)}`")
        lines.append(f"- KV cache memory (GB): `{_format_float(metric.memory_kvcache_gb, 4)}`")
        lines.append(f"- Share of model time: `{_format_float(metric.share_of_model_time, 3)}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analytic_layer_docs(report: AnalyticModelReport) -> None:
    """Render per-layer and summary Markdown docs for an analytic report.

    Parameters
    ----------
    report : AnalyticModelReport
        Analytic model report with module metadata and metrics.
    """

    out_dir = Path(report.layer_docs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_summary(report, out_dir)

    modules_by_id = _index_modules(report)
    metrics_by_id = _index_metrics(report)
    for module_id, module in modules_by_id.items():
        metric = metrics_by_id.get(module_id)
        _write_module_doc(out_dir, module, metric)

