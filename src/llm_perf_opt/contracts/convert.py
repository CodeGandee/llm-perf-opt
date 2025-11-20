"""Contract/domain conversion utilities using `cattrs`.

Provides a shared converter for transforming between domain models and
contract models.
"""

from __future__ import annotations

from typing import List

from cattrs import Converter

from llm_perf_opt.contracts.models import (
    AnalyticModuleSummary,
    DeepSeekOCRAnalyticReportSummary,
)
from llm_perf_opt.data.deepseek_ocr_analytic import AnalyticModelReport

# Public converter instance; register hooks as needed.
converter = Converter()


def _build_deepseek_ocr_summary(report: AnalyticModelReport) -> DeepSeekOCRAnalyticReportSummary:
    """Construct a DeepSeekOCRAnalyticReportSummary from an AnalyticModelReport."""

    module_index = {m.module_id: m for m in report.modules}

    summaries: List[AnalyticModuleSummary] = []
    for snap in report.module_metrics:
        module = module_index.get(snap.module_id)
        name = module.name if module is not None else snap.module_id
        stage = module.stage if module is not None else "other"
        summaries.append(
            AnalyticModuleSummary(
                module_id=snap.module_id,
                name=name,
                stage=stage,  # type: ignore[arg-type]
                share_of_model_time=snap.share_of_model_time,
                total_time_ms=snap.total_time_ms,
                total_flops_tflops=snap.total_flops_tflops,
                memory_activations_gb=snap.memory_activations_gb,
            ),
        )

    # Order modules by descending share of model time.
    summaries.sort(key=lambda s: s.share_of_model_time, reverse=True)

    return DeepSeekOCRAnalyticReportSummary(
        report_id=report.report_id,
        model_variant=report.model.model_variant,
        workload_profile_id=report.workload.profile_id,
        predicted_total_time_ms=report.predicted_total_time_ms,
        measured_total_time_ms=report.measured_total_time_ms,
        predicted_vs_measured_ratio=report.predicted_vs_measured_ratio,
        top_modules=summaries,
        notes=report.notes,
    )


def register_deepseek_ocr_hooks(conv: Converter) -> None:
    """Register AnalyticModelReport <-> DeepSeek-OCR contract hooks.

    The primary use case is to expose a summary-friendly JSON view over
    AnalyticModelReport for the ``/analytic/deepseek-ocr/{report_id}/summary``
    endpoint while keeping the full-model payload aligned with the internal
    attrs domain model.
    """

    def _unstructure_report(report: AnalyticModelReport) -> dict:
        summary = _build_deepseek_ocr_summary(report)
        return conv.unstructure(summary)

    conv.register_unstructure_hook(AnalyticModelReport, _unstructure_report)


# Configure the shared converter on import so downstream callers can rely on it.
register_deepseek_ocr_hooks(converter)

