"""Unit tests for DeepSeek-OCR compatibility after shared schema refactor."""

from __future__ import annotations

from llm_perf_opt.data import analytic_common
from llm_perf_opt.data import deepseek_ocr_analytic


def test_deepseek_ocr_reexports_shared_schema_types() -> None:
    """DeepSeek-OCR module re-exports shared schema types for compatibility."""

    assert deepseek_ocr_analytic.AnalyticModuleNode is analytic_common.AnalyticModuleNode
    assert deepseek_ocr_analytic.OperatorCategory is analytic_common.OperatorCategory
    assert deepseek_ocr_analytic.OperatorMetrics is analytic_common.OperatorMetrics
    assert deepseek_ocr_analytic.ModuleMetricsSnapshot is analytic_common.ModuleMetricsSnapshot
