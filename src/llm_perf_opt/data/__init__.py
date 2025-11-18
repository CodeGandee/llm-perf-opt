"""Data utilities and domain models for ``llm_perf_opt``.

This package hosts small data helpers plus attrs-based domain models
for profiling (Stage 1/2) and analytic modeling features.
"""

from __future__ import annotations

from .models import KernelRecord, LLMProfileReport, OperatorSummary, StageTiming, Stats
from .deepseek_ocr_analytic import (
    AnalyticModelReport,
    AnalyticModuleNode,
    DeepSeekOCRModelSpec,
    ModuleMetricsSnapshot,
    OCRWorkloadProfile,
    OperatorCategory,
    OperatorMetrics,
    OperatorSpec,
    TargetOperatorList,
)

__all__ = [
    # Stage 1/2 profiling domain models
    "KernelRecord",
    "StageTiming",
    "OperatorSummary",
    "Stats",
    "LLMProfileReport",
    # DeepSeek-OCR analytic modeling domain models
    "DeepSeekOCRModelSpec",
    "OCRWorkloadProfile",
    "AnalyticModuleNode",
    "OperatorCategory",
    "OperatorMetrics",
    "ModuleMetricsSnapshot",
    "AnalyticModelReport",
    "OperatorSpec",
    "TargetOperatorList",
]

