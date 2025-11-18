# Implementation Guide: Phase 6 – Core Aggregation & Analytic Pipeline (US1, Part 4)

**Phase**: 6 | **Feature**: DeepSeek-OCR Analytic Modeling in ModelMeter (`001-deepseek-ocr-modelmeter`) | **Tasks**: T029–T034

## Goal

Wire vision, LLaMA, and decoder analytic layers into a root `DeepseekOCRModel` aggregator, expose an analytic mode in `DeepseekOCRStaticAnalyzer`, and generate both structured JSON/YAML artifacts and Markdown layer/operator documentation for DeepSeek-OCR.

## Public APIs

### T029/T030: Manual script and unit tests

- `T029` – Manual performance-report script  
  `tests/manual/deepseek_ocr/manual_deepseek_ocr_performance_report.py`
- `T030` – Unit tests for analytic layer scaling  
  `tests/unit/deepseek_ocr/test_analytic_layers_scaling.py`

These validate end-to-end artifact generation and basic metric monotonicity.

### T031: `DeepseekOCRModel(BaseLayer)` aggregator – `core/deepseek_ocr_model.py`

```python
# extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py

from __future__ import annotations

from extern.modelmeter.layers.base import BaseLayer


class DeepseekOCRModel(BaseLayer):
    """Aggregate analytic model for DeepSeek-OCR."""

    def __init__(
        self,
        *,
        vision: BaseLayer,
        decoder_blocks: list[BaseLayer],
        head: BaseLayer | None = None,
    ) -> None:
        super().__init__()
        self.m_vision = vision
        self.m_decoder_blocks = decoder_blocks
        self.m_head = head

    def forward_tensor_core_flops(self) -> float:
        total = self.m_vision.forward_tensor_core_flops() or 0.0
        for block in self.m_decoder_blocks:
            total += block.forward_tensor_core_flops() or 0.0
        if self.m_head is not None:
            total += self.m_head.forward_tensor_core_flops() or 0.0
        return total

    # Implement remaining BaseLayer metrics by aggregating sublayers.
```

### T032: Analytic mode in `DeepseekOCRStaticAnalyzer` – `src/llm_perf_opt/runners/dsocr_analyzer.py`

Add a method (or path) that constructs analytic layers and returns an `AnalyticModelReport`.

```python
from llm_perf_opt.data.deepseek_ocr_analytic import AnalyticModelReport
from llm_perf_opt.utils.paths import analytic_model_dir, analytic_layer_docs_dir


class DeepseekOCRStaticAnalyzer:
    ...

    def run_analytic(self, config: AnalysisConfig) -> AnalyticModelReport:
        """Build analytic model report for DeepSeek-OCR."""
        ...
```

### T033: Markdown generation utilities – `src/llm_perf_opt/visualize/analytic_layers.py`

Render per-layer and summary Markdown docs from `AnalyticModelReport`.

### T034: Docs updates – READMEs and quickstart

- Update `extern/modelmeter/models/deepseek_ocr/README.md`.
- Update `specs/001-deepseek-ocr-modelmeter/quickstart.md` to document analytic CLI and artifact layout.

---

## Phase Integration

```mermaid
graph LR
    V[Vision layers (Phase 3)] --> M[DeepseekOCRModel]
    L[LLaMA layers (Phase 4)] --> D[Decoder layers (Phase 5)]
    D --> M
    M --> A[run_analytic in dsocr_analyzer]
    A --> R[AnalyticModelReport JSON/YAML]
    A --> MD[Markdown layer docs]
    R --> T029[Manual performance script]
    R --> T030[Unit scaling tests]
```

---

## Testing

### Test Input

- All analytic layers from Phases 3–5 implemented.
- DeepSeek-OCR model weights and TorchInfo artifacts available.

### Test Procedure

```bash
cd /workspace/code/llm-perf-opt

pixi run -e rtx5090 python tests/manual/deepseek_ocr/manual_deepseek_ocr_performance_report.py
pixi run -e rtx5090 pytest tests/unit/deepseek_ocr/test_analytic_layers_scaling.py
```

### Test Output

- JSON/YAML analytic reports under `tmp/profile-output/<run_id>/static_analysis/analytic_model/`.
- Markdown layer docs under the `layers/` subdirectory referenced by `AnalyticModelReport.layer_docs_dir`.
- All scaling tests pass for representative shapes.

---

## References

- Tasks: `specs/001-deepseek-ocr-modelmeter/tasks.md` (Phase 6, T029–T034)
- Data model: `specs/001-deepseek-ocr-modelmeter/data-model.md`

---

## Implementation Summary

*(to be filled after implementation)*

### What has been implemented

- (after implementation) Summarize aggregator, analytic runner, and Markdown generation wiring.

### How to verify

- (after implementation) Describe how to locate and inspect all artifacts for a given `report_id`.

