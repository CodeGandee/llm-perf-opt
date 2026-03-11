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

Layer docs: `context/hints/dsocr-kb/ops/op-DeepseekOCRModel.md`

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

### What has been implemented

- Extended `DeepseekOCRModel(BaseLayer)` in `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  to aggregate all BaseLayer metrics (FLOPs, I/O, memory) across a configurable vision stack and a repeated decoder
  stack, with an internal `_CompositeLayer` helper used for grouping vision components (`ImageEncoderViT`, `VitModel`,
  `MlpProjector`).
- Added an analytic pipeline to `DeepseekOCRStaticAnalyzer` in `src/llm_perf_opt/runners/dsocr_analyzer.py`:
  - `_build_model_spec` / `_build_workload_profile` construct `DeepSeekOCRModelSpec` and `OCRWorkloadProfile` for the
    `dsocr-standard-v1` workload.
  - `_build_vision_layers` and `_build_decoder_layer` instantiate the DeepSeek-OCR analytic layers using the shapes and
    configs from the Phase 3–5 guides.
  - `_build_module_nodes_and_metrics` builds `AnalyticModuleNode` and `ModuleMetricsSnapshot` entries for the vision
    modules, decoder stack, and root `DeepseekOCRModel`, including theoretical time estimates derived from TFLOPs and
    `get_peak_tflops`.
  - `run_analytic(...)` assembles an `AnalyticModelReport`, writes `report.json` / `report.yaml` under
    `tmp/profile-output/<run_id>/static_analysis/analytic_model/`, and calls the Markdown renderer.
- Implemented `write_analytic_layer_docs(report)` in `src/llm_perf_opt/visualize/analytic_layers.py` to emit a
  `summary.md` table plus per-module Markdown pages under the directory referenced by
  `AnalyticModelReport.layer_docs_dir`.
- Added a manual smoke test `tests/manual/deepseek_ocr/manual_deepseek_ocr_performance_report.py` that runs
  `python -m llm_perf_opt.runners.dsocr_analyzer --mode analytic` and asserts the presence of
  `report.json` / `report.yaml` and `layers/` for a chosen `run_id`.
- Added unit tests in `tests/unit/deepseek_ocr/test_analytic_layers_scaling.py` to ensure representative analytic layers
  (`LlamaFlashAttention2`, `DeepseekV2DecoderLayer`, `VitModel`) report non-negative metrics and scale monotonically
  with sequence length and hidden size.
- Updated `extern/modelmeter/models/deepseek_ocr/README.md` and
  `specs/001-deepseek-ocr-modelmeter/quickstart.md` to document the analytic CLI invocation, the analytic artifact
  layout under `tmp/profile-output/<run_id>/static_analysis/analytic_model/`, and the location of the generated
  Markdown layer docs.

### How to verify

- Ensure a DeepSeek-OCR checkpoint is available at
  `/workspace/code/llm-perf-opt/models/deepseek-ocr` and run:

  ```bash
  cd /workspace/code/llm-perf-opt
  pixi run -e rtx5090 python tests/manual/deepseek_ocr/manual_deepseek_ocr_performance_report.py
  ```

  This will invoke `python -m llm_perf_opt.runners.dsocr_analyzer --mode analytic` with a fresh `run_id` and check for:
  - `tmp/profile-output/<run_id>/static_analysis/analytic_model/report.json`
  - `tmp/profile-output/<run_id>/static_analysis/analytic_model/report.yaml`
  - `tmp/profile-output/<run_id>/static_analysis/analytic_model/layers/summary.md` and per-module `.md` files.

- Optionally, run the scaling tests for additional confidence:

  ```bash
  pixi run -e rtx5090 pytest tests/unit/deepseek_ocr/test_analytic_layers_scaling.py
  ```
