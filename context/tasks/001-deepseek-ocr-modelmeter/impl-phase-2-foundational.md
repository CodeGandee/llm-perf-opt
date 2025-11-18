# Implementation Guide: Phase 2 – Foundational (Blocking Prerequisites)

**Phase**: 2 | **Feature**: DeepSeek-OCR Analytic Modeling in ModelMeter (`001-deepseek-ocr-modelmeter`) | **Tasks**: T004–T008

## Goal

Define the core analytic data models, package structure, and helpers required to implement, run, and serialize DeepSeek-OCR analytic models. This phase produces the `attrs` data structures, ModelMeter layer package skeletons, and path/operator-list utilities used by all user stories.

## Public APIs

### T004: DeepSeek-OCR analytic layer package skeleton

Create the hierarchical package under `extern/modelmeter/models/deepseek_ocr/layers/` reflecting the TorchInfo module hierarchy and `BaseLayer` subclass stubs.

```python
# extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py

from __future__ import annotations

from extern.modelmeter.layers.base import BaseLayer


class DeepseekOCRModel(BaseLayer):
    """Analytic root for DeepSeek-OCR.

    Responsibilities:
    - Aggregate FLOPs/IO/memory metrics across vision, decoder, and LLaMA submodules.
    - Provide a single entry point for model-level analytic estimates.
    """

    def __init__(self, *, vision_layer: BaseLayer, decoder_layer: BaseLayer) -> None:
        super().__init__()
        self.m_vision_layer = vision_layer
        self.m_decoder_layer = decoder_layer

    def forward_tensor_core_flops(self) -> float:
        return (
            (self.m_vision_layer.forward_tensor_core_flops() or 0.0)
            + (self.m_decoder_layer.forward_tensor_core_flops() or 0.0)
        )

    # Other BaseLayer methods follow a similar aggregation pattern.
```

**Usage Flow**:

```mermaid
graph LR
    V[Vision BaseLayer subclasses] --> R[DeepseekOCRModel]
    D[Decoder/LLaMA BaseLayer subclasses] --> R
    R --> A[Analytic pipeline in dsocr_analyzer]
```

### T005/T006: Core analytic domain models in `src/llm_perf_opt/data/deepseek_ocr_analytic.py`

Implement the entities defined in `data-model.md` as `attrs` models.

```python
# src/llm_perf_opt/data/deepseek_ocr_analytic.py

from __future__ import annotations
from typing import Literal
from attrs import define, field
from attrs.validators import instance_of


@define(kw_only=True)
class DeepSeekOCRModelSpec:
    """Model specification for DeepSeek-OCR analytic reports."""

    model_id: str = field(validator=[instance_of(str)])
    model_variant: str = field(validator=[instance_of(str)])
    hf_revision: str | None = field(default=None)
    config_path: str = field(validator=[instance_of(str)])
    hidden_size: int = field(validator=[instance_of(int)])
    intermediate_size: int = field(validator=[instance_of(int)])
    num_layers: int = field(validator=[instance_of(int)])
    num_attention_heads: int = field(validator=[instance_of(int)])
    vision_backbone: Literal["sam_vit_b", "clip_vit_l", "other"] = field(validator=[instance_of(str)])
    uses_moe: bool = field(validator=[instance_of(bool)])
    notes: str = field(default="")


@define(kw_only=True)
class OCRWorkloadProfile:
    """Synthetic workload profile (e.g., dsocr-standard-v1)."""

    profile_id: str = field(validator=[instance_of(str)])
    description: str = field(validator=[instance_of(str)])
    seq_len: int = field(validator=[instance_of(int)])
    base_size: int = field(validator=[instance_of(int)])
    image_size: int = field(validator=[instance_of(int)])
    crop_mode: bool = field(validator=[instance_of(bool)])
    max_new_tokens: int = field(validator=[instance_of(int)])
    doc_kind: Literal["text_heavy", "mixed_layout", "image_rich", "synthetic"] = field(
        validator=[instance_of(str)]
    )
    num_pages: int = field(validator=[instance_of(int)])
```

Expose these models via `src/llm_perf_opt/data/__init__.py` so other modules can import them.

### T007: Analytic artifact path helpers in `src/llm_perf_opt/utils/paths.py`

Add helpers that construct absolute paths for analytic artifacts based on a run ID.

```python
# src/llm_perf_opt/utils/paths.py

import os


def analytic_model_dir(run_id: str) -> str:
    """Return directory for analytic artifacts for a given run."""

    from llm_perf_opt.utils.paths import workspace_root  # if already exists

    root = workspace_root()
    return os.path.join(root, "tmp", "profile-output", run_id, "static_analysis", "analytic_model")


def analytic_layer_docs_dir(run_id: str) -> str:
    """Return directory for per-layer Markdown docs."""

    return os.path.join(analytic_model_dir(run_id), "layers")
```

### T008: Load `TargetOperatorList` from TorchInfo artifacts

Provide a parser for TorchInfo outputs.

```python
# src/llm_perf_opt/utils/dsocr_callgraph_parse.py

from __future__ import annotations
import json
from pathlib import Path
from llm_perf_opt.data.deepseek_ocr_analytic import TargetOperatorList, OperatorSpec


def load_target_operator_list(artifact_dir: str) -> TargetOperatorList:
    """Load TorchInfo operator snapshot into a TargetOperatorList."""

    base = Path(artifact_dir)
    layers_json = base / "torchinfo-unique-layers.json"
    with layers_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    operators = [
        OperatorSpec(
            class_name=entry["class_name"],
            class_name_qualified=entry["class_name_qualified"],
            is_pytorch_builtin=entry["is_torch_builtin"],
            is_custom=not entry["is_torch_builtin"],
            children_classes=entry.get("children", []),
            default_category_id="",
        )
        for entry in data["layers"]
    ]

    return TargetOperatorList(
        snapshot_id=data["generated_at"],
        source_artifact_dir=str(base.resolve()),
        layers_md_path=str((base / "torchinfo-unique-layers.md").resolve()),
        layers_json_path=str(layers_json.resolve()),
        stages_json_path=str((base / "torchinfo-stages.json").resolve()),
        operators=operators,
    )
```

---

## Phase Integration

```mermaid
graph LR
    T004[T004: Layer package skeleton] --> T011[US1 analytic formulas]
    T005[T005: Core analytic models] --> T014[US1 analytic pipeline]
    T006[T006: Supporting models] --> T020[US2 metrics aggregation]
    T007[T007: Path helpers] --> T014
    T007 --> T015[US1 Markdown generation]
    T008[T008: TargetOperatorList loader] --> T020
```

Phase 2 produces the building blocks used by the analytic runner and exporters in later phases.

---

## Testing

### Test Input

- TorchInfo artifacts under `reports/20211117-dsorc-op-analysis/static-20251118-130533/`.
- Python environment from Phase 1.

### Test Procedure

```bash
cd /workspace/code/llm-perf-opt

# 1. Import domain models
pixi run -e rtx5090 python -c "from llm_perf_opt.data.deepseek_ocr_analytic import DeepSeekOCRModelSpec, AnalyticModelReport"

# 2. Import path helpers
pixi run -e rtx5090 python -c "from llm_perf_opt.utils.paths import analytic_model_dir; print(analytic_model_dir('test-run'))"

# 3. Load TargetOperatorList
pixi run -e rtx5090 python - << 'EOF'
from llm_perf_opt.utils.dsocr_callgraph_parse import load_target_operator_list
tol = load_target_operator_list('reports/20211117-dsorc-op-analysis/static-20251118-130533')
print(len(tol.operators))
EOF
```

### Test Output

- Imports succeed without raising exceptions.
- Path helpers return absolute paths under `tmp/profile-output/<run_id>/static_analysis/analytic_model/`.
- `load_target_operator_list` loads a non-zero number of `OperatorSpec` instances.

---

## References

- Spec: `specs/001-deepseek-ocr-modelmeter/spec.md`
- Data model: `specs/001-deepseek-ocr-modelmeter/data-model.md`
- Contracts: `specs/001-deepseek-ocr-modelmeter/contracts/`

---

## Implementation Summary

*(to be filled after implementation)*

### What has been implemented

- (after implementation) Summarize concrete classes in `deepseek_ocr_analytic.py`, path helpers, and layer package files.

### How to verify

- (after implementation) Document the script or test suite used to validate model construction and TorchInfo parsing.

