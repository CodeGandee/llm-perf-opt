"""Unit tests for shared analytic report schema types."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from attrs import asdict
from cattrs import Converter

from llm_perf_opt.data.analytic_common import (
    AnalyticModuleNode,
    ModuleMetricsSnapshot,
    OperatorCategory,
    OperatorMetrics,
    ScalarParam,
    _validate_absolute_path,
)


def test_validate_absolute_path_rejects_relative() -> None:
    """Absolute-path validator rejects relative paths."""

    with pytest.raises(ValueError, match="must be an absolute path"):
        _validate_absolute_path(object(), type("A", (), {"name": "path"})(), "relative/path.json")  # type: ignore[arg-type]


def test_module_metrics_snapshot_rejects_negative_flops() -> None:
    """Metric validators reject negative FLOP values."""

    with pytest.raises(ValueError, match="must be non-negative"):
        _ = ModuleMetricsSnapshot(
            module_id="m",
            profile_id="p",
            calls=1,
            total_time_ms=0.0,
            total_flops_tflops=-1.0,
            total_io_tb=0.0,
            memory_weights_gb=0.0,
            memory_activations_gb=0.0,
            memory_kvcache_gb=0.0,
            share_of_model_time=0.0,
            operator_breakdown=[],
        )


def test_cattrs_roundtrip(tmp_path: Path) -> None:
    """Shared schema objects can be round-tripped via cattrs."""

    node = AnalyticModuleNode(
        module_id="root",
        name="Root",
        qualified_class_name="modelmeter.models.wan2_1.layers.core.Root",
        stage="diffusion",
        parent_id=None,
        children=[],
        repetition="none",
        repetition_count=None,
        constructor_params={"hidden_size": 16},
    )
    cat = OperatorCategory(
        category_id="matmul",
        display_name="Matmul",
        description="Matrix multiplications.",
        match_classes=["torch.nn.modules.linear.Linear"],
    )
    snap = ModuleMetricsSnapshot(
        module_id=node.module_id,
        profile_id="dummy",
        calls=1,
        total_time_ms=0.0,
        total_flops_tflops=1.0,
        total_io_tb=0.0,
        memory_weights_gb=0.0,
        memory_activations_gb=0.0,
        memory_kvcache_gb=0.0,
        share_of_model_time=1.0,
        operator_breakdown=[
            OperatorMetrics(
                category_id=cat.category_id,
                calls=1,
                flops_tflops=1.0,
                io_tb=0.0,
                share_of_module_flops=1.0,
            ),
        ],
    )

    conv = Converter()
    conv.register_structure_hook(ScalarParam, lambda v, _: v)
    node_payload = json.loads(json.dumps(conv.unstructure(node)))
    snap_payload = json.loads(json.dumps(conv.unstructure(snap)))
    node_rt = conv.structure(node_payload, AnalyticModuleNode)
    snap_rt = conv.structure(snap_payload, ModuleMetricsSnapshot)

    assert node_rt == node
    assert snap_rt == snap
    # Sanity: attrs.asdict must succeed on shared schema objects.
    assert asdict(node)["module_id"] == node.module_id
    assert asdict(snap)["total_flops_tflops"] == snap.total_flops_tflops
