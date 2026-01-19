"""Unit tests for Wan2.1 analytic report structural invariants."""

from __future__ import annotations

import math

from llm_perf_opt.data.analytic_common import AnalyticModuleNode, ModuleMetricsSnapshot, OperatorCategory, OperatorMetrics
from llm_perf_opt.data.wan2_1_analytic import Wan2_1AnalyticModelReport, Wan2_1ModelSpec, Wan2_1WorkloadProfile


def _assert_finite_non_negative(value: float) -> None:
    """Assert that a float is finite and non-negative."""

    assert math.isfinite(value)
    assert value >= 0.0


def _validate_report_invariants(report: Wan2_1AnalyticModelReport) -> None:
    """Validate basic invariants on the analytic report structure."""

    modules_by_id = {m.module_id: m for m in report.modules}
    assert len(modules_by_id) == len(report.modules)

    metrics_by_id = {m.module_id: m for m in report.module_metrics}
    assert len(metrics_by_id) == len(report.module_metrics)

    for snap in report.module_metrics:
        assert snap.module_id in modules_by_id
        _assert_finite_non_negative(snap.total_flops_tflops)
        _assert_finite_non_negative(snap.share_of_model_time)
        for op in snap.operator_breakdown:
            _assert_finite_non_negative(op.flops_tflops)
            _assert_finite_non_negative(op.share_of_module_flops)

    # Parent FLOPs should equal sum of children FLOPs when all are present.
    for node in report.modules:
        if not node.children:
            continue
        parent = metrics_by_id.get(node.module_id)
        if parent is None:
            continue
        child_snaps = [metrics_by_id.get(cid) for cid in node.children]
        if any(s is None for s in child_snaps):
            continue
        child_total = sum(float(s.total_flops_tflops) for s in child_snaps if s is not None)
        assert float(parent.total_flops_tflops) == child_total


def test_report_invariants_for_minimal_tree() -> None:
    """A minimal hierarchical report satisfies invariants used by tooling."""

    root_id = "diffusion/dit"
    block_id = f"{root_id}/block_00"
    attn_id = f"{block_id}/attn"
    mlp_id = f"{block_id}/mlp"

    modules = [
        AnalyticModuleNode(
            module_id=root_id,
            name="root",
            qualified_class_name="modelmeter.models.wan2_1.layers.core.wan2_1_dit_model.Wan2_1DiTModel",
            stage="diffusion",
            parent_id=None,
            children=[block_id],
            repetition="none",
            repetition_count=None,
            constructor_params={},
        ),
        AnalyticModuleNode(
            module_id=block_id,
            name="block",
            qualified_class_name="modelmeter.models.wan2_1.layers.transformer.wan2_1_transformer_block.Wan2_1TransformerBlock",
            stage="diffusion",
            parent_id=root_id,
            children=[attn_id, mlp_id],
            repetition="none",
            repetition_count=None,
            constructor_params={},
        ),
        AnalyticModuleNode(
            module_id=attn_id,
            name="attn",
            qualified_class_name="modelmeter.models.wan2_1.layers.transformer.wan2_1_attention.Wan2_1Attention",
            stage="diffusion",
            parent_id=block_id,
            children=[],
            repetition="none",
            repetition_count=None,
            constructor_params={},
        ),
        AnalyticModuleNode(
            module_id=mlp_id,
            name="mlp",
            qualified_class_name="modelmeter.models.wan2_1.layers.transformer.wan2_1_mlp.Wan2_1MLP",
            stage="diffusion",
            parent_id=block_id,
            children=[],
            repetition="none",
            repetition_count=None,
            constructor_params={},
        ),
    ]

    cats = [
        OperatorCategory(category_id="attention_proj", display_name="Attention projections", description="Q/K/V/O GEMMs.", match_classes=[]),
        OperatorCategory(category_id="attention_core", display_name="Attention core", description="QK^T and PV matmuls.", match_classes=[]),
        OperatorCategory(category_id="mlp_proj", display_name="MLP projections", description="MLP GEMMs.", match_classes=[]),
    ]

    attn_flops = 2.0
    mlp_flops = 3.0
    block_flops = attn_flops + mlp_flops
    root_flops = block_flops

    metrics = [
        ModuleMetricsSnapshot(
            module_id=attn_id,
            profile_id="wan2-1-ci-tiny",
            calls=1,
            total_time_ms=0.0,
            total_flops_tflops=attn_flops,
            total_io_tb=0.0,
            memory_weights_gb=0.0,
            memory_activations_gb=0.0,
            memory_kvcache_gb=0.0,
            share_of_model_time=attn_flops / root_flops,
            operator_breakdown=[
                OperatorMetrics(category_id="attention_proj", calls=1, flops_tflops=1.0, io_tb=0.0, share_of_module_flops=0.5),
                OperatorMetrics(category_id="attention_core", calls=1, flops_tflops=1.0, io_tb=0.0, share_of_module_flops=0.5),
            ],
        ),
        ModuleMetricsSnapshot(
            module_id=mlp_id,
            profile_id="wan2-1-ci-tiny",
            calls=1,
            total_time_ms=0.0,
            total_flops_tflops=mlp_flops,
            total_io_tb=0.0,
            memory_weights_gb=0.0,
            memory_activations_gb=0.0,
            memory_kvcache_gb=0.0,
            share_of_model_time=mlp_flops / root_flops,
            operator_breakdown=[OperatorMetrics(category_id="mlp_proj", calls=1, flops_tflops=mlp_flops, io_tb=0.0, share_of_module_flops=1.0)],
        ),
        ModuleMetricsSnapshot(
            module_id=block_id,
            profile_id="wan2-1-ci-tiny",
            calls=1,
            total_time_ms=0.0,
            total_flops_tflops=block_flops,
            total_io_tb=0.0,
            memory_weights_gb=0.0,
            memory_activations_gb=0.0,
            memory_kvcache_gb=0.0,
            share_of_model_time=block_flops / root_flops,
            operator_breakdown=[],
        ),
        ModuleMetricsSnapshot(
            module_id=root_id,
            profile_id="wan2-1-ci-tiny",
            calls=1,
            total_time_ms=0.0,
            total_flops_tflops=root_flops,
            total_io_tb=0.0,
            memory_weights_gb=0.0,
            memory_activations_gb=0.0,
            memory_kvcache_gb=0.0,
            share_of_model_time=1.0,
            operator_breakdown=[],
        ),
    ]

    report = Wan2_1AnalyticModelReport(
        report_id="dummy",
        model=Wan2_1ModelSpec(
            model_id="wan2.1-t2v-14b",
            model_variant="t2v-14b",
            config_path="/abs/config.json",
            hidden_size=64,
            num_layers=1,
            num_attention_heads=4,
            head_dim=16,
            mlp_intermediate_size=128,
            vae_downsample_factor=8,
            patch_size=2,
            latent_channels=16,
            notes="",
        ),
        workload=Wan2_1WorkloadProfile(
            profile_id="wan2-1-ci-tiny",
            description="dummy",
            batch_size=1,
            num_frames=4,
            height=256,
            width=256,
            num_inference_steps=1,
            text_len=64,
        ),
        modules=modules,
        operator_categories=cats,
        module_metrics=metrics,
        profile_run_id=None,
        predicted_total_time_ms=0.0,
        notes="",
        layer_docs_dir=None,
    )

    _validate_report_invariants(report)
