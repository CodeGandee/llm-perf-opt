"""Unit tests for Wan2.1 hotspot extraction and aggregation helpers."""

from __future__ import annotations

from llm_perf_opt.data.analytic_common import AnalyticModuleNode, ModuleMetricsSnapshot, OperatorCategory, OperatorMetrics
from llm_perf_opt.data.wan2_1_analytic import Wan2_1AnalyticModelReport, Wan2_1ModelSpec, Wan2_1WorkloadProfile
from llm_perf_opt.visualize.wan2_1_analytic_summary import aggregate_category_flops, top_k_layers_by_flops, top_k_categories_by_flops


def _make_report() -> Wan2_1AnalyticModelReport:
    """Create a small synthetic report fixture for hotspot tests."""

    root = AnalyticModuleNode(
        module_id="diffusion/dit",
        name="root",
        qualified_class_name="m.Root",
        stage="diffusion",
        parent_id=None,
        children=["a", "b"],
        repetition="none",
        repetition_count=None,
        constructor_params={},
    )
    a = AnalyticModuleNode(
        module_id="a",
        name="A",
        qualified_class_name="m.A",
        stage="diffusion",
        parent_id="diffusion/dit",
        children=[],
        repetition="none",
        repetition_count=None,
        constructor_params={},
    )
    b = AnalyticModuleNode(
        module_id="b",
        name="B",
        qualified_class_name="m.B",
        stage="diffusion",
        parent_id="diffusion/dit",
        children=[],
        repetition="none",
        repetition_count=None,
        constructor_params={},
    )
    modules = [root, a, b]

    cats = [
        OperatorCategory(category_id="x", display_name="X", description="x", match_classes=[]),
        OperatorCategory(category_id="y", display_name="Y", description="y", match_classes=[]),
    ]

    metrics = [
        ModuleMetricsSnapshot(
            module_id="a",
            profile_id="p",
            calls=1,
            total_time_ms=0.0,
            total_flops_tflops=2.0,
            total_io_tb=0.0,
            memory_weights_gb=0.0,
            memory_activations_gb=0.0,
            memory_kvcache_gb=0.0,
            share_of_model_time=0.5,
            operator_breakdown=[
                OperatorMetrics(category_id="x", calls=1, flops_tflops=2.0, io_tb=0.0, share_of_module_flops=1.0),
            ],
        ),
        ModuleMetricsSnapshot(
            module_id="b",
            profile_id="p",
            calls=1,
            total_time_ms=0.0,
            total_flops_tflops=2.0,
            total_io_tb=0.0,
            memory_weights_gb=0.0,
            memory_activations_gb=0.0,
            memory_kvcache_gb=0.0,
            share_of_model_time=0.5,
            operator_breakdown=[
                OperatorMetrics(category_id="y", calls=1, flops_tflops=2.0, io_tb=0.0, share_of_module_flops=1.0),
            ],
        ),
        # Include a parent snapshot to ensure leaf_only avoids double counting.
        ModuleMetricsSnapshot(
            module_id="diffusion/dit",
            profile_id="p",
            calls=1,
            total_time_ms=0.0,
            total_flops_tflops=4.0,
            total_io_tb=0.0,
            memory_weights_gb=0.0,
            memory_activations_gb=0.0,
            memory_kvcache_gb=0.0,
            share_of_model_time=1.0,
            operator_breakdown=[],
        ),
    ]

    return Wan2_1AnalyticModelReport(
        report_id="r",
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
            profile_id="p",
            description="p",
            batch_size=1,
            num_frames=1,
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


def test_top_k_layers_stable_ordering_on_ties() -> None:
    """Top-k layer selection is stable and deterministic under ties."""

    report = _make_report()
    rows = top_k_layers_by_flops(report, k=2, leaf_only=True)
    assert [m.module_id for m, _s in rows] == ["a", "b"]


def test_top_k_categories_stable_ordering_on_ties() -> None:
    """Top-k category selection is stable and deterministic under ties."""

    report = _make_report()
    rows = top_k_categories_by_flops(report, k=2, leaf_only=True)
    assert rows[0][1] == rows[1][1]  # tie on flops
    assert [cat for cat, _flops in rows] == ["x", "y"]


def test_leaf_only_category_aggregation_avoids_double_counting() -> None:
    """Leaf-only aggregation avoids double counting when parent snapshots exist."""

    report = _make_report()
    totals = aggregate_category_flops(report, leaf_only=True)
    assert totals == {"x": 2.0, "y": 2.0}
