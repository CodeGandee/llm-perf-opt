"""Unit tests for Wan2.1 summary helpers."""

from __future__ import annotations

from llm_perf_opt.data.analytic_common import AnalyticModuleNode, ModuleMetricsSnapshot, OperatorCategory
from llm_perf_opt.data.wan2_1_analytic import Wan2_1AnalyticModelReport, Wan2_1ModelSpec, Wan2_1WorkloadProfile
from llm_perf_opt.visualize.wan2_1_analytic_summary import total_flops_tflops


def test_total_flops_prefers_root_module_when_pipeline_present() -> None:
    report = Wan2_1AnalyticModelReport(
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
            height=64,
            width=64,
            num_inference_steps=1,
            text_len=16,
        ),
        modules=[
            AnalyticModuleNode(
                module_id="pipeline",
                name="pipeline",
                qualified_class_name="m.Pipeline",
                stage="pipeline",
                parent_id=None,
                children=["diffusion/dit"],
                repetition="none",
                repetition_count=None,
                constructor_params={},
            ),
            AnalyticModuleNode(
                module_id="diffusion/dit",
                name="diffusion",
                qualified_class_name="m.Diffusion",
                stage="diffusion",
                parent_id="pipeline",
                children=[],
                repetition="none",
                repetition_count=None,
                constructor_params={},
            ),
        ],
        operator_categories=[OperatorCategory(category_id="x", display_name="X", description="x", match_classes=[])],
        module_metrics=[
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
                share_of_model_time=0.4,
                operator_breakdown=[],
            ),
            ModuleMetricsSnapshot(
                module_id="pipeline",
                profile_id="p",
                calls=1,
                total_time_ms=0.0,
                total_flops_tflops=10.0,
                total_io_tb=0.0,
                memory_weights_gb=0.0,
                memory_activations_gb=0.0,
                memory_kvcache_gb=0.0,
                share_of_model_time=1.0,
                operator_breakdown=[],
            ),
        ],
        profile_run_id=None,
        predicted_total_time_ms=0.0,
        notes="",
        layer_docs_dir=None,
    )

    assert total_flops_tflops(report) == 10.0

