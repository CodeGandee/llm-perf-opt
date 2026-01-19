"""Integration test: Wan2.1 report totals scale monotonically across workloads."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from llm_perf_opt.runners.wan2_1_analyzer import Wan2_1AnalyzerConfig, Wan2_1StaticAnalyzer
from llm_perf_opt.utils.paths import workspace_root
from llm_perf_opt.visualize.wan2_1_analytic_summary import aggregate_category_flops, total_flops_tflops


def _has_local_wan2_1_metadata() -> bool:
    """Return True when local Wan2.1 model metadata is available."""

    cfg = Path(workspace_root()) / "models" / "wan2.1-t2v-14b" / "source-data" / "config.json"
    return cfg.is_file()


@pytest.mark.integration
def test_hotspots_monotonic_scaling_across_workloads() -> None:
    """Larger workload produces larger total FLOPs and higher attention-core share."""

    if not _has_local_wan2_1_metadata():
        pytest.skip("Wan2.1 local model metadata not available (run models/wan2.1-t2v-14b/bootstrap.sh).")

    analyzer = Wan2_1StaticAnalyzer()
    overrides = [
        "hf.hidden_size=64",
        "hf.num_layers=2",
        "hf.num_attention_heads=4",
        "hf.head_dim=16",
        "hf.mlp_intermediate_size=128",
    ]

    report_tiny = analyzer.run(
        cfg=Wan2_1AnalyzerConfig(workload_profile_id="wan2-1-ci-tiny", run_id=f"wan2-1-hot-{uuid4().hex[:8]}"),
        overrides=overrides,
    )
    report_big = analyzer.run(
        cfg=Wan2_1AnalyzerConfig(workload_profile_id="wan2-1-512p", run_id=f"wan2-1-hot-{uuid4().hex[:8]}"),
        overrides=overrides,
    )

    total_tiny = total_flops_tflops(report_tiny)
    total_big = total_flops_tflops(report_big)
    assert total_big > total_tiny

    cats_tiny = aggregate_category_flops(report_tiny, leaf_only=True)
    cats_big = aggregate_category_flops(report_big, leaf_only=True)
    share_core_tiny = float(cats_tiny.get("attention_core", 0.0)) / total_tiny
    share_core_big = float(cats_big.get("attention_core", 0.0)) / total_big
    assert share_core_big >= share_core_tiny
