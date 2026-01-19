"""Integration test: Wan2.1 analyzer writes `report.json` artifacts."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from llm_perf_opt.runners.wan2_1_analyzer import Wan2_1AnalyzerConfig, Wan2_1StaticAnalyzer
from llm_perf_opt.utils.paths import wan2_1_report_path, workspace_root


def _has_local_wan2_1_metadata() -> bool:
    """Return True when local Wan2.1 model metadata is available."""

    cfg = Path(workspace_root()) / "models" / "wan2.1-t2v-14b" / "source-data" / "config.json"
    return cfg.is_file()


@pytest.mark.integration
def test_wan2_1_analyzer_writes_report_json() -> None:
    """Analyzer emits a structured report and persists `report.json`."""

    if not _has_local_wan2_1_metadata():
        pytest.skip("Wan2.1 local model metadata not available (run models/wan2.1-t2v-14b/bootstrap.sh).")

    run_id = f"wan2-1-it-{uuid4().hex[:8]}"
    analyzer = Wan2_1StaticAnalyzer()
    report = analyzer.run(
        cfg=Wan2_1AnalyzerConfig(workload_profile_id="wan2-1-ci-tiny", run_id=run_id),
        overrides=[
            "hf.hidden_size=64",
            "hf.num_layers=2",
            "hf.num_attention_heads=4",
            "hf.head_dim=16",
            "hf.mlp_intermediate_size=128",
        ],
    )

    report_path = Path(wan2_1_report_path(run_id))
    assert report_path.is_file()
    assert report.report_id == run_id
    assert report.workload.profile_id == "wan2-1-ci-tiny"
    assert report.modules
    assert report.module_metrics
