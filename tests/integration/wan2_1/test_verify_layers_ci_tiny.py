"""Integration test: verify Wan2.1 per-layer FLOPs on a tiny workload."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_perf_opt.utils.paths import workspace_root


def _has_local_wan2_1_metadata() -> bool:
    """Return True when local Wan2.1 model metadata is available."""

    cfg = Path(workspace_root()) / "models" / "wan2.1-t2v-14b" / "source-data" / "config.json"
    return cfg.is_file()


@pytest.mark.integration
def test_verify_layers_ci_tiny() -> None:
    """Per-layer verification runs and passes for a tiny workload."""

    if not _has_local_wan2_1_metadata():
        pytest.skip("Wan2.1 local model metadata not available (run models/wan2.1-t2v-14b/bootstrap.sh).")

    from modelmeter.models.wan2_1.scripts.verify.run_verify_layers import main

    assert main(["--workload", "wan2-1-ci-tiny", "--accept-rel-diff", "0.05"]) == 0
