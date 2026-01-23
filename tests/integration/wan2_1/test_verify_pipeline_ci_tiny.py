"""Integration tests: verify Wan2.1 full-pipeline FLOPs on a tiny workload."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_verify_text_encoder_ci_tiny() -> None:
    from modelmeter.models.wan2_1.scripts.verify.run_verify_text_encoder import main

    assert main(["--workload", "wan2-1-ci-tiny", "--accept-rel-diff", "1e-6"]) == 0


@pytest.mark.integration
def test_verify_vae_decode_ci_tiny() -> None:
    from modelmeter.models.wan2_1.scripts.verify.run_verify_vae_decode import main

    assert main(["--workload", "wan2-1-ci-tiny", "--accept-rel-diff", "1e-6"]) == 0


@pytest.mark.integration
def test_verify_pipeline_invariants_ci_tiny() -> None:
    from modelmeter.models.wan2_1.scripts.verify.run_verify_pipeline_invariants import main

    assert main(["--workload", "wan2-1-ci-tiny"]) == 0

