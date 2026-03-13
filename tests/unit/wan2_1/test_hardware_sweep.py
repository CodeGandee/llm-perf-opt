from __future__ import annotations

import pytest

from modelmeter.models.wan2_1.hardware_sweep import precision_metadata_from_rows
from modelmeter.models.wan2_1.hardware_sweep import resolve_device_profile
from modelmeter.models.wan2_1.hardware_sweep import supported_device_selectors
from modelmeter.models.wan2_1.hardware_sweep import tensor_peak_tflops_from_metadata
from modelmeter.models.wan2_1.scripts.sizing.run_hardware_concurrency_sweep import _compute_precision_for_profile_name


def test_resolved_dv_device_profile_supports_fp4_tensor_peak() -> None:
    profile = resolve_device_profile("dv200")

    assert profile.tensor_peak_tflops("fp4") == pytest.approx(12000.0)


def test_supported_device_selectors_include_h20_and_b200() -> None:
    selectors = supported_device_selectors()

    assert "h20" in selectors
    assert "b200" in selectors


def test_resolve_device_profile_supports_h20_and_b200_metadata() -> None:
    h20 = resolve_device_profile("h20")
    b200 = resolve_device_profile("b200")
    h20_metadata = h20.to_metadata()
    b200_metadata = b200.to_metadata()

    assert h20.display_name == "H20"
    assert h20.family == "nvidia"
    assert h20.fp8_tflops == pytest.approx(296.0)
    assert h20.io_tb_s == pytest.approx(4.0)
    assert h20.p2p_gb_s == pytest.approx(300.0)
    assert h20_metadata["selector"] == "h20"
    assert h20_metadata["fp8_tflops"] == pytest.approx(296.0)
    assert h20_metadata["io_tb_s"] == pytest.approx(4.0)
    assert h20_metadata["p2p_gb_s"] == pytest.approx(300.0)
    assert h20_metadata["bisection_gb_s"] == pytest.approx(2400.0)

    assert b200.display_name == "B200"
    assert b200.family == "nvidia"
    assert b200.fp8_tflops == pytest.approx(32000.0)
    assert b200.io_tb_s == pytest.approx(7.7)
    assert b200.p2p_gb_s == pytest.approx(14.4 * 1024.0)
    assert b200_metadata["selector"] == "b200"
    assert b200_metadata["fp8_tflops"] == pytest.approx(32000.0)
    assert b200_metadata["io_tb_s"] == pytest.approx(7.7)
    assert b200_metadata["p2p_gb_s"] == pytest.approx(14.4 * 1024.0)
    assert b200_metadata["bisection_gb_s"] == pytest.approx(4.0 * 14.4 * 1024.0)


@pytest.mark.parametrize("selector", ["ngu800p", "h20", "b200"])
def test_device_without_fp4_support_fails_fast(selector: str) -> None:
    profile = resolve_device_profile(selector)

    with pytest.raises(ValueError, match="does not expose fp4_tflops"):
        profile.tensor_peak_tflops("fp4")


def test_precision_metadata_from_rows_preserves_fp4_values() -> None:
    metadata = precision_metadata_from_rows(
        [
            {
                "precision": "fp4",
                "compute_precision": "fp4",
                "storage_bits": 4,
            },
        ],
    )

    assert metadata == {"name": "fp4", "compute_precision": "fp4", "storage_bits": 4}


def test_tensor_peak_tflops_from_metadata_supports_fp4() -> None:
    peak = tensor_peak_tflops_from_metadata(
        {
            "precision": {
                "name": "fp4",
                "compute_precision": "fp4",
                "storage_bits": 4,
            },
            "device": {
                "fp4_tflops": 24000.0,
            },
        },
    )

    assert peak == pytest.approx(24000.0)


def test_tensor_peak_tflops_from_metadata_rejects_missing_fp4_peak() -> None:
    with pytest.raises(ValueError, match="missing fp4_tflops"):
        tensor_peak_tflops_from_metadata(
            {
                "precision": {
                    "name": "fp4",
                    "compute_precision": "fp4",
                    "storage_bits": 4,
                },
                "device": {},
            },
        )


def test_compute_precision_for_profile_name_supports_fp4_and_fp8_mixed() -> None:
    assert _compute_precision_for_profile_name("fp4") == "fp4"
    assert _compute_precision_for_profile_name("fp8_mixed") == "fp8"
