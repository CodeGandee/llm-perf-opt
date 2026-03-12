from __future__ import annotations

import pytest

from modelmeter.models.wan2_1.hardware_sweep import precision_metadata_from_rows
from modelmeter.models.wan2_1.hardware_sweep import resolve_device_profile
from modelmeter.models.wan2_1.hardware_sweep import tensor_peak_tflops_from_metadata
from modelmeter.models.wan2_1.scripts.sizing.run_hardware_concurrency_sweep import _compute_precision_for_profile_name


def test_resolved_dv_device_profile_supports_fp4_tensor_peak() -> None:
    profile = resolve_device_profile("dv200")

    assert profile.tensor_peak_tflops("fp4") == pytest.approx(12000.0)


def test_device_without_fp4_support_fails_fast() -> None:
    profile = resolve_device_profile("ngu800p")

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
