from __future__ import annotations

import pytest
from mdutils.mdutils import MdUtils

from modelmeter.models.wan2_1.hardware_sweep import stage_breakdown_key
from modelmeter.models.wan2_1.hardware_sweep_reporting import compute_device_summary
from modelmeter.models.wan2_1.scripts.reporting.run_make_dv_stakeholder_report import _add_image
from modelmeter.models.wan2_1.scripts.reporting.run_make_dv_stakeholder_report import _utilization_percent_text
from modelmeter.models.wan2_1.scripts.reporting.run_make_dv_stakeholder_report import _write_stakeholder_reports


def _metadata(*, selector: str, display_name: str, fp8_tflops: float, cuda_tflops: float, io_tb_s: float, p2p_gb_s: float = 1000.0) -> dict[str, object]:
    return {
        "precision": {
            "name": "fp8",
            "compute_precision": "fp8",
            "storage_bits": 8,
        },
        "device": {
            "selector": selector,
            "display_name": display_name,
            "fp8_tflops": fp8_tflops,
            "cuda_tflops": cuda_tflops,
            "io_tb_s": io_tb_s,
            "p2p_gb_s": p2p_gb_s,
        },
    }


def _row(
    *,
    batch_size: int,
    device_num: int,
    effective_gpus: int,
    total_cost_s: float,
    throughput_videos_per_s: float,
    tensor_cost_s: float,
    io_cost_s: float,
    model_tensor_tflops: float,
    model_io_tb: float,
    model_cuda_tflops: float = 0.0,
) -> dict[str, object]:
    return {
        "model_mode": "full_pipeline",
        "util_profile": "optimistic",
        "device_num": device_num,
        "batch_size": batch_size,
        "effective_gpus": effective_gpus,
        "workload_size": "1280*720",
        "workload_frame_num": 81,
        "workload_steps": 50,
        "workload_text_len": 256,
        "total_cost_s": total_cost_s,
        "tensor_cost_s": tensor_cost_s,
        "cuda_cost_s": 0.0,
        "io_cost_s": io_cost_s,
        "throughput_videos_per_s": throughput_videos_per_s,
        "model_tensor_tflops": model_tensor_tflops,
        "model_cuda_tflops": model_cuda_tflops,
        "model_io_tb": model_io_tb,
        "used_tensor_tflops_s": model_tensor_tflops / total_cost_s,
        "used_io_tb_s": model_io_tb / total_cost_s,
    }


def test_compute_device_summary_uses_memio_gap_for_io_bound_device() -> None:
    metadata = _metadata(
        selector="dv100",
        display_name="DV100",
        fp8_tflops=2000.0,
        cuda_tflops=1.0,
        io_tb_s=2.0,
    )
    rows = [
        _row(
            batch_size=1,
            device_num=1,
            effective_gpus=1,
            total_cost_s=424.45702457653533,
            throughput_videos_per_s=0.002356,
            tensor_cost_s=100.0,
            io_cost_s=424.45702457653533,
            model_tensor_tflops=309279.99113301194,
            model_io_tb=848.9140491530707,
        ),
        _row(
            batch_size=8,
            device_num=8,
            effective_gpus=8,
            total_cost_s=424.259054,
            throughput_videos_per_s=0.018857346514532913,
            tensor_cost_s=100.0,
            io_cost_s=424.259054,
            model_tensor_tflops=309279.99113301194,
            model_io_tb=848.9140491530707,
        ),
    ]

    summary = compute_device_summary(metadata, rows)

    assert summary["primary_bottleneck"] == "io"
    assert summary["dominant_gap_resource_kind"] == "io"
    assert summary["dominant_gap_resource_label"] == "MemIO bandwidth"
    assert summary["required_dominant_vs_peak_8gpu"] == pytest.approx(848.9140491530707 / 16.0)
    assert summary["required_dominant_vs_peak_8gpu"] == pytest.approx(summary["required_memio_vs_peak_8gpu"])
    assert summary["used_memio_vs_peak_per_gpu_at_peak_throughput"] == pytest.approx(
        summary["required_memio_vs_peak_8gpu"] * summary["peak_throughput_8gpu_videos_s"],
    )
    assert summary["used_tensor_vs_peak_per_gpu_at_peak_throughput"] < summary["used_memio_vs_peak_per_gpu_at_peak_throughput"]


def test_compute_device_summary_uses_tensor_gap_for_tensor_bound_device() -> None:
    metadata = _metadata(
        selector="dv300",
        display_name="DV300",
        fp8_tflops=8000.0,
        cuda_tflops=1.0,
        io_tb_s=50.0,
    )
    rows = [
        _row(
            batch_size=1,
            device_num=1,
            effective_gpus=1,
            total_cost_s=38.65999889162649,
            throughput_videos_per_s=0.025866,
            tensor_cost_s=38.65999889162649,
            io_cost_s=6.0,
            model_tensor_tflops=309279.99113301194,
            model_io_tb=848.9140491530707,
        ),
        _row(
            batch_size=8,
            device_num=8,
            effective_gpus=8,
            total_cost_s=38.65999889162649,
            throughput_videos_per_s=0.20693223562747565,
            tensor_cost_s=38.65999889162649,
            io_cost_s=6.0,
            model_tensor_tflops=309279.99113301194,
            model_io_tb=848.9140491530707,
        ),
    ]

    summary = compute_device_summary(metadata, rows)

    assert summary["primary_bottleneck"] == "tensor"
    assert summary["dominant_gap_resource_kind"] == "tensor"
    assert summary["dominant_gap_resource_label"] == "tensor throughput"
    assert summary["required_dominant_vs_peak_8gpu"] == pytest.approx(309279.99113301194 / 64000.0)
    assert summary["required_dominant_vs_peak_8gpu"] == pytest.approx(summary["required_tensor_vs_peak_8gpu"])
    assert summary["required_memio_vs_peak_8gpu"] < summary["required_dominant_vs_peak_8gpu"]
    assert summary["used_tensor_vs_peak_per_gpu_at_peak_throughput"] == pytest.approx(
        summary["required_tensor_vs_peak_8gpu"] * summary["peak_throughput_8gpu_videos_s"],
    )
    assert summary["used_memio_vs_peak_per_gpu_at_peak_throughput"] < summary["used_tensor_vs_peak_per_gpu_at_peak_throughput"]


def test_add_image_starts_on_fresh_markdown_block() -> None:
    md = MdUtils(file_name="dummy")
    md.new_paragraph("Paragraph before image.", wrap_width=0)

    _add_image(md, "figures/example.svg")

    assert "Paragraph before image.\n\n![](figures/example.svg)\n\n" in md.file_data_text
    assert "Paragraph before image.![](figures/example.svg)" not in md.file_data_text


def test_utilization_percent_text_clamps_to_100_percent() -> None:
    assert _utilization_percent_text(1.0005166491209174) == "100.0%"
    assert _utilization_percent_text(0.9112812441885738) == "91.1%"


def test_write_stakeholder_reports_generates_english_and_chinese_variants(tmp_path) -> None:
    input_struct = ("1280*720", 81, 50, 256)
    run_ids = {"dv100": "run-dv100", "dv200": "run-dv200", "dv300": "run-dv300"}
    metadata_by_selector = {
        "dv100": _metadata(selector="dv100", display_name="DV100", fp8_tflops=2000.0, cuda_tflops=1.0, io_tb_s=2.0, p2p_gb_s=1100.0),
        "dv200": _metadata(selector="dv200", display_name="DV200", fp8_tflops=4000.0, cuda_tflops=1.0, io_tb_s=10.0, p2p_gb_s=1800.0),
        "dv300": _metadata(selector="dv300", display_name="DV300", fp8_tflops=8000.0, cuda_tflops=1.0, io_tb_s=50.0, p2p_gb_s=3600.0),
    }
    rows_by_selector = {
        "dv100": [
            _row(
                batch_size=1,
                device_num=1,
                effective_gpus=1,
                total_cost_s=424.45702457653533,
                throughput_videos_per_s=0.002356,
                tensor_cost_s=100.0,
                io_cost_s=424.45702457653533,
                model_tensor_tflops=309279.99113301194,
                model_io_tb=848.9140491530707,
            ),
            _row(
                batch_size=8,
                device_num=8,
                effective_gpus=8,
                total_cost_s=424.259054,
                throughput_videos_per_s=0.018857346514532913,
                tensor_cost_s=100.0,
                io_cost_s=424.259054,
                model_tensor_tflops=309279.99113301194,
                model_io_tb=848.9140491530707,
            ),
        ],
        "dv200": [
            _row(
                batch_size=1,
                device_num=1,
                effective_gpus=1,
                total_cost_s=84.89140491530706,
                throughput_videos_per_s=0.011779,
                tensor_cost_s=42.0,
                io_cost_s=84.89140491530706,
                model_tensor_tflops=309279.99113301194,
                model_io_tb=848.9140491530707,
            ),
            _row(
                batch_size=8,
                device_num=8,
                effective_gpus=8,
                total_cost_s=84.847811,
                throughput_videos_per_s=0.09428673257266457,
                tensor_cost_s=42.0,
                io_cost_s=84.847811,
                model_tensor_tflops=309279.99113301194,
                model_io_tb=848.9140491530707,
            ),
        ],
        "dv300": [
            _row(
                batch_size=1,
                device_num=1,
                effective_gpus=1,
                total_cost_s=38.65999889162649,
                throughput_videos_per_s=0.025866,
                tensor_cost_s=38.65999889162649,
                io_cost_s=6.0,
                model_tensor_tflops=309279.99113301194,
                model_io_tb=848.9140491530707,
            ),
            _row(
                batch_size=8,
                device_num=8,
                effective_gpus=8,
                total_cost_s=38.65999889162649,
                throughput_videos_per_s=0.20693223562747565,
                tensor_cost_s=38.65999889162649,
                io_cost_s=6.0,
                model_tensor_tflops=309279.99113301194,
                model_io_tb=848.9140491530707,
            ),
        ],
    }
    payloads_by_device = {}
    summaries_by_device = {}
    stage_payload = {
        stage_breakdown_key(input_struct): {
            "text_encoder": {"tensor_tflops": 2.39659, "memio_tb": 0.00486469},
            "diffusion_core": {"tensor_tflops": 308617.0, "memio_tb": 848.724},
            "vae_decode": {"tensor_tflops": 660.444, "memio_tb": 0.185534},
        }
    }
    for selector, metadata in metadata_by_selector.items():
        payloads_by_device[selector] = {
            "metadata": metadata,
            "stage_breakdown_by_input_struct": stage_payload,
        }
        summaries_by_device[selector] = compute_device_summary(metadata, rows_by_selector[selector])

    report_texts = _write_stakeholder_reports(
        run_ids=run_ids,
        payloads_by_device=payloads_by_device,
        summaries_by_device=summaries_by_device,
        comparison_dir=tmp_path,
    )

    english_report = tmp_path / "stakeholder-report.en.md"
    chinese_report = tmp_path / "stakeholder-report.cn.md"
    assert english_report.exists()
    assert chinese_report.exists()
    assert set(report_texts.keys()) == {"en", "cn"}

    english_text = english_report.read_text()
    chinese_text = chinese_report.read_text()
    assert "DV hardware bottlenecks for Wan2.1-T2V-14B" in english_text
    assert "DV 系列 Wan2.1-T2V-14B 的硬件瓶颈" in chinese_text
    assert "stakeholder-report.cn.md" not in chinese_text
    assert "MemIO" in chinese_text
    assert "batch size" in chinese_text
    assert "Data-parallel replication" in chinese_text
    assert "\n\n![](figures/dv100_full_pipeline_8gpu_throughput.svg)\n\n" in english_text
    assert "\n\n![](figures/dv100_full_pipeline_8gpu_throughput.svg)\n\n" in chinese_text
