from __future__ import annotations

import csv
import json
import pytest
from mdutils.mdutils import MdUtils

from modelmeter.models.wan2_1.hardware_sweep import device_run_dir
from modelmeter.models.wan2_1.hardware_sweep import write_results_csv
from modelmeter.models.wan2_1.hardware_sweep import stage_breakdown_key
from modelmeter.models.wan2_1.hardware_sweep_reporting import compute_device_summary
from modelmeter.models.wan2_1.scripts.reporting import run_make_dv_stakeholder_report
from modelmeter.models.wan2_1.scripts.reporting.run_make_dv_stakeholder_report import _add_image
from modelmeter.models.wan2_1.scripts.reporting.run_make_dv_stakeholder_report import _utilization_percent_text
from modelmeter.models.wan2_1.scripts.reporting.run_make_dv_stakeholder_report import _validate_comparable_runs
from modelmeter.models.wan2_1.scripts.reporting.run_make_dv_stakeholder_report import _write_detailed_bundle
from modelmeter.models.wan2_1.scripts.reporting.run_make_dv_stakeholder_report import _write_stakeholder_reports
from modelmeter.models.wan2_1.scripts.reporting.run_make_dv_stakeholder_report import DeviceRunSpec
from modelmeter.models.wan2_1.scripts.reporting.run_make_comparative_stakeholder_summary import _write_comparative_summary_bundle


def _metadata(
    *,
    selector: str,
    display_name: str,
    fp8_tflops: float,
    cuda_tflops: float,
    io_tb_s: float,
    p2p_gb_s: float = 1000.0,
    fp4_tflops: float | None = None,
    precision_name: str = "fp8",
    compute_precision: str = "fp8",
    storage_bits: int = 8,
    family: str | None = None,
) -> dict[str, object]:
    return {
        "precision": {
            "name": precision_name,
            "compute_precision": compute_precision,
            "storage_bits": storage_bits,
        },
        "device": {
            "selector": selector,
            "display_name": display_name,
            "fp8_tflops": fp8_tflops,
            "fp4_tflops": fp4_tflops,
            "cuda_tflops": cuda_tflops,
            "io_tb_s": io_tb_s,
            "p2p_gb_s": p2p_gb_s,
            "family": family,
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
    assert "Hardware bottlenecks for Wan2.1-T2V-14B" in english_text
    assert "analytic sizing, fp8" in english_text
    assert "Wan2.1-T2V-14B 的硬件瓶颈" in chinese_text
    assert "stakeholder-report.cn.md" not in chinese_text
    assert "MemIO" in chinese_text
    assert "batch size" in chinese_text
    assert "Data-parallel replication" in chinese_text
    assert "Avg time per video (s/video)" in english_text
    assert "单视频平均耗时 (s/video)" in chinese_text
    assert "Throughput (videos/s)" not in english_text
    assert "吞吐量 (videos/s)" not in chinese_text
    assert "\n\n![](figures/dv100_full_pipeline_8gpu_throughput.svg)\n\n" in english_text
    assert "\n\n![](figures/dv100_full_pipeline_8gpu_throughput.svg)\n\n" in chinese_text


def _payload_for_rows(
    *,
    metadata: dict[str, object],
    rows: list[dict[str, object]],
    input_struct: tuple[str, int, int, int] = ("1280*720", 81, 50, 256),
) -> dict[str, object]:
    enriched_metadata = {
        **metadata,
        "workloads": [
            {
                "size": input_struct[0],
                "frame_num": input_struct[1],
                "steps": input_struct[2],
                "text_len": input_struct[3],
            },
        ],
        "batch_sizes": sorted({int(row["batch_size"]) for row in rows}),
        "device_nums": sorted({int(row["device_num"]) for row in rows}),
    }
    return {
        "metadata": enriched_metadata,
        "stage_breakdown_by_input_struct": {
            stage_breakdown_key(input_struct): {
                "text_encoder": {"tensor_tflops": 2.39659, "memio_tb": 0.00486469},
                "diffusion_core": {"tensor_tflops": 308617.0, "memio_tb": 848.724},
                "vae_decode": {"tensor_tflops": 660.444, "memio_tb": 0.185534},
            },
        },
        "rows": rows,
    }


def test_write_detailed_bundle_generates_generic_nvidia_bundle(tmp_path) -> None:
    wan2_1_dir = tmp_path / "wan2_1"
    comparison_dir = wan2_1_dir / "reports" / "hardware_sweeps" / "comparisons" / "nvidia-detailed"
    input_struct = ("1280*720", 81, 50, 256)
    h20_rows = [
        _row(
            batch_size=1,
            device_num=1,
            effective_gpus=1,
            total_cost_s=61.0,
            throughput_videos_per_s=1.0 / 61.0,
            tensor_cost_s=61.0,
            io_cost_s=14.0,
            model_tensor_tflops=18000.0,
            model_io_tb=850.0,
        ),
        _row(
            batch_size=8,
            device_num=8,
            effective_gpus=8,
            total_cost_s=61.0,
            throughput_videos_per_s=8.0 / 61.0,
            tensor_cost_s=61.0,
            io_cost_s=14.0,
            model_tensor_tflops=18000.0,
            model_io_tb=850.0,
        ),
    ]
    b200_rows = [
        _row(
            batch_size=1,
            device_num=1,
            effective_gpus=1,
            total_cost_s=6.0,
            throughput_videos_per_s=1.0 / 6.0,
            tensor_cost_s=6.0,
            io_cost_s=2.0,
            model_tensor_tflops=18000.0,
            model_io_tb=850.0,
        ),
        _row(
            batch_size=8,
            device_num=8,
            effective_gpus=8,
            total_cost_s=6.0,
            throughput_videos_per_s=8.0 / 6.0,
            tensor_cost_s=6.0,
            io_cost_s=2.0,
            model_tensor_tflops=18000.0,
            model_io_tb=850.0,
        ),
    ]
    payload_by_selector = {
        "h20": _payload_for_rows(
            metadata=_metadata(
                selector="h20",
                display_name="H20",
                fp8_tflops=296.0,
                cuda_tflops=1.0,
                io_tb_s=4.0,
                p2p_gb_s=300.0,
                family="nvidia",
            ),
            rows=h20_rows,
            input_struct=input_struct,
        ),
        "b200": _payload_for_rows(
            metadata=_metadata(
                selector="b200",
                display_name="B200",
                fp8_tflops=32000.0,
                cuda_tflops=1.0,
                io_tb_s=7.7,
                p2p_gb_s=14.4 * 1024.0,
                family="nvidia",
            ),
            rows=b200_rows,
            input_struct=input_struct,
        ),
    }
    for selector, run_id in (("h20", "run-h20"), ("b200", "run-b200")):
        run_dir = device_run_dir(wan2_1_dir=wan2_1_dir, device_selector=selector, run_id=run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(json.dumps(payload_by_selector[selector], indent=2) + "\n")

    _write_detailed_bundle(
        comparison_dir=comparison_dir,
        wan2_1_dir=wan2_1_dir,
        device_runs=(
            DeviceRunSpec("h20", "run-h20"),
            DeviceRunSpec("b200", "run-b200"),
        ),
    )

    english_text = (comparison_dir / "stakeholder-report.en.md").read_text()
    chinese_text = (comparison_dir / "stakeholder-report.cn.md").read_text()
    metadata = json.loads((comparison_dir / "bundle-metadata.json").read_text())

    assert metadata["devices"] == ["b200", "h20"]
    assert "B200 and H20" in english_text
    assert "B200 和 H20" in chinese_text
    assert "first-pass analytic assumptions pending external validation" in english_text
    assert "5) Cross-device conclusions" in english_text
    assert "5) 跨设备结论" in chinese_text
    assert "DV hardware bottlenecks" not in english_text
    assert (comparison_dir / "figures" / "h20_full_pipeline_8gpu_throughput.svg").exists()
    assert (comparison_dir / "figures" / "b200_full_pipeline_8gpu_throughput.svg").exists()
    assert (comparison_dir / "figures" / "full_pipeline_8gpu_throughput_compare.svg").exists()
    compare_svg_text = (comparison_dir / "figures" / "full_pipeline_8gpu_throughput_compare.svg").read_text()
    assert "NVIDIA devices" in compare_svg_text


def test_dv_main_keeps_wrapper_behavior(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_write_detailed_bundle(*, comparison_dir, wan2_1_dir, device_runs):
        captured["comparison_dir"] = comparison_dir
        captured["wan2_1_dir"] = wan2_1_dir
        captured["device_runs"] = device_runs
        return {"en": "ok", "cn": "ok"}

    monkeypatch.setattr(
        run_make_dv_stakeholder_report,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "comparison_run_id": "bundle-run",
                "dv100_run_id": "dv100-run",
                "dv200_run_id": "dv200-run",
                "dv300_run_id": "dv300-run",
            },
        )(),
    )
    monkeypatch.setattr(run_make_dv_stakeholder_report, "_write_detailed_bundle", fake_write_detailed_bundle)
    monkeypatch.setattr(run_make_dv_stakeholder_report.Path, "resolve", lambda self: tmp_path / "scripts" / "reporting" / "run_make_dv_stakeholder_report.py")

    assert run_make_dv_stakeholder_report.main() == 0
    assert captured["comparison_dir"] == tmp_path / "reports" / "hardware_sweeps" / "comparisons" / "bundle-run"
    assert captured["wan2_1_dir"] == tmp_path
    assert captured["device_runs"] == (
        DeviceRunSpec("dv100", "dv100-run"),
        DeviceRunSpec("dv200", "dv200-run"),
        DeviceRunSpec("dv300", "dv300-run"),
    )


def _comparison_row(summary: dict[str, object], *, run_id: str) -> dict[str, object]:
    return {
        "device_selector": summary["device_selector"],
        "device_name": summary["device_name"],
        "run_id": run_id,
        "precision": summary["precision_name"],
        "compute_precision": summary["compute_precision"],
        "storage_bits": summary["storage_bits"],
        "workload_size": summary["input_struct"][0],
        "workload_frame_num": summary["input_struct"][1],
        "workload_steps": summary["input_struct"][2],
        "workload_text_len": summary["input_struct"][3],
        "model_mode": summary["model_mode"],
        "util_profile": summary["util_profile"],
        "device_num": summary["device_num"],
        "primary_bottleneck": summary["primary_bottleneck"],
        "latency_batch1_s": summary["latency_batch1_s"],
        "peak_throughput_8gpu_videos_s": summary["peak_throughput_8gpu_videos_s"],
        "required_tensor_tflops_s": summary["required_tensor_tflops_s"],
        "required_cuda_tflops_s": summary["required_cuda_tflops_s"],
        "required_memio_tb_s": summary["required_memio_tb_s"],
        "tensor_peak_8gpu": summary["tensor_peak_8gpu"],
        "cuda_peak_8gpu": summary["cuda_peak_8gpu"],
        "memio_peak_8gpu": summary["memio_peak_8gpu"],
        "required_tensor_vs_peak_8gpu": summary["required_tensor_vs_peak_8gpu"],
        "required_cuda_vs_peak_8gpu": summary["required_cuda_vs_peak_8gpu"],
        "required_memio_vs_peak_8gpu": summary["required_memio_vs_peak_8gpu"],
        "used_memio_vs_peak_per_gpu_at_peak_throughput": summary["used_memio_vs_peak_per_gpu_at_peak_throughput"],
        "used_tensor_vs_peak_per_gpu_at_peak_throughput": summary["used_tensor_vs_peak_per_gpu_at_peak_throughput"],
        "dominant_gap_resource_kind": summary["dominant_gap_resource_kind"],
        "dominant_gap_resource_label": summary["dominant_gap_resource_label"],
        "dominant_gap_resource_short_label": summary["dominant_gap_resource_short_label"],
        "dominant_gap_peak_label": summary["dominant_gap_peak_label"],
        "dominant_gap_unit": summary["dominant_gap_unit"],
        "required_dominant_value": summary["required_dominant_value"],
        "dominant_peak_8gpu": summary["dominant_peak_8gpu"],
        "required_dominant_vs_peak_8gpu": summary["required_dominant_vs_peak_8gpu"],
        "saturation_batch_size": summary["saturation_batch_size"],
    }


def _write_detailed_bundle_fixture(
    *,
    bundle_dir,
    precision_name: str,
    compute_precision: str,
    storage_bits: int,
    summaries_by_selector: dict[str, dict[str, object]],
    run_ids: dict[str, str],
    input_struct: tuple[str, int, int, int],
    device_selectors: tuple[str, ...] | None = None,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    ordered_selectors = device_selectors or tuple(summaries_by_selector.keys())
    write_results_csv(
        bundle_dir / "comparison-table.csv",
        [_comparison_row(summaries_by_selector[selector], run_id=run_ids[selector]) for selector in ordered_selectors],
    )
    (bundle_dir / "bundle-metadata.json").write_text(
        json.dumps(
            {
                "bundle_kind": "detailed",
                "devices": list(ordered_selectors),
                "precision": {
                    "name": precision_name,
                    "compute_precision": compute_precision,
                    "storage_bits": storage_bits,
                },
                "input_struct": {
                    "workload_size": input_struct[0],
                    "workload_frame_num": input_struct[1],
                    "workload_steps": input_struct[2],
                    "workload_text_len": input_struct[3],
                },
                "model_mode": "full_pipeline",
                "util_profile": "optimistic",
                "device_num": 8,
                "run_ids": run_ids,
            },
            indent=2,
        )
        + "\n",
    )


def _read_csv(path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_validate_comparable_runs_rejects_precision_mismatch() -> None:
    input_struct = ("1280*720", 81, 50, 256)
    metadata_fp8 = _metadata(selector="dv100", display_name="DV100", fp8_tflops=2000.0, fp4_tflops=4000.0, cuda_tflops=1.0, io_tb_s=2.0)
    metadata_fp4 = _metadata(
        selector="dv200",
        display_name="DV200",
        fp8_tflops=4000.0,
        fp4_tflops=12000.0,
        cuda_tflops=1.0,
        io_tb_s=10.0,
        precision_name="fp4",
        compute_precision="fp4",
        storage_bits=4,
    )
    rows_io = [
        _row(
            batch_size=1,
            device_num=1,
            effective_gpus=1,
            total_cost_s=100.0,
            throughput_videos_per_s=0.01,
            tensor_cost_s=10.0,
            io_cost_s=100.0,
            model_tensor_tflops=1000.0,
            model_io_tb=200.0,
        ),
        _row(
            batch_size=8,
            device_num=8,
            effective_gpus=8,
            total_cost_s=100.0,
            throughput_videos_per_s=0.08,
            tensor_cost_s=10.0,
            io_cost_s=100.0,
            model_tensor_tflops=1000.0,
            model_io_tb=200.0,
        ),
    ]
    rows_tensor = [
        _row(
            batch_size=1,
            device_num=1,
            effective_gpus=1,
            total_cost_s=10.0,
            throughput_videos_per_s=0.1,
            tensor_cost_s=10.0,
            io_cost_s=1.0,
            model_tensor_tflops=1000.0,
            model_io_tb=200.0,
        ),
        _row(
            batch_size=8,
            device_num=8,
            effective_gpus=8,
            total_cost_s=10.0,
            throughput_videos_per_s=0.8,
            tensor_cost_s=10.0,
            io_cost_s=1.0,
            model_tensor_tflops=1000.0,
            model_io_tb=200.0,
        ),
    ]
    payloads_by_device = {
        "dv100": {"metadata": metadata_fp8, "workloads": [], "batch_sizes": [], "device_nums": []},
        "dv200": {"metadata": metadata_fp4, "workloads": [], "batch_sizes": [], "device_nums": []},
        "dv300": {"metadata": metadata_fp8, "workloads": [], "batch_sizes": [], "device_nums": []},
    }
    summaries_by_device = {
        "dv100": compute_device_summary(metadata_fp8, rows_io),
        "dv200": compute_device_summary(metadata_fp4, rows_tensor),
        "dv300": compute_device_summary(metadata_fp8, rows_io),
    }
    for summary in summaries_by_device.values():
        assert summary["input_struct"] == input_struct

    with pytest.raises(ValueError, match="precision mismatch"):
        _validate_comparable_runs(payloads_by_device=payloads_by_device, summaries_by_device=summaries_by_device)


def test_write_comparative_summary_bundle_generates_bilingual_reports(tmp_path) -> None:
    input_struct = ("1280*720", 81, 50, 256)
    run_ids = {"dv100": "dv100-run", "dv200": "dv200-run", "dv300": "dv300-run"}
    fp8_summaries = {
        "dv100": compute_device_summary(
            _metadata(selector="dv100", display_name="DV100", fp8_tflops=2000.0, fp4_tflops=4000.0, cuda_tflops=1.0, io_tb_s=2.0),
            [
                _row(batch_size=1, device_num=1, effective_gpus=1, total_cost_s=400.0, throughput_videos_per_s=0.0025, tensor_cost_s=100.0, io_cost_s=400.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
                _row(batch_size=8, device_num=8, effective_gpus=8, total_cost_s=400.0, throughput_videos_per_s=0.02, tensor_cost_s=100.0, io_cost_s=400.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
            ],
        ),
        "dv200": compute_device_summary(
            _metadata(selector="dv200", display_name="DV200", fp8_tflops=4000.0, fp4_tflops=12000.0, cuda_tflops=1.0, io_tb_s=10.0),
            [
                _row(batch_size=1, device_num=1, effective_gpus=1, total_cost_s=80.0, throughput_videos_per_s=0.0125, tensor_cost_s=40.0, io_cost_s=80.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
                _row(batch_size=8, device_num=8, effective_gpus=8, total_cost_s=80.0, throughput_videos_per_s=0.1, tensor_cost_s=40.0, io_cost_s=80.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
            ],
        ),
        "dv300": compute_device_summary(
            _metadata(selector="dv300", display_name="DV300", fp8_tflops=8000.0, fp4_tflops=24000.0, cuda_tflops=1.0, io_tb_s=50.0),
            [
                _row(batch_size=1, device_num=1, effective_gpus=1, total_cost_s=40.0, throughput_videos_per_s=0.025, tensor_cost_s=40.0, io_cost_s=5.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
                _row(batch_size=8, device_num=8, effective_gpus=8, total_cost_s=40.0, throughput_videos_per_s=0.2, tensor_cost_s=40.0, io_cost_s=5.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
            ],
        ),
    }
    fp4_summaries = {
        "dv100": compute_device_summary(
            _metadata(selector="dv100", display_name="DV100", fp8_tflops=2000.0, fp4_tflops=4000.0, cuda_tflops=1.0, io_tb_s=2.0, precision_name="fp4", compute_precision="fp4", storage_bits=4),
            [
                _row(batch_size=1, device_num=1, effective_gpus=1, total_cost_s=200.0, throughput_videos_per_s=0.005, tensor_cost_s=50.0, io_cost_s=200.0, model_tensor_tflops=300000.0, model_io_tb=400.0),
                _row(batch_size=8, device_num=8, effective_gpus=8, total_cost_s=200.0, throughput_videos_per_s=0.04, tensor_cost_s=50.0, io_cost_s=200.0, model_tensor_tflops=300000.0, model_io_tb=400.0),
            ],
        ),
        "dv200": compute_device_summary(
            _metadata(selector="dv200", display_name="DV200", fp8_tflops=4000.0, fp4_tflops=12000.0, cuda_tflops=1.0, io_tb_s=10.0, precision_name="fp4", compute_precision="fp4", storage_bits=4),
            [
                _row(batch_size=1, device_num=1, effective_gpus=1, total_cost_s=30.0, throughput_videos_per_s=0.0333, tensor_cost_s=25.0, io_cost_s=30.0, model_tensor_tflops=300000.0, model_io_tb=400.0),
                _row(batch_size=8, device_num=8, effective_gpus=8, total_cost_s=30.0, throughput_videos_per_s=0.2667, tensor_cost_s=25.0, io_cost_s=30.0, model_tensor_tflops=300000.0, model_io_tb=400.0),
            ],
        ),
        "dv300": compute_device_summary(
            _metadata(selector="dv300", display_name="DV300", fp8_tflops=8000.0, fp4_tflops=24000.0, cuda_tflops=1.0, io_tb_s=50.0, precision_name="fp4", compute_precision="fp4", storage_bits=4),
            [
                _row(batch_size=1, device_num=1, effective_gpus=1, total_cost_s=13.0, throughput_videos_per_s=0.0769, tensor_cost_s=13.0, io_cost_s=2.0, model_tensor_tflops=300000.0, model_io_tb=400.0),
                _row(batch_size=8, device_num=8, effective_gpus=8, total_cost_s=13.0, throughput_videos_per_s=0.6154, tensor_cost_s=13.0, io_cost_s=2.0, model_tensor_tflops=300000.0, model_io_tb=400.0),
            ],
        ),
    }

    fp8_dir = tmp_path / "fp8-detailed"
    fp4_dir = tmp_path / "fp4-detailed"
    summary_dir = tmp_path / "comparative-summary"
    _write_detailed_bundle_fixture(
        bundle_dir=fp8_dir,
        precision_name="fp8",
        compute_precision="fp8",
        storage_bits=8,
        summaries_by_selector=fp8_summaries,
        run_ids=run_ids,
        input_struct=input_struct,
    )
    _write_detailed_bundle_fixture(
        bundle_dir=fp4_dir,
        precision_name="fp4",
        compute_precision="fp4",
        storage_bits=4,
        summaries_by_selector=fp4_summaries,
        run_ids=run_ids,
        input_struct=input_struct,
    )

    report_texts = _write_comparative_summary_bundle(
        comparison_dirs=[fp8_dir, fp4_dir],
        summary_dir=summary_dir,
    )

    assert set(report_texts.keys()) == {"en", "cn"}
    assert (summary_dir / "bundle-metadata.json").exists()
    assert (summary_dir / "comparison-table.csv").exists()
    assert (summary_dir / "stakeholder-summary.en.md").exists()
    assert (summary_dir / "stakeholder-summary.cn.md").exists()
    assert (summary_dir / "figures" / "comparative_batch1_latency.svg").exists()
    assert (summary_dir / "figures" / "comparative_avg_time_per_video.svg").exists()

    english_text = (summary_dir / "stakeholder-summary.en.md").read_text()
    chinese_text = (summary_dir / "stakeholder-summary.cn.md").read_text()
    assert "Wan2.1-T2V-14B hardware comparative summary" in english_text
    assert "Wan2.1-T2V-14B 硬件对比摘要" in chinese_text
    assert "At-a-glance scenario table" in english_text
    assert "场景一览表" in chinese_text
    assert "How to read these columns, by intuition" in english_text
    assert "这些列可以这样读" in chinese_text
    assert "Avg time per video at peak 8-GPU load (s/video)" in english_text
    assert "8-GPU 满载时单视频平均耗时 (s/video)" in chinese_text
    assert "best full-node operating point" in english_text
    assert "整机表现最好的那个运行点" in chinese_text
    assert "Used/peak MemIO per GPU" in english_text
    assert "Used/peak compute per GPU" in english_text
    assert "每卡 Used/peak MemIO" in chinese_text
    assert "每卡 Used/peak compute" in chinese_text
    assert _utilization_percent_text(float(fp8_summaries["dv100"]["used_memio_vs_peak_per_gpu_at_peak_throughput"])) in english_text
    assert _utilization_percent_text(float(fp8_summaries["dv100"]["used_tensor_vs_peak_per_gpu_at_peak_throughput"])) in english_text

    metadata = json.loads((summary_dir / "bundle-metadata.json").read_text())
    assert metadata["bundle_kind"] == "comparative-summary"
    assert metadata["scenario_ids"] == [
        "dv100-fp8",
        "dv100-fp4",
        "dv200-fp8",
        "dv200-fp4",
        "dv300-fp8",
        "dv300-fp4",
    ]

    comparison_rows = _read_csv(summary_dir / "comparison-table.csv")
    assert [row["scenario_id"] for row in comparison_rows] == metadata["scenario_ids"]
    assert comparison_rows[0]["scenario_label"] == "DV100 fp8"
    assert comparison_rows[1]["scenario_label"] == "DV100 fp4"
    assert comparison_rows[-1]["source_comparison_run_id"] == "fp4-detailed"


def test_write_comparative_summary_bundle_lists_dv_series_before_other_devices(tmp_path) -> None:
    input_struct = ("1280*720", 81, 50, 256)
    dv_summary = compute_device_summary(
        _metadata(selector="dv200", display_name="DV200", fp8_tflops=4000.0, fp4_tflops=12000.0, cuda_tflops=1.0, io_tb_s=10.0),
        [
            _row(batch_size=1, device_num=1, effective_gpus=1, total_cost_s=80.0, throughput_videos_per_s=0.0125, tensor_cost_s=40.0, io_cost_s=80.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
            _row(batch_size=8, device_num=8, effective_gpus=8, total_cost_s=80.0, throughput_videos_per_s=0.1, tensor_cost_s=40.0, io_cost_s=80.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
        ],
    )
    b200_summary = compute_device_summary(
        _metadata(selector="b200", display_name="B200", fp8_tflops=32000.0, cuda_tflops=1.0, io_tb_s=7.7, family="nvidia"),
        [
            _row(batch_size=1, device_num=1, effective_gpus=1, total_cost_s=12.0, throughput_videos_per_s=1.0 / 12.0, tensor_cost_s=4.0, io_cost_s=12.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
            _row(batch_size=8, device_num=8, effective_gpus=8, total_cost_s=12.0, throughput_videos_per_s=8.0 / 12.0, tensor_cost_s=4.0, io_cost_s=12.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
        ],
    )

    dv_dir = tmp_path / "dv-detailed"
    nvidia_dir = tmp_path / "nvidia-detailed"
    _write_detailed_bundle_fixture(
        bundle_dir=dv_dir,
        precision_name="fp8",
        compute_precision="fp8",
        storage_bits=8,
        summaries_by_selector={"dv200": dv_summary},
        run_ids={"dv200": "dv200-run"},
        input_struct=input_struct,
        device_selectors=("dv200",),
    )
    _write_detailed_bundle_fixture(
        bundle_dir=nvidia_dir,
        precision_name="fp8",
        compute_precision="fp8",
        storage_bits=8,
        summaries_by_selector={"b200": b200_summary},
        run_ids={"b200": "b200-run"},
        input_struct=input_struct,
        device_selectors=("b200",),
    )

    summary_dir = tmp_path / "mixed-summary"
    _write_comparative_summary_bundle(
        comparison_dirs=[nvidia_dir, dv_dir],
        summary_dir=summary_dir,
    )

    comparison_rows = _read_csv(summary_dir / "comparison-table.csv")
    assert [row["scenario_label"] for row in comparison_rows] == ["DV200 fp8", "B200 fp8"]
    english_text = (summary_dir / "stakeholder-summary.en.md").read_text()
    assert "Compared scenarios: `DV200 fp8`, `B200 fp8`." in english_text


def test_write_comparative_summary_bundle_rejects_mismatched_context(tmp_path) -> None:
    input_struct_fp8 = ("1280*720", 81, 50, 256)
    input_struct_fp4 = ("832*480", 81, 50, 256)
    run_ids = {"dv100": "dv100-run", "dv200": "dv200-run", "dv300": "dv300-run"}
    fp8_summaries = {
        "dv100": compute_device_summary(
            _metadata(selector="dv100", display_name="DV100", fp8_tflops=2000.0, fp4_tflops=4000.0, cuda_tflops=1.0, io_tb_s=2.0),
            [
                _row(batch_size=1, device_num=1, effective_gpus=1, total_cost_s=400.0, throughput_videos_per_s=0.0025, tensor_cost_s=100.0, io_cost_s=400.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
                _row(batch_size=8, device_num=8, effective_gpus=8, total_cost_s=400.0, throughput_videos_per_s=0.02, tensor_cost_s=100.0, io_cost_s=400.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
            ],
        ),
        "dv200": compute_device_summary(
            _metadata(selector="dv200", display_name="DV200", fp8_tflops=4000.0, fp4_tflops=12000.0, cuda_tflops=1.0, io_tb_s=10.0),
            [
                _row(batch_size=1, device_num=1, effective_gpus=1, total_cost_s=80.0, throughput_videos_per_s=0.0125, tensor_cost_s=40.0, io_cost_s=80.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
                _row(batch_size=8, device_num=8, effective_gpus=8, total_cost_s=80.0, throughput_videos_per_s=0.1, tensor_cost_s=40.0, io_cost_s=80.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
            ],
        ),
        "dv300": compute_device_summary(
            _metadata(selector="dv300", display_name="DV300", fp8_tflops=8000.0, fp4_tflops=24000.0, cuda_tflops=1.0, io_tb_s=50.0),
            [
                _row(batch_size=1, device_num=1, effective_gpus=1, total_cost_s=40.0, throughput_videos_per_s=0.025, tensor_cost_s=40.0, io_cost_s=5.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
                _row(batch_size=8, device_num=8, effective_gpus=8, total_cost_s=40.0, throughput_videos_per_s=0.2, tensor_cost_s=40.0, io_cost_s=5.0, model_tensor_tflops=300000.0, model_io_tb=800.0),
            ],
        ),
    }
    fp4_rows = [
        {
            **_row(batch_size=1, device_num=1, effective_gpus=1, total_cost_s=200.0, throughput_videos_per_s=0.005, tensor_cost_s=50.0, io_cost_s=200.0, model_tensor_tflops=300000.0, model_io_tb=400.0),
            "workload_size": input_struct_fp4[0],
        },
        {
            **_row(batch_size=8, device_num=8, effective_gpus=8, total_cost_s=200.0, throughput_videos_per_s=0.04, tensor_cost_s=50.0, io_cost_s=200.0, model_tensor_tflops=300000.0, model_io_tb=400.0),
            "workload_size": input_struct_fp4[0],
        },
    ]
    fp4_summaries = {
        "dv100": compute_device_summary(
            _metadata(selector="dv100", display_name="DV100", fp8_tflops=2000.0, fp4_tflops=4000.0, cuda_tflops=1.0, io_tb_s=2.0, precision_name="fp4", compute_precision="fp4", storage_bits=4),
            fp4_rows,
        ),
        "dv200": compute_device_summary(
            _metadata(selector="dv200", display_name="DV200", fp8_tflops=4000.0, fp4_tflops=12000.0, cuda_tflops=1.0, io_tb_s=10.0, precision_name="fp4", compute_precision="fp4", storage_bits=4),
            fp4_rows,
        ),
        "dv300": compute_device_summary(
            _metadata(selector="dv300", display_name="DV300", fp8_tflops=8000.0, fp4_tflops=24000.0, cuda_tflops=1.0, io_tb_s=50.0, precision_name="fp4", compute_precision="fp4", storage_bits=4),
            fp4_rows,
        ),
    }

    fp8_dir = tmp_path / "fp8-detailed"
    fp4_dir = tmp_path / "fp4-detailed"
    _write_detailed_bundle_fixture(
        bundle_dir=fp8_dir,
        precision_name="fp8",
        compute_precision="fp8",
        storage_bits=8,
        summaries_by_selector=fp8_summaries,
        run_ids=run_ids,
        input_struct=input_struct_fp8,
    )
    _write_detailed_bundle_fixture(
        bundle_dir=fp4_dir,
        precision_name="fp4",
        compute_precision="fp4",
        storage_bits=4,
        summaries_by_selector=fp4_summaries,
        run_ids=run_ids,
        input_struct=input_struct_fp4,
    )

    with pytest.raises(ValueError, match="Incompatible comparative summary inputs"):
        _write_comparative_summary_bundle(
            comparison_dirs=[fp8_dir, fp4_dir],
            summary_dir=tmp_path / "summary",
        )
