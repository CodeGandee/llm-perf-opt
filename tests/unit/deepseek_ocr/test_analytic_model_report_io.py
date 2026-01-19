"""Round-trip serialization tests for AnalyticModelReport.

These tests ensure that AnalyticModelReport instances can be converted to
JSON/YAML and back without losing key structural information. This mirrors
the export format written by the DeepSeek-OCR analytic runner.
"""

from __future__ import annotations

import json
from pathlib import Path

from attrs import asdict
from ruamel.yaml import YAML  # type: ignore[import-untyped]

from llm_perf_opt.data.deepseek_ocr_analytic import (
    AnalyticModelReport,
    AnalyticModuleNode,
    DeepSeekOCRModelSpec,
    ModuleMetricsSnapshot,
    OCRWorkloadProfile,
    OperatorCategory,
    OperatorMetrics,
)


def _make_dummy_report(tmp_path: Path) -> AnalyticModelReport:
    config_path = str((tmp_path / "dummy-config.json").resolve())
    layer_docs_dir = str((tmp_path / "layers").resolve())

    model = DeepSeekOCRModelSpec(
        model_id="deepseek-ai/DeepSeek-OCR",
        model_variant="deepseek-ocr-v1-base",
        hf_revision=None,
        config_path=config_path,
        hidden_size=1280,
        intermediate_size=5120,
        num_layers=2,
        num_attention_heads=16,
        vision_backbone="clip_vit_l",
        uses_moe=False,
        notes="dummy",
    )
    workload = OCRWorkloadProfile(
        profile_id="dsocr-standard-v1",
        description="dummy workload",
        seq_len=512,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        max_new_tokens=512,
        doc_kind="mixed_layout",
        num_pages=1,
    )
    modules = [
        AnalyticModuleNode(
            module_id="decoder/dummy",
            name="DummyModule",
            qualified_class_name="extern.modelmeter.models.deepseek_ocr.layers.dummy.DummyModule",
            stage="decode",
            parent_id=None,
            children=[],
            repetition="none",
            repetition_count=None,
            constructor_params={},
        ),
    ]
    operator_categories = [
        OperatorCategory(
            category_id="other",
            display_name="Other",
            description="Fallback category",
            match_classes=[],
        ),
    ]
    module_metrics = [
        ModuleMetricsSnapshot(
            module_id="decoder/dummy",
            profile_id=workload.profile_id,
            calls=1,
            total_time_ms=10.0,
            total_flops_tflops=1.0,
            total_io_tb=0.1,
            memory_weights_gb=0.5,
            memory_activations_gb=0.5,
            memory_kvcache_gb=0.0,
            share_of_model_time=1.0,
            operator_breakdown=[
                OperatorMetrics(
                    category_id="other",
                    calls=1,
                    flops_tflops=1.0,
                    io_tb=0.1,
                    share_of_module_flops=1.0,
                ),
            ],
        ),
    ]

    return AnalyticModelReport(
        report_id="dummy-report",
        model=model,
        workload=workload,
        modules=modules,
        operator_categories=operator_categories,
        module_metrics=module_metrics,
        profile_run_id=None,
        predicted_total_time_ms=10.0,
        measured_total_time_ms=None,
        predicted_vs_measured_ratio=None,
        notes="dummy report",
        layer_docs_dir=layer_docs_dir,
    )


def test_json_roundtrip(tmp_path: Path) -> None:
    report = _make_dummy_report(tmp_path)
    payload = asdict(report)
    json_data = json.dumps(payload)
    loaded = json.loads(json_data)
    # JSON round-trip should preserve the top-level structure and key fields.
    assert isinstance(loaded, dict)
    assert loaded["report_id"] == report.report_id
    assert loaded["model"]["model_variant"] == report.model.model_variant
    assert loaded["workload"]["profile_id"] == report.workload.profile_id
    assert len(loaded["modules"]) == len(report.modules)
    assert len(loaded["module_metrics"]) == len(report.module_metrics)


def test_yaml_roundtrip(tmp_path: Path) -> None:
    report = _make_dummy_report(tmp_path)
    payload = asdict(report)

    yaml = YAML(typ="safe")
    tmp_file = tmp_path / "report.yaml"
    with tmp_file.open("w", encoding="utf-8") as f:
        yaml.dump(payload, f)

    with tmp_file.open("r", encoding="utf-8") as f:
        loaded = yaml.load(f)

    assert isinstance(loaded, dict)
    assert loaded["report_id"] == report.report_id
    assert loaded["model"]["model_id"] == report.model.model_id
    assert loaded["layer_docs_dir"] == report.layer_docs_dir
