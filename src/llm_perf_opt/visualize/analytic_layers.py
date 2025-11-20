"""Markdown generation for DeepSeek-OCR analytic layer reports.

This module renders lightweight per-layer and summary Markdown
documentation from :class:`AnalyticModelReport` artifacts produced by
the analytic modeling pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from mdutils.mdutils import MdUtils  # type: ignore[import-untyped]

from modelmeter.models.deepseek_ocr.layers.decoder.deepseek_v2_decoder_layer import (
    DeepseekV2DecoderLayer,
)
from modelmeter.models.deepseek_ocr.layers.decoder.deepseek_v2_mlp import (
    DeepseekV2MLP,
)
from modelmeter.models.deepseek_ocr.layers.decoder.deepseek_v2_moe import (
    DeepseekV2MoE,
)
from modelmeter.models.deepseek_ocr.layers.llama.llama_flash_attention2 import (
    LlamaFlashAttention2,
)
from modelmeter.models.deepseek_ocr.layers.vision.image_encoder_vit import (
    ImageEncoderViT,
)
from modelmeter.models.deepseek_ocr.layers.vision.mlp_block import MLPBlock

from llm_perf_opt.data.deepseek_ocr_analytic import (
    AnalyticModelReport,
    AnalyticModuleNode,
    ModuleMetricsSnapshot,
)


def _index_modules(report: AnalyticModelReport) -> Dict[str, AnalyticModuleNode]:
    return {m.module_id: m for m in report.modules}


def _index_metrics(report: AnalyticModelReport) -> Dict[str, ModuleMetricsSnapshot]:
    return {m.module_id: m for m in report.module_metrics}


def _format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _write_summary(report: AnalyticModelReport, out_dir: Path) -> None:
    modules_by_id = _index_modules(report)
    metrics_by_id = _index_metrics(report)

    # Sort modules with metrics by descending share_of_model_time.
    rows: list[tuple[AnalyticModuleNode, ModuleMetricsSnapshot]] = []
    for module_id, metric in metrics_by_id.items():
        module = modules_by_id.get(module_id)
        if module is not None:
            rows.append((module, metric))
    rows.sort(key=lambda pair: pair[1].share_of_model_time, reverse=True)

    file_base = str(out_dir / "summary")
    md = MdUtils(file_name=file_base)
    md.new_header(level=1, title="DeepSeek-OCR Analytic Model – Summary")

    md.new_list(
        items=[
            f"Report ID: `{report.report_id}`",
            f"Model variant: `{report.model.model_variant}`",
            f"Workload profile: `{report.workload.profile_id}`",
        ],
    )

    md.new_header(level=2, title="Workload and input shapes")
    md.new_list(
        items=[
            f"Sequence length (tokens): `{report.workload.seq_len}`",
            f"Base image size (pixels): `{report.workload.base_size}`",
            f"Crop image size (pixels): `{report.workload.image_size}`",
            f"Crop mode: `{'enabled' if report.workload.crop_mode else 'disabled'}`",
            f"Max new tokens: `{report.workload.max_new_tokens}`",
            f"Document kind: `{report.workload.doc_kind}`",
            f"Pages: `{report.workload.num_pages}`",
        ],
    )

    md.new_header(level=2, title="Module Metrics")

    header = [
        "Module",
        "Stage",
        "Calls",
        "Time (ms)",
        "FLOPs (TFLOPs)",
        "Weights (GB)",
        "Activations (GB)",
        "KV cache (GB)",
        "Time share",
    ]
    table_data: list[str] = header.copy()
    for module, metric in rows:
        table_data.extend(
            [
                module.name,
                str(module.stage),
                str(metric.calls),
                _format_float(metric.total_time_ms, digits=3),
                _format_float(metric.total_flops_tflops, digits=3),
                _format_float(metric.memory_weights_gb, digits=4),
                _format_float(metric.memory_activations_gb, digits=4),
                _format_float(metric.memory_kvcache_gb, digits=4),
                _format_float(metric.share_of_model_time, digits=3),
            ],
        )

    md.new_table(columns=len(header), rows=len(rows) + 1, text=table_data, text_align="center")

    # Per-layer configuration/shape summary derived from constructor parameters.
    md.new_header(level=2, title="Module parameters (per-layer shapes/config)")

    param_header = ["Module", "Stage", "Parameters"]
    param_table: list[str] = param_header.copy()
    for module, _metric in rows:
        if module.constructor_params:
            parts = [f"{k}={v}" for k, v in module.constructor_params.items()]
            params_str = ", ".join(parts)
        else:
            params_str = "n/a"
        param_table.extend([module.name, str(module.stage), params_str])

    md.new_table(columns=len(param_header), rows=len(rows) + 1, text=param_table, text_align="left")

    # Analytic primitives (representative BaseLayer configurations) to expose deeper
    # per-layer shapes such as MLP blocks and attention primitives.
    primitive_header = ["Layer", "Scope", "Shape/config"]
    primitive_rows: list[tuple[str, str, str]] = []

    # Vision stack primitives derived from ImageEncoderViT.
    try:
        vision_node = modules_by_id.get("vision/image_encoder_vit")
        if vision_node is not None:
            vp = vision_node.constructor_params
            image_encoder = ImageEncoderViT(
                img_size=int(vp.get("img_size", 1024)),
                patch_size=int(vp.get("patch_size", 16)),
                embed_dim=int(vp.get("embed_dim", 768)),
                depth=int(vp.get("depth", 12)),
                num_heads=int(vp.get("num_heads", 12)),
                batch_size=1,
            )
            blocks = list(image_encoder.blocks)
            if blocks:
                block = blocks[0]
                attn = block.attention
                mlp = block.mlp
                # Attention primitive (vision).
                primitive_rows.append(
                    (
                        "Attention (vision)",
                        "vision/block",
                        (
                            f"B≈{block.batch_size}, S={block.seq_len}, "
                            f"H={block.dim}, heads={block.num_heads}, "
                            f"window_area={block.window_area}, "
                            f"num_windows={block.num_windows}"
                        ),
                    ),
                )
                # MLPBlock primitive (vision).
                if isinstance(mlp, MLPBlock):
                    primitive_rows.append(
                        (
                            "MLPBlock (vision)",
                            "vision/block",
                            (
                                f"B={mlp.batch_size}, S={mlp.seq_len}, "
                                f"D={mlp.embedding_dim}, MLP={mlp.mlp_dim}"
                            ),
                        ),
                    )
    except Exception:
        # Best-effort: if vision primitives fail, leave this section empty.
        primitive_rows = primitive_rows

    # Decoder primitives derived from DeepseekV2DecoderLayer.
    try:
        decoder_node = modules_by_id.get("decoder/deepseek_v2_decoder_layer")
        if decoder_node is not None:
            dp = decoder_node.constructor_params
            decoder = DeepseekV2DecoderLayer(
                hidden_size=int(dp.get("hidden_size", 0)),
                num_heads=int(dp.get("num_heads", 0)),
                seq_len=int(dp.get("seq_len", 0)),
                intermediate_size=int(dp.get("intermediate_size", 0)),
                num_experts=dp.get("num_experts"),
                batch_size=int(dp.get("batch_size", 1)),
                num_key_value_heads=int(
                    dp.get("num_key_value_heads") or dp.get("num_heads", 0),
                ),
                k_active=int(dp.get("k_active", 2)),
                num_shared_experts=int(dp.get("num_shared_experts", 0)),
            )
            attn = decoder.self_attn
            if isinstance(attn, LlamaFlashAttention2):
                primitive_rows.append(
                    (
                        "LlamaFlashAttention2 (decoder)",
                        "decoder",
                        (
                            f"B={attn.batch_size}, S={attn.seq_len}, "
                            f"H={attn.hidden_size}, heads={attn.num_heads}, "
                            f"kv_heads={attn.num_key_value_heads}"
                        ),
                    ),
                )
            mlp = decoder.mlp
            if isinstance(mlp, DeepseekV2MLP):
                primitive_rows.append(
                    (
                        "DeepseekV2MLP (decoder)",
                        "decoder",
                        (
                            f"B={mlp.batch_size}, S={mlp.seq_len}, "
                            f"H={mlp.hidden_size}, I={mlp.intermediate_size}"
                        ),
                    ),
                )
            elif isinstance(mlp, DeepseekV2MoE):
                primitive_rows.append(
                    (
                        "DeepseekV2MoE (decoder)",
                        "decoder",
                        (
                            f"B={mlp.batch_size}, S={mlp.seq_len}, "
                            f"H={mlp.hidden_size}, I={mlp.intermediate_size}, "
                            f"experts={mlp.num_experts}, "
                            f"k_active={mlp.k_active}, "
                            f"shared_experts={mlp.num_shared_experts}"
                        ),
                    ),
                )
    except Exception:
        primitive_rows = primitive_rows

    if primitive_rows:
        md.new_header(level=2, title="Analytic primitives (representative per-layer shapes)")
        prim_table: list[str] = primitive_header.copy()
        for layer_name, scope, shape_str in primitive_rows:
            prim_table.extend([layer_name, scope, shape_str])
        md.new_table(
            columns=len(primitive_header),
            rows=len(primitive_rows) + 1,
            text=prim_table,
            text_align="left",
        )

    md.create_md_file()


def _write_module_doc(
    out_dir: Path,
    module: AnalyticModuleNode,
    metric: Optional[ModuleMetricsSnapshot],
) -> None:
    safe_id = module.module_id.replace("/", "_")
    file_base = str(out_dir / safe_id)
    md = MdUtils(file_name=file_base)

    md.new_header(level=1, title=f"{module.name} (`{module.module_id}`)")

    parent = module.parent_id or "-"
    children = ", ".join(module.children) if module.children else "none"
    summary_items = [
        f"Stage: `{module.stage}`",
        f"Qualified class: `{module.qualified_class_name}`",
        f"Parent: `{parent}`",
        f"Children: {children}",
        f"Repetition: `{module.repetition}`",
    ]
    if module.repetition_count is not None:
        summary_items.append(f"Repetition count: `{module.repetition_count}`")
    md.new_list(items=summary_items)

    if module.constructor_params:
        md.new_header(level=2, title="Constructor parameters")
        ctor_items = [f"`{key}`: `{value}`" for key, value in module.constructor_params.items()]
        md.new_list(items=ctor_items)

    md.new_header(level=2, title="Analytic metrics")
    if metric is None:
        md.new_paragraph("No metrics recorded for this module in the current report.")
    else:
        metric_items = [
            f"Calls: `{metric.calls}`",
            f"Total time (ms): `{_format_float(metric.total_time_ms, 3)}`",
            f"Total FLOPs (TFLOPs): `{_format_float(metric.total_flops_tflops, 3)}`",
            f"Total I/O (Tb): `{_format_float(metric.total_io_tb, 3)}`",
            f"Weight memory (GB): `{_format_float(metric.memory_weights_gb, 4)}`",
            f"Activation memory (GB): `{_format_float(metric.memory_activations_gb, 4)}`",
            f"KV cache memory (GB): `{_format_float(metric.memory_kvcache_gb, 4)}`",
            f"Share of model time: `{_format_float(metric.share_of_model_time, 3)}`",
        ]
        md.new_list(items=metric_items)

    md.create_md_file()


def write_analytic_layer_docs(report: AnalyticModelReport) -> None:
    """Render per-layer and summary Markdown docs for an analytic report.

    Parameters
    ----------
    report : AnalyticModelReport
        Analytic model report with module metadata and metrics.
    """

    out_dir = Path(report.layer_docs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_summary(report, out_dir)

    modules_by_id = _index_modules(report)
    metrics_by_id = _index_metrics(report)
    for module_id, module in modules_by_id.items():
        metric = metrics_by_id.get(module_id)
        _write_module_doc(out_dir, module, metric)
