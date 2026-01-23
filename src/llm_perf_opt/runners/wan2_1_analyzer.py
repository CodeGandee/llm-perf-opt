"""Wan2.1 static analytic report generation.

This runner composes the Wan2.1 ModelMeter analytic model via Hydra and writes
machine-readable artifacts under:

    tmp/profile-output/<run_id>/static_analysis/wan2_1/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

from attrs import asdict
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from modelmeter.layers.conv2d import Conv2d as SharedConv2d
from modelmeter.layers.conv3d import Conv3d as SharedConv3d
from modelmeter.layers.linear import Linear as SharedLinear
from modelmeter.layers.mlp import Mlp as SharedMlp
from modelmeter.layers.self_attn import SelfAttention as SharedSelfAttention
from modelmeter.models.wan2_1.layers.core.wan2_1_dit_model import Wan2_1DiTModel
from modelmeter.models.wan2_1.layers.core.wan2_1_pipeline_model import Wan2_1PipelineModel
from modelmeter.models.wan2_1.layers.text_encoder.umt5_encoder_block import Wan2_1Umt5EncoderBlock
from modelmeter.models.wan2_1.layers.text_encoder.umt5_encoder_model import Wan2_1Umt5EncoderModel
from modelmeter.models.wan2_1.layers.text_encoder.umt5_mlp import Wan2_1Umt5GatedMlp
from modelmeter.models.wan2_1.layers.transformer.wan2_1_attention import Wan2_1Attention
from modelmeter.models.wan2_1.layers.transformer.wan2_1_mlp import Wan2_1MLP
from modelmeter.models.wan2_1.layers.transformer.wan2_1_transformer_block import Wan2_1TransformerBlock
from modelmeter.models.wan2_1.layers.transformer.step_scaled_layer import Wan2_1StepScaledLayer
from modelmeter.models.wan2_1.layers.vae.wan2_1_vae_decode_model import Wan2_1VaeDecodeModel

from llm_perf_opt.data.analytic_common import AnalyticModuleNode, ModuleMetricsSnapshot, OperatorCategory, OperatorMetrics
from llm_perf_opt.data.wan2_1_analytic import Wan2_1AnalyticModelReport, Wan2_1ModelSpec, Wan2_1WorkloadProfile
from llm_perf_opt.utils.paths import wan2_1_analytic_dir, wan2_1_report_path, wan2_1_summary_path
from llm_perf_opt.visualize.wan2_1_analytic_summary import render_wan2_1_summary_md


def _workspace_root() -> Path:
    """Return the workspace root directory (repo root)."""

    return Path(__file__).resolve().parents[3]


def _default_config_json_path() -> Path:
    """Return the default local Wan2.1 `config.json` path (absolute)."""

    return (
        _workspace_root()
        / "models"
        / "wan2.1-t2v-14b"
        / "source-data"
        / "config.json"
    ).resolve()


def _load_cfg(overrides: List[str] | None = None, *, config_name: str = "wan2_1_t2v_14b") -> DictConfig:
    """Load the Wan2.1 analytic Hydra config from `extern/modelmeter`."""

    from modelmeter.models.wan2_1 import configs_dir

    config_dir = Path(configs_dir()).resolve()
    overrides = overrides or []
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg: DictConfig = compose(config_name=str(config_name), overrides=overrides)
    return cfg


def _apply_workload_profile(cfg: DictConfig) -> None:
    """Resolve `workload.profile_id` into explicit workload fields."""

    if "workload" not in cfg:
        raise ValueError("Wan2.1 config must provide a `workload` section")

    workload = cfg.workload
    profile_id = str(getattr(workload, "profile_id", "") or "").strip()
    if not profile_id:
        raise ValueError("workload.profile_id must be set")

    presets = getattr(workload, "presets", None)
    if presets is not None and profile_id in presets:
        preset = presets[profile_id]
        for k, v in dict(preset).items():
            if k == "description":
                workload.description = v
            else:
                setattr(workload, k, v)

    overrides = getattr(workload, "overrides", None)
    if overrides:
        for k, v in dict(overrides).items():
            setattr(workload, k, v)

    # Keep profile_id stable even when presets override other fields.
    workload.profile_id = profile_id


def _qualified_name(obj: object) -> str:
    """Return a stable qualified name for an object's class."""

    cls = obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _safe_float(value: float) -> float:
    """Validate a float metric value is finite, then return it."""

    if value != value:  # NaN
        raise ValueError("metric value must be finite (got NaN)")
    if value == float("inf") or value == float("-inf"):
        raise ValueError("metric value must be finite (got inf)")
    return float(value)


def _operator_categories() -> list[OperatorCategory]:
    """Return the operator category list used for Wan2.1 reports."""

    return [
        OperatorCategory(
            category_id="attention_proj",
            display_name="Attention projections",
            description="Q/K/V/O projections (GEMMs).",
            match_classes=["torch.nn.modules.linear.Linear"],
        ),
        OperatorCategory(
            category_id="attention_core",
            display_name="Attention core",
            description="QK^T and PV matmuls (attention core).",
            match_classes=["torch.matmul"],
        ),
        OperatorCategory(
            category_id="mlp_proj",
            display_name="MLP projections",
            description="MLP linear projections (GEMMs).",
            match_classes=["torch.nn.modules.linear.Linear"],
        ),
        OperatorCategory(
            category_id="conv",
            display_name="Convolutions",
            description="Conv2d/Conv3d layers (GEMMs/implicit GEMMs).",
            match_classes=["torch.nn.modules.conv.Conv2d", "torch.nn.modules.conv.Conv3d"],
        ),
        OperatorCategory(
            category_id="other",
            display_name="Other",
            description="Fallback category.",
            match_classes=[],
        ),
    ]


def _block_id(root_id: str, idx: int) -> str:
    """Return a stable module_id for a diffusion transformer block."""

    return f"{root_id}/block_{idx:02d}"


def _diffusion_root_id() -> str:
    """Return the module id used for the diffusion core root node."""

    return "diffusion/dit"


def _pipeline_root_id() -> str:
    """Return the module id used for the full pipeline root node."""

    return "pipeline"


def _build_diffusion_modules(*, model: Wan2_1DiTModel, parent_id: str | None) -> tuple[list[AnalyticModuleNode], dict[str, object]]:
    """Build a hierarchical module tree for the diffusion core and a module_id→layer mapping."""

    root_id = _diffusion_root_id()
    geom = model.token_geometry
    modules: list[AnalyticModuleNode] = []
    layer_by_id: dict[str, object] = {}

    root_node = AnalyticModuleNode(
        module_id=root_id,
        name="Wan2.1 DiT (diffusion core)",
        qualified_class_name=_qualified_name(model),
        stage="diffusion",
        parent_id=parent_id,
        children=[_block_id(root_id, i) for i in range(len(model.m_blocks))],
        repetition="none",
        repetition_count=None,
        constructor_params={
            "batch_size": geom.batch_size,
            "num_frames": geom.num_frames,
            "height": geom.height,
            "width": geom.width,
            "vae_downsample_factor": geom.vae_downsample_factor,
            "patch_size": geom.patch_size,
            "latent_h": geom.latent_h,
            "latent_w": geom.latent_w,
            "tokens_per_frame": geom.tokens_per_frame,
            "dit_seq_len": geom.dit_seq_len,
        },
    )
    modules.append(root_node)
    layer_by_id[root_id] = model

    for block in model.m_blocks:
        block_id = _block_id(root_id, block.block_idx)
        attn_id = f"{block_id}/attn"
        mlp_id = f"{block_id}/mlp"
        modules.append(
            AnalyticModuleNode(
                module_id=block_id,
                name=f"Block {block.block_idx:02d}",
                qualified_class_name=_qualified_name(block),
                stage="diffusion",
                parent_id=root_id,
                children=[attn_id, mlp_id],
                repetition="none",
                repetition_count=None,
                constructor_params={"block_idx": block.block_idx, "dit_seq_len": geom.dit_seq_len},
            ),
        )
        layer_by_id[block_id] = block

        attn = block.m_attn
        mlp = block.m_mlp
        if attn is None or mlp is None:
            raise RuntimeError(f"Block {block.block_idx} must have attention and MLP configured")

        modules.append(
            AnalyticModuleNode(
                module_id=attn_id,
                name="Attention",
                qualified_class_name=_qualified_name(attn),
                stage="diffusion",
                parent_id=block_id,
                children=[],
                repetition="none",
                repetition_count=None,
                constructor_params={"block_idx": block.block_idx, "dit_seq_len": geom.dit_seq_len},
            ),
        )
        layer_by_id[attn_id] = attn

        modules.append(
            AnalyticModuleNode(
                module_id=mlp_id,
                name="MLP",
                qualified_class_name=_qualified_name(mlp),
                stage="diffusion",
                parent_id=block_id,
                children=[],
                repetition="none",
                repetition_count=None,
                constructor_params={"block_idx": block.block_idx, "dit_seq_len": geom.dit_seq_len},
            ),
        )
        layer_by_id[mlp_id] = mlp

    return modules, layer_by_id


def _build_text_encoder_modules(*, model: Wan2_1Umt5EncoderModel, parent_id: str) -> tuple[list[AnalyticModuleNode], dict[str, object]]:
    """Build a module tree for the UMT5 encoder stage."""

    root_id = "text_encoder/umt5"
    modules: list[AnalyticModuleNode] = []
    layer_by_id: dict[str, object] = {}

    embedding_id = f"{root_id}/embedding"
    block_ids = [f"{root_id}/block_{i:02d}" for i in range(len(model.m_blocks))]
    children = [embedding_id] + block_ids

    modules.append(
        AnalyticModuleNode(
            module_id=root_id,
            name="UMT5 encoder (text encoder)",
            qualified_class_name=_qualified_name(model),
            stage="text_encoder",
            parent_id=parent_id,
            children=children,
            repetition="none",
            repetition_count=None,
            constructor_params={},
        ),
    )
    layer_by_id[root_id] = model

    embedding = model.m_embedding
    if embedding is not None:
        modules.append(
            AnalyticModuleNode(
                module_id=embedding_id,
                name="Token embedding",
                qualified_class_name=_qualified_name(embedding),
                stage="text_encoder",
                parent_id=root_id,
                children=[],
                repetition="none",
                repetition_count=None,
                constructor_params={},
            ),
        )
        layer_by_id[embedding_id] = embedding

    for idx, block in enumerate(model.m_blocks):
        block_id = block_ids[idx]
        attn_id = f"{block_id}/attn"
        mlp_id = f"{block_id}/mlp"

        modules.append(
            AnalyticModuleNode(
                module_id=block_id,
                name=f"Encoder block {idx:02d}",
                qualified_class_name=_qualified_name(block),
                stage="text_encoder",
                parent_id=root_id,
                children=[attn_id, mlp_id],
                repetition="none",
                repetition_count=None,
                constructor_params={},
            ),
        )
        layer_by_id[block_id] = block

        if block.m_attn is None or block.m_mlp is None:
            raise RuntimeError(f"UMT5 encoder block {idx} must have attention and MLP configured")

        modules.append(
            AnalyticModuleNode(
                module_id=attn_id,
                name="Self-attention",
                qualified_class_name=_qualified_name(block.m_attn),
                stage="text_encoder",
                parent_id=block_id,
                children=[],
                repetition="none",
                repetition_count=None,
                constructor_params={},
            ),
        )
        layer_by_id[attn_id] = block.m_attn

        modules.append(
            AnalyticModuleNode(
                module_id=mlp_id,
                name="Gated MLP",
                qualified_class_name=_qualified_name(block.m_mlp),
                stage="text_encoder",
                parent_id=block_id,
                children=[],
                repetition="none",
                repetition_count=None,
                constructor_params={},
            ),
        )
        layer_by_id[mlp_id] = block.m_mlp

    return modules, layer_by_id


def _build_vae_decode_modules(*, model: Wan2_1VaeDecodeModel, parent_id: str) -> tuple[list[AnalyticModuleNode], dict[str, object]]:
    """Build a module tree for the VAE decode stage."""

    root_id = "vae/decode"
    modules: list[AnalyticModuleNode] = []
    layer_by_id: dict[str, object] = {}

    layer_ids = [f"{root_id}/layer_{idx:02d}" for idx in range(len(model.m_layers))]
    modules.append(
        AnalyticModuleNode(
            module_id=root_id,
            name="Wan-VAE decode",
            qualified_class_name=_qualified_name(model),
            stage="vae",
            parent_id=parent_id,
            children=layer_ids,
            repetition="none",
            repetition_count=None,
            constructor_params={},
        ),
    )
    layer_by_id[root_id] = model

    for idx, layer in enumerate(model.m_layers):
        layer_id = layer_ids[idx]
        modules.append(
            AnalyticModuleNode(
                module_id=layer_id,
                name=layer.__class__.__name__,
                qualified_class_name=_qualified_name(layer),
                stage="vae",
                parent_id=root_id,
                children=[],
                repetition="none",
                repetition_count=None,
                constructor_params={},
            ),
        )
        layer_by_id[layer_id] = layer

    return modules, layer_by_id


def _build_pipeline_modules(model: Wan2_1PipelineModel) -> tuple[str, list[AnalyticModuleNode], dict[str, object]]:
    """Build a module tree for the full pipeline (text encoder + diffusion + VAE)."""

    root_id = _pipeline_root_id()

    text_root_id = "text_encoder/umt5"
    diff_root_id = _diffusion_root_id()
    vae_root_id = "vae/decode"

    modules: list[AnalyticModuleNode] = [
        AnalyticModuleNode(
            module_id=root_id,
            name="Wan2.1 pipeline (text + diffusion + VAE)",
            qualified_class_name=_qualified_name(model),
            stage="pipeline",
            parent_id=None,
            children=[text_root_id, diff_root_id, vae_root_id],
            repetition="none",
            repetition_count=None,
            constructor_params={},
        ),
    ]
    layer_by_id: dict[str, object] = {root_id: model}

    te_modules, te_layers = _build_text_encoder_modules(model=model.text_encoder, parent_id=root_id)
    diff_modules, diff_layers = _build_diffusion_modules(model=model.diffusion_core, parent_id=root_id)
    vae_modules, vae_layers = _build_vae_decode_modules(model=model.vae_decode, parent_id=root_id)

    modules.extend(te_modules)
    modules.extend(diff_modules)
    modules.extend(vae_modules)
    layer_by_id.update(te_layers)
    layer_by_id.update(diff_layers)
    layer_by_id.update(vae_layers)

    return root_id, modules, layer_by_id


def _build_module_tree(model: Wan2_1DiTModel | Wan2_1PipelineModel) -> tuple[str, list[AnalyticModuleNode], dict[str, object]]:
    """Build a module tree for either diffusion-core-only or full-pipeline mode."""

    if isinstance(model, Wan2_1DiTModel):
        modules, layer_by_id = _build_diffusion_modules(model=model, parent_id=None)
        return _diffusion_root_id(), modules, layer_by_id
    if isinstance(model, Wan2_1PipelineModel):
        return _build_pipeline_modules(model)
    raise TypeError(f"Unsupported Wan2.1 analytic model type: {type(model)!r}")


def _sum_breakdowns(breakdowns: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Sum multiple `{category_id: flops}` breakdown dicts."""

    out: dict[str, float] = {}
    for bd in breakdowns:
        for k, v in bd.items():
            out[k] = out.get(k, 0.0) + float(v)
    return out


def _layer_breakdown_tflops(layer: object) -> dict[str, float]:
    """Return a `{category_id: TFLOPs}` breakdown for a known layer type."""

    if isinstance(layer, Wan2_1PipelineModel):
        return _sum_breakdowns(
            [
                _layer_breakdown_tflops(layer.text_encoder),
                _layer_breakdown_tflops(layer.diffusion_core),
                _layer_breakdown_tflops(layer.vae_decode),
            ],
        )

    if isinstance(layer, Wan2_1Umt5EncoderModel):
        parts: list[dict[str, float]] = []
        if layer.m_embedding is not None:
            parts.append(_layer_breakdown_tflops(layer.m_embedding))
        parts.extend(_layer_breakdown_tflops(b) for b in layer.m_blocks)
        return _sum_breakdowns(parts)

    if isinstance(layer, Wan2_1Umt5EncoderBlock):
        if layer.m_attn is None or layer.m_mlp is None:
            raise RuntimeError("UMT5 encoder block missing attention/MLP")
        return _sum_breakdowns([_layer_breakdown_tflops(layer.m_attn), _layer_breakdown_tflops(layer.m_mlp)])

    if isinstance(layer, Wan2_1Umt5GatedMlp):
        return {"mlp_proj": float(layer.forward_tensor_core_flops() or 0.0)}

    if isinstance(layer, Wan2_1VaeDecodeModel):
        return _sum_breakdowns([_layer_breakdown_tflops(sublayer) for sublayer in layer.m_layers])

    if isinstance(layer, Wan2_1StepScaledLayer):
        inner = layer.inner
        steps = float(layer.num_inference_steps)
        if isinstance(inner, SharedSelfAttention):
            proj = float(
                (inner.q_proj.forward_tensor_core_flops() or 0.0)
                + (inner.k_proj.forward_tensor_core_flops() or 0.0)
                + (inner.v_proj.forward_tensor_core_flops() or 0.0)
                + (inner.o_proj.forward_tensor_core_flops() or 0.0),
            )
            core = float(inner.attn.forward_tensor_core_flops() or 0.0)
            return {"attention_proj": float(proj * steps), "attention_core": float(core * steps)}
        if isinstance(inner, SharedMlp):
            proj = float(
                (inner.linear1.forward_tensor_core_flops() or 0.0)
                + (inner.linear2.forward_tensor_core_flops() or 0.0),
            )
            return {"mlp_proj": float(proj * steps)}

    if isinstance(layer, SharedSelfAttention):
        proj = float(
            (layer.q_proj.forward_tensor_core_flops() or 0.0)
            + (layer.k_proj.forward_tensor_core_flops() or 0.0)
            + (layer.v_proj.forward_tensor_core_flops() or 0.0)
            + (layer.o_proj.forward_tensor_core_flops() or 0.0),
        )
        core = float(layer.attn.forward_tensor_core_flops() or 0.0)
        return {"attention_proj": float(proj), "attention_core": float(core)}

    if isinstance(layer, SharedMlp):
        proj = float((layer.linear1.forward_tensor_core_flops() or 0.0) + (layer.linear2.forward_tensor_core_flops() or 0.0))
        return {"mlp_proj": float(proj)}

    if isinstance(layer, Wan2_1Attention):
        bd = layer.forward_tensor_core_flops_breakdown()
        return {"attention_proj": float(bd.proj_tflops), "attention_core": float(bd.core_tflops)}
    if isinstance(layer, Wan2_1MLP):
        bd = layer.forward_tensor_core_flops_breakdown()
        return {"mlp_proj": float(bd.proj_tflops)}
    if isinstance(layer, Wan2_1TransformerBlock):
        if layer.m_attn is None or layer.m_mlp is None:
            raise RuntimeError(f"Block {layer.block_idx} missing attention/MLP")
        return _sum_breakdowns([_layer_breakdown_tflops(layer.m_attn), _layer_breakdown_tflops(layer.m_mlp)])
    if isinstance(layer, Wan2_1DiTModel):
        return _sum_breakdowns([_layer_breakdown_tflops(b) for b in layer.m_blocks])

    if isinstance(layer, SharedLinear):
        return {"mlp_proj": float(layer.forward_tensor_core_flops() or 0.0)}

    if isinstance(layer, (SharedConv2d, SharedConv3d)):
        return {"conv": float(layer.forward_tensor_core_flops() or 0.0)}

    # Fallback: treat unknown layer as "other" (best-effort).
    flops = float(getattr(layer, "forward_tensor_core_flops", lambda: 0.0)() or 0.0)
    return {"other": float(flops)}


def _root_module_id(modules: list[AnalyticModuleNode]) -> str:
    """Return the report root module id."""

    roots = [m.module_id for m in modules if m.parent_id is None]
    if len(roots) == 1:
        return str(roots[0])
    if _diffusion_root_id() in {m.module_id for m in modules}:
        return _diffusion_root_id()
    raise ValueError("Unable to determine a unique report root module id")


def _calls_for_node(*, node: AnalyticModuleNode, num_inference_steps: int) -> int:
    """Return a best-effort call count for a module node.

    Notes
    -----
    - The diffusion core executes once per sampling step.
    - The text encoder and VAE decode stages execute once per generated video.
    - The analytic FLOPs reported in this runner are already scaled by steps
      for diffusion blocks (via `Wan2_1StepScaledLayer`), so `calls` should be
      interpreted as a logical execution count, not a multiplier for FLOPs.
    """

    if node.stage == "diffusion" and node.module_id != _diffusion_root_id():
        return int(num_inference_steps)
    return 1


def _build_metrics(
    *,
    workload_profile_id: str,
    root_id: str,
    modules: list[AnalyticModuleNode],
    layer_by_id: dict[str, object],
    num_inference_steps: int,
) -> list[ModuleMetricsSnapshot]:
    """Compute module-level metric snapshots for report generation."""

    root_layer = layer_by_id[root_id]
    root_breakdown = _layer_breakdown_tflops(root_layer)
    root_flops = float(sum(root_breakdown.values()))
    if root_flops <= 0.0:
        raise ValueError("root FLOPs must be positive")

    snapshots: list[ModuleMetricsSnapshot] = []
    for node in modules:
        layer = layer_by_id.get(node.module_id)
        if layer is None:
            continue
        breakdown = _layer_breakdown_tflops(layer)
        total_flops = float(sum(breakdown.values()))

        calls = _calls_for_node(node=node, num_inference_steps=num_inference_steps)

        op_breakdown: list[OperatorMetrics] = []
        for cat_id, flops in breakdown.items():
            if flops <= 0.0:
                continue
            op_breakdown.append(
                OperatorMetrics(
                    category_id=str(cat_id),
                    calls=calls,
                    flops_tflops=_safe_float(float(flops)),
                    io_tb=0.0,
                    share_of_module_flops=_safe_float(float(flops / total_flops)) if total_flops > 0.0 else 0.0,
                ),
            )

        snapshots.append(
            ModuleMetricsSnapshot(
                module_id=node.module_id,
                profile_id=workload_profile_id,
                calls=calls,
                total_time_ms=0.0,
                total_flops_tflops=_safe_float(total_flops),
                total_io_tb=0.0,
                memory_weights_gb=0.0,
                memory_activations_gb=0.0,
                memory_kvcache_gb=0.0,
                share_of_model_time=_safe_float(float(total_flops / root_flops)),
                operator_breakdown=op_breakdown,
            ),
        )

    return snapshots


@dataclass(frozen=True)
class Wan2_1AnalyzerConfig:
    """Inputs for a Wan2.1 static analysis run."""

    workload_profile_id: str
    run_id: str
    modelmeter_config_name: str = "wan2_1_t2v_14b"


class Wan2_1StaticAnalyzer:
    """Generate Wan2.1 analytic reports (filesystem artifacts under tmp/profile-output)."""

    def run(self, *, cfg: Wan2_1AnalyzerConfig, overrides: List[str] | None = None) -> Wan2_1AnalyticModelReport:
        """Generate a Wan2.1 report.json (and summary.md) for a workload profile.

        Parameters
        ----------
        cfg:
            Workload id and run id used for output paths under `tmp/profile-output/<run_id>/`.
        overrides:
            Optional Hydra override strings applied when composing the ModelMeter analytic config.

        Returns
        -------
        Wan2_1AnalyticModelReport
            The generated report object (also written to disk).
        """

        overrides = list(overrides or [])
        overrides.append(f"workload.profile_id={cfg.workload_profile_id}")
        hydra_cfg = _load_cfg(overrides, config_name=cfg.modelmeter_config_name)
        _apply_workload_profile(hydra_cfg)

        model: Wan2_1DiTModel | Wan2_1PipelineModel = instantiate(hydra_cfg.model)

        root_id, modules, layer_by_id = _build_module_tree(model)
        num_steps = int(hydra_cfg.workload.num_inference_steps)
        metrics = _build_metrics(
            workload_profile_id=str(hydra_cfg.workload.profile_id),
            root_id=root_id,
            modules=modules,
            layer_by_id=layer_by_id,
            num_inference_steps=num_steps,
        )

        cfg_path = _default_config_json_path()
        if not cfg_path.is_file():
            raise FileNotFoundError(
                "Wan2.1 model metadata not found. Run `bash models/wan2.1-t2v-14b/bootstrap.sh` "
                f"and ensure {cfg_path} exists.",
            )

        model_spec = Wan2_1ModelSpec(
            model_id="wan2.1-t2v-14b",
            model_variant="t2v-14b",
            config_path=str(cfg_path),
            hidden_size=int(hydra_cfg.hf.hidden_size),
            num_layers=int(hydra_cfg.hf.num_layers),
            num_attention_heads=int(hydra_cfg.hf.num_attention_heads),
            head_dim=int(hydra_cfg.hf.head_dim),
            mlp_intermediate_size=int(hydra_cfg.hf.mlp_intermediate_size),
            vae_downsample_factor=int(hydra_cfg.hf.vae_downsample_factor),
            patch_size=int(hydra_cfg.hf.patch_size),
            latent_channels=int(hydra_cfg.hf.latent_channels),
            notes="",
        )

        workload = Wan2_1WorkloadProfile(
            profile_id=str(hydra_cfg.workload.profile_id),
            description=str(getattr(hydra_cfg.workload, "description", "")),
            batch_size=int(hydra_cfg.workload.batch_size),
            num_frames=int(hydra_cfg.workload.num_frames),
            height=int(hydra_cfg.workload.height),
            width=int(hydra_cfg.workload.width),
            num_inference_steps=int(hydra_cfg.workload.num_inference_steps),
            text_len=int(hydra_cfg.workload.text_len),
        )

        report = Wan2_1AnalyticModelReport(
            report_id=str(cfg.run_id),
            model=model_spec,
            workload=workload,
            modules=modules,
            operator_categories=_operator_categories(),
            module_metrics=metrics,
            profile_run_id=None,
            predicted_total_time_ms=0.0,
            notes=(
                "share_of_model_time is reported as share of total FLOPs (no time model in v1). "
                f"modelmeter_config_name={cfg.modelmeter_config_name}"
            ),
            layer_docs_dir=None,
        )

        out_dir = Path(wan2_1_analytic_dir(cfg.run_id))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(wan2_1_report_path(cfg.run_id))
        out_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        Path(wan2_1_summary_path(cfg.run_id)).write_text(render_wan2_1_summary_md(report) + "\n", encoding="utf-8")
        return report


def main(argv: List[str] | None = None) -> int:
    """Hydra entrypoint: generate report artifacts for the configured workload."""

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Wan2.1 static analysis (analytic FLOP breakdown).", add_help=True)
    parser.add_argument("--pipeline", action="store_true", help="Use full-pipeline config (wan2_1_t2v_14b_pipeline).")
    parser.add_argument(
        "--config-name",
        type=str,
        default="wan2_1_t2v_14b",
        help="ModelMeter config name (default: wan2_1_t2v_14b).",
    )
    args, overrides = parser.parse_known_args(argv)
    config_name = "wan2_1_t2v_14b_pipeline" if bool(args.pipeline) else str(args.config_name)

    cfg = _load_cfg(overrides, config_name=config_name)
    _apply_workload_profile(cfg)
    if "run_id" in cfg:
        run_id = str(cfg.run_id)
    else:
        hydra_dir = None
        try:
            hydra_dir = str(cfg.hydra.run.dir)
        except Exception:
            hydra_dir = None
        run_id = Path(hydra_dir).name if hydra_dir else datetime.now().strftime("%Y%m%d-%H%M%S")

    analyzer = Wan2_1StaticAnalyzer()
    report = analyzer.run(
        cfg=Wan2_1AnalyzerConfig(
            workload_profile_id=str(cfg.workload.profile_id),
            run_id=run_id,
            modelmeter_config_name=config_name,
        ),
        overrides=overrides,
    )
    out_dir = wan2_1_analytic_dir(report.report_id)
    print(f"Wan2.1 static analysis report written: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
