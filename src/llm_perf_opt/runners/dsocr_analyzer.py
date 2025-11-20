"""Static computational analysis for DeepSeek-OCR models.

This module provides comprehensive static analysis using fvcore to compute
parameters, FLOPs, and activations at both model-level and per-stage granularity.

The analyzer follows a composition pattern, taking an initialized DeepSeekOCRSession
as input and producing detailed reports without modifying the session or model.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from attrs import asdict
from ruamel.yaml import YAML  # type: ignore[import-untyped]

from modelmeter.models.deepseek_ocr.layers.core.deepseek_ocr_model import (
    DeepseekOCRModel,
    _CompositeLayer,
)
from modelmeter.models.deepseek_ocr.layers.decoder.deepseek_v2_decoder_layer import (
    DeepseekV2DecoderLayer,
)
from modelmeter.models.deepseek_ocr.layers.vision.clip_vision_embeddings import (
    CLIPVisionEmbeddings,
)
from modelmeter.models.deepseek_ocr.layers.vision.image_encoder_vit import (
    ImageEncoderViT,
)
from modelmeter.models.deepseek_ocr.layers.vision.mlp_projector import MlpProjector
from modelmeter.models.deepseek_ocr.layers.vision.notp_attention import (
    NoTPAttention,
)
from modelmeter.models.deepseek_ocr.layers.vision.notp_feedforward import (
    NoTPFeedForward,
)
from modelmeter.models.deepseek_ocr.layers.vision.notp_transformer import (
    NoTPTransformer,
)
from modelmeter.models.deepseek_ocr.layers.vision.notp_transformer_block import NoTPTransformerBlock
from modelmeter.models.deepseek_ocr.layers.vision.vit_model import VitModel
from llm_perf_opt.data.deepseek_ocr_analytic import (
    AnalyticModelReport,
    AnalyticModuleNode,
    DeepSeekOCRModelSpec,
    ModuleMetricsSnapshot,
    OCRWorkloadProfile,
    OperatorCategory,
    OperatorMetrics,
    build_operator_categories_from_target_list,
    build_operator_category_index,
    categorize_operator_class_name,
)
from llm_perf_opt.profiling.hw import get_device_name, get_peak_tflops
from llm_perf_opt.utils.dsocr_callgraph_parse import load_target_operator_list
from llm_perf_opt.utils.paths import analytic_layer_docs_dir, analytic_model_dir, workspace_root
from llm_perf_opt.visualize.analytic_layers import write_analytic_layer_docs


@dataclass
class AnalysisConfig:
    """Configuration for static analysis.

    Attributes
    ----------
    image_h : int
        Representative image height for vision analysis
    image_w : int
        Representative image width for vision analysis
    base_size : int
        Global view padding size (e.g., 1024)
    image_size : int
        Crop size for local views (e.g., 640)
    seq_len : int
        Representative sequence length for LLM analysis
    crop_mode : bool
        Whether to include local crops (affects vision FLOPs)
    patch_size : int
        Patch size for vision encoder (default 16)
    downsample_ratio : int
        Downsample ratio for image tokens (default 4)
    use_analytic_fallback : bool
        Use analytical formulas where fvcore fails
    use_synthetic_inputs : bool
        Use synthetic tensors (True) or load real sample (False)
    sample_image_path : Optional[str]
        Path to real image sample if use_synthetic_inputs=False
    """

    image_h: int = 1024
    image_w: int = 1024
    base_size: int = 1024
    image_size: int = 640
    seq_len: int = 1024
    crop_mode: bool = True
    patch_size: int = 16
    downsample_ratio: int = 4
    use_analytic_fallback: bool = True
    use_synthetic_inputs: bool = True
    sample_image_path: Optional[str] = None


@dataclass
class StageAnalysisResult:
    """Results for a single stage analysis.

    Attributes
    ----------
    stage_name : str
        Stage identifier (sam, clip, projector, prefill, decode)
    params : int
        Total parameters in this stage
    flops : float
        Total FLOPs for this stage
    activations : float
        Total activation elements for this stage
    operators : dict[str, float]
        Breakdown by operator type (conv, matmul, etc.)
    flops_isolated : Optional[float]
        FLOPs from isolated stage analysis (if available)
    flops_analytic : Optional[float]
        FLOPs from analytical formula (if available)
    notes : list[str]
        Analysis notes, warnings, or caveats
    """

    stage_name: str
    params: int = 0
    flops: float = 0.0
    activations: float = 0.0
    operators: dict[str, float] = field(default_factory=dict)
    flops_isolated: Optional[float] = None
    flops_analytic: Optional[float] = None
    notes: list[str] = field(default_factory=list)


class DeepseekOCRStaticAnalyzer:
    """Static computational analysis for DeepSeek-OCR using fvcore.

    This class provides comprehensive static analysis of a DeepSeek-OCR model,
    computing parameters, FLOPs, and activations at both full-model and per-stage
    granularity. It uses fvcore for accurate FLOP counting and falls back to
    analytical formulas where needed.

    Usage
    -----
    >>> from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession
    >>> session = DeepSeekOCRSession.from_local("/path/to/model")
    >>> analyzer = DeepseekOCRStaticAnalyzer(session)
    >>> config = AnalysisConfig(base_size=1024, image_size=640, seq_len=512)
    >>> report = analyzer.generate_report(config)
    >>> print(report["total"]["flops"])

    Attributes
    ----------
    m_session : DeepSeekOCRSession
        Initialized session with loaded model and tokenizer
    m_logger : logging.Logger
        Logger instance for this analyzer
    m_stage_module_map : dict[str, list[str]]
        Mapping from stage names to module name prefixes
    """

    def __init__(self, session: Any):
        """Initialize analyzer with a loaded session.

        Parameters
        ----------
        session : DeepSeekOCRSession
            Initialized session with model and tokenizer loaded

        Raises
        ------
        ValueError
            If session is not properly initialized
        """
        if session.m_model is None or session.m_tokenizer is None:
            raise ValueError("Session must be initialized with model and tokenizer")

        self.m_session = session
        self.m_logger = logging.getLogger(__name__)

        # Stage-to-module mapping for extraction from full model analysis
        # Note: Modules are nested under model.* in DeepseekOCRForCausalLM
        self.m_stage_module_map = {
            "sam": ["model.sam_model"],
            "clip": ["model.vision_model"],
            "projector": ["model.projector"],
            "prefill": ["model.embed_tokens", "model.layers", "model.norm"],
            "decode": ["lm_head"],
        }

    def prepare_inputs(
        self, config: AnalysisConfig
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Prepare representative inputs for static analysis.

        Builds input tensors matching the preprocessing used in inference,
        either from a real sample image or synthetic data.

        Parameters
        ----------
        config : AnalysisConfig
            Configuration specifying input shapes and modes

        Returns
        -------
        input_ids : torch.Tensor
            Tokenized prompt with image token spans [1, seq_len]
        model_kwargs : dict
            Dictionary with keys: images, images_seq_mask, images_spatial_crop
        """
        self.m_logger.info(
            "Preparing inputs | synthetic=%s base=%d img=%d crop=%s",
            config.use_synthetic_inputs,
            config.base_size,
            config.image_size,
            config.crop_mode,
        )

        # 1. Create or load image
        from PIL import Image, ImageOps
        from torchvision import transforms

        if config.use_synthetic_inputs:
            img = Image.new("RGB", (config.image_w, config.image_h), color=(128, 128, 128))
        else:
            if config.sample_image_path is None:
                raise ValueError("sample_image_path required when use_synthetic_inputs=False")
            img = Image.open(config.sample_image_path).convert("RGB")
            img = ImageOps.exif_transpose(img)

        # 2. Build global view (pad to base_size)
        mean = (0.5, 0.5, 0.5)
        global_view = ImageOps.pad(
            img, (config.base_size, config.base_size), color=tuple(int(x * 255) for x in mean)
        )

        # 3. Apply transforms
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=mean),
        ])
        images_ori = tfm(global_view).to(torch.bfloat16)
        images_ori = images_ori.unsqueeze(0).to(self.m_session.m_device)  # [1,3,H,W]

        # 4. Generate crops if needed
        w_crop, h_crop = 1, 1
        if config.crop_mode:
            # Replicate dynamic_preprocess logic
            orig_w, orig_h = img.size
            aspect_ratio = orig_w / float(orig_h)

            # Candidate ratios
            target_ratios = sorted({
                (i, j)
                for n in range(2, 10)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if (i * j) <= 9 and (i * j) >= 2
            }, key=lambda x: x[0] * x[1])

            # Find closest ratio
            best = (1, 1)
            best_diff = float("inf")
            area = orig_w * orig_h
            for (w_r, h_r) in target_ratios:
                target_ar = w_r / float(h_r)
                diff = abs(aspect_ratio - target_ar)
                if diff < best_diff:
                    best = (w_r, h_r)
                    best_diff = diff
                elif diff == best_diff:
                    if area > 0.5 * config.image_size * config.image_size * w_r * h_r:
                        best = (w_r, h_r)

            w_crop, h_crop = best
            target_w = config.image_size * w_crop
            target_h = config.image_size * h_crop
            resized = img.resize((target_w, target_h))

            crops = []
            for i in range(w_crop * h_crop):
                c = i % w_crop
                r = i // w_crop
                box = (
                    c * config.image_size,
                    r * config.image_size,
                    (c + 1) * config.image_size,
                    (r + 1) * config.image_size,
                )
                crops.append(resized.crop(box))

            crop_tensors = [tfm(cr).to(torch.bfloat16) for cr in crops]
            images_crop = torch.stack(crop_tensors, dim=0).to(self.m_session.m_device)
        else:
            images_crop = torch.zeros(
                (1, 3, config.base_size, config.base_size),
                dtype=torch.bfloat16,
                device=self.m_session.m_device,
            )

        images_spatial_crop = torch.tensor(
            [[w_crop, h_crop]], dtype=torch.long, device=self.m_session.m_device
        )
        images = [(images_crop, images_ori)]

        # 5. Build input_ids and images_seq_mask
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
        IMAGE_TOKEN_ID = 128815

        text_splits = prompt.split("<image>")
        tokenized_str: list[int] = []
        mask_list: list[bool] = []

        for text_piece in text_splits[:-1]:
            toks = self.m_session.m_tokenizer.encode(text_piece, add_special_tokens=False)
            tokenized_str += toks
            mask_list += [False] * len(toks)

            # Mirror vendor token span sizing
            num_queries = math.ceil((config.image_size // config.patch_size) / config.downsample_ratio)
            num_queries_base = math.ceil((config.base_size // config.patch_size) / config.downsample_ratio)

            # Base tokens (global view)
            tokenized_image = ([IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]) * num_queries_base
            tokenized_image += [IMAGE_TOKEN_ID]

            # Local crops (if any)
            if config.crop_mode and (w_crop > 1 or h_crop > 1):
                tokenized_image += (
                    ([IMAGE_TOKEN_ID] * (num_queries * w_crop) + [IMAGE_TOKEN_ID])
                    * (num_queries * h_crop)
                )

            tokenized_str += tokenized_image
            mask_list += [True] * len(tokenized_image)

        toks_tail = self.m_session.m_tokenizer.encode(text_splits[-1], add_special_tokens=False)
        tokenized_str = [0] + tokenized_str + toks_tail  # BOS=0
        mask_list = [False] + mask_list + [False] * len(toks_tail)

        input_ids = torch.tensor(
            tokenized_str, dtype=torch.long, device=self.m_session.m_device
        ).unsqueeze(0)
        images_seq_mask = torch.tensor(
            mask_list, dtype=torch.bool, device=self.m_session.m_device
        ).unsqueeze(0)

        self.m_logger.info(
            "Inputs prepared | input_len=%d grid=%dx%d",
            input_ids.shape[-1],
            w_crop,
            h_crop,
        )

        model_kwargs = {
            "images": images,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

        return input_ids, model_kwargs

    def analyze_full_model(self, config: AnalysisConfig) -> dict[str, Any]:
        """Run full model static analysis using fvcore.

        Traces the entire model with representative inputs and computes
        total parameters, FLOPs, and activations. Extracts per-stage
        statistics from the full model analysis using module name mapping.

        Parameters
        ----------
        config : AnalysisConfig
            Configuration for input preparation and analysis

        Returns
        -------
        dict
            Full model analysis with keys:
            - total_params : int
            - total_flops : float
            - total_activations : float
            - by_module : dict[str, float] (module name -> FLOPs)
            - by_module_and_operator : dict[str, dict[str, float]]
            - by_operator : dict[str, float]
            - acts_by_module : dict[str, float]

        Raises
        ------
        RuntimeError
            If fvcore analysis fails and no fallback is available
        """
        self.m_logger.info("Running full model analysis with fvcore")

        try:
            from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis, parameter_count

            input_ids, model_kwargs = self.prepare_inputs(config)

            # Create a wrapper to convert kwargs to positional args for fvcore
            import torch.nn as nn

            class ModelWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, input_ids, images, images_seq_mask, images_spatial_crop):
                    return self.model(
                        input_ids=input_ids,
                        images=images,
                        images_seq_mask=images_seq_mask,
                        images_spatial_crop=images_spatial_crop,
                    )

            wrapper = ModelWrapper(self.m_session.m_model)
            wrapper_inputs = (
                input_ids,
                model_kwargs["images"],
                model_kwargs["images_seq_mask"],
                model_kwargs["images_spatial_crop"],
            )

            # Full model FLOPs
            self.m_logger.info("FlopCountAnalysis starting...")
            flops_full = FlopCountAnalysis(wrapper, wrapper_inputs)

            total_flops = float(flops_full.total())
            by_module = dict(flops_full.by_module())
            by_module_and_op = {k: dict(v) for k, v in flops_full.by_module_and_operator().items()}
            by_operator = dict(flops_full.by_operator())

            self.m_logger.info("FlopCountAnalysis complete | total_flops=%.2fG", total_flops / 1e9)

            # Activations
            self.m_logger.info("ActivationCountAnalysis starting...")
            acts_full = ActivationCountAnalysis(wrapper, wrapper_inputs)
            total_acts = float(acts_full.total())
            acts_by_module = dict(acts_full.by_module())

            self.m_logger.info("ActivationCountAnalysis complete | total_acts=%.2fM", total_acts / 1e6)

            # Parameters
            self.m_logger.info("parameter_count starting...")
            params = parameter_count(self.m_session.m_model)
            total_params = int(params.get("", 0))

            self.m_logger.info("parameter_count complete | total_params=%.2fM", total_params / 1e6)

            # Fix module names: fvcore adds "model." prefix to wrapper, we need to strip it
            by_module_fixed = {}
            for mod_name, flops in by_module.items():
                # Strip "model." prefix from wrapper
                fixed_name = mod_name.replace("model.model.", "model.")
                by_module_fixed[fixed_name] = flops

            by_module_and_op_fixed = {}
            for mod_name, ops in by_module_and_op.items():
                fixed_name = mod_name.replace("model.model.", "model.")
                by_module_and_op_fixed[fixed_name] = ops

            acts_by_module_fixed = {}
            for mod_name, acts in acts_by_module.items():
                fixed_name = mod_name.replace("model.model.", "model.")
                acts_by_module_fixed[fixed_name] = acts

            return {
                "total_params": total_params,
                "total_flops": total_flops,
                "total_activations": total_acts,
                "by_module": by_module_fixed,
                "by_module_and_operator": by_module_and_op_fixed,
                "by_operator": by_operator,
                "acts_by_module": acts_by_module_fixed,
                "params": params,
            }

        except Exception as e:
            self.m_logger.error(f"Full model fvcore analysis failed: {e}", exc_info=True)
            if not config.use_analytic_fallback:
                raise RuntimeError(f"fvcore analysis failed: {e}") from e

            # Return empty results
            return {
                "total_params": 0,
                "total_flops": 0.0,
                "total_activations": 0.0,
                "by_module": {},
                "by_module_and_operator": {},
                "by_operator": {},
                "acts_by_module": {},
                "params": {},
            }

    def _extract_stage_from_full_analysis(
        self,
        stage_name: str,
        by_module: dict[str, float],
        by_module_and_operator: dict[str, dict[str, float]],
        params: dict[str, int],
        acts_by_module: dict[str, float],
    ) -> StageAnalysisResult:
        """Extract per-stage statistics from full model analysis results.

        Uses m_stage_module_map to identify which modules belong to each stage
        and aggregates their FLOPs, parameters, activations, and operators.

        Parameters
        ----------
        stage_name : str
            Stage identifier
        by_module : dict[str, float]
            Full model by_module() results
        by_module_and_operator : dict[str, dict[str, float]]
            Full model by_module_and_operator() results
        params : dict[str, int]
            Full model parameter_count() results
        acts_by_module : dict[str, float]
            Full model activation by_module() results

        Returns
        -------
        StageAnalysisResult
            Aggregated stage results
        """
        module_prefixes = self.m_stage_module_map.get(stage_name, [])

        stage_flops = 0.0
        stage_acts = 0.0
        stage_params = 0
        stage_ops: dict[str, float] = {}

        for mod_name, flops in by_module.items():
            # Check if module name matches any of our stage prefixes
            if any(mod_name.startswith(prefix) or mod_name == prefix for prefix in module_prefixes):
                stage_flops += float(flops)
                stage_acts += float(acts_by_module.get(mod_name, 0))
                stage_params += int(params.get(mod_name, 0))
                # Operator breakdown for this module
                if mod_name in by_module_and_operator:
                    for op, op_flops in by_module_and_operator[mod_name].items():
                        prev = float(stage_ops.get(op, 0.0))
                        stage_ops[op] = prev + float(op_flops)

        return StageAnalysisResult(
            stage_name=stage_name,
            params=stage_params,
            flops=stage_flops,
            activations=stage_acts,
            operators=dict(stage_ops),
            notes=["Extracted from full model analysis"],
        )

    def _analyze_sam_isolated(self, config: AnalysisConfig) -> StageAnalysisResult:
        """Isolated analysis of SAM encoder stage."""
        self.m_logger.info("Analyzing SAM stage (isolated)")

        try:
            from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis, parameter_count

            # Access nested model structure: model.model.sam_model
            base_model = getattr(self.m_session.m_model, "model", self.m_session.m_model)
            sam_model = getattr(base_model, "sam_model", None)
            if sam_model is None:
                return StageAnalysisResult(
                    stage_name="sam",
                    notes=["SAM model not found in session"],
                )

            # Prepare inputs: use crops or global view
            _, model_kwargs = self.prepare_inputs(config)
            images_crop = model_kwargs["images"][0][0]  # [N, 3, H, W]

            flops_sam = FlopCountAnalysis(sam_model, (images_crop,))
            total_flops = float(flops_sam.total())
            operators = dict(flops_sam.by_operator())

            acts_sam = ActivationCountAnalysis(sam_model, (images_crop,))
            total_acts = float(acts_sam.total())

            params = parameter_count(sam_model)
            total_params = int(params.get("", 0))

            self.m_logger.info("SAM isolated | flops=%.2fG params=%.2fM", total_flops / 1e9, total_params / 1e6)

            return StageAnalysisResult(
                stage_name="sam",
                params=total_params,
                flops_isolated=total_flops,
                activations=total_acts,
                operators=operators,
                notes=["Isolated fvcore analysis on SAM module"],
            )

        except Exception as e:
            self.m_logger.warning(f"SAM isolated analysis failed: {e}")
            return StageAnalysisResult(
                stage_name="sam",
                notes=[f"Isolated analysis failed: {e}"],
            )

    def _analyze_clip_isolated(self, config: AnalysisConfig) -> StageAnalysisResult:
        """Isolated analysis of CLIP encoder stage."""
        self.m_logger.info("Analyzing CLIP stage (isolated)")

        try:
            from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis, parameter_count

            # Access nested model structure: model.model.vision_model
            base_model = getattr(self.m_session.m_model, "model", self.m_session.m_model)
            vision_model = getattr(base_model, "vision_model", None)
            if vision_model is None:
                return StageAnalysisResult(
                    stage_name="clip",
                    notes=["CLIP vision model not found in session"],
                )

            # Get real SAM features by running SAM first
            sam_model = getattr(base_model, "sam_model", None)
            _, model_kwargs = self.prepare_inputs(config)
            images_ori = model_kwargs["images"][0][1]  # Global view: [1, 3, H, W]

            if sam_model is None:
                # Fall back to mock features with corrected dimensions
                self.m_logger.warning("SAM not available, using mock features for CLIP")
                batch_size = images_ori.shape[0]
                # SAM outputs downsampled features: base_size // 16 = 64
                H_sam = config.base_size // 16  # e.g., 1024 // 16 = 64
                W_sam = config.base_size // 16
                C_sam = 1024  # SAM feature dim
                sam_features = torch.randn(
                    batch_size, H_sam, W_sam, C_sam,
                    dtype=torch.bfloat16,
                    device=self.m_session.m_device,
                )
            else:
                # Use real SAM features from GLOBAL VIEW (not crops)
                self.m_logger.info("Running SAM on global view to get features for CLIP analysis")
                with torch.no_grad():
                    sam_features = sam_model(images_ori)  # Use global view [1, 3, H, W]

            # Note: This may fail if CLIP forward signature doesn't match
            # We'll catch and report
            try:
                flops_clip = FlopCountAnalysis(vision_model, (images_ori, sam_features))
                total_flops = float(flops_clip.total())
                operators = dict(flops_clip.by_operator())

                acts_clip = ActivationCountAnalysis(vision_model, (images_ori, sam_features))
                total_acts = float(acts_clip.total())

                params = parameter_count(vision_model)
                total_params = int(params.get("", 0))

                self.m_logger.info("CLIP isolated | flops=%.2fG params=%.2fM", total_flops / 1e9, total_params / 1e6)

                return StageAnalysisResult(
                    stage_name="clip",
                    params=total_params,
                    flops_isolated=total_flops,
                    activations=total_acts,
                    operators=operators,
                    notes=["Isolated fvcore analysis on CLIP module (with real SAM features)"],
                )
            except Exception as inner_e:
                self.m_logger.warning(f"CLIP forward tracing failed: {inner_e}")
                # Fallback: just count params
                params = parameter_count(vision_model)
                total_params = int(params.get("", 0))
                return StageAnalysisResult(
                    stage_name="clip",
                    params=total_params,
                    notes=[f"fvcore tracing failed, params only: {inner_e}"],
                )

        except Exception as e:
            self.m_logger.warning(f"CLIP isolated analysis failed: {e}")
            return StageAnalysisResult(
                stage_name="clip",
                notes=[f"Isolated analysis failed: {e}"],
            )

    def _analyze_projector_isolated(self, config: AnalysisConfig) -> StageAnalysisResult:
        """Isolated analysis of projector stage."""
        self.m_logger.info("Analyzing projector stage (isolated)")

        try:
            from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis, parameter_count

            # Access nested model structure: model.model.projector
            base_model = getattr(self.m_session.m_model, "model", self.m_session.m_model)
            projector = getattr(base_model, "projector", None)
            if projector is None:
                return StageAnalysisResult(
                    stage_name="projector",
                    notes=["Projector not found in session"],
                )

            # Mock vision features: [B, T_vision, D_vision]
            # Vision features are concatenated SAM + CLIP features
            batch_size = 1
            num_vision_tokens = (config.base_size // config.patch_size) ** 2  # e.g., 64*64 = 4096
            # SAM 1024 + CLIP 1024 = 2048 (typical)
            D_vision = 2048
            mock_vision_feats = torch.randn(
                batch_size, num_vision_tokens, D_vision,
                dtype=torch.bfloat16,
                device=self.m_session.m_device,
            )

            flops_proj = FlopCountAnalysis(projector, (mock_vision_feats,))
            total_flops = float(flops_proj.total())
            operators = dict(flops_proj.by_operator())

            acts_proj = ActivationCountAnalysis(projector, (mock_vision_feats,))
            total_acts = float(acts_proj.total())

            params = parameter_count(projector)
            total_params = int(params.get("", 0))

            self.m_logger.info("Projector isolated | flops=%.2fG params=%.2fM", total_flops / 1e9, total_params / 1e6)

            return StageAnalysisResult(
                stage_name="projector",
                params=total_params,
                flops_isolated=total_flops,
                activations=total_acts,
                operators=operators,
                notes=["Isolated fvcore analysis on projector module"],
            )

        except Exception as e:
            self.m_logger.warning(f"Projector isolated analysis failed: {e}")
            return StageAnalysisResult(
                stage_name="projector",
                notes=[f"Isolated analysis failed: {e}"],
            )

    def _analyze_prefill_analytic(self, config: AnalysisConfig) -> StageAnalysisResult:
        """Analytical FLOP estimation for prefill stage."""
        self.m_logger.info("Analyzing prefill stage (analytical)")

        try:
            from llm_perf_opt.profiling.mfu import estimate_prefill_flops_total

            model_config = self.m_session.m_model.config
            d_model = model_config.hidden_size
            d_ff = model_config.intermediate_size
            n_layers = model_config.num_hidden_layers

            prefill_flops = estimate_prefill_flops_total(
                d_model=d_model,
                d_ff=d_ff,
                n_layers=n_layers,
                seq_len=config.seq_len,
            )

            # Estimate params for LLM layers (embed_tokens + layers + norm)
            # This is approximate; we can refine if needed
            embed_params = d_model * getattr(model_config, "vocab_size", 50000)
            layer_params = n_layers * (
                4 * d_model * d_model +  # QKV + output projection
                2 * d_model * d_ff  # MLP
            )
            norm_params = d_model

            total_params = embed_params + layer_params + norm_params

            self.m_logger.info("Prefill analytic | flops=%.2fG params=%.2fM", prefill_flops / 1e9, total_params / 1e6)

            return StageAnalysisResult(
                stage_name="prefill",
                params=int(total_params),
                flops_analytic=prefill_flops,
                operators={"analytical": prefill_flops},
                notes=[f"Analytical estimate for seq_len={config.seq_len}"],
            )

        except Exception as e:
            self.m_logger.warning(f"Prefill analytical analysis failed: {e}")
            return StageAnalysisResult(
                stage_name="prefill",
                notes=[f"Analytical analysis failed: {e}"],
            )

    def _analyze_decode_analytic(self, config: AnalysisConfig) -> StageAnalysisResult:
        """Analytical FLOP estimation for decode stage (per-token)."""
        self.m_logger.info("Analyzing decode stage (analytical per-token)")

        try:
            from llm_perf_opt.profiling.mfu import estimate_decode_flops_per_token

            model_config = self.m_session.m_model.config
            d_model = model_config.hidden_size
            d_ff = model_config.intermediate_size
            n_layers = model_config.num_hidden_layers

            # Representative context length
            ctx_len = config.seq_len

            decode_flops_per_token = estimate_decode_flops_per_token(
                d_model=d_model,
                d_ff=d_ff,
                n_layers=n_layers,
                ctx_len=ctx_len,
            )

            # Estimate params for lm_head
            lm_head_params = d_model * getattr(model_config, "vocab_size", 50000)

            self.m_logger.info(
                "Decode analytic | flops_per_token=%.2fM params=%.2fM",
                decode_flops_per_token / 1e6,
                lm_head_params / 1e6,
            )

            return StageAnalysisResult(
                stage_name="decode",
                params=int(lm_head_params),
                flops_analytic=decode_flops_per_token,
                operators={"analytical_per_token": decode_flops_per_token},
                notes=[f"Analytical per-token estimate at ctx_len={ctx_len}"],
            )

        except Exception as e:
            self.m_logger.warning(f"Decode analytical analysis failed: {e}")
            return StageAnalysisResult(
                stage_name="decode",
                notes=[f"Analytical analysis failed: {e}"],
            )

    def analyze_stage_isolated(
        self, stage_name: str, config: AnalysisConfig
    ) -> StageAnalysisResult:
        """Analyze a specific stage in isolation.

        Runs fvcore analysis on a single stage (module) with appropriate
        inputs. This provides more accurate per-stage FLOPs compared to
        extracting from full model analysis, especially for stages with
        control flow or dynamic behavior.

        Parameters
        ----------
        stage_name : str
            Stage identifier (sam, clip, projector, prefill, decode)
        config : AnalysisConfig
            Configuration for input preparation

        Returns
        -------
        StageAnalysisResult
            Isolated analysis results for this stage

        Raises
        ------
        ValueError
            If stage_name is not recognized
        """
        stage_lower = stage_name.lower()

        if stage_lower == "sam":
            return self._analyze_sam_isolated(config)
        elif stage_lower == "clip":
            return self._analyze_clip_isolated(config)
        elif stage_lower == "projector":
            return self._analyze_projector_isolated(config)
        elif stage_lower == "prefill":
            if config.use_analytic_fallback:
                return self._analyze_prefill_analytic(config)
            else:
                return StageAnalysisResult(
                    stage_name="prefill",
                    notes=["Prefill requires analytical fallback; enable use_analytic_fallback"],
                )
        elif stage_lower == "decode":
            if config.use_analytic_fallback:
                return self._analyze_decode_analytic(config)
            else:
                return StageAnalysisResult(
                    stage_name="decode",
                    notes=["Decode requires analytical fallback; enable use_analytic_fallback"],
                )
        else:
            raise ValueError(f"Unknown stage: {stage_name}")

    def analyze_all_stages(self, config: AnalysisConfig) -> dict[str, StageAnalysisResult]:
        """Analyze all stages (sam, clip, projector, prefill, decode).

        Runs both full model analysis (for extraction) and isolated stage
        analyses. Combines results and uses analytical fallbacks where needed.

        Parameters
        ----------
        config : AnalysisConfig
            Configuration for analysis

        Returns
        -------
        dict[str, StageAnalysisResult]
            Mapping from stage name to analysis results
        """
        self.m_logger.info("Analyzing all stages")

        # Full model analysis first
        full_results = self.analyze_full_model(config)

        # Extract stages from full model
        extracted_stages: dict[str, StageAnalysisResult] = {}
        for stage_name in self.m_stage_module_map.keys():
            extracted = self._extract_stage_from_full_analysis(
                stage_name,
                full_results["by_module"],
                full_results["by_module_and_operator"],
                full_results["params"],
                full_results["acts_by_module"],
            )
            extracted_stages[stage_name] = extracted

        # Run isolated analyses for each stage
        isolated_stages: dict[str, StageAnalysisResult] = {}
        for stage_name in ["sam", "clip", "projector", "prefill", "decode"]:
            isolated = self.analyze_stage_isolated(stage_name, config)
            isolated_stages[stage_name] = isolated

        # Merge results: prefer isolated > extracted > analytical
        merged_stages: dict[str, StageAnalysisResult] = {}
        for stage_name in ["sam", "clip", "projector", "prefill", "decode"]:
            extracted = extracted_stages.get(stage_name, StageAnalysisResult(stage_name=stage_name))
            isolated = isolated_stages.get(stage_name, StageAnalysisResult(stage_name=stage_name))

            # Merge: take isolated flops if available, otherwise extracted
            merged = StageAnalysisResult(
                stage_name=stage_name,
                params=isolated.params if isolated.params > 0 else extracted.params,
                flops=isolated.flops_isolated or extracted.flops or isolated.flops_analytic or 0.0,
                activations=isolated.activations if isolated.activations > 0 else extracted.activations,
                operators=isolated.operators if isolated.operators else extracted.operators,
                flops_isolated=isolated.flops_isolated,
                flops_analytic=isolated.flops_analytic,
                notes=extracted.notes + isolated.notes,
            )
            merged_stages[stage_name] = merged

        return merged_stages

    def generate_report(self, config: AnalysisConfig) -> dict[str, Any]:
        """Generate comprehensive static analysis report.

        This is the main entry point for static analysis. It orchestrates
        full model analysis, per-stage analyses, analytical fallbacks, and
        produces a structured report ready for serialization or formatting.

        Parameters
        ----------
        config : AnalysisConfig
            Configuration for analysis

        Returns
        -------
        dict
            Comprehensive report with structure:
            {
                "metadata": {
                    "image_h": int, "image_w": int, "base_size": int,
                    "image_size": int, "seq_len": int, "crop_mode": bool,
                    "fvcore_version": str, ...
                },
                "total": {
                    "params": int, "flops": float, "activations": float
                },
                "stages": {
                    "sam": {
                        "params": int, "flops": float, "activations": float,
                        "operators": {"conv": float, ...},
                        "flops_isolated": float, "flops_analytic": float,
                        "notes": [str, ...]
                    },
                    ... (clip, projector, prefill, decode)
                },
                "notes": [str, ...]
            }
        """
        self.m_logger.info("Generating comprehensive static analysis report")

        # Analyze all stages
        stages_results = self.analyze_all_stages(config)

        # Get full model totals
        full_results = self.analyze_full_model(config)

        # Build metadata
        try:
            import fvcore
            fvcore_version = getattr(fvcore, "__version__", "unknown")
        except Exception:
            fvcore_version = "unknown"

        metadata = {
            "image_h": config.image_h,
            "image_w": config.image_w,
            "base_size": config.base_size,
            "image_size": config.image_size,
            "seq_len": config.seq_len,
            "crop_mode": config.crop_mode,
            "patch_size": config.patch_size,
            "downsample_ratio": config.downsample_ratio,
            "use_synthetic_inputs": config.use_synthetic_inputs,
            "fvcore_version": fvcore_version,
        }

        # Build total (from full model or sum of stages)
        total_params = full_results["total_params"]
        total_flops = full_results["total_flops"]
        total_activations = full_results["total_activations"]

        # If full model totals are zero, sum from stages
        if total_params == 0:
            total_params = sum(s.params for s in stages_results.values())
        if total_flops == 0:
            total_flops = sum(s.flops for s in stages_results.values())
        if total_activations == 0:
            total_activations = sum(s.activations for s in stages_results.values())

        # Build stages dict
        stages_dict = {}
        for stage_name, result in stages_results.items():
            stages_dict[stage_name] = {
                "params": result.params,
                "flops": result.flops,
                "activations": result.activations,
                "operators": result.operators,
                "flops_isolated": result.flops_isolated,
                "flops_analytic": result.flops_analytic,
                "notes": result.notes,
            }

        # Build notes
        notes = [
            "FLOPs are shape-dependent; values based on representative inputs",
            "Prefill/decode use analytical estimates for accuracy",
            "Vision stages (SAM/CLIP/projector) use fvcore where feasible",
        ]

        if config.use_analytic_fallback:
            notes.append("Analytical fallback enabled for LLM stages")

        report = {
            "metadata": metadata,
            "total": {
                "params": int(total_params),
                "flops": float(total_flops),
                "activations": float(total_activations),
            },
            "stages": stages_dict,
            "notes": notes,
        }

        self.m_logger.info(
            "Report generated | total_params=%.2fM total_flops=%.2fG",
            total_params / 1e6,
            total_flops / 1e9,
        )

        return report

    # ------------------------------------------------------------------
    # Analytic modeling pipeline (ModelMeter-based)
    # ------------------------------------------------------------------

    def _infer_model_dims(self) -> tuple[int, int, int, int]:
        """Best-effort extraction of (hidden_size, intermediate_size, num_layers, num_heads)."""

        hidden_size = 1280
        intermediate_size = 4 * hidden_size
        num_layers = 12
        num_heads = 16

        try:
            cfg = getattr(self.m_session.m_model, "config", None)
            if cfg is not None:
                hidden_size = int(getattr(cfg, "hidden_size", hidden_size))
                intermediate_size = int(getattr(cfg, "intermediate_size", intermediate_size))
                num_layers = int(getattr(cfg, "num_hidden_layers", num_layers))
                num_heads = int(getattr(cfg, "num_attention_heads", num_heads))
        except Exception:
            # Fall back to defaults if anything goes wrong.
            pass

        return hidden_size, intermediate_size, num_layers, num_heads

    def _build_model_spec(self, config_path_fallback: Optional[str] = None) -> DeepSeekOCRModelSpec:
        """Construct DeepSeek-OCR model metadata for analytic reports."""

        hidden_size, intermediate_size, num_layers, num_heads = self._infer_model_dims()
        cfg = getattr(self.m_session.m_model, "config", None)

        model_id = "deepseek-ai/DeepSeek-OCR"
        model_variant = "deepseek-ocr-v1-base"

        model_path = getattr(self.m_session, "m_model_path", None)
        if isinstance(model_path, str):
            config_path = str(Path(model_path).resolve())
        elif config_path_fallback is not None:
            config_path = str(Path(config_path_fallback).resolve())
        else:
            config_path = str(Path.cwd().resolve())

        vision_backbone = "clip_vit_l"
        uses_moe = False
        if cfg is not None:
            try:
                routed = int(getattr(cfg, "n_routed_experts", 0))
                uses_moe = routed > 0
            except Exception:
                uses_moe = False

        notes = (
            "Analytic model for DeepSeek-OCR (vision SAM-B + CLIP-L, "
            "decoder DeepseekV2 stack)."
        )

        return DeepSeekOCRModelSpec(
            model_id=model_id,
            model_variant=model_variant,
            hf_revision=None,
            config_path=config_path,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            vision_backbone=vision_backbone,
            uses_moe=uses_moe,
            notes=notes,
        )

    def _build_workload_profile(
        self,
        config: AnalysisConfig,
        workload_profile_id: str,
    ) -> OCRWorkloadProfile:
        """Construct OCRWorkloadProfile for a given AnalysisConfig."""

        description = (
            "Standard DeepSeek-OCR synthetic workload "
            f"({workload_profile_id}, single-page mixed-layout document)."
        )

        # For this phase we treat max_new_tokens as a fixed calibration value.
        max_new_tokens = 512

        return OCRWorkloadProfile(
            profile_id=workload_profile_id,
            description=description,
            seq_len=config.seq_len,
            base_size=config.base_size,
            image_size=config.image_size,
            crop_mode=config.crop_mode,
            max_new_tokens=max_new_tokens,
            doc_kind="mixed_layout",
            num_pages=1,
        )

    def _build_decoder_layer(self, config: AnalysisConfig) -> tuple[DeepseekV2DecoderLayer, int]:
        """Instantiate analytic decoder layer and infer decoder depth."""

        hidden_size, intermediate_size, num_layers, num_heads = self._infer_model_dims()
        cfg = getattr(self.m_session.m_model, "config", None)

        num_experts: Optional[int] = None
        num_key_value_heads: Optional[int] = None
        k_active = 2
        num_shared_experts = 0

        if cfg is not None:
            try:
                routed = int(getattr(cfg, "n_routed_experts", 0))
                if routed > 0:
                    num_experts = routed
            except Exception:
                num_experts = None

            try:
                kv_heads = getattr(cfg, "num_key_value_heads", None)
                if kv_heads is not None:
                    num_key_value_heads = int(kv_heads)
            except Exception:
                num_key_value_heads = None

            try:
                k_val = getattr(cfg, "num_experts_per_tok", None)
                if k_val is not None:
                    k_active = max(int(k_val), 1)
            except Exception:
                k_active = 2

            try:
                shared_val = getattr(cfg, "n_shared_experts", 0)
                num_shared_experts = max(int(shared_val), 0)
            except Exception:
                num_shared_experts = 0

        decoder_layer = DeepseekV2DecoderLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            seq_len=config.seq_len,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            batch_size=1,
            num_key_value_heads=num_key_value_heads,
            k_active=k_active,
            num_shared_experts=num_shared_experts,
        )

        return decoder_layer, num_layers

    def _build_vision_layers(self, config: AnalysisConfig) -> dict[str, object]:
        """Instantiate analytic vision layers for DeepSeek-OCR."""

        # SAM-style encoder for high-resolution features.
        image_encoder = ImageEncoderViT(
            img_size=config.base_size,
            patch_size=config.patch_size,
            batch_size=1,
        )

        # CLIP-L vision tower over SAM features.
        clip_embeddings = CLIPVisionEmbeddings(
            hidden_size=1024,
            image_size=224,
            patch_size=14,
            num_channels=3,
            batch_size=1,
            use_precomputed_patch_embeds=True,
        )
        notp_attn = NoTPAttention(hidden_size=1024, num_heads=16, seq_len=257, batch_size=1)
        notp_ff = NoTPFeedForward(dim=1024, hidden_dim=4096, batch_size=1, seq_len=257)
        notp_block = NoTPTransformerBlock(attention=notp_attn, mlp=notp_ff)
        notp_stack = NoTPTransformer(blocks=[notp_block] * 24)
        vit_model = VitModel(
            embeddings=clip_embeddings,
            transformer=notp_stack,
            hidden_size=1024,
            seq_len=257,
            batch_size=1,
        )

        # Vision-to-decoder projector (SAM+CLIP → decoder embeddings).
        hidden_size, _, _, _ = self._infer_model_dims()
        projector = MlpProjector(
            input_dim=2048,
            output_dim=hidden_size,
            num_tokens=577,
            batch_size=1,
            depth=2,
            projector_type="mlp_gelu",
        )

        return {
            "vision/image_encoder_vit": image_encoder,
            "vision/vit_model": vit_model,
            "vision/mlp_projector": projector,
        }

    def _build_module_nodes_and_metrics(
        self,
        *,
        model_layer: DeepseekOCRModel,
        vision_layers: dict[str, object],
        decoder_layer: DeepseekV2DecoderLayer,
        num_decoder_layers: int,
        model_spec: DeepSeekOCRModelSpec,
        workload: OCRWorkloadProfile,
    ) -> tuple[list[AnalyticModuleNode], list[ModuleMetricsSnapshot], float]:
        """Construct AnalyticModuleNode and ModuleMetricsSnapshot lists."""

        modules: list[AnalyticModuleNode] = []
        metrics: list[ModuleMetricsSnapshot] = []

        profile_id = workload.profile_id

        # Root module.
        root_id = "model/deepseek_ocr_model"
        modules.append(
            AnalyticModuleNode(
                module_id=root_id,
                name="DeepseekOCRModel",
                qualified_class_name=(
                    "extern.modelmeter.models.deepseek_ocr.layers.core."
                    "deepseek_ocr_model.DeepseekOCRModel"
                ),
                stage="other",
                parent_id=None,
                children=[
                    "vision/image_encoder_vit",
                    "vision/vit_model",
                    "vision/mlp_projector",
                    "decoder/deepseek_v2_decoder_layer",
                ],
                repetition="none",
                repetition_count=None,
                constructor_params={"num_layers": model_spec.num_layers},
            ),
        )

        # Vision modules.
        img_enc: ImageEncoderViT = vision_layers["vision/image_encoder_vit"]  # type: ignore[assignment]
        vit_model: VitModel = vision_layers["vision/vit_model"]  # type: ignore[assignment]
        projector: MlpProjector = vision_layers["vision/mlp_projector"]  # type: ignore[assignment]

        # Best-effort extraction of vision encoder head count; fall back to an internal
        # attribute if the public property is unavailable in a given ModelMeter rev.
        img_num_heads = getattr(img_enc, "num_heads", None)
        if img_num_heads is None:
            img_num_heads = getattr(img_enc, "m_num_heads", 0)

        modules.append(
            AnalyticModuleNode(
                module_id="vision/image_encoder_vit",
                name="ImageEncoderViT",
                qualified_class_name=(
                    "extern.modelmeter.models.deepseek_ocr.layers.vision."
                    "image_encoder_vit.ImageEncoderViT"
                ),
                stage="vision",
                parent_id=root_id,
                children=[],
                repetition="none",
                repetition_count=None,
                constructor_params={
                    "img_size": img_enc.img_size,
                    "patch_size": img_enc.patch_size,
                    "embed_dim": img_enc.embed_dim,
                    "depth": img_enc.depth,
                    "num_heads": img_num_heads,
                },
            ),
        )
        modules.append(
            AnalyticModuleNode(
                module_id="vision/vit_model",
                name="VitModel",
                qualified_class_name=(
                    "extern.modelmeter.models.deepseek_ocr.layers.vision."
                    "vit_model.VitModel"
                ),
                stage="vision",
                parent_id=root_id,
                children=[],
                repetition="none",
                repetition_count=None,
                constructor_params={
                    "hidden_size": vit_model.hidden_size,
                    "seq_len": vit_model.seq_len,
                    "batch_size": vit_model.batch_size,
                },
            ),
        )
        modules.append(
            AnalyticModuleNode(
                module_id="vision/mlp_projector",
                name="MlpProjector",
                qualified_class_name=(
                    "extern.modelmeter.models.deepseek_ocr.layers.vision."
                    "mlp_projector.MlpProjector"
                ),
                stage="vision",
                parent_id=root_id,
                children=[],
                repetition="none",
                repetition_count=None,
                constructor_params={
                    "input_dim": projector.input_dim,
                    "output_dim": projector.output_dim,
                    "num_tokens": projector.num_tokens,
                    "depth": projector.depth,
                    "projector_type": projector.projector_type,
                },
            ),
        )

        # Decoder module (stack of identical layers).
        modules.append(
            AnalyticModuleNode(
                module_id="decoder/deepseek_v2_decoder_layer",
                name="DeepseekV2DecoderLayer",
                qualified_class_name=(
                    "extern.modelmeter.models.deepseek_ocr.layers.decoder."
                    "deepseek_v2_decoder_layer.DeepseekV2DecoderLayer"
                ),
                stage="decode",
                parent_id=root_id,
                children=[],
                repetition="for",
                repetition_count=num_decoder_layers,
                constructor_params={
                    "hidden_size": decoder_layer.hidden_size,
                    "num_heads": decoder_layer.num_heads,
                    "seq_len": decoder_layer.seq_len,
                    "intermediate_size": decoder_layer.intermediate_size,
                    "num_experts": decoder_layer.num_experts,
                    "batch_size": decoder_layer.batch_size,
                    "num_key_value_heads": decoder_layer.num_key_value_heads,
                    "k_active": decoder_layer.k_active,
                    "num_shared_experts": decoder_layer.num_shared_experts,
                },
            ),
        )

        def _module_metrics(
            module_id: str,
            layer: object,
            calls: int,
        ) -> ModuleMetricsSnapshot:
            # All BaseLayer metrics are per-call; aggregate over calls.
            calls_int = max(int(calls), 1)

            def _val(name: str) -> float:
                fn = getattr(layer, name)
                v = fn()
                return float(v or 0.0)

            flops_tflops = _val("forward_tensor_core_flops") + _val("forward_cuda_core_flops")
            io_tb = _val("forward_cal_io")
            mem_w = _val("forward_memory_weight")
            mem_a = _val("forward_memory_activation")
            mem_kv = _val("forward_memory_kvcache")

            device_name = get_device_name()
            peak_tflops = get_peak_tflops(device_name, precision="bf16")
            # time_ms = FLOPs / peak_TFLOPs * 1e3
            total_flops_tflops = flops_tflops * float(calls_int)
            total_time_ms = 1000.0 * total_flops_tflops / max(peak_tflops, 1e-6)

            return ModuleMetricsSnapshot(
                module_id=module_id,
                profile_id=profile_id,
                calls=calls_int,
                total_time_ms=total_time_ms,
                total_flops_tflops=total_flops_tflops,
                total_io_tb=io_tb * float(calls_int),
                memory_weights_gb=mem_w,
                memory_activations_gb=mem_a,
                memory_kvcache_gb=mem_kv,
                share_of_model_time=0.0,  # filled below
                operator_breakdown=[],
            )

        # Leaf modules' metrics.
        metrics.append(_module_metrics("vision/image_encoder_vit", img_enc, calls=1))
        metrics.append(_module_metrics("vision/vit_model", vit_model, calls=1))
        metrics.append(_module_metrics("vision/mlp_projector", projector, calls=1))
        metrics.append(
            _module_metrics(
                "decoder/deepseek_v2_decoder_layer",
                decoder_layer,
                calls=num_decoder_layers,
            ),
        )

        # Compute shares based on leaf modules only.
        total_leaf_time = sum(m.total_time_ms for m in metrics)
        if total_leaf_time > 0.0:
            for m in metrics:
                m.share_of_model_time = m.total_time_ms / total_leaf_time

        # Model-level time for AnalyticModelReport.
        predicted_total_time_ms = model_layer.forward_tensor_core_flops()
        predicted_total_time_ms += model_layer.forward_cuda_core_flops()
        device_name = get_device_name()
        peak_tflops = get_peak_tflops(device_name, precision="bf16")
        predicted_total_time_ms = 1000.0 * predicted_total_time_ms / max(peak_tflops, 1e-6)

        return modules, metrics, predicted_total_time_ms

    def run_analytic(
        self,
        config: AnalysisConfig,
        *,
        workload_profile_id: str = "dsocr-standard-v1",
        run_id: Optional[str] = None,
    ) -> AnalyticModelReport:
        """Build AnalyticModelReport and write JSON/YAML and Markdown artifacts.

        This method constructs analytic ModelMeter layers for the
        DeepSeek-OCR vision and decoder stacks, aggregates their
        metrics into an :class:`AnalyticModelReport`, and writes:

        - ``report.json`` / ``report.yaml`` under
          ``tmp/profile-output/<run_id>/static_analysis/analytic_model/``
        - per-layer Markdown docs under the ``layers/`` subdirectory
          referenced by :attr:`AnalyticModelReport.layer_docs_dir`.
        """

        if run_id is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_id = f"dsocr-analytic-{ts}"

        model_spec = self._build_model_spec()
        workload = self._build_workload_profile(config, workload_profile_id)

        vision_layers = self._build_vision_layers(config)
        decoder_layer, num_decoder_layers = self._build_decoder_layer(config)

        # Build composite vision layer for model-level aggregation.
        vision_stack = _CompositeLayer(
            layers=[
                vision_layers["vision/image_encoder_vit"],
                vision_layers["vision/vit_model"],
                vision_layers["vision/mlp_projector"],
            ],
        )
        model_layer = DeepseekOCRModel.from_layers(
            vision_stack,
            decoder_layer,
            num_decoder_layers=num_decoder_layers,
        )

        modules, module_metrics, predicted_total_time_ms = self._build_module_nodes_and_metrics(
            model_layer=model_layer,
            vision_layers=vision_layers,
            decoder_layer=decoder_layer,
            num_decoder_layers=num_decoder_layers,
            model_spec=model_spec,
            workload=workload,
        )

        # Derive operator categories from the TorchInfo operator snapshot.
        operator_categories: list[OperatorCategory] = []
        try:
            artifact_dir = (
                Path(workspace_root())
                / "reports"
                / "20211117-dsorc-op-analysis"
                / "static-20251118-130533"
            )
            target_ops = load_target_operator_list(str(artifact_dir))
            operator_categories = build_operator_categories_from_target_list(target_ops)
        except Exception as exc:  # noqa: BLE001
            self.m_logger.warning(
                "Failed to load TorchInfo operator snapshot for operator categories: %s",
                exc,
            )
            operator_categories = []

        # Attach a coarse operator breakdown to each module snapshot using
        # the module's analytic layer class as the operator family.
        category_index = build_operator_category_index(operator_categories)
        modules_by_id = {m.module_id: m for m in modules}
        known_category_ids = {c.category_id for c in operator_categories}

        for snapshot in module_metrics:
            module = modules_by_id.get(snapshot.module_id)
            if module is None:
                continue

            qualified_name = module.qualified_class_name
            category_id = category_index.get(qualified_name)
            if category_id is None:
                category_id = categorize_operator_class_name(qualified_name)
                if category_id not in known_category_ids:
                    display = category_id.replace("_", " ").title()
                    operator_categories.append(
                        OperatorCategory(
                            category_id=category_id,
                            display_name=display,
                            description="Analytic module category derived from layer class.",
                            match_classes=[qualified_name],
                        ),
                    )
                    known_category_ids.add(category_id)
                    category_index[qualified_name] = category_id

            if snapshot.total_flops_tflops > 0.0:
                share = 1.0
            else:
                share = 0.0

            snapshot.operator_breakdown = [
                OperatorMetrics(
                    category_id=category_id,
                    calls=snapshot.calls,
                    flops_tflops=snapshot.total_flops_tflops,
                    io_tb=snapshot.total_io_tb,
                    share_of_module_flops=share,
                ),
            ]

        layer_docs_dir = analytic_layer_docs_dir(run_id)
        analytic_dir = analytic_model_dir(run_id)

        report = AnalyticModelReport(
            report_id=run_id,
            model=model_spec,
            workload=workload,
            modules=modules,
            operator_categories=operator_categories,
            module_metrics=module_metrics,
            profile_run_id=None,
            predicted_total_time_ms=predicted_total_time_ms,
            measured_total_time_ms=None,
            predicted_vs_measured_ratio=None,
            notes="DeepSeek-OCR analytic model generated via static layer formulas.",
            layer_docs_dir=layer_docs_dir,
        )

        out_dir = Path(analytic_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        Path(layer_docs_dir).mkdir(parents=True, exist_ok=True)

        payload = asdict(report)

        # JSON
        (out_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # YAML
        yaml = YAML(typ="safe")
        with (out_dir / "report.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(payload, f)

        # Markdown docs
        write_analytic_layer_docs(report)

        self.m_logger.info(
            "Analytic model written | run_id=%s dir=%s", run_id, out_dir.as_posix()
        )

        return report


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR static and analytic modeling utilities.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["analytic"],
        help="Execution mode (currently only 'analytic' is supported).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Local model path (default: <workspace_root>/models/deepseek-ocr).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device specifier for loading the model (e.g., 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--workload-profile-id",
        type=str,
        default="dsocr-standard-v1",
        help="Workload profile identifier (default: dsocr-standard-v1).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier used for output directory naming.",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=1024,
        help="Base padded size for vision analysis (default: 1024).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Image tile size for vision analysis (default: 640).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Representative sequence length for decoder analysis (default: 512).",
    )
    parser.add_argument(
        "--crop-mode",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable dynamic crops (1) or use only the global view (0).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for ``python -m llm_perf_opt.runners.dsocr_analyzer``."""

    args = _parse_cli_args()
    if args.mode != "analytic":
        print("ERROR: only --mode analytic is supported in this entry point.", file=sys.stderr)
        return 1

    # Resolve model path
    if args.model is None:
        model_path = Path(workspace_root()) / "models" / "deepseek-ocr"
    else:
        model_path = Path(args.model)
    model_path = model_path.resolve()
    if not model_path.exists():
        print(f"ERROR: model path not found: {model_path}", file=sys.stderr)
        return 1

    from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession

    try:
        session = DeepSeekOCRSession.from_local(
            model_path=str(model_path),
            device=args.device,
            use_flash_attn=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to initialize DeepSeekOCRSession: {exc}", file=sys.stderr)
        return 1

    analyzer = DeepseekOCRStaticAnalyzer(session)
    config = AnalysisConfig(
        image_h=args.base_size,
        image_w=args.base_size,
        base_size=args.base_size,
        image_size=args.image_size,
        seq_len=args.seq_len,
        crop_mode=bool(args.crop_mode),
        use_analytic_fallback=True,
        use_synthetic_inputs=True,
    )

    try:
        report = analyzer.run_analytic(
            config,
            workload_profile_id=args.workload_profile_id,
            run_id=args.run_id,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: analytic modeling failed: {exc}", file=sys.stderr)
        return 1

    out_dir = analytic_model_dir(report.report_id)
    print(f"Analytic report generated | report_id={report.report_id}")
    print(f"Artifacts directory: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
