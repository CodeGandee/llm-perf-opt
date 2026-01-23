"""Unit tests for Wan2.1 static analyzer pipeline-mode module tree."""

from __future__ import annotations

from modelmeter.models.wan2_1.layers.core.wan2_1_pipeline_model import Wan2_1PipelineModel

from llm_perf_opt.runners.wan2_1_analyzer import _build_module_tree


def test_runner_builds_full_pipeline_module_tree() -> None:
    model = Wan2_1PipelineModel.from_config(
        # Workload.
        batch_size=1,
        num_frames=4,
        height=64,
        width=64,
        num_inference_steps=2,
        text_len=16,
        # Diffusion core.
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        head_dim=16,
        mlp_intermediate_size=128,
        latent_channels=16,
        vae_downsample_factor=8,
        vae_temporal_downsample_factor=2,
        patch_size=2,
        transformer_bits=16,
        transformer_use_bias=False,
        attention_sram_kb=1024,
        layer_impl="shared",
        # Text encoder (tiny).
        text_vocab_size=128,
        text_hidden_size=64,
        text_num_layers=2,
        text_num_attention_heads=4,
        text_head_dim=16,
        text_mlp_intermediate_size=128,
        text_attention_sram_kb=1024,
        text_bits=16,
        # VAE knobs.
        vae_bits=16,
        include_vae_mid_attention=False,
    )

    root_id, modules, layer_by_id = _build_module_tree(model)
    assert root_id == "pipeline"

    module_ids = [m.module_id for m in modules]
    assert len(module_ids) == len(set(module_ids))

    for expected in ["pipeline", "text_encoder/umt5", "diffusion/dit", "vae/decode"]:
        assert expected in layer_by_id

    modules_by_id = {m.module_id: m for m in modules}
    assert modules_by_id["text_encoder/umt5"].parent_id == "pipeline"
    assert modules_by_id["diffusion/dit"].parent_id == "pipeline"
    assert modules_by_id["vae/decode"].parent_id == "pipeline"

