"""Unit tests for Wan2.1 full-pipeline analytic model invariants."""

from __future__ import annotations

import pytest

from modelmeter.models.wan2_1.layers.core.wan2_1_pipeline_model import Wan2_1PipelineModel


def _build_tiny_pipeline(*, steps: int, frames: int, height: int, width: int) -> Wan2_1PipelineModel:
    return Wan2_1PipelineModel.from_config(
        # Workload.
        batch_size=1,
        num_frames=frames,
        height=height,
        width=width,
        num_inference_steps=steps,
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


def test_pipeline_stage_costs_sum_to_total() -> None:
    model = _build_tiny_pipeline(steps=2, frames=4, height=64, width=64)
    total = model.get_forward_cost()
    stages = model.stage_costs()

    assert total.flops_tflops == pytest.approx(
        stages.text_encoder.flops_tflops + stages.diffusion_core.flops_tflops + stages.vae_decode.flops_tflops,
        rel=0.0,
        abs=0.0,
    )
    assert total.io_tb == pytest.approx(
        stages.text_encoder.io_tb + stages.diffusion_core.io_tb + stages.vae_decode.io_tb,
        rel=0.0,
        abs=0.0,
    )
    assert total.weights_gb == pytest.approx(
        stages.text_encoder.weights_gb + stages.diffusion_core.weights_gb + stages.vae_decode.weights_gb,
        rel=0.0,
        abs=0.0,
    )
    assert total.kv_gb == pytest.approx(
        stages.text_encoder.kv_gb + stages.diffusion_core.kv_gb + stages.vae_decode.kv_gb,
        rel=0.0,
        abs=0.0,
    )
    assert total.activations_gb == pytest.approx(
        max(stages.text_encoder.activations_gb, stages.diffusion_core.activations_gb, stages.vae_decode.activations_gb),
        rel=0.0,
        abs=0.0,
    )


def test_pipeline_flops_scale_with_steps_for_diffusion_stage() -> None:
    model_1 = _build_tiny_pipeline(steps=1, frames=4, height=64, width=64)
    model_3 = _build_tiny_pipeline(steps=3, frames=4, height=64, width=64)

    dit_1 = float(model_1.diffusion_core.get_forward_cost().flops_tflops)
    dit_3 = float(model_3.diffusion_core.get_forward_cost().flops_tflops)
    assert dit_1 > 0.0
    assert dit_3 == pytest.approx(3.0 * dit_1, rel=1e-6, abs=0.0)

    total_1 = float(model_1.get_forward_cost().flops_tflops)
    total_3 = float(model_3.get_forward_cost().flops_tflops)
    assert total_3 == pytest.approx(total_1 + 2.0 * dit_1, rel=1e-6, abs=0.0)


def test_pipeline_flops_monotonic_with_resolution() -> None:
    small = _build_tiny_pipeline(steps=1, frames=4, height=64, width=64)
    large = _build_tiny_pipeline(steps=1, frames=4, height=128, width=128)

    f_small = float(small.get_forward_cost().flops_tflops)
    f_large = float(large.get_forward_cost().flops_tflops)
    assert f_large >= f_small


def test_pipeline_flops_monotonic_with_frames() -> None:
    f4 = _build_tiny_pipeline(steps=1, frames=4, height=64, width=64)
    f5 = _build_tiny_pipeline(steps=1, frames=5, height=64, width=64)

    flops4 = float(f4.get_forward_cost().flops_tflops)
    flops5 = float(f5.get_forward_cost().flops_tflops)
    assert flops5 >= flops4
