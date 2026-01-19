"""Unit tests for Wan2.1 token geometry and FLOP scaling invariants."""

from __future__ import annotations

import pytest

from modelmeter.models.wan2_1.layers.core.wan2_1_dit_model import Wan2_1DiTModel
from modelmeter.models.wan2_1.layers.geometry import build_token_geometry


def test_build_token_geometry_monotonic_with_frames() -> None:
    """Token sequence length scales linearly with frames when resolution is fixed."""

    g1 = build_token_geometry(
        batch_size=1,
        num_frames=4,
        height=256,
        width=256,
        vae_downsample_factor=8,
        patch_size=2,
    )
    g2 = build_token_geometry(
        batch_size=1,
        num_frames=8,
        height=256,
        width=256,
        vae_downsample_factor=8,
        patch_size=2,
    )
    assert g2.tokens_per_frame == g1.tokens_per_frame
    assert g2.dit_seq_len > g1.dit_seq_len
    assert g2.dit_seq_len == 2 * g1.dit_seq_len


def test_build_token_geometry_monotonic_with_resolution() -> None:
    """Token sequence length increases with spatial resolution when frames are fixed."""

    g1 = build_token_geometry(
        batch_size=1,
        num_frames=4,
        height=256,
        width=256,
        vae_downsample_factor=8,
        patch_size=2,
    )
    g2 = build_token_geometry(
        batch_size=1,
        num_frames=4,
        height=512,
        width=512,
        vae_downsample_factor=8,
        patch_size=2,
    )
    assert g2.latent_h >= g1.latent_h
    assert g2.latent_w >= g1.latent_w
    assert g2.tokens_per_frame > g1.tokens_per_frame
    assert g2.dit_seq_len > g1.dit_seq_len


def test_dit_flops_scale_linearly_with_steps() -> None:
    """DiT FLOPs scale linearly with the diffusion step count."""

    model_1 = Wan2_1DiTModel.from_config(
        batch_size=1,
        num_frames=4,
        height=256,
        width=256,
        num_inference_steps=1,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        head_dim=16,
        mlp_intermediate_size=128,
        vae_downsample_factor=8,
        patch_size=2,
        attention_sram_kb=1024,
        bits=16,
        use_bias=False,
    )
    model_3 = Wan2_1DiTModel.from_config(
        batch_size=1,
        num_frames=4,
        height=256,
        width=256,
        num_inference_steps=3,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        head_dim=16,
        mlp_intermediate_size=128,
        vae_downsample_factor=8,
        patch_size=2,
        attention_sram_kb=1024,
        bits=16,
        use_bias=False,
    )
    f1 = float(model_1.forward_tensor_core_flops() or 0.0)
    f3 = float(model_3.forward_tensor_core_flops() or 0.0)
    assert f1 > 0.0
    assert f3 == pytest.approx(3.0 * f1, rel=1e-6, abs=0.0)
