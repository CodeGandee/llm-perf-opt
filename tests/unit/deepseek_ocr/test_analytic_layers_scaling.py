from __future__ import annotations

"""Scaling tests for DeepSeek-OCR analytic layers.

These unit tests validate that selected analytic layers:

- produce non-negative FLOPs, I/O, and memory estimates; and
- exhibit monotonic (non-decreasing) behavior when sequence length
  or hidden size increases.
"""

from modelmeter.models.deepseek_ocr.layers.decoder.deepseek_v2_decoder_layer import (
    DeepseekV2DecoderLayer,
)
from modelmeter.models.deepseek_ocr.layers.llama.llama_flash_attention2 import (
    LlamaFlashAttention2,
)
from modelmeter.models.deepseek_ocr.layers.vision.clip_vision_embeddings import (
    CLIPVisionEmbeddings,
)
from modelmeter.models.deepseek_ocr.layers.vision.notp_attention import (
    NoTPAttention,
)
from modelmeter.models.deepseek_ocr.layers.vision.notp_feedforward import (
    NoTPFeedForward,
)
from modelmeter.models.deepseek_ocr.layers.vision.notp_transformer import (
    NoTPTransformer,
)
from modelmeter.models.deepseek_ocr.layers.vision.notp_transformer_block import (
    NoTPTransformerBlock,
)
from modelmeter.models.deepseek_ocr.layers.vision.vit_model import VitModel


def _assert_non_negative(*values: float) -> None:
    for v in values:
        assert v >= 0.0


def _llama_metrics(seq_len: int, hidden_size: int) -> tuple[float, float]:
    layer = LlamaFlashAttention2(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=16,
        batch_size=1,
    )
    flops = (layer.forward_tensor_core_flops() or 0.0) + (layer.forward_cuda_core_flops() or 0.0)
    io_tb = layer.forward_cal_io() or 0.0
    _assert_non_negative(flops, io_tb)
    return flops, io_tb


def test_llama_flash_attention_scales_with_seq_len() -> None:
    flops_small, io_small = _llama_metrics(seq_len=128, hidden_size=1280)
    flops_large, io_large = _llama_metrics(seq_len=256, hidden_size=1280)
    assert flops_large >= flops_small
    assert io_large >= io_small


def test_llama_flash_attention_scales_with_hidden_size() -> None:
    flops_small, io_small = _llama_metrics(seq_len=256, hidden_size=1024)
    flops_large, io_large = _llama_metrics(seq_len=256, hidden_size=1536)
    assert flops_large >= flops_small
    assert io_large >= io_small


def _decoder_metrics(
    *,
    hidden_size: int,
    seq_len: int,
) -> tuple[float, float, float]:
    layer = DeepseekV2DecoderLayer(
        hidden_size=hidden_size,
        num_heads=16,
        seq_len=seq_len,
        intermediate_size=4 * hidden_size,
        num_experts=None,
        batch_size=1,
    )
    flops = (layer.forward_tensor_core_flops() or 0.0) + (layer.forward_cuda_core_flops() or 0.0)
    io_tb = layer.forward_cal_io() or 0.0
    mem_act = layer.forward_memory_activation() or 0.0
    _assert_non_negative(flops, io_tb, mem_act)
    return flops, io_tb, mem_act


def test_decoder_layer_scales_with_seq_len() -> None:
    flops_small, io_small, mem_small = _decoder_metrics(hidden_size=1280, seq_len=256)
    flops_large, io_large, mem_large = _decoder_metrics(hidden_size=1280, seq_len=512)
    assert flops_large >= flops_small
    assert io_large >= io_small
    assert mem_large >= mem_small


def test_decoder_layer_scales_with_hidden_size() -> None:
    flops_small, io_small, mem_small = _decoder_metrics(hidden_size=1024, seq_len=256)
    flops_large, io_large, mem_large = _decoder_metrics(hidden_size=1536, seq_len=256)
    assert flops_large >= flops_small
    assert io_large >= io_small
    assert mem_large >= mem_small


def _make_vit(seq_len: int) -> VitModel:
    # CLIP-style embeddings (Conv2d path treated as negligible here).
    emb = CLIPVisionEmbeddings(
        hidden_size=1024,
        image_size=224,
        patch_size=14,
        num_channels=3,
        batch_size=1,
        use_precomputed_patch_embeds=True,
    )
    attn = NoTPAttention(hidden_size=1024, num_heads=16, seq_len=seq_len, batch_size=1)
    ff = NoTPFeedForward(dim=1024, hidden_dim=4096, batch_size=1, seq_len=seq_len)
    blk = NoTPTransformerBlock(attention=attn, mlp=ff)
    stack = NoTPTransformer(blocks=[blk] * 4)
    return VitModel(
        embeddings=emb,
        transformer=stack,
        hidden_size=1024,
        seq_len=seq_len,
        batch_size=1,
    )


def test_vit_model_scales_with_seq_len() -> None:
    small = _make_vit(seq_len=128)
    large = _make_vit(seq_len=256)
    flops_small = (small.forward_tensor_core_flops() or 0.0) + (small.forward_cuda_core_flops() or 0.0)
    flops_large = (large.forward_tensor_core_flops() or 0.0) + (large.forward_cuda_core_flops() or 0.0)
    io_small = small.forward_cal_io() or 0.0
    io_large = large.forward_cal_io() or 0.0
    mem_small = small.forward_memory_activation() or 0.0
    mem_large = large.forward_memory_activation() or 0.0
    _assert_non_negative(flops_small, flops_large, io_small, io_large, mem_small, mem_large)
    assert flops_large >= flops_small
    assert io_large >= io_small
    assert mem_large >= mem_small
