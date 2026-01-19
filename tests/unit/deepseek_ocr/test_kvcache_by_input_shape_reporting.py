from __future__ import annotations

from typing import Dict

from modelmeter.models.deepseek_ocr.scripts.reporting.kvcache_by_input_shape import (
    KVCachePoint,
    _merge_sweeps_to_kvcache_points,
)


def _make_prefill_sweep() -> Dict[str, object]:
    point: Dict[str, object] = {
        "image_tokens_total": 100,
        "image_tokens_crops": 90,
        "height_crop_num": 1,
        "width_crop_num": 2,
        "batch_size": 1,
        "context_len": 110,
        "decode_steps": 64,
        "prefill_stagecost": {
            "prefill_eager": {"kv_gb": 1.0},
            "prefill_flash": {"kv_gb": 0.8},
            "prefill_vendor": {"kv_gb": 0.9},
        },
    }
    return {
        "model_id": "dummy-model",
        "min_image_tokens": 100,
        "max_image_tokens": 100,
        "global_image_tokens": 10,
        "batch_size": 1,
        "decode_steps": 64,
        "points": [point],
    }


def _make_decode_sweep() -> Dict[str, object]:
    point: Dict[str, object] = {
        "image_tokens_total": 100,
        "image_tokens_crops": 90,
        "height_crop_num": 1,
        "width_crop_num": 2,
        "batch_size": 1,
        "context_len": 110,
        "decode_steps": 64,
        "decode_stagecost": {
            "decode_eager": {"kv_gb": 2.0},
            "decode_flash": {"kv_gb": 1.5},
            "decode_vendor": {"kv_gb": 1.5},
        },
    }
    return {
        "model_id": "dummy-model",
        "min_image_tokens": 100,
        "max_image_tokens": 100,
        "global_image_tokens": 10,
        "batch_size": 1,
        "decode_steps": 64,
        "points": [point],
    }


def test_merge_sweeps_to_kvcache_points_single_point() -> None:
    prefill_sweep = _make_prefill_sweep()
    decode_sweep = _make_decode_sweep()

    json_points, kv_points = _merge_sweeps_to_kvcache_points(prefill_sweep, decode_sweep)

    assert len(json_points) == 1
    assert len(kv_points) == 1

    kv_point: KVCachePoint = kv_points[0]
    assert kv_point.image_tokens_total == 100
    assert kv_point.height_crop_num == 1
    assert kv_point.width_crop_num == 2
    assert kv_point.decode_steps == 64

    # Analytic KV-cache components.
    assert kv_point.analytic_normal_prefill_kv_gb == 1.0
    assert kv_point.analytic_flash_prefill_kv_gb == 0.8
    assert kv_point.analytic_normal_decode_kv_gb == 2.0
    assert kv_point.analytic_flash_decode_kv_gb == 1.5

    # Analytic totals should sum prefill + decode.
    assert kv_point.analytic_normal_total_kv_gb == 3.0
    assert kv_point.analytic_flash_total_kv_gb == 2.3
    # Vendor KV-cache totals sum prefill + decode; the test data
    # expresses vendor KV in the same units/precision as analytic KV.
    assert kv_point.vendor_total_kv_gb == 2.4

    record = json_points[0]
    kvcache_gb = record.get("kvcache_gb")
    assert isinstance(kvcache_gb, dict)

    assert kvcache_gb["analytic_normal_attention_full_prefill"] == 1.0
    assert kvcache_gb["analytic_flash_attention_full_prefill"] == 0.8
    assert kvcache_gb["analytic_normal_attention_full_decode"] == 2.0
    assert kvcache_gb["analytic_flash_attention_full_decode"] == 1.5
    assert kvcache_gb["analytic_normal_attention_full_total"] == 3.0
    assert kvcache_gb["analytic_flash_attention_full_total"] == 2.3
    assert kvcache_gb["vendor_total"] == 2.4
