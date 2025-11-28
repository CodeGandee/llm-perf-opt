from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_table(rows: Iterable[Dict[str, object]]) -> str:
    header = (
        "|Num crops|Crop grid (H×W)|Image tokens (global + crops)|"
        "Tensor Core FLOPs (TFLOPs)|CUDA-core FLOPs (TFLOPs)|Forward I/O (GB)|"
        "KV-cache size (GB)|Peak activation size (GB)|VRAM needed (GB)|"
    )
    align = (
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    lines: List[str] = [header, align]
    for row in rows:
        lines.append(
            "|{crop_count}|{crop_shape}|{image_tokens}|{tensor_tflops:.6e}|"
            "{cuda_tflops:.6e}|{io_gb:.3f}|{kv_gb:.3f}|{act_gb:.3f}|{vram_gb:.3f}|".format(
                **row
            )
        )
    return "\n".join(lines)


def load_static_weights_gb(sweep_root: Path) -> float:
    """
    Load static model weight size (GB) from the torchinfo snapshot.

    This is used instead of StageCost.weights_gb because the current
    analytic model underestimates decoder MoE weights; the static
    report from the vendor model is treated as ground truth for
    parameter bytes.
    """

    # sweep_root is typically reports/sweep/<timestamp>, so reports_root
    # is its parent directory (../..).
    reports_root = sweep_root.parent.parent
    torchinfo_path = (
        reports_root
        / "20211117-dsorc-op-analysis"
        / "static-20251118-130533"
        / "torchinfo-layers.json"
    )
    data = load_json(torchinfo_path)
    root = data["layers_flat"][0]
    param_bytes = int(root["param_bytes"])
    # Use decimal GB here to match report conventions.
    return param_bytes / 1.0e9


def build_rows(sweep_root: Path) -> List[Dict[str, object]]:
    """Build per-crop memory requirement rows from kvcache + StageCost JSON.

    We use:
    - tensor/cuda TFLOPs and IO from the vision+prefill StageCost (flash attention path).
    - weights/activations from the same StageCost.
    - total KV-cache (prefill + decode, for K decode steps) from kvcache-by-input-shape.json.
    """

    data = load_json(
        sweep_root / "e2e_vision_prefill" / "kvcache-by-input-shape.json"
    )

    static_weights_gb = load_static_weights_gb(sweep_root)

    rows: List[Dict[str, object]] = []
    for point in data["points"]:
        h = int(point["height_crop_num"])
        w = int(point["width_crop_num"])
        crop_count = h * w
        crop_shape = f"{h}x{w}"

        image_tokens = int(point["image_tokens_total"])

        stage = point["prefill_stagecost"]["prefill_flash"]
        tensor_tflops = float(stage["flops_tensor_tflops"])
        cuda_tflops = float(stage["flops_cuda_tflops"])
        io_tb = float(stage["io_tb"])
        # Convert activation I/O from terabits to gigabytes.
        # 1 Tb = 10^12 bits, 1 GB = 10^9 bytes = 8×10^9 bits ⇒ 1 Tb = 125 GB.
        io_gb = io_tb * 125.0

        # Prefer static vendor weight size over analytic weights_gb.
        weights_gb = static_weights_gb
        activations_gb = float(stage["activations_gb"])

        kv_total_gb = float(
            point["kvcache_gb"]["analytic_flash_attention_full_total"]
        )

        vram_gb = weights_gb + activations_gb + kv_total_gb

        rows.append(
            {
                "crop_count": crop_count,
                "crop_shape": crop_shape,
                "image_tokens": image_tokens,
                "tensor_tflops": tensor_tflops,
                "cuda_tflops": cuda_tflops,
                "io_gb": io_gb,
                "kv_gb": kv_total_gb,
                "act_gb": activations_gb,
                "vram_gb": vram_gb,
            }
        )

    # Keep natural sweep ordering (already sorted by image_tokens_total).
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a memory-requirement table for DeepSeek-OCR "
            "from e2e vision+prefill and decode sweeps."
        )
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        required=True,
        help="Path to a sweep run directory (e.g., reports/sweep/20251128-152354).",
    )
    args = parser.parse_args()

    rows = build_rows(args.sweep_root)
    print(format_table(rows))


if __name__ == "__main__":
    main()
