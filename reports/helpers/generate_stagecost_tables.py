from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class StageCostRow:
    num_crops: int
    crop_grid: str
    image_tokens: int
    flops_tensor_tflops: float
    flops_cuda_tflops: float

    @property
    def tensor_cuda_ratio(self) -> float:
        if self.flops_cuda_tflops == 0.0:
            return float("inf")
        return self.flops_tensor_tflops / self.flops_cuda_tflops


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_table(rows: Iterable[StageCostRow]) -> str:
    header = (
        "|Num crops|Crop grid (H×W)|Image tokens (global + crops)|"
        "Tensor Core FLOPs (TFLOPs)|CUDA-core FLOPs (TFLOPs)|Tensor:CUDA ratio (CUDA=1.0)|"
    )
    align = "| :---: | :---: | :---: | :---: | :---: | :---: |"
    lines: List[str] = [header, align]
    for row in rows:
        tensor = f"{row.flops_tensor_tflops:.6e}"
        cuda = f"{row.flops_cuda_tflops:.6e}"
        if row.tensor_cuda_ratio == float("inf"):
            ratio_str = "inf:1.0"
        else:
            ratio_str = f"{row.tensor_cuda_ratio:.3f}:1.0"
        lines.append(
            f"|{row.num_crops}|{row.crop_grid}|{row.image_tokens}|"
            f"{tensor}|{cuda}|{ratio_str}|"
        )
    return "\n".join(lines)


def build_vision_rows(sweep_root: Path) -> List[StageCostRow]:
    data = load_json(sweep_root / "vision_crops" / "vision_sweep.json")
    rows: List[StageCostRow] = []
    for point in data["points"]:
        stage = point["stage_costs"]["vision"]
        num_crops = int(point["height_crop_num"]) * int(point["width_crop_num"])
        crop_grid = f"{int(point['height_crop_num'])}x{int(point['width_crop_num'])}"
        rows.append(
            StageCostRow(
                num_crops=num_crops,
                crop_grid=crop_grid,
                image_tokens=int(point["image_tokens_total"]),
                flops_tensor_tflops=float(stage["flops_tensor_tflops"]),
                flops_cuda_tflops=float(stage["flops_cuda_tflops"]),
            )
        )
    return rows


def build_decode_rows(sweep_root: Path) -> List[StageCostRow]:
    data = load_json(sweep_root / "e2e_decode" / "e2e_decode_sweep.json")
    rows: List[StageCostRow] = []
    for point in data["points"]:
        stage = point["decode_stagecost"]["decode_eager"]
        num_crops = int(point["height_crop_num"]) * int(point["width_crop_num"])
        crop_grid = f"{int(point['height_crop_num'])}x{int(point['width_crop_num'])}"
        rows.append(
            StageCostRow(
                num_crops=num_crops,
                crop_grid=crop_grid,
                image_tokens=int(point["image_tokens_total"]),
                flops_tensor_tflops=float(stage["flops_tensor_tflops"]),
                flops_cuda_tflops=float(stage["flops_cuda_tflops"]),
            )
        )
    return rows


def build_prefill_rows(sweep_root: Path) -> List[StageCostRow]:
    data = load_json(sweep_root / "e2e_vision_prefill" / "e2e_vision_prefill_sweep.json")
    rows: List[StageCostRow] = []
    for point in data["points"]:
        stage = point["prefill_stagecost"]["prefill_eager"]
        num_crops = int(point["height_crop_num"]) * int(point["width_crop_num"])
        crop_grid = f"{int(point['height_crop_num'])}x{int(point['width_crop_num'])}"
        rows.append(
            StageCostRow(
                num_crops=num_crops,
                crop_grid=crop_grid,
                image_tokens=int(point["image_tokens_total"]),
                flops_tensor_tflops=float(stage["flops_tensor_tflops"]),
                flops_cuda_tflops=float(stage["flops_cuda_tflops"]),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate StageCost FLOP-split tables from DeepSeek-OCR sweep outputs."
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        required=True,
        help="Path to a sweep run directory (e.g., reports/sweep/20251128-152354).",
    )
    parser.add_argument(
        "--mode",
        choices=("vision", "decode", "prefill"),
        required=True,
        help="Which StageCost table to generate.",
    )
    args = parser.parse_args()

    if args.mode == "vision":
        rows = build_vision_rows(args.sweep_root)
    elif args.mode == "decode":
        rows = build_decode_rows(args.sweep_root)
    else:
        rows = build_prefill_rows(args.sweep_root)

    print(format_table(rows))


if __name__ == "__main__":
    main()

