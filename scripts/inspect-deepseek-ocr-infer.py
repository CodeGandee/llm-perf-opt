#!/usr/bin/env python
"""Inspect where DeepSeek-OCR `model.infer` comes from at runtime.

Loads the model with trust_remote_code and prints the source file, module,
and line numbers for `infer`, plus class/module information.

Usage:
  pixi run python scripts/inspect-deepseek-ocr-infer.py \
    --model ./models/deepseek-ocr --device cuda:0
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect DeepSeek-OCR infer() provenance")
    p.add_argument("--model", default=None, help="Path to local model repo (default: <repo>/models/deepseek-ocr)")
    p.add_argument("--device", default=None, help="Device, e.g. cuda:0 or cpu (default: auto)")
    return p.parse_args()


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "pyproject.toml").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).parent)
    model_path = args.model or str((repo_root / "models" / "deepseek-ocr").resolve())
    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    print(f"[cfg] model={model_path}\n[cfg] device={device}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
        local_files_only=True,
    )
    model = model.to(device).eval()

    print("[model] class:", model.__class__)  # type: ignore
    try:
        cls_file = inspect.getsourcefile(model.__class__)  # type: ignore[arg-type]
        print("[model] class file:", cls_file)
    except Exception as e:
        print("[model] class file: <unavailable>", e)

    has_infer = hasattr(model, "infer")
    print("[infer] hasattr:", has_infer)
    if not has_infer:
        print("infer() is not present on model; check trust_remote_code and local files.")
        return

    meth = getattr(model, "infer")
    print("[infer] obj:", meth)
    func = meth
    if hasattr(meth, "__func__"):
        func = meth.__func__  # type: ignore[attr-defined]

    # Print module and file locations
    try:
        modname = getattr(func, "__module__", None)
        print("[infer] __module__:", modname)
    except Exception:
        pass
    try:
        src_file = inspect.getsourcefile(func)  # type: ignore[arg-type]
        src_lines, first_line = inspect.getsourcelines(func)  # type: ignore[arg-type]
        print("[infer] source file:", src_file)
        print("[infer] first line:", first_line)
        # Print a small preview of the function definition header
        header = "".join(src_lines[:5])
        print("[infer] source preview:\n" + header)
    except Exception as e:
        print("[infer] could not get source:", e)

    # Also try to locate the vendor files by path
    vendor_modeling = (Path(model_path) / "modeling_deepseekocr.py").resolve()
    vendor_conv = (Path(model_path) / "conversation.py").resolve()
    print("[vendor] modeling_deepseekocr.py exists:", vendor_modeling.is_file(), vendor_modeling)
    print("[vendor] conversation.py exists:", vendor_conv.is_file(), vendor_conv)

    # Sanity: show tokenizer/model module names to confirm remote code usage
    print("[modules] model class module:", model.__class__.__module__)  # type: ignore
    print("[modules] tokenizer class module:", tok.__class__.__module__)


if __name__ == "__main__":
    main()
