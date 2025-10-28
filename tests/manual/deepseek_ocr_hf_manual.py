#!/usr/bin/env python
"""
Manual script: load DeepSeek-OCR (HF, no vLLM) once, then iterate multiple
images and run inference in bf16.

Run with Pixi (not system Python):
  pixi run python tests/manual/deepseek_ocr_hf_manual.py

Guidelines
- Global-scope script for easy Jupyter use (no main guard)
- Load model/tokenizer once; iterate images separately
- Never modify input images; write results under ./tmp

Prereqs: torch, transformers, tokenizers, einops, addict, easydict, pillow;
optional: flash-attn. Uses offline files from models/deepseek-ocr.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

# Configuration via env vars
MODEL_ID = os.environ.get("DSOCR_MODEL", str(Path("models/deepseek-ocr").resolve()))
DEVICE_STR = os.environ.get("DSOCR_DEVICE")
USE_FLASH_ATTN = os.environ.get("DSOCR_USE_FLASH_ATTN", "1").lower() not in ("0", "false", "no")

# Device and dtype
DEVICE = torch.device(DEVICE_STR) if DEVICE_STR else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
DTYPE = torch.bfloat16

# Discover input images
images: list[Path] = []
env_image = os.environ.get("DSOCR_IMAGE")
if env_image:
    p = Path(env_image)
    if p.is_dir():
        images = sorted([*p.glob("*.png"), *p.glob("*.jpg")])
    elif p.is_file():
        images = [p]
    else:
        raise FileNotFoundError(f"DSOCR_IMAGE not found: {env_image}")
else:
    samples_dir = Path("data/samples")
    if not samples_dir.is_dir():
        raise FileNotFoundError("Missing directory: data/samples/")
    images = sorted([*samples_dir.glob("*.png"), *samples_dir.glob("*.jpg")])
if not images:
    raise FileNotFoundError("No input images found (.png/.jpg)")

# Output root
TMP = Path.cwd() / "tmp"
OUT_ROOT = TMP / "dsocr_outputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Load tokenizer + model once
print(f"[info] Loading tokenizer from: {MODEL_ID} (offline)")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, local_files_only=True
)

attn_impl = "flash_attention_2" if USE_FLASH_ATTN else "eager"
print(f"[info] Loading model from: {MODEL_ID} (attn={attn_impl}, offline)")
try:
    model = AutoModel.from_pretrained(
        MODEL_ID,
        _attn_implementation=attn_impl,
        trust_remote_code=True,
        use_safetensors=True,
        local_files_only=True,
    )
except Exception as e:
    if USE_FLASH_ATTN:
        print(f"[warn] flash-attn load failed: {e}. Retrying with eager...")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            _attn_implementation="eager",
            trust_remote_code=True,
            use_safetensors=True,
            local_files_only=True,
        )
    else:
        raise

model = model.eval()
try:
    model = model.to(DEVICE).to(DTYPE)
except Exception as e:  # noqa: BLE001
    print(f"[warn] dtype/device move failed ({e}), using default precision/device")

# Prompt (from models/deepseek-ocr README)
PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "

# Iterate images; prefer custom .infer when available
for img_path in images:
    out_dir = OUT_ROOT / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Running DeepSeek-OCR .infer(...) on image: {img_path}")

    if hasattr(model, "infer"):
        try:
            res = model.infer(
                tokenizer,
                prompt=PROMPT,
                image_file=str(img_path),
                output_path=str(out_dir),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
                test_compress=True,
            )
            print("\n===== DeepSeek-OCR Manual Test Output =====\n")
            print(str(res))
            print("\n==========================================\n")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"[warn] .infer failed on {img_path}: {e}")

    # Fallback: basic text-only generate
    try:
        inputs = tokenizer("Free OCR.", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64)
        text = cast(str, tokenizer.decode(out[0], skip_special_tokens=False))
        print("\n===== Fallback Output =====\n")
        print(text)
        print("\n==========================\n")
    except Exception as e:  # noqa: BLE001
        print(f"[error] Fallback generation failed on {img_path}: {e}")
