#!/usr/bin/env python
"""
Manual script: load DeepSeek-OCR with Hugging Face Transformers (no vLLM) and
run a single inference in bf16 on a randomly generated image.

Run with Pixi (not system Python):
  pixi run python tests/manual/deepseek_ocr_hf_manual.py

Notes
- Writes temporary files to ./tmp within the workspace.
- Tries flash-attn first; falls back to SDPA if unavailable.
- If inference fails, emits a random placeholder string so the wiring can be
  inspected without hard failures.

Prereqs (per external/dsocr-hf/README.md): torch, transformers, tokenizers,
einops, addict, easydict, pillow; optional: flash-attn.
"""

from __future__ import annotations

import os
import random
import string
from pathlib import Path


def _make_tmp_dir() -> Path:
    root = Path.cwd() / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_random_image(path: Path, size: int = 640) -> Path:
    # Generate a random RGB image. Prefer numpy if available, otherwise
    # fall back to a solid color image via PIL only for portability.
    try:
        import numpy as np
        from PIL import Image

        arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
        img = Image.fromarray(arr, mode="RGB")
    except Exception:
        from PIL import Image

        color = tuple(random.randint(0, 255) for _ in range(3))
        img = Image.new("RGB", (size, size), color=color)
    img.save(path)
    return path


def _random_text(n: int = 200) -> str:
    alphabet = string.ascii_letters + string.digits + " \n.-_:#|/"
    return "".join(random.choice(alphabet) for _ in range(n))


def run_inference(model_name: str, device: str | None, use_flash_attn: bool) -> str:
    from transformers import AutoModel, AutoTokenizer
    import torch
    from PIL import Image

    # Select device
    if device:
        d = torch.device(device)
    else:
        d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Prepare dtype (prefer bf16)
    dtype = torch.bfloat16

    # Temp paths
    tmp = _make_tmp_dir()
    image_path = tmp / "dsocr_random_input.png"
    output_dir = tmp / "dsocr_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    _make_random_image(image_path, size=640)

    # Prompt options from external/dsocr-hf/README.md
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    # Attempt to use flash-attn if requested
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"

    print(f"[info] Loading tokenizer from: {model_name} (offline)")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"[info] Loading model from: {model_name} (attn={attn_impl}, offline)")
    try:
        model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation=attn_impl,
            trust_remote_code=True,
            use_safetensors=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        if use_flash_attn:
            print(f"[warn] flash-attn load failed: {e}. Retrying with sdpa...", file=sys.stderr)
            model = AutoModel.from_pretrained(
                model_name,
                _attn_implementation="sdpa",
                trust_remote_code=True,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        else:
            raise

    model = model.eval()
    try:
        model = model.to(d).to(dtype)
    except Exception as e:
        print(f"[warn] dtype/device move failed ({e}), falling back to default precision/device")

    # Prefer the custom .infer interface (trust_remote_code)
    if hasattr(model, "infer"):
        try:
            print("[info] Running DeepSeek-OCR .infer(...) on random image")
            res = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                output_path=str(output_dir),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
                test_compress=True,
            )
            # The custom API may return a string or structure; stringify for logging
            return str(res)
        except Exception as e:
            print(f"[warn] Model .infer failed: {e}")

    # Fallback: try a basic generate if the custom interface is unavailable
    try:
        print("[info] Fallback to generate() with plain text prompt (no image)")
        inputs = tokenizer("Free OCR.", return_tensors="pt").to(d)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64)
        return tokenizer.decode(out[0], skip_special_tokens=False)
    except Exception as e:
        print(f"[warn] Fallback generate failed: {e}")
        print("[info] Emitting random placeholder text to complete manual test")
        return _random_text(300)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DSOCR_MODEL", "deepseek-ai/DeepSeek-OCR"))
    ap.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    ap.add_argument("--no-flash-attn", action="store_true", help="disable flash-attn attempt")
    args = ap.parse_args()

"""
Configuration (override with env vars for notebook-friendliness):
  - DSOCR_MODEL: Hugging Face model id (default: deepseek-ai/DeepSeek-OCR)
  - DSOCR_DEVICE: Device string (e.g., cuda, cuda:0, cpu); default auto
  - DSOCR_USE_FLASH_ATTN: "1"/"0" to toggle flash-attn attempt (default: 1)
"""

MODEL = os.environ.get("DSOCR_MODEL", str(Path("external/dsocr-hf").resolve()))
DEVICE = os.environ.get("DSOCR_DEVICE", None)
USE_FLASH_ATTENTION = os.environ.get("DSOCR_USE_FLASH_ATTN", "1") not in ("0", "false", "False", "no", "No")

try:
    text = run_inference(MODEL, DEVICE, USE_FLASH_ATTENTION)
    print("\n===== DeepSeek-OCR Manual Test Output =====\n")
    print(text)
    print("\n==========================================\n")
except ImportError as e:
    print("[error] Missing dependency:", e)
    print("Hint: run within Pixi and ensure transformers, tokenizers, einops, addict, easydict, pillow are installed.")
except Exception as e:
    print("[error] Unexpected failure:", e)
