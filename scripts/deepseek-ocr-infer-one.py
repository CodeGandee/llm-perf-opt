#!/usr/bin/env python
"""DeepSeek-OCR: run vendor infer() over images.

Usage
-----
Run with Pixi environment:

    pixi run python scripts/deepseek-ocr-infer-one.py \
      -i <image-path|image-dir|glob:pattern> -o <output-dir>

Arguments
---------
-i / --input
    One of:
    - path to a single image file (jpg/jpeg/png)
    - path to a directory tree (recursively scans for images)
    - "glob:<pattern>" (uses Python glob; pattern may include **, absolute or relative)
    - path to a text file listing image paths, one per line
      (relative paths are resolved against current working directory)

-o / --output
    Output root directory. Subdir strategy:
    - single image path: subdir = image basename without extension
    - image-dir: replicate directory structure relative to the input dir and
      add a leaf subdir named by the image basename
    - glob pattern: subdir = f"{image-basename}-{index}" where index is the
      1-based order in the string-sorted glob file list
    - file list: subdir = f"{image-basename}-{index}" where index follows
      the order in the list file (no additional sorting)

Optional
--------
--model <path>         Local model path (default: <repo_root>/models/deepseek-ocr)
--device <device>      Device spec (default: auto cuda/cpu)
--base-size <int>      Base padded size (default: 1024)
--image-size <int>     Image tile size (default: 640)
--crop-mode <0/1>      Enable dynamic crops (default: 1)
--prompt <str>         Prompt (default: "<image>\n<|grounding|>Convert the document to markdown.")

Notes
-----
- Uses model.infer(...) from trust_remote_code to perform full preprocessing
  and saving of results (result.mmd, *_with_boxes.jpg, images/*.jpg).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "pyproject.toml").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start


def list_images_from_dir(dir_path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    out: list[Path] = []
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out, key=lambda x: str(x))


def list_images_from_glob(pattern: str) -> list[Path]:
    paths = [Path(p) for p in glob.glob(pattern, recursive=True)]
    paths = [p for p in paths if p.is_file()]
    return sorted(paths, key=lambda x: str(x))


def list_images_from_filelist(filelist: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    out: list[Path] = []
    lines = filelist.read_text(encoding="utf-8").splitlines()
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        p = Path(s)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def build_output_dir_for_file(
    input_mode: str,
    inp: str,
    out_root: Path,
    file_path: Path,
    index: int | None,
) -> Path:
    # input_mode: "file" | "dir" | "glob"
    if input_mode == "file":
        return out_root / file_path.stem
    if input_mode == "dir":
        input_dir = Path(inp).resolve()
        rel_parent = file_path.resolve().parent.relative_to(input_dir)
        return out_root / rel_parent / file_path.stem
    if input_mode == "glob":
        suffix = f"-{index}" if index is not None else ""
        return out_root / f"{file_path.stem}{suffix}"
    return out_root / file_path.stem


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR vendor infer() over images",
    )
    parser.add_argument("-i", "--input", required=True, help="image path | image dir | glob:pattern")
    parser.add_argument("-o", "--output", required=True, help="output root directory")
    parser.add_argument("--model", default=None, help="path to local model (default: <repo_root>/models/deepseek-ocr)")
    parser.add_argument("--device", default=None, help="device (e.g., cuda:0)")
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--crop-mode", type=int, default=1)
    parser.add_argument(
        "--prompt",
        default="<image>\n<|grounding|>Convert the document to markdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()

    here = Path(__file__).parent
    repo_root = find_repo_root(here)
    out_root = Path(args.output).resolve()
    ensure_dir(out_root)

    model_path = args.model or str((repo_root / "models" / "deepseek-ocr").resolve())

    # Resolve images by mode
    input_mode = "file"
    image_list: list[Path] = []
    inp = args.input
    if inp.startswith("glob:"):
        input_mode = "glob"
        pattern = inp[len("glob:") :]
        image_list = list_images_from_glob(pattern)
    else:
        p = Path(inp)
        if p.is_dir():
            input_mode = "dir"
            image_list = list_images_from_dir(p)
        elif p.is_file():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                input_mode = "file"
                image_list = [p]
            else:
                input_mode = "filelist"
                image_list = list_images_from_filelist(p)
        else:
            raise FileNotFoundError(f"Input not found: {inp}")

    if not image_list:
        raise RuntimeError("No images found to process")

    # Load model/tokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
        local_files_only=True,
    )
    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.eval().to(device)
    try:
        model = model.to(dtype=torch.bfloat16)
    except Exception:
        pass

    # Inference loop (vendor helper)
    crop_mode = bool(int(args.crop_mode))
    for idx, img_path in enumerate(image_list, start=1):
        img_abs = str(Path(img_path).resolve())
        subdir = build_output_dir_for_file(
            input_mode,
            inp,
            out_root,
            Path(img_abs),
            index=(idx if input_mode in {"glob", "filelist"} else None),
        )
        ensure_dir(subdir)
        print(f"[infer] image={img_abs}\n        output={subdir}")
        model.infer(
            tok,
            prompt=str(args.prompt),
            image_file=img_abs,
            output_path=str(subdir),
            base_size=int(args.base_size),
            image_size=int(args.image_size),
            crop_mode=crop_mode,
            save_results=True,
            test_compress=False,
        )


if __name__ == "__main__":
    main()
