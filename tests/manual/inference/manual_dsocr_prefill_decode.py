#!/usr/bin/env python
"""
Manual script: separate prefill and decode for DeepSeek-OCR.

This script loads a local DeepSeek-OCR snapshot and performs a two-phase
inference: a prefill forward pass (building the KV cache and performing
vision embedding fusion) followed by a token-by-token decode loop that reuses
the cache. It writes vendor-style artifacts and an interpretable JSON.

Usage
-----
From the repository root:

    pixi run -e rtx5090 python tests/manual/inference/manual_dsocr_prefill_decode.py \
        --image models/deepseek-ocr/assets/show1.jpg \
        --max_new_tokens 512

Outputs
-------
Under ``tmp/<YYYYmmdd-HHMMSS>/``:
  - ``result.txt`` (decoded text)
  - ``result.mmd`` (vendor-like markdown with image references)
  - ``result_with_boxes.jpg`` (overlay if <|ref|>/<|det|> spans present)
  - ``images/`` (crops for label == 'image')
  - ``info.json`` (interpretable JSON summary)
"""

import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from llm_perf_opt.visualize.annotations import (
    render_vendor_style,
    write_vendor_result_mmd,
)


# ----------------------
# Utilities (learned from models/deepseek-ocr/modeling_deepseekocr.py)
# Source refs in comments
# ----------------------

def load_image(image_path: str) -> Image.Image:
    """Load an image with EXIF orientation handling.

    Mirrors vendor logic (modeling_deepseekocr.py:14–30).

    Parameters
    ----------
    image_path : str
        Path to an image file.

    Returns
    -------
    PIL.Image.Image
        Loaded RGB image with EXIF orientation applied when present.
    """
    try:
        image = Image.open(image_path)
        return ImageOps.exif_transpose(image)
    except Exception:
        return Image.open(image_path)


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Choose the closest aspect ratio grid from candidates.

    Mirrors vendor heuristic (modeling_deepseekocr.py:91–106).
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 2,
    max_num: int = 9,
    image_size: int = 640,
    use_thumbnail: bool = False,
):
    """Split image into a grid of local crops closest to its aspect ratio.

    Mirrors vendor dynamic_preprocess (modeling_deepseekocr.py:108–142).
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


class BasicImageTransform:
    """Basic tensor+normalize transform matching vendor defaults.

    Reference: modeling_deepseekocr.py:196–223
    """
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True):
        transform_pipelines = [transforms.ToTensor()]
        if normalize:
            transform_pipelines.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(transform_pipelines)

    def __call__(self, x: Image.Image) -> torch.Tensor:
        return self.transform(x)


def text_encode(tokenizer, text: str, bos: bool = False, eos: bool = False) -> List[int]:
    """Encode text without special tokens, with optional BOS/EOS insertion.

    Reference: modeling_deepseekocr.py:250–262 (BOS=0, EOS=1).
    """
    t = tokenizer.encode(text, add_special_tokens=False)
    if bos:
        t = [0] + t
    if eos:
        t = t + [1]
    return t


def re_match(text: str):
    """Find vendor <|ref|>/<|det|> spans in model output.

    Reference: modeling_deepseekocr.py:33–51.
    """
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    mathes_image, mathes_other = [], []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width: int, image_height: int):
    """Extract label and normalized coordinates list from a ref span.

    Reference: modeling_deepseekocr.py:53–66.
    """
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception:
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image: Image.Image, refs, output_path: str) -> Image.Image:
    """Draw boxes on a copy of the image and save crops for label==image.

    Simplified reference: modeling_deepseekocr.py:68–138.
    """
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    img_idx = 0
    import numpy as np
    for ref in refs:
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if not result:
                continue
            label_type, points_list = result
            color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
            color_a = color + (20,)
            for points in points_list:
                x1, y1, x2, y2 = points
                x1 = int(x1 / 999 * image_width)
                y1 = int(y1 / 999 * image_height)
                x2 = int(x2 / 999 * image_width)
                y2 = int(y2 / 999 * image_height)
                if label_type == 'image':
                    try:
                        cropped = image.crop((x1, y1, x2, y2))
                        cropped.save(os.path.join(output_path, 'images', f"{img_idx}.jpg"))
                    except Exception:
                        pass
                    img_idx += 1
                try:
                    if label_type == 'title':
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                        draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                    else:
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                    text_x = x1
                    text_y = max(0, y1 - 15)
                    text_bbox = draw.textbbox((0, 0), label_type, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=(255, 255, 255, 30))
                    draw.text((text_x, text_y), label_type, font=font, fill=color)
                except Exception:
                    pass
        except Exception:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def main() -> None:
    """Entry point for vendor-style separate prefill/decode demo."""
    parser = argparse.ArgumentParser(description="DeepSeek-OCR: separate prefill and decode")
    parser.add_argument("--image", type=str, default="models/deepseek-ocr/assets/show1.jpg")
    # Always use vendor-style grounding prompt; the provided text is inserted
    # after <|grounding|> as the user instruction.
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Given the layout of the image. "
            "Extract text and provide bounding boxes for titles, tables, equations, and figures."
        ),
    )
    parser.add_argument("--base_size", type=int, default=1024)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--output_root", type=str, default="tmp")
    parser.add_argument("--crop_mode", action="store_true", default=True)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.output_root, ts)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    model_dir = "models/deepseek-ocr"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer/model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    # Construct vendor-style prompt with grounding tokens
    prompt = f"<|User|>\n<image>\n<|grounding|>{args.prompt.strip()}\n<|Assistant|>"

    # Load and prepare image(s)
    pil_img = load_image(args.image).convert("RGB")
    image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)

    patch_size = 16
    downsample_ratio = 4
    image_token_id = 128815  # modeling_deepseekocr.py:761

    # The original code splits prompt by '<image>' and pairs with images; with plain prompt, we still insert image tokens
    text_splits = prompt.split('<image>')
    tokenized_str: List[int] = []
    images_seq_mask: List[bool] = []
    images_list = []
    images_crop_list = []
    images_spatial_crop = []

    # In original implementation, they iterate over (text_sep, image) pairs. Here single image case.
    for idx, text_sep in enumerate(text_splits[:1]):  # single image
        tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        # crop_mode branch
        if args.crop_mode:
            if pil_img.size[0] <= 640 and pil_img.size[1] <= 640:
                crop_ratio = [1, 1]
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(pil_img)

            # Global view (padded to base_size)
            global_view = ImageOps.pad(
                pil_img,
                (args.base_size, args.base_size),
                color=tuple(int(x * 255) for x in (0.5, 0.5, 0.5)),
            )
            images_list.append(image_transform(global_view).to(torch.bfloat16))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            if width_crop_num > 1 or height_crop_num > 1:
                for im in images_crop_raw:
                    images_crop_list.append(image_transform(im).to(torch.bfloat16))

            # Compose image token pattern (global + local)
            num_queries = math.ceil((args.image_size // patch_size) / downsample_ratio)
            num_queries_base = math.ceil((args.base_size // patch_size) / downsample_ratio)
            tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
            tokenized_image += [image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += (
                    [image_token_id] * (num_queries * width_crop_num) + [image_token_id]
                ) * (num_queries * height_crop_num)
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
        else:
            # Non-crop path (resize to image_size and one grid)
            img_resized = pil_img.resize((args.image_size, args.image_size))
            global_view = ImageOps.pad(
                img_resized, (args.image_size, args.image_size), color=tuple(int(x * 255) for x in (0.5, 0.5, 0.5))
            )
            images_list.append(image_transform(global_view).to(torch.bfloat16))
            width_crop_num, height_crop_num = 1, 1
            images_spatial_crop.append([width_crop_num, height_crop_num])
            num_queries = math.ceil((args.image_size // patch_size) / downsample_ratio)
            tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
            tokenized_image += [image_token_id]
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)

    # Append the last text split per original logic
    tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    # Add BOS
    tokenized_str = [0] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    input_ids = torch.tensor(tokenized_str, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    images_seq_mask_tensor = torch.tensor(images_seq_mask, dtype=torch.bool).unsqueeze(0)
    if len(images_list) == 0:
        images_ori = torch.zeros((1, 3, args.image_size, args.image_size), dtype=torch.bfloat16)
        images_spatial_crop_tensor = torch.zeros((1, 2), dtype=torch.long)
        images_crop = torch.zeros((1, 3, args.base_size, args.base_size), dtype=torch.bfloat16)
    else:
        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)
        if len(images_crop_list) > 0:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, args.base_size, args.base_size), dtype=torch.bfloat16)

    # Move tensors
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    images_seq_mask_tensor = images_seq_mask_tensor.to(device)
    images = [(images_crop.to(device), images_ori.to(device))]

    # ----------------------
    # Prefill: forward once with images to build cache
    # ----------------------
    print("Running prefill ...")
    from time import perf_counter
    t0 = perf_counter()
    with torch.no_grad(), torch.autocast(device_type=(device.type if hasattr(device, 'type') else 'cuda'), dtype=torch.bfloat16):
        outputs_prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            images=images,
            images_seq_mask=images_seq_mask_tensor,
            images_spatial_crop=images_spatial_crop_tensor,
            return_dict=True,
        )
    prefill_ms = (perf_counter() - t0) * 1000.0

    past_kv = outputs_prefill.past_key_values
    last_logits = outputs_prefill.logits[:, -1, :]
    next_token = torch.argmax(last_logits, dim=-1)

    # ----------------------
    # Decode: token-by-token with cache, no images needed
    # ----------------------
    print("Running decode ...")
    t1 = perf_counter()
    generated_tokens = []
    next_input_ids = next_token.unsqueeze(1)
    eos_id = tokenizer.eos_token_id
    max_steps = args.max_new_tokens

    for _ in range(max_steps):
        # extend attention mask for this new token BEFORE preparing inputs
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )

        prepared = model.prepare_inputs_for_generation(
            next_input_ids,
            past_key_values=past_kv,
            attention_mask=attention_mask,
            use_cache=True,
        )
        with torch.no_grad(), torch.autocast(device_type=(device.type if hasattr(device, 'type') else 'cuda'), dtype=torch.bfloat16):
            out = model(**prepared, return_dict=True)
        step_logits = out.logits[:, -1, :]
        next_token = torch.argmax(step_logits, dim=-1)
        generated_tokens.append(next_token)

        # update state
        next_input_ids = next_token.unsqueeze(1)
        past_kv = out.past_key_values
        if eos_id is not None and (next_token == eos_id).all().item():
            break

    if len(generated_tokens) > 0:
        gen = torch.stack(generated_tokens, dim=1)
        text = tokenizer.decode(gen[0].tolist(), skip_special_tokens=False)
    else:
        text = ""
    decode_ms = (perf_counter() - t1) * 1000.0

    # Save outputs
    result_txt = os.path.join(out_dir, "result.txt")
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out_dir, "result.mmd"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved text to {result_txt}")

    # Render vendor-style overlay and write vendor-like markdown, plus JSON summary
    try:
        out_img_path, boxes = render_vendor_style(args.image, text, out_dir)
        mmd_path = write_vendor_result_mmd(text, out_dir)
        print("Wrote annotated image and result.mmd")
    except Exception as e:
        print(f"Annotation render failed: {e}")
        out_img_path, boxes = (None, [])
        mmd_path = None

    # Write info.json similar to runner
    info = {
        "source_image": str(Path(args.image).resolve()),
        "result_image": (Path(out_img_path).name if out_img_path else None),
        "text_raw": text,
        "result_mmd": (Path(mmd_path).name if mmd_path else None),
        "timings_ms": {"prefill": prefill_ms, "decode": decode_ms},
        "tokens": int(len(generated_tokens)),
        "prefill_len": int(input_ids.shape[-1]),
        "boxes": boxes,
    }
    try:
        with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
