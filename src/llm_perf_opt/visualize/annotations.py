"""Vendor-style annotation rendering for DeepSeek-OCR results.

Parses <|ref|>label</|ref|><|det|>[[x1,y1,x2,y2], ...]</|det|> spans from the
model output text and renders colored boxes with a semi-transparent overlay.
Also saves cropped images for label == 'image' into an images/ subfolder.
"""

from __future__ import annotations

import ast
import random
import re
from pathlib import Path
from typing import Iterable  # noqa: F401

from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-untyped]


_REF_DET_RE = re.compile(
    r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)",
    re.DOTALL,
)


def _parse_spans(text: str) -> list[tuple[str, list[list[int]], str]]:
    """Return list of (label, coords_list, segment_text_after_block).

    segment_text_after_block is the substring from the end of the block until
    the start of the next block (or end-of-text), stripped. This approximates
    the lines that will appear in result.mmd for that region.
    """
    spans: list[tuple[str, list[list[int]], str]] = []
    src = text or ""
    matches = list(_REF_DET_RE.finditer(src))
    for idx, m in enumerate(matches):
        label = m.group(2)
        coords_str = m.group(3)
        try:
            coords = ast.literal_eval(coords_str)
        except Exception:
            continue
        if not isinstance(coords, list):
            continue
        next_start = matches[idx + 1].start(0) if (idx + 1) < len(matches) else len(src)
        seg_text = src[m.end(0):next_start].strip()
        spans.append((str(label).strip(), coords, seg_text))
    return spans


def render_vendor_style(image_path: str, result_text: str, output_dir: str) -> tuple[Path, list[dict]]:
    """Render vendor-style overlay and crops.

    Parameters
    ----------
    image_path : str
        Path to the original image.
    result_text : str
        Model output text containing <|ref|>/<|det|> spans.
    output_dir : str
        Directory to write outputs into. This function writes:
        - result_with_boxes.jpg
        - images/ (crops for label == 'image')

    Returns
    -------
    tuple[Path, list[dict]]
        (Path to the output annotated image, list of box dicts)
        Each box dict has keys: 'x', 'y', 'w', 'h' (pixel ints in original
        image coordinates) and 'text' (recognized/label text for the box).
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Try to load a default font; fallback if not available
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover - very unlikely
        font = None  # type: ignore[assignment]

    spans = _parse_spans(result_text)
    boxes: list[dict] = []
    crop_idx = 0
    for label, coords_list, seg_text in spans:
        # Soft color palette
        base = (
            random.randint(80, 200),
            random.randint(80, 200),
            random.randint(100, 240),
        )
        fill_a = base + (32,)  # semi-transparent fill
        outline_w = 4 if label == "title" else 2
        for coords in coords_list:
            if not (isinstance(coords, (list, tuple)) and len(coords) == 4):
                continue
            x1, y1, x2, y2 = coords
            # Vendor mapping uses normalization by 999
            try:
                x1p = int(x1 / 999.0 * W)
                y1p = int(y1 / 999.0 * H)
                x2p = int(x2 / 999.0 * W)
                y2p = int(y2 / 999.0 * H)
            except Exception:
                continue

            # Filled overlay + outline
            draw_overlay.rectangle([x1p, y1p, x2p, y2p], fill=fill_a)
            draw.rectangle([x1p, y1p, x2p, y2p], outline=base, width=outline_w)

            # Label text box (small)
            if font is not None and label:
                try:
                    tw, th = draw.textbbox((0, 0), label, font=font)[2:]
                except Exception:
                    tw = th = 0
                pad = 2
                draw.rectangle([x1p, max(0, y1p - th - 2 * pad), x1p + tw + 2 * pad, y1p], fill=(255, 255, 255))
                draw.text((x1p + pad, max(0, y1p - th - pad)), label, fill=base, font=font)

            # Save crops for label == image
            if label == "image":
                try:
                    crop = im.crop((x1p, y1p, x2p, y2p))
                    crop.save(images_dir / f"{crop_idx}.jpg")
                    crop_idx += 1
                except Exception:
                    pass

            # Record box structure
            try:
                # Normalize a couple vendor tokens similar to write_vendor_result_mmd
                _txt = (seg_text or "").replace("\\\\coloneqq", ":=").replace("\\\\eqqcolon", "=:")
                boxes.append(
                    {
                        "x": int(x1p),
                        "y": int(y1p),
                        "w": int(max(0, x2p - x1p)),
                        "h": int(max(0, y2p - y1p)),
                        "text": _txt,
                    }
                )
            except Exception:
                pass

    # Composite overlay
    im_rgba = im.convert("RGBA")
    im_rgba.alpha_composite(overlay)
    out_path = out_dir / "result_with_boxes.jpg"
    im_rgba.convert("RGB").save(out_path, format="JPEG", quality=90)
    return out_path, boxes


def write_vendor_result_mmd(result_text: str, output_dir: str) -> Path:
    """Write a vendor-like ``result.mmd`` replacing image refs with images.

    This mirrors the logic in the vendor's ``infer(..., save_results=True)``:
    - Replace image refs with ``![](images/<idx>.jpg)``
    - Remove other refs
    - Normalize a couple of LaTeX tokens

    Parameters
    ----------
    result_text : str
        Raw decoded text from model output (keep special tokens).
    output_dir : str
        Directory containing an ``images/`` subdir populated by ``render_vendor_style``.

    Returns
    -------
    Path
        Path to the written ``result.mmd`` file.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Match full ref spans and split by label "image" vs others
    matches = list(_REF_DET_RE.findall(result_text or ""))
    matches_image: list[str] = []
    matches_other: list[str] = []
    for full, label, _coords in matches:
        if "<|ref|>image<|/ref|>" in full:
            matches_image.append(full)
        else:
            matches_other.append(full)

    out = (result_text or "").strip()
    # Replace image refs in order with links to saved images
    for idx, m in enumerate(matches_image):
        out = out.replace(m, f"![](images/{idx}.jpg)\n")
    # Remove non-image refs and normalize a couple of tokens used upstream
    for m in matches_other:
        out = out.replace(m, "")
    out = out.replace("\\\\coloneqq", ":=").replace("\\\\eqqcolon", "=:")

    p = out_dir / "result.mmd"
    p.write_text(out, encoding="utf-8")
    return p
