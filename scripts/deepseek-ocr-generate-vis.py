#!/usr/bin/env python3
"""CLI tool to generate visualizations from DeepSeek-OCR prediction results.

Reads predictions.jsonl from a result directory (e.g., tmp/stage1/20251029-053945/)
and generates vendor-style visualizations with bounding boxes and galleries.

Usage:
    python scripts/deepseek-ocr-generate-vis.py -i tmp/stage1/20251029-053945 -o viz_output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from mdutils.mdutils import MdUtils  # type: ignore[import-untyped]

# Import visualization functions from the package
from llm_perf_opt.visualize.annotations import render_vendor_style, write_vendor_result_mmd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from DeepSeek-OCR prediction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input directory containing predictions.jsonl (e.g., tmp/stage1/20251029-053945)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to visualize (default: all)",
    )
    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=480,
        help="Width of thumbnail images in pixels (default: 480)",
    )
    parser.add_argument(
        "--no-gallery",
        action="store_true",
        help="Skip generating the Markdown gallery",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def load_predictions(input_dir: Path) -> list[dict[str, Any]]:
    """Load predictions from predictions.jsonl file.

    Parameters
    ----------
    input_dir : Path
        Directory containing predictions.jsonl

    Returns
    -------
    list[dict[str, Any]]
        List of prediction records

    Raises
    ------
    FileNotFoundError
        If predictions.jsonl is not found
    """
    jsonl_path = input_dir / "predictions.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"predictions.jsonl not found in {input_dir}")

    predictions: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                predictions.append(record)
            except json.JSONDecodeError as e:
                logging.warning("Skipping invalid JSON on line %d: %s", line_num, e)
                continue

    return predictions


def generate_visualizations(
    predictions: list[dict[str, Any]],
    output_dir: Path,
    max_images: int | None,
    thumbnail_width: int,
    make_gallery: bool,
) -> None:
    """Generate vendor-style visualizations and optional gallery.

    Parameters
    ----------
    predictions : list[dict[str, Any]]
        List of prediction records from predictions.jsonl
    output_dir : Path
        Output directory for visualizations
    max_images : int | None
        Maximum number of images to process (None = all)
    thumbnail_width : int
        Width for thumbnail images
    make_gallery : bool
        Whether to generate a Markdown gallery
    """
    logger = logging.getLogger(__name__)

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_root = output_dir / "viz"
    viz_root.mkdir(parents=True, exist_ok=True)
    thumb_dir = viz_root / "_thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # Initialize gallery if needed
    md: MdUtils | None = None
    if make_gallery:
        md = MdUtils(file_name=str((output_dir / "gallery").as_posix()))
        md.new_header(level=1, title="Predictions Gallery")

    # Process each prediction
    count = 0
    for idx, rec in enumerate(predictions):
        if max_images is not None and count >= max_images:
            logger.info("Reached max_images limit (%d), stopping", max_images)
            break

        img_path_str = str(rec.get("image", ""))
        img_path = Path(img_path_str)
        if not img_path.is_file():
            logger.warning("Image not found: %s (skipping)", img_path_str)
            continue

        text_raw = str(rec.get("text_raw", ""))
        text_clean = str(rec.get("text_clean", ""))

        logger.info("Processing %d/%d: %s", idx + 1, len(predictions), img_path.name)

        # Render vendor-style annotations per-image subdir
        annotated_img_rel: Path | None = None
        try:
            per_image_dir = viz_root / img_path.stem
            per_image_dir.mkdir(parents=True, exist_ok=True)
            out_annotated = render_vendor_style(str(img_path), text_raw, str(per_image_dir))
            logger.debug("  Rendered boxes: %s", out_annotated)

            # Write vendor-style result.mmd
            try:
                result_mmd = write_vendor_result_mmd(text_raw, str(per_image_dir))
                logger.debug("  Wrote markdown: %s", result_mmd)
            except Exception as e:
                logger.warning("  Failed to write result.mmd: %s", e)

            annotated_img_rel = out_annotated.relative_to(output_dir)
        except Exception as e:
            logger.error("  Failed to render vendor-style annotations: %s", e)
            annotated_img_rel = None

        # Generate thumbnail and add to gallery
        if make_gallery and md is not None:
            try:
                im = Image.open(img_path).convert("RGB")
                w, h = im.size
                if w > thumbnail_width:
                    ratio = thumbnail_width / float(w)
                    im = im.resize((thumbnail_width, int(h * ratio)))
                thumb_name = img_path.stem + ".jpg"
                thumb_path = thumb_dir / thumb_name
                im.save(thumb_path, format="JPEG", quality=90)
                rel_thumb = thumb_path.relative_to(output_dir)

                # Add to gallery
                md.new_header(level=2, title=img_path.name)
                md.new_paragraph(f"![{img_path.name}]({rel_thumb.as_posix()})")

                if annotated_img_rel is not None:
                    md.new_paragraph("**Annotated (with boxes)**")
                    md.new_paragraph(f"![annotated]({annotated_img_rel.as_posix()})")

            except Exception as e:
                logger.warning("  Failed to generate thumbnail: %s", e)
                md.new_header(level=2, title=img_path.name)

            # Add prediction text
            md.new_paragraph("**Prediction (clean)**")
            md.new_line("```text")
            md.new_line(text_clean)
            md.new_line("```")
            md.new_paragraph("**Raw (with special tokens)**")
            md.new_line("```text")
            md.new_line(text_raw)
            md.new_line("```")

            # Add metrics if available
            metrics_parts = []
            if "prefill_ms" in rec:
                metrics_parts.append(f"prefill={rec['prefill_ms']:.2f}ms")
            if "decode_ms" in rec:
                metrics_parts.append(f"decode={rec['decode_ms']:.2f}ms")
            if "tokens" in rec:
                metrics_parts.append(f"tokens={rec['tokens']}")
            if "tokens_per_s" in rec:
                metrics_parts.append(f"tokens/s={rec['tokens_per_s']:.2f}")
            if metrics_parts:
                md.new_paragraph(f"*Metrics: {', '.join(metrics_parts)}*")

        count += 1

    # Write gallery
    if make_gallery and md is not None:
        md.create_md_file()
        logger.info("Gallery written to: %s", output_dir / "gallery.md")

    logger.info("Processed %d images", count)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Validate inputs
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1

    output_dir = Path(args.output)
    logger.info("Input directory: %s", input_dir)
    logger.info("Output directory: %s", output_dir)

    # Load predictions
    try:
        predictions = load_predictions(input_dir)
        logger.info("Loaded %d predictions from %s", len(predictions), input_dir / "predictions.jsonl")
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    except Exception as e:
        logger.error("Failed to load predictions: %s", e)
        return 1

    if not predictions:
        logger.warning("No predictions to visualize")
        return 0

    # Generate visualizations
    try:
        generate_visualizations(
            predictions=predictions,
            output_dir=output_dir,
            max_images=args.max_images,
            thumbnail_width=args.thumbnail_width,
            make_gallery=not args.no_gallery,
        )
        logger.info("Visualizations complete!")
        logger.info("Output directory: %s", output_dir)
        if not args.no_gallery:
            logger.info("Gallery: %s", output_dir / "gallery.md")
    except Exception as e:
        logger.error("Failed to generate visualizations: %s", e)
        if args.verbose:
            logger.exception("Detailed traceback:")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
