#!/usr/bin/env python
"""Inspect DeepSeek-OCR model structure to find vision modules."""

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession


def inspect_model(model, prefix="", max_depth=3, current_depth=0):
    """Recursively inspect model structure."""
    if current_depth >= max_depth:
        return

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(f"{'  ' * current_depth}{full_name}: {type(module).__name__}")

        # Check for vision-related modules
        if any(keyword in name.lower() for keyword in ['sam', 'vision', 'clip', 'projector']):
            print(f"{'  ' * current_depth}  -> VISION MODULE FOUND: {full_name}")
            print(f"{'  ' * current_depth}     Type: {type(module)}")
            print(f"{'  ' * current_depth}     Has parameters: {sum(p.numel() for p in module.parameters())}")

        # Recurse
        inspect_model(module, full_name, max_depth, current_depth + 1)


def main():
    print("Loading DeepSeek-OCR model...")
    session = DeepSeekOCRSession.from_local(
        model_path=str(repo_root / "models" / "deepseek-ocr"),
        device="cuda:1",
        use_flash_attn=True,
    )

    print("\n" + "=" * 80)
    print("Model Structure")
    print("=" * 80)

    # Check top-level attributes
    print("\nTop-level model type:", type(session.m_model).__name__)
    print("Top-level attributes:")
    for attr in dir(session.m_model):
        if not attr.startswith('_') and not callable(getattr(session.m_model, attr, None)):
            obj = getattr(session.m_model, attr, None)
            if hasattr(obj, 'parameters'):
                print(f"  {attr}: {type(obj).__name__}")

    # Check if there's a .model attribute
    if hasattr(session.m_model, 'model'):
        print("\nModel has .model attribute:", type(session.m_model.model).__name__)
        print(".model attributes:")
        for attr in dir(session.m_model.model):
            if not attr.startswith('_') and not callable(getattr(session.m_model.model, attr, None)):
                obj = getattr(session.m_model.model, attr, None)
                if hasattr(obj, 'parameters'):
                    print(f"  {attr}: {type(obj).__name__}")

    print("\n" + "=" * 80)
    print("Hierarchical Structure (depth=3)")
    print("=" * 80 + "\n")

    inspect_model(session.m_model, max_depth=3)

    print("\n" + "=" * 80)
    print("Attempting Direct Access")
    print("=" * 80 + "\n")

    # Try different attribute paths
    paths_to_try = [
        "sam_model",
        "model.sam_model",
        "vision_model",
        "model.vision_model",
        "projector",
        "model.projector",
    ]

    for path in paths_to_try:
        parts = path.split('.')
        obj = session.m_model
        try:
            for part in parts:
                obj = getattr(obj, part)
            print(f"✓ {path}: {type(obj).__name__} ({sum(p.numel() for p in obj.parameters()) / 1e6:.2f}M params)")
        except AttributeError:
            print(f"✗ {path}: Not found")


if __name__ == "__main__":
    main()
