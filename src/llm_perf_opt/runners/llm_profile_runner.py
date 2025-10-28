"""CLI entry skeleton for LLM profiling.

This module defines a thin CLI that parses arguments for the LLM profiling
workflow. Core wiring to the profiling session is added in foundational phases.

Functions
---------
build_parser
    Construct the argument parser for the CLI.
main
    Entry point that parses CLI arguments.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build an ``argparse`` parser for the profile runner.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(description="LLM Profile Runner (Stage 1)")
    parser.add_argument("--model-path", required=True, help="Absolute path to model directory")
    parser.add_argument("--input-dir", required=True, help="Absolute path to input images directory")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated passes")
    parser.add_argument("--device", default="cuda:0", help="Target device, e.g., cuda:0")
    parser.add_argument("--use-flash-attn", type=int, default=1, help="1 to enable flash-attn if available")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens for fallback generate")
    return parser


def main() -> None:  # pragma: no cover - thin CLI wrapper
    """Parse CLI arguments.

    Notes
    -----
    Wiring into the profiling session is added in foundational phases; this
    function currently validates argument parsing only.
    """

    parser = build_parser()
    args = parser.parse_args()

    # Wire session in foundational phase: keep it minimal
    from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession

    session = DeepSeekOCRSession.from_local(
        model_path=args.model_path,
        device=args.device,
        use_flash_attn=bool(int(args.use_flash_attn)),
    )

    # Image discovery happens in the next phase; for now just validate session
    _ = session.device


if __name__ == "__main__":  # pragma: no cover
    main()
