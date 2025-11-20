"""CLI wrapper for DeepSeek-OCR analytic modeling.

This entry point accepts arguments that mirror the
``DeepSeekOCRAnalyticRequest`` contract and dispatches an analytic run via
the :mod:`llm_perf_opt.runners.dsocr_analyzer` module.

It is intended as a thin façade over ``python -m llm_perf_opt.runners.dsocr_analyzer``
so that external orchestration tooling can bind directly to the contract
shapes defined in ``specs/001-deepseek-ocr-modelmeter/contracts``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_perf_opt.contracts.models import DeepSeekOCRAnalyticRequest
from llm_perf_opt.runners.dsocr_analyzer import AnalysisConfig, DeepseekOCRStaticAnalyzer
from llm_perf_opt.utils.paths import analytic_model_dir, workspace_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR analytic modeling (contract-oriented wrapper).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="Canonical model identifier (matches DeepSeekOCRAnalyticRequest.model_id).",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="deepseek-ocr-v1-base",
        help="Internal model variant (matches DeepSeekOCRAnalyticRequest.model_variant).",
    )
    parser.add_argument(
        "--workload-profile-id",
        type=str,
        default="dsocr-standard-v1",
        help="Workload profile id (matches DeepSeekOCRAnalyticRequest.workload_profile_id).",
    )
    parser.add_argument(
        "--profile-run-id",
        type=str,
        default=None,
        help="Optional Stage1/Stage2 profiling run id used for later validation.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force recomputation even if a matching analytic report already exists.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Local model path (default: <workspace_root>/models/deepseek-ocr).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device specifier for loading the model (e.g., 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier used for output directory naming.",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=1024,
        help="Base padded size for vision analysis (default: 1024).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Image tile size for vision analysis (default: 640).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Representative sequence length for decoder analysis (default: 512).",
    )
    parser.add_argument(
        "--crop-mode",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable dynamic crops (1) or use only the global view (0).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Build a request object to mirror the contract, even though the current
    # implementation uses fixed model_id/model_variant internally.
    _ = DeepSeekOCRAnalyticRequest(
        model_id=args.model_id,
        model_variant=args.model_variant,
        workload_profile_id=args.workload_profile_id,
        profile_run_id=args.profile_run_id,
        force_rebuild=bool(args.force_rebuild),
    )

    # Resolve model path.
    if args.model is None:
        model_path = Path(workspace_root()) / "models" / "deepseek-ocr"
    else:
        model_path = Path(args.model)
    model_path = model_path.resolve()
    if not model_path.exists():
        print(f"ERROR: model path not found: {model_path}", file=sys.stderr)
        return 1

    from llm_perf_opt.runners.dsocr_session import DeepSeekOCRSession

    try:
        session = DeepSeekOCRSession.from_local(
            model_path=str(model_path),
            device=args.device,
            use_flash_attn=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to initialize DeepSeekOCRSession: {exc}", file=sys.stderr)
        return 1

    analyzer = DeepseekOCRStaticAnalyzer(session)
    config = AnalysisConfig(
        image_h=args.base_size,
        image_w=args.base_size,
        base_size=args.base_size,
        image_size=args.image_size,
        seq_len=args.seq_len,
        crop_mode=bool(args.crop_mode),
        use_analytic_fallback=True,
        use_synthetic_inputs=True,
    )

    try:
        report = analyzer.run_analytic(
            config,
            workload_profile_id=args.workload_profile_id,
            run_id=args.run_id,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: analytic modeling failed: {exc}", file=sys.stderr)
        return 1

    out_dir = analytic_model_dir(report.report_id)
    print("DeepSeek-OCR analytic modeling accepted:")
    print(f"  report_id={report.report_id}")
    print(f"  artifacts_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

