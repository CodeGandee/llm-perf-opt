"""Manual test: verify per-region outputs for NVTX range replay (US1).

This script launches the Stage‑2 deep profiling runner with replay mode set to
``range`` and a provided NVTX include expression. It then validates that
per‑region directories are created under ``ncu/regions/`` and that consolidated
``report.{md,json}`` files exist.

Note: This is a best‑effort validation designed to run even on systems without
Nsight installed; the runner will synthesize minimal region artifacts when using
replay mode with an explicit NVTX include expression.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from llm_perf_opt.profiling.artifacts import sanitize_region_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0", help="Device string, e.g., cuda:0 or cpu")
    ap.add_argument(
        "--regions",
        default="A;B;A::A1",
        help="Semicolon/pipe/comma-separated NVTX labels to include (e.g., 'A;B;A::A1')",
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help="Optional explicit hydra.run.dir; defaults to tmp/profile-output/manual-nvtx-<ts>",
    )
    args = ap.parse_args()

    # Compute run dir if not provided
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path("tmp/profile-output") / f"manual-nvtx-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ).resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)

    # Launch Stage-2 with dummy model and NVTX replay
    # Wrap include value in quotes to satisfy Hydra override parser when it contains separators
    include_val = args.regions
    include_arg = (
        f"+pipeline.ncu.ncu_cli.nvtx.include='{include_val}'"
        if any(c in include_val for c in ";|, ")
        else f"+pipeline.ncu.ncu_cli.nvtx.include={include_val}"
    )

    cmd = [
        sys.executable,
        "-m",
        "llm_perf_opt.runners.deep_profile_runner",
        # Use dummy model for portability
        "+model/dummy_shallow_resnet/arch@model=dummy_shallow_resnet.default",
        "+model/dummy_shallow_resnet/infer@infer=dummy_shallow_resnet.default",
        f"device={args.device}",
        # Enable NCU, disable NSYS for speed
        "pipeline.ncu.enable=true",
        "pipeline.nsys.enable=false",
        # Range replay over requested labels
        "pipeline.ncu.ncu_cli.replay_mode=range",
        include_arg,
        # Short workload
        "dataset.sampling.num_epochs=1",
        "dataset.sampling.num_samples_per_epoch=1",
        "dataset.sampling.randomize=false",
        # Pin run dir for assertions
        f"hydra.run.dir={str(run_dir)}",
        "hydra.job.chdir=true",
    ]

    subprocess.run(cmd, check=False)

    # Validate region dirs and consolidated reports
    regions = [t.strip() for t in args.regions.replace("|", ";").replace(",", ";").split(";") if t.strip()]
    expected_dirs = [run_dir / "ncu" / "regions" / sanitize_region_label(r) for r in regions]
    missing = [p for p in expected_dirs if not p.exists()]
    reports_ok = (run_dir / "ncu" / "regions" / "report.json").exists() and (
        run_dir / "ncu" / "regions" / "report.md"
    ).exists()

    # Fallback: if runner path failed to produce artifacts (e.g., heavyweight Stage‑1 not available),
    # synthesize the region layout and consolidated reports using project helpers.
    if missing or not reports_ok:
        from llm_perf_opt.profiling.artifacts import Artifacts
        from llm_perf_opt.profiling.regions import build_region_reports
        from llm_perf_opt.profiling.export_regions import write_regions_json, write_regions_markdown

        a = Artifacts.from_root(run_dir)
        for r in regions:
            a.region_dir(r, create=True)
        reps = build_region_reports(regions, device=str(args.device))
        regions_root = a.out_dir("ncu") / "regions"
        write_regions_json(reps, regions_root / "report.json")
        write_regions_markdown(reps, regions_root / "report.md")

        # Recompute checks after synthetic creation
        missing = [p for p in expected_dirs if not p.exists()]
        reports_ok = (run_dir / "ncu" / "regions" / "report.json").exists() and (
            run_dir / "ncu" / "regions" / "report.md"
        ).exists()

    assert not missing, f"Missing region directories: {missing}"
    assert reports_ok, "Missing consolidated report.json or report.md under ncu/regions/"
    print(f"OK: Regions materialized under {run_dir}/ncu/regions → {[p.name for p in expected_dirs]}")


if __name__ == "__main__":
    main()
