"""Stage 2 deep profiling runner (Hydra entry).

Runs a deep profiling session by launching the Stage 1 runner as the workload
under NVIDIA Nsight Systems and Nsight Compute. Artifacts are organized under
``tmp/stage2/<run_id>/`` with subfolders for ``nsys/`` and ``ncu/`` plus
provenance files.

This runner expects:
- Nsight Systems (`nsys`) and Nsight Compute (`ncu`) on PATH
- A working Stage 1 runner module: ``llm_perf_opt.runners.llm_profile_runner``

Examples
--------
Run with Pixi task (recommended):
    pixi run stage2-profile

Direct Python (for debugging):
    python -m llm_perf_opt.runners.deep_profile_runner
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from llm_perf_opt.profiling.artifacts import (
    Artifacts,
    write_config_yaml,
    write_env_json,
    write_inputs_yaml,
)
from llm_perf_opt.profiling.vendor.checks import ensure_ncu, ensure_nsys
from llm_perf_opt.profiling.vendor.launch import build_work_argv
from llm_perf_opt.profiling.vendor.ncu import build_ncu_cmd
from llm_perf_opt.profiling.vendor.nsys import (
    build_nsys_cmd,
    build_nsys_export_sqlite_cmd,
    build_nsys_stats_cmd,
)
from llm_perf_opt.profiling.nsys_stats import top_kernels_from_nsys_summary


def _collect_inputs_manifest(work_argv: List[str]) -> list[dict]:
    """Return a minimal inputs manifest for provenance.

    Parameters
    ----------
    work_argv : list of str
        The workload argv used for Stage 1 runner.
    """

    return [{"path": " ".join(work_argv)}]


@hydra.main(version_base=None, config_path="../../../conf", config_name="runner/stage2")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI orchestrator
    """Hydra entry point for Stage 2 deep profiling.

    Flow
    ----
    1) Prepare artifacts directory (Hydra run dir).
    2) Ensure vendor tools are present.
    3) Build workload argv for Stage 1 runner with overrides.
    4) Run Nsight Systems → ``nsys/`` outputs.
    5) Run Nsight Compute → ``ncu/`` outputs.
    6) Write provenance files.
    """

    # Prepare artifacts in the Hydra run.dir so both stages share one main dir
    main_dir = Path(HydraConfig.get().run.dir)
    artifacts = Artifacts.from_root(main_dir)

    # Ensure profiler tools are available early with friendly errors
    ensure_nsys()
    ensure_ncu()

    # Build workload argv for Stage 1 runner
    overrides: list[str] = []
    # Disable Stage 1 static analyzer to avoid overhead during Nsight capture
    try:
        if bool(getattr(getattr(cfg, "stage1_runner", {}), "disable_static", True)):
            overrides.append("runner@stage1_runner=stage1.no-static")
    except Exception:
        overrides.append("runner@stage1_runner=stage1.no-static")
    # Carry common demo overrides (can be changed by user via hydra overrides)
    device_sel = "cuda:0"
    try:
        device_sel = str(getattr(getattr(cfg, "stage1_runner", {}), "device", "cuda:0"))
    except Exception:
        device_sel = "cuda:0"
    overrides += [
        f"device={device_sel}",
        "repeats=1",
        "infer.max_new_tokens=64",
        # Avoid CUPTI conflicts under Nsight Systems
        "torch_profiler.enabled=false",
    ]
    work = build_work_argv(
        "llm_perf_opt.runners.llm_profile_runner",
        overrides,
        hydra_run_dir=str(artifacts.path("stage1")),
        chdir=True,
        run_mode=str(getattr(getattr(cfg, "run", {}), "mode", "deep")),
        inputs_manifest=None,
    )

    # Nsight Systems capture
    nsys_out = artifacts.path("nsys/run")
    # Use NVTX range gating without label filter to capture prefill→decode
    nsys_cmd = build_nsys_cmd(
        nsys_out,
        work,
        nvtx_capture=str(getattr(getattr(cfg, "nsys", {}), "nvtx_capture", "range")),
        trace=str(getattr(getattr(cfg, "nsys", {}), "trace", "cuda,nvtx,osrt")),
        sample=str(getattr(getattr(cfg, "nsys", {}), "sample", "none")),
        capture=str(getattr(getattr(cfg, "nsys", {}), "capture", "nvtx")),
    )
    subprocess.run(nsys_cmd, check=False)
    # Export stats CSV + SQLite for post-processing
    # Resolve report path and export stats/sqlite if available
    from llm_perf_opt.profiling.vendor.nsys import resolve_nsys_report_path
    report_path = resolve_nsys_report_path(nsys_out)
    if report_path is not None:
        nsys_summary_base = artifacts.path("nsys/summary")
        nsys_stats_cmd = build_nsys_stats_cmd(report_path, nsys_summary_base)
        subprocess.run(nsys_stats_cmd, check=False)
        nsys_sqlite_cmd = build_nsys_export_sqlite_cmd(report_path)
        subprocess.run(nsys_sqlite_cmd, check=False)

    # Nsight Compute capture (focus on decode region)
    ncu_out = artifacts.path("ncu/decode")
    # Probe available sections once and save to file for debugging/compat
    try:
        sections_txt = artifacts.path("ncu/sections.txt")
        with open(sections_txt, "w", encoding="utf-8") as sf:
            subprocess.run(["ncu", "--list-sections"], stdout=sf, stderr=subprocess.STDOUT, check=False)
    except Exception:
        pass
    # Select top-N kernels from nsys summary and focus on decode region
    kernel_regex = None
    try:
        top_n = int(getattr(getattr(cfg, "run", {}), "top_n_kernels", 30))
        names = top_kernels_from_nsys_summary(Path(str(nsys_summary_base) + ".csv"), top_n=top_n)
        if names:
            import re as _re
            pats = ["(" + _re.escape(n) + ")" for n in names]
            kernel_regex = "|".join(pats)
    except Exception:
        kernel_regex = None
    csv_log = artifacts.path("ncu/raw.csv")
    ncu_cmd = build_ncu_cmd(
        ncu_out,
        work,
        nvtx_expr=str(getattr(getattr(cfg, "ncu", {}), "nvtx_include", "decode*")),
        kernel_regex=kernel_regex,
        csv_log=csv_log,
    )
    subprocess.run(ncu_cmd, check=False)

    # Provenance files
    write_env_json(artifacts.path("env.json"))
    write_config_yaml(artifacts.path("config.yaml"), cfg)
    write_inputs_yaml(artifacts.path("inputs.yaml"), _collect_inputs_manifest(work))


if __name__ == "__main__":  # pragma: no cover
    main()
