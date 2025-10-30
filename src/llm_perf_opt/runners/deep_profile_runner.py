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
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from llm_perf_opt.profiling.artifacts import (
    Artifacts,
    write_config_yaml,
    write_env_json,
    write_inputs_yaml,
)
from llm_perf_opt.profiling.vendor.checks import ensure_ncu, ensure_nsys
from llm_perf_opt.profiling.vendor.launch import build_work_argv
from llm_perf_opt.profiling.vendor.ncu import build_ncu_cmd
from llm_perf_opt.profiling.vendor.nsys import build_nsys_cmd


def _collect_inputs_manifest(work_argv: List[str]) -> list[dict]:
    """Return a minimal inputs manifest for provenance.

    Parameters
    ----------
    work_argv : list of str
        The workload argv used for Stage 1 runner.
    """

    return [{"path": " ".join(work_argv)}]


@hydra.main(version_base=None, config_path="../../../conf/runner", config_name="stage2")
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

    # Prepare artifacts directory (Hydra run.dir is set by stage2.yaml)
    run_dir = Path(HydraConfig.get().run.dir)
    artifacts = Artifacts.from_root(run_dir)

    # Ensure profiler tools are available early with friendly errors
    ensure_nsys()
    ensure_ncu()

    # Build workload argv for Stage 1 runner
    overrides: list[str] = []
    # Disable Stage 1 static analyzer to avoid overhead during Nsight capture
    try:
        if bool(getattr(getattr(cfg, "stage1_runner", {}), "disable_static", True)):
            overrides.append("runners=stage1.no-static")
    except Exception:
        overrides.append("runners=stage1.no-static")
    # Carry common demo overrides (can be changed by user via hydra overrides)
    overrides += [
        "device=cuda:0",
        "repeats=1",
        "infer.max_new_tokens=64",
    ]
    work = build_work_argv("llm_perf_opt.runners.llm_profile_runner", overrides)

    # Nsight Systems capture
    nsys_out = artifacts.path("nsys/run")
    # Use NVTX range gating without label filter to capture prefill→decode
    nsys_cmd = build_nsys_cmd(nsys_out, work, nvtx_capture="range")
    subprocess.run(nsys_cmd, check=False)

    # Nsight Compute capture (focus on decode region)
    ncu_out = artifacts.path("ncu/decode")
    # Focus on decode region using existing NVTX label from the session
    ncu_cmd = build_ncu_cmd(ncu_out, work, nvtx_expr="decode*")
    subprocess.run(ncu_cmd, check=False)

    # Provenance files
    write_env_json(artifacts.path("env.json"))
    write_config_yaml(artifacts.path("config.yaml"), cfg)
    write_inputs_yaml(artifacts.path("inputs.yaml"), _collect_inputs_manifest(work))


if __name__ == "__main__":  # pragma: no cover
    main()
