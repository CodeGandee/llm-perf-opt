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
from llm_perf_opt.profiling.kernels import parse_ncu_csv, kernels_from_ncu_rows
from llm_perf_opt.profiling.export import (
    top_n_kernels,
    write_kernel_markdown,
    write_stakeholder_summary,
)


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
    # Repeats and dataset subset for Stage 1 workload
    stage1_repeats = 1
    try:
        stage1_repeats = int(getattr(getattr(cfg, "run", {}), "stage1_repeats", 1))
    except Exception:
        stage1_repeats = 1
    overrides += [
        f"device={device_sel}",
        f"repeats={stage1_repeats}",
        "infer.max_new_tokens=64",
        # Avoid CUPTI conflicts under Nsight Systems
        "torch_profiler.enabled=false",
    ]
    try:
        subset_filelist = getattr(getattr(cfg, "run", {}), "dataset_subset_filelist", None)
        if subset_filelist:
            overrides.append(f"dataset.subset_filelist={subset_filelist}")
    except Exception:
        pass
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
    # Use NVTX gating unless disabled via config
    gating_nvtx_nsys = bool(getattr(getattr(cfg, "nsys", {}), "gating_nvtx", True))
    nsys_cmd = build_nsys_cmd(
        nsys_out,
        work,
        nvtx_capture=str(getattr(getattr(cfg, "nsys", {}), "nvtx_capture", "range")) if gating_nvtx_nsys else "none",
        trace=str(getattr(getattr(cfg, "nsys", {}), "trace", "cuda,nvtx,osrt")),
        sample=str(getattr(getattr(cfg, "nsys", {}), "sample", "none")),
        capture=str(getattr(getattr(cfg, "nsys", {}), "capture", "nvtx")) if gating_nvtx_nsys else "none",
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
    gating_nvtx_ncu = bool(getattr(getattr(cfg, "ncu", {}), "gating_nvtx", True))
    ncu_cmd = build_ncu_cmd(
        ncu_out,
        work,
        nvtx_expr=str(getattr(getattr(cfg, "ncu", {}), "nvtx_include", "decode*")),
        kernel_regex=kernel_regex,
        csv_log=csv_log,
        use_nvtx=gating_nvtx_ncu,
    )
    subprocess.run(ncu_cmd, check=False)

    # Generate kernels.md from Nsight Compute CSV (if present)
    try:
        ncu_csv_path = artifacts.path("ncu/raw.csv")
        topkern_list = None
        if Path(ncu_csv_path).exists():
            ncu_rows = parse_ncu_csv(ncu_csv_path)
            device_for_records = device_sel
            krecs = kernels_from_ncu_rows(ncu_rows, device=device_for_records)
            topkern_list = top_n_kernels(krecs, n=int(getattr(getattr(cfg, "run", {}), "top_n_kernels", 20)))
            write_kernel_markdown(topkern_list, str(artifacts.path("kernels.md")), top_k=len(topkern_list))
    except Exception:
        # Best-effort: do not block run completion on export failures
        pass

    # Provenance files
    write_env_json(artifacts.path("env.json"))
    write_config_yaml(artifacts.path("config.yaml"), cfg)
    write_inputs_yaml(artifacts.path("inputs.yaml"), _collect_inputs_manifest(work))

    # US3: Stakeholder summary and Stage 2 report
    try:
        import json as _json
        # Load Stage 1 metrics for aggregates/MFU
        stats_dict: dict | None = None
        metrics_path = artifacts.path("stage1/metrics.json")
        if Path(metrics_path).exists():
            with open(metrics_path, "r", encoding="utf-8") as mf:
                m = _json.load(mf)
            aggr = m.get("aggregates", {}) if isinstance(m.get("aggregates"), dict) else {}
            stats_dict = {
                "aggregates": aggr,
                "mfu_model": float(m.get("mfu_model_level", 0.0)),
                "mfu_stages": m.get("mfu_per_stage", {}) if isinstance(m.get("mfu_per_stage"), dict) else {},
                "peak_tflops": float(m.get("peak_tflops", 0.0)),
                "device": "",
            }
        # Heuristic stage takeaways
        stage_msgs: dict[str, str] = {}
        try:
            if stats_dict is not None:
                ag = stats_dict.get("aggregates", {}) if isinstance(stats_dict.get("aggregates"), dict) else {}
                pf = float(ag.get("prefill_ms", {}).get("mean", 0.0))
                dc = float(ag.get("decode_ms", {}).get("mean", 0.0))
                tps = float(ag.get("tokens_per_s", {}).get("mean", 0.0))
                mfu = stats_dict.get("mfu_stages", {}) if isinstance(stats_dict.get("mfu_stages"), dict) else {}
                mfu_d = float(mfu.get("decode", 0.0))
                mfu_p = float(mfu.get("prefill", 0.0))
                if dc >= max(1.0, pf) * 1.2:
                    stage_msgs["decode"] = (
                        f"Decode dominates runtime (≈ {dc:.1f} ms per run). Tokens/s ≈ {tps:.2f}; MFU(decode) ≈ {mfu_d:.6f}."
                    )
                elif pf >= max(1.0, dc) * 1.2:
                    stage_msgs["prefill"] = (
                        f"Prefill dominates runtime (≈ {pf:.1f} ms per run). MFU(prefill) ≈ {mfu_p:.6f}."
                    )
                else:
                    stage_msgs["decode"] = (
                        f"Decode and prefill comparable (decode ≈ {dc:.1f} ms). Tokens/s ≈ {tps:.2f}; MFU(decode) ≈ {mfu_d:.6f}."
                    )
        except Exception:
            stage_msgs = {}

        write_stakeholder_summary(
            str(artifacts.path("stakeholder_summary.md")),
            top_ops=[],  # Stage 2 does not collect operators; see Stage 1 operators.md
            stage_takeaways=stage_msgs,
            stats=stats_dict,
            top_kernels=topkern_list if 'topkern_list' in locals() else None,
        )

        # Lightweight report for Stage 2
        rep_path = artifacts.path("report.md")
        lines = [
            "# Stage 2 Profiling Report",
            "",
            "Artifacts:",
            f"- NSYS: {artifacts.path('nsys/run.nsys-rep').name} (if present), summary CSV",
            f"- NCU: {artifacts.path('ncu/raw.csv').name}",
            f"- Top kernels: {artifacts.path('kernels.md').name}",
            f"- Stakeholder: {artifacts.path('stakeholder_summary.md').name}",
            f"- Stage 1 report: {artifacts.path('stage1/report.md').name} (see Stage 1 dir)",
            "",
        ]
        try:
            if stats_dict is not None:
                ag = stats_dict.get("aggregates", {}) if isinstance(stats_dict.get("aggregates"), dict) else {}
                lines += [
                    "## Per-Stage Timings (ms)",
                    (
                        f"- Prefill: mean={float(ag.get('prefill_ms',{}).get('mean',0.0)):.3f}, "
                        f"std={float(ag.get('prefill_ms',{}).get('std',0.0)):.3f}"
                    ),
                    (
                        f"- Decode: mean={float(ag.get('decode_ms',{}).get('mean',0.0)):.3f}, "
                        f"std={float(ag.get('decode_ms',{}).get('std',0.0)):.3f}"
                    ),
                    "",
                    "## MFU",
                    f"- Model-level: {float(stats_dict.get('mfu_model',0.0)):.6f}",
                ]
        except Exception:
            pass
        rep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
