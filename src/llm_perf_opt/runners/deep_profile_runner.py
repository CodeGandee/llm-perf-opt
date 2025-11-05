"""Stage‑2 Deep Profiling Runner (Hydra entry).

Runs a deep profiling session by launching the Stage‑1 runner as the workload
under NVIDIA Nsight Systems and Nsight Compute. All artifacts are written under
the Hydra run directory (e.g., ``tmp/profile-output/<run_id>/``) with
subfolders for ``nsys/`` and ``ncu/`` plus provenance files.

Requirements
------------
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


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
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

    # Prepare artifacts in the Hydra run.dir so all pipeline stages share one main dir
    run_dir_cfg = Path(HydraConfig.get().run.dir)
    base_cwd = Path(HydraConfig.get().runtime.cwd)
    main_dir = run_dir_cfg if run_dir_cfg.is_absolute() else (base_cwd / run_dir_cfg)
    artifacts = Artifacts.from_root(main_dir)

    # Ensure required profiler tools are available based on pipeline toggles
    try:
        if bool(getattr(getattr(cfg, "pipeline", {}), "nsys", {}).get("enable", False)):
            ensure_nsys()
    except Exception:
        raise
    try:
        if bool(getattr(getattr(cfg, "pipeline", {}), "ncu", {}).get("enable", False)):
            ensure_ncu()
    except Exception:
        raise

    # Build workload argv for Stage 1 runner
    overrides: list[str] = []
    # Carry common demo overrides (can be changed by user via hydra overrides)
    device_sel = str(getattr(cfg, "device", "cuda:0"))
    # Repeats and dataset subset for Stage 1 workload
    stage1_repeats = 1
    try:
        stage1_repeats = int(getattr(getattr(cfg, "run", {}), "stage1_repeats", 1))
    except Exception:
        stage1_repeats = 1
    # Force model.path to repo-absolute path to avoid Hydra CWD ambiguity under nested runs
    try:
        repo_root = Path(HydraConfig.get().runtime.cwd)
        model_path_abs = str(repo_root / "models" / "deepseek-ocr")
        ds_root_abs = str(repo_root / "datasets" / "omnidocbench" / "source-data")
        overrides.append(f"model.path={model_path_abs}")
        overrides.append(f"dataset.root={ds_root_abs}")
    except Exception:
        pass

    # Reuse Stage-1 infer.max_new_tokens when available; allow a run-level override
    try:
        rep_mnt = getattr(getattr(cfg, "run", {}), "representative_max_new_tokens", None)
    except Exception:
        rep_mnt = None
    try:
        if rep_mnt in (None, "null"):
            rep_mnt = getattr(getattr(cfg, "infer", {}), "max_new_tokens", 64)
    except Exception:
        rep_mnt = 64
    try:
        rep_mnt = int(rep_mnt)  # enforce integer-only policy (Stage-1 runner enforces the same)
    except Exception:
        rep_mnt = 64

    overrides += [
        f"device={device_sel}",
        f"infer.max_new_tokens={rep_mnt}",
        # Avoid CUPTI conflicts under Nsight Systems (disable PyTorch profiler in workload)
        "pipeline.torch_profiler.enable=false",
        # Disable static analysis during Nsight capture
        "pipeline.static_analysis.enable=false",
        # Explicit sampling plan for Stage-1 workload (map legacy stage1_repeats → samples/epoch)
        # Note: preset is already mounted via conf/config.yaml; no need to append the dataset/sampling preset here.
        "dataset.sampling.num_epochs=1",
        f"dataset.sampling.num_samples_per_epoch={stage1_repeats}",
        "dataset.sampling.randomize=false",
    ]
    try:
        subset_filelist = getattr(getattr(cfg, "run", {}), "dataset_subset_filelist", None)
        if subset_filelist:
            overrides.append(f"dataset.subset_filelist={subset_filelist}")
    except Exception:
        pass
    # For NSYS/NCU captures, run the internal workload in a temp area, not a user-facing pipeline dir
    workload_dir = artifacts.tmp_dir("workload")
    work = build_work_argv(
        "llm_perf_opt.runners.llm_profile_runner",
        overrides,
        hydra_run_dir=str(workload_dir),
        chdir=True,
        # run_mode is already provided in conf/config.yaml under run.mode
        run_mode=None,
        inputs_manifest=None,
    )

    # Predeclare NSYS summary base for later post-processing (CSV may or may not exist)
    nsys_summary_base = artifacts.out_dir("nsys") / "summary"

    # Nsight Systems capture (guarded by pipeline.nsys.enable)
    if bool(getattr(getattr(cfg, "pipeline", {}), "nsys", {}).get("enable", False)):
        nsys_out = artifacts.out_dir("nsys") / "run"
        # Use capture_range semantics matching Nsight Systems CLI. When
        # `capture_range=nvtx`, an explicit `nvtx_capture` is REQUIRED; empty/omitted
        # is an error (nsys default is 'none' which never triggers capture).
        nsys_cfg = getattr(getattr(cfg, "pipeline", {}), "nsys", {})
        gating_nvtx_nsys = bool(getattr(nsys_cfg, "gating_nvtx", True))
        try:
            _nvx_raw = getattr(nsys_cfg, "nvtx_capture", None)
        except Exception:
            _nvx_raw = None
        _nvx_str = None if _nvx_raw is None else str(_nvx_raw)
        _nvx_empty = isinstance(_nvx_raw, str) and (_nvx_str.strip() == "")
        # Align with official args: only NVTX gating uses nvtx_capture
        # Default to 'none' (capture all) to avoid empty reports when NVTX labels
        # are missing or misconfigured.
        _cap_range_cfg = str(getattr(nsys_cfg, "capture_range", "none")).lower()
        # Optional: capture-range-end behavior; omit if unset/empty
        try:
            _cap_end_cfg_raw = getattr(nsys_cfg, "capture_range_end", None)
        except Exception:
            _cap_end_cfg_raw = None
        _cap_end_cfg = None if _cap_end_cfg_raw is None else str(_cap_end_cfg_raw).strip() or None

        # NVTX gating fallback: if capture_range=nvtx but nvtx_capture is omitted/empty,
        # fall back to capture all (none) to avoid empty reports. Log a warning.
        _use_nvtx_gate = False
        if gating_nvtx_nsys and _cap_range_cfg == "nvtx":
            if (_nvx_str is None) or _nvx_empty:
                try:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "Nsight Systems: capture_range=nvtx but nvtx_capture omitted/empty; "
                        "falling back to capture_range=none (capture all) to avoid empty report."
                    )
                except Exception:
                    pass
                _cap_range_cfg = "none"
            else:
                _use_nvtx_gate = True

        # Determine final capture-range to pass to NSYS
        _cap_final = str(_cap_range_cfg) if gating_nvtx_nsys else "none"
        # If gating is disabled, force none regardless of config
        if not gating_nvtx_nsys:
            _cap_final = "none"

        # Convert optional sampling/capture settings: pass None to omit flags
        _sample_cfg_raw = getattr(nsys_cfg, "sample", None)
        _sample_arg = None
        try:
            if isinstance(_sample_cfg_raw, str) and _sample_cfg_raw.strip().lower() != "none":
                _sample_arg = str(_sample_cfg_raw)
        except Exception:
            _sample_arg = None
        _capture_arg = None if _cap_final == "none" else _cap_final

        nsys_cmd = build_nsys_cmd(
            nsys_out,
            work,
            # When using NVTX gating, nvtx_capture must be explicitly provided.
            nvtx_capture=(_nvx_str if (_use_nvtx_gate and _cap_final == "nvtx") else None),
            trace=str(getattr(nsys_cfg, "trace", "cuda,nvtx,osrt")),
            sample=_sample_arg,
            capture=_capture_arg,
            capture_end=_cap_end_cfg if (_cap_final != "none") else None,
        )
        # Persist the constructed NSYS command for reproducibility/triage
        try:
            import shlex as _shlex
            (artifacts.out_dir("nsys") / "cmd.txt").write_text(_shlex.join(nsys_cmd) + "\n", encoding="utf-8")
        except Exception:
            pass
        subprocess.run(nsys_cmd, check=False)
        # Export stats CSV + SQLite for post-processing
        # Resolve report path and export stats/sqlite if available
        from llm_perf_opt.profiling.vendor.nsys import resolve_nsys_report_path
        report_path = resolve_nsys_report_path(nsys_out)
        if report_path is not None:
            nsys_stats_cmd = build_nsys_stats_cmd(report_path, nsys_summary_base)
            subprocess.run(nsys_stats_cmd, check=False)
            nsys_sqlite_cmd = build_nsys_export_sqlite_cmd(report_path)
            subprocess.run(nsys_sqlite_cmd, check=False)

    # Nsight Compute capture (focus on decode region)
    ncu_out = artifacts.out_dir("ncu") / "decode"
    # Probe available sections once and save to file for debugging/compat
    try:
        sections_txt = artifacts.out_dir("ncu") / "sections.txt"
        with open(sections_txt, "w", encoding="utf-8") as sf:
            subprocess.run(["ncu", "--list-sections"], stdout=sf, stderr=subprocess.STDOUT, check=False)
    except Exception:
        pass
    # Select top-N kernels from nsys summary (if present) and focus on decode region
    kernel_regex = None
    try:
        top_n = int(getattr(getattr(cfg, "run", {}), "top_n_kernels", 30))
    except Exception:
        top_n = 30
    try:
        csv_candidate = Path(str(nsys_summary_base) + ".csv")
        if csv_candidate.exists():
            names = top_kernels_from_nsys_summary(csv_candidate, top_n=top_n)
            if names:
                import re as _re
                pats = ["(" + _re.escape(n) + ")" for n in names]
                kernel_regex = "|".join(pats)
    except Exception:
        kernel_regex = None
    ncu_cfg = getattr(getattr(cfg, "pipeline", {}), "ncu", {})
    ncu_cli = getattr(ncu_cfg, "ncu_cli", {})
    gating_nvtx_ncu = bool(getattr(ncu_cfg, "gating_nvtx", True))
    # Read NCU config knobs from new schema
    try:
        nvtx_val = getattr(ncu_cli, "nvtx", None)
        nvtx_expr = None
        if isinstance(nvtx_val, dict):
            inc = nvtx_val.get("include", None)
            if inc is not None and str(inc).strip():
                nvtx_expr = str(inc).strip()
    except Exception:
        nvtx_expr = None
    try:
        ncu_set = str(getattr(ncu_cli, "set", "roofline"))
    except Exception:
        ncu_set = "roofline"
    # metrics may be list[str] or None
    try:
        ncu_metrics = getattr(ncu_cli, "metrics", None)
    except Exception:
        ncu_metrics = None
    # sections optional list[str]
    ncu_sections = None
    try:
        secs = getattr(ncu_cli, "sections", None)
        if isinstance(secs, (list, tuple)):
            ncu_sections = [str(s) for s in secs if s]
    except Exception:
        ncu_sections = None
    # target processes optional
    try:
        ncu_target_procs = str(getattr(ncu_cli, "target_processes", "all"))
    except Exception:
        ncu_target_procs = "all"
    # force overwrite optional (default true per preset)
    try:
        ncu_force = bool(getattr(ncu_cli, "force_overwrite", True))
    except Exception:
        ncu_force = True

    if bool(getattr(getattr(cfg, "pipeline", {}), "ncu", {}).get("enable", False)):
        ncu_cmd = build_ncu_cmd(
            ncu_out,
            work,
            nvtx_expr=nvtx_expr,
            kernel_regex=kernel_regex,
            csv_log=csv_log,
            use_nvtx=gating_nvtx_ncu,
            set_name=ncu_set,
            metrics=ncu_metrics,
            sections=ncu_sections,
            target_processes=ncu_target_procs,
            force_overwrite=ncu_force,
        )
        # Persist the constructed NCU command as well
        try:
            import shlex as _shlex
            (artifacts.out_dir("ncu") / "cmd.txt").write_text(_shlex.join(ncu_cmd) + "\n", encoding="utf-8")
        except Exception:
            pass
        subprocess.run(ncu_cmd, check=False)

    # If sections were requested, import the .ncu-rep and render sections to a text report
    if bool(getattr(getattr(cfg, "pipeline", {}), "ncu", {}).get("enable", False)):
        try:
            if ncu_sections:
                ncu_rep = Path(str(ncu_out) + ".ncu-rep")
                if ncu_rep.exists():
                    from llm_perf_opt.profiling.vendor.ncu import build_ncu_import_sections_cmd
                    sec_cmd = build_ncu_import_sections_cmd(ncu_rep, ncu_sections, page="raw")
                    sec_out = artifacts.path("ncu/sections_report.txt")
                    with open(sec_out, "w", encoding="utf-8") as sf:
                        subprocess.run(sec_cmd, check=False, stdout=sf, stderr=subprocess.STDOUT)
        except Exception:
            pass

    # Fallback: if NCU reported no kernels, rerun without NVTX gating and without sections
    if bool(getattr(getattr(cfg, "pipeline", {}), "ncu", {}).get("enable", False)):
        try:
            ncu_csv_path = artifacts.out_dir("ncu") / "raw.csv"
            rerun = False
            if Path(ncu_csv_path).exists():
                head = Path(ncu_csv_path).read_text(encoding="utf-8", errors="ignore")[:1000]
                if "No kernels were profiled" in head:
                    rerun = True
            if rerun:
                ncu_cmd2 = build_ncu_cmd(
                    ncu_out,
                    work,
                    nvtx_expr=nvtx_expr,
                    kernel_regex=None,
                    csv_log=ncu_csv_path,
                    use_nvtx=False,
                    set_name=ncu_set,
                    metrics=None,  # let tool choose
                    sections=None,
                    target_processes=ncu_target_procs,
                    force_overwrite=ncu_force,
                )
                subprocess.run(ncu_cmd2, check=False)
                try:
                    import shlex as _shlex
                    (artifacts.out_dir("ncu") / "cmd-rerun.txt").write_text(_shlex.join(ncu_cmd2) + "\n", encoding="utf-8")
                except Exception:
                    pass
                # Best-effort import of sections from the new rep as well
                try:
                    ncu_rep2 = Path(str(ncu_out) + ".ncu-rep")
                    if ncu_rep2.exists() and ncu_sections:
                        from llm_perf_opt.profiling.vendor.ncu import build_ncu_import_sections_cmd
                        sec_cmd2 = build_ncu_import_sections_cmd(ncu_rep2, ncu_sections, page="raw")
                        sec_out2 = artifacts.path("ncu/sections_report.txt")
                        with open(sec_out2, "w", encoding="utf-8") as sf2:
                            subprocess.run(sec_cmd2, check=False, stdout=sf2, stderr=subprocess.STDOUT)
                except Exception:
                    pass
        except Exception:
            pass

    # Generate kernels.md from Nsight Compute CSV (if present)
    try:
        ncu_csv_path = artifacts.out_dir("ncu") / "raw.csv"
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

    # US3: Stakeholder summary and deep profiling report
    try:
        import json as _json
        # Load workload metrics for aggregates/MFU
        stats_dict: dict | None = None
        metrics_path = Path(workload_dir) / "torch_profiler" / "metrics.json"
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
            "# Deep Profiling Report",
            "",
            "Artifacts:",
            f"- NSYS: {(artifacts.out_dir('nsys') / 'run.nsys-rep').name} (if present), summary CSV",
            f"- NCU: {(artifacts.out_dir('ncu') / 'raw.csv').name}",
            f"- Top kernels: {artifacts.path('kernels.md').name}",
            f"- Stakeholder: {artifacts.path('stakeholder_summary.md').name}",
            f"- Workload report: {(Path(workload_dir) / 'torch_profiler' / 'report.md').name} (internal workload)",
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
    # export.csv controls whether we request raw CSV log from ncu
    try:
        export_csv = bool(getattr(getattr(ncu_cli, "export", {}), "csv", True))
    except Exception:
        export_csv = True
    csv_log = (artifacts.out_dir("ncu") / "raw.csv") if export_csv else None
