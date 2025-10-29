"""Export helpers for operator summaries and stakeholder outputs.

Functions
---------
top_n_operators
    Sort operator dicts by total time and return top-N.
write_operator_markdown
    Emit a Markdown table for a list of operator records.
write_stakeholder_summary
    Write a concise stakeholder-facing summary in Markdown.
build_summary
    Convenience: derive a stakeholder summary from metrics.json only.
"""

from __future__ import annotations

from typing import Iterable, Any
from pathlib import Path
from datetime import datetime
from mdutils.mdutils import MdUtils  # type: ignore[import-untyped]


def top_n_operators(records: Iterable[dict], n: int = 10) -> list[dict]:
    """Return top-N operator dicts by ``total_time_ms``.

    Parameters
    ----------
    records : Iterable[dict]
        Operator dictionaries containing at least ``op_name``,
        ``total_time_ms``, ``cuda_time_ms``, and ``calls``.
    n : int, default=20
        Maximum number of records to return.

    Returns
    -------
    list[dict]
        Sorted records limited to the top-N by ``total_time_ms``.
    """

    sorted_recs = sorted(records, key=lambda r: float(r.get("total_time_ms", 0.0)), reverse=True)
    return sorted_recs[:n]


def write_operator_markdown(records: Iterable[dict], path: str, top_k: int = 20) -> None:
    """Write a top‑K operator summary as a Markdown table using mdutils.

    Parameters
    ----------
    records : Iterable[dict]
        Operator dictionaries to summarize.
    path : str
        Destination file path (created/overwritten). Accepts ``.md`` suffix; it is stripped
        to satisfy mdutils' file naming (which appends ``.md`` automatically).
    top_k : int, default=20
        Number of rows to include in the output table.
    """

    rows = top_n_operators(list(records), n=top_k)
    # mdutils expects a flattened list row-wise (including header)
    header = ["op_name", "total_time_ms", "cuda_time_ms", "calls"]
    table_data: list[str] = header.copy()
    for r in rows:
        table_data.extend(
            [
                str(r.get("op_name", "")),
                f"{float(r.get('total_time_ms', 0.0)):.3f}",
                f"{float(r.get('cuda_time_ms', 0.0)):.3f}",
                str(int(r.get("calls", 0))),
            ]
        )

    file_base = path[:-3] if path.endswith(".md") else path
    md = MdUtils(file_name=file_base)
    md.new_header(level=1, title="Operator Summary (Top‑K)")
    md.new_paragraph(f"Rows: {len(rows)} (k={int(top_k)})")
    md.new_table(columns=4, rows=len(rows) + 1, text=table_data, text_align="center")
    md.create_md_file()


def write_stakeholder_summary(
    path: str,
    top_ops: list[dict],
    stage_takeaways: dict[str, str],
    stats: dict | None = None,
) -> None:
    """Write a concise stakeholder summary in Markdown.

    Parameters
    ----------
    path : str
        Destination Markdown file path.
    top_ops : list[dict]
        Pre-sorted top operator records. Each dict may include keys
        ``op_name``, ``total_time_ms``, ``cuda_time_ms``, and ``calls``.
        If empty, the Top Operators section will note absence of data.
    stage_takeaways : dict[str, str]
        Free-form messages keyed by stage (e.g., "prefill", "decode", "vision")
        summarizing attribution and recommendations.
    """

    file_base = path[:-3] if path.endswith(".md") else path
    md = MdUtils(file_name=file_base)
    md.new_header(level=1, title="Stakeholder Summary")
    md.new_list(items=[f"Generated: {datetime.utcnow().isoformat()}Z"])

    # Overview statistics as tables
    if isinstance(stats, dict):
        # Environment table
        device = str(stats.get("device", ""))
        peak = stats.get("peak_tflops", None)
        env_rows = [
            ["Device", device],
            ["Peak TFLOPs (est.)", f"{float(peak):.2f}" if peak is not None else ""],
        ]
        env_header = ["Key", "Value"]
        env_table_data: list[str] = env_header.copy()
        for r in env_rows:
            env_table_data.extend([str(r[0]), str(r[1])])
        md.new_header(level=2, title="Environment")
        md.new_table(columns=2, rows=len(env_rows) + 1, text=env_table_data, text_align="center")

        # Aggregates table (Metric, Mean, Std)
        aggr = stats.get("aggregates", {}) if isinstance(stats.get("aggregates"), dict) else {}
        def _m_s(d: dict, k: str) -> tuple[float, float]:
            v = d.get(k, {})
            if not isinstance(v, dict):
                return 0.0, 0.0
            return float(v.get("mean", 0.0)), float(v.get("std", 0.0))

        rows_aggr: list[list[str]] = []
        pf_m, pf_s = _m_s(aggr, "prefill_ms")
        dc_m, dc_s = _m_s(aggr, "decode_ms")
        tk_m, tk_s = _m_s(aggr, "tokens")
        tps_m, tps_s = _m_s(aggr, "tokens_per_s")
        rows_aggr.append(["Prefill ms", f"{pf_m:.3f}", f"{pf_s:.3f}"])
        rows_aggr.append(["Decode ms", f"{dc_m:.3f}", f"{dc_s:.3f}"])
        rows_aggr.append(["Tokens", f"{tk_m:.1f}", f"{tk_s:.1f}"])
        rows_aggr.append(["Tokens/s", f"{tps_m:.3f}", f"{tps_s:.3f}"])
        aggr_header = ["Metric", "Mean", "Std"]
        aggr_table: list[str] = aggr_header.copy()
        for r in rows_aggr:
            aggr_table.extend(r)
        md.new_header(level=2, title="Aggregates")
        md.new_table(columns=3, rows=len(rows_aggr) + 1, text=aggr_table, text_align="center")

        # Per-Stage Timings table (if available)
        stage_ms = stats.get("stage_ms") if isinstance(stats.get("stage_ms"), dict) else None
        if stage_ms is None and isinstance(aggr, dict):
            stage_ms = aggr.get("stage_ms") if isinstance(aggr.get("stage_ms"), dict) else None
        if isinstance(stage_ms, dict) and stage_ms:
            # Show prefill/decode and sub-stages; omit 'vision' from the table to avoid
            # implying it is a separate top-level stage (it is nested within prefill).
            ordered = ["prefill", "decode", "sam", "clip", "projector"]
            rows_stage: list[list[str]] = []
            for k in ordered:
                if k in stage_ms and isinstance(stage_ms[k], dict):
                    m = float(stage_ms[k].get("mean", 0.0))
                    s = float(stage_ms[k].get("std", 0.0))
                    rows_stage.append([k, f"{m:.3f}", f"{s:.3f}"])
            if rows_stage:
                st_header = ["Stage", "Mean (ms)", "Std (ms)"]
                st_table: list[str] = st_header.copy()
                for r in rows_stage:
                    st_table.extend(r)
                md.new_header(level=2, title="Per-Stage Timings (ms)")
                md.new_table(columns=3, rows=len(rows_stage) + 1, text=st_table, text_align="center")
                # Show 'vision' as a note line with attribution (inside prefill)
                if "vision" in stage_ms and isinstance(stage_ms["vision"], dict):
                    vm = float(stage_ms["vision"].get("mean", 0.0))
                    vs = float(stage_ms["vision"].get("std", 0.0))
                    md.new_paragraph(
                        (
                            f"Note: Vision = sam + clip + projector (nested within prefill). "
                            f"Vision ≈ {vm:.3f} ± {vs:.3f} ms; do not add to prefill."
                        )
                    )

        # MFU table (Scope, Value)
        mfu_model = float(stats.get("mfu_model", 0.0))
        mfu_stages = stats.get("mfu_stages", {}) if isinstance(stats.get("mfu_stages"), dict) else {}
        rows_mfu = [
            ["Model-level", f"{mfu_model:.6f}"],
            ["Vision", f"{float(mfu_stages.get('vision', 0.0)):.6f}"],
            ["Prefill", f"{float(mfu_stages.get('prefill', 0.0)):.6f}"],
            ["Decode", f"{float(mfu_stages.get('decode', 0.0)):.6f}"],
        ]
        mfu_header = ["Scope", "MFU"]
        mfu_table: list[str] = mfu_header.copy()
        for r in rows_mfu:
            mfu_table.extend(r)
        md.new_header(level=2, title="MFU")
        md.new_table(columns=2, rows=len(rows_mfu) + 1, text=mfu_table, text_align="center")

    # Stage takeaways (narrative summary after the tables)
    if stage_takeaways:
        md.new_header(level=2, title="Stage Takeaways")
        items: list[str] = []
        for stage in ("decode", "prefill", "vision"):
            if stage in stage_takeaways:
                items.append(f"{stage.capitalize()}: {stage_takeaways[stage]}")
        for k, v in stage_takeaways.items():
            if k not in ("decode", "prefill", "vision"):
                items.append(f"{k}: {v}")
        if items:
            md.new_list(items=items)

    # Top operators table (if any)
    md.new_header(level=2, title="Top Operators")
    if not top_ops:
        md.new_paragraph("No operator records available. See operators.md for details if present.")
    else:
        rows = top_ops
        header = ["Operator", "Total ms", "CUDA ms", "Calls"]
        table_data: list[str] = header.copy()
        for r in rows:
            table_data.extend(
                [
                    str(r.get("op_name", "")),
                    f"{float(r.get('total_time_ms', 0.0)):.3f}",
                    f"{float(r.get('cuda_time_ms', 0.0)):.3f}",
                    str(int(r.get("calls", 0))),
                ]
            )
        md.new_table(columns=4, rows=len(rows) + 1, text=table_data, text_align="center")

    # Recommendations (template)
    md.new_header(level=2, title="Recommendations")
    md.new_list(
        items=[
            "Decode-heavy: consider FlashAttention/fused attention, reduce context, ensure KV cache efficiency.",
            "Prefill-heavy: check image resolution and preprocessing (base_size/patch_size), consider lighter vision encoder.",
            "General: verify BF16/FP16 paths, avoid CPU/GPU syncs, repeat runs (≥3) to stabilize metrics.",
            "Next: Validate with Nsight (Stage 2) for kernel-level attribution and MFU cross-check.",
        ]
    )

    md.create_md_file()


def build_summary(metrics_path: str, out_md: str) -> None:
    """Build a stakeholder summary using only metrics.json.

    This helper loads Stage 1 ``metrics.json`` (as produced by the runner)
    and emits a minimal stakeholder summary without operator detail. If
    operator records are available separately, prefer calling
    :func:`write_stakeholder_summary` with ``top_ops``.

    Parameters
    ----------
    metrics_path : str
        Path to metrics.json produced by the Stage 1 runner.
    out_md : str
        Destination Markdown path for the summary.
    """

    import json

    mp = Path(metrics_path)
    if not mp.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    data = json.loads(mp.read_text(encoding="utf-8"))

    aggr = data.get("aggregates", {}) if isinstance(data.get("aggregates"), dict) else {}
    prefill_mean = float(aggr.get("prefill_ms", {}).get("mean", 0.0))
    decode_mean = float(aggr.get("decode_ms", {}).get("mean", 0.0))
    tps_mean = float(aggr.get("tokens_per_s", {}).get("mean", 0.0))
    mfu_model = float(data.get("mfu_model_level", 0.0))
    mfu_stages = data.get("mfu_per_stage", {}) if isinstance(data.get("mfu_per_stage"), dict) else {}
    mfu_decode = float(mfu_stages.get("decode", 0.0))
    mfu_prefill = float(mfu_stages.get("prefill", 0.0))
    mfu_vision = float(mfu_stages.get("vision", 0.0))

    # Heuristic takeaways based on relative stage timings
    stage_msgs: dict[str, str] = {}
    if decode_mean >= max(1.0, prefill_mean) * 1.2:
        stage_msgs["decode"] = (
            f"Decode dominates runtime (≈ {decode_mean:.1f} ms per run). "
            f"Tokens/s ≈ {tps_mean:.2f}; MFU(decode) ≈ {mfu_decode:.6f}."
        )
    elif prefill_mean >= max(1.0, decode_mean) * 1.2:
        stage_msgs["prefill"] = (
            f"Prefill dominates runtime (≈ {prefill_mean:.1f} ms per run). "
            f"MFU(prefill) ≈ {mfu_prefill:.6f}."
        )
    else:
        stage_msgs["decode"] = (
            f"Decode and prefill comparable (decode ≈ {decode_mean:.1f} ms). "
            f"Tokens/s ≈ {tps_mean:.2f}; MFU(decode) ≈ {mfu_decode:.6f}."
        )
        stage_msgs["prefill"] = f"Prefill ≈ {prefill_mean:.1f} ms; MFU(prefill) ≈ {mfu_prefill:.6f}."
    if mfu_vision > 0.0:
        stage_msgs["vision"] = f"Vision compute contributes to MFU ≈ {mfu_vision:.6f}."
    stage_msgs["model"] = f"Model-level MFU ≈ {mfu_model:.6f}."

    stats = {
        "aggregates": aggr,
        "mfu_model": mfu_model,
        "mfu_stages": mfu_stages,
        "peak_tflops": float(data.get("peak_tflops", 0.0)),
        "device": "",  # unknown in this context
    }
    write_stakeholder_summary(out_md, top_ops=[], stage_takeaways=stage_msgs, stats=stats)


# -----------------------------------------------------------------------------
# Static Compute Report Utilities (for fvcore-based analysis)
# -----------------------------------------------------------------------------


def write_static_compute_json(report: dict, output_path: Path) -> None:
    """Write static compute report as JSON.

    Parameters
    ----------
    report : dict
        Static analysis report dictionary from DeepseekOCRStaticAnalyzer
    output_path : Path
        Destination file path for JSON output
    """
    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def write_static_compute_markdown(report: dict, output_path: Path) -> None:
    """Write static compute report as formatted markdown table.

    Produces a table with columns:
    - Stage
    - Parameters (M)
    - FLOPs (G)
    - Activations (M)
    - Top operators

    Parameters
    ----------
    report : dict
        Static analysis report dictionary from DeepseekOCRStaticAnalyzer
    output_path : Path
        Destination file path for Markdown output
    """
    file_base = str(output_path)[:-3] if str(output_path).endswith(".md") else str(output_path)
    md = MdUtils(file_name=file_base)

    md.new_header(level=1, title="Static Compute Analysis Report")

    # Metadata
    md.new_header(level=2, title="Configuration")
    metadata = report.get("metadata", {})
    meta_items = []
    for k, v in metadata.items():
        meta_items.append(f"**{k}**: {v}")
    md.new_list(items=meta_items)

    # Total
    md.new_header(level=2, title="Total Model")
    total = report.get("total", {})
    md.new_list(
        items=[
            f"**Parameters**: {float(total.get('params', 0)) / 1e6:.2f}M",
            f"**FLOPs**: {float(total.get('flops', 0)) / 1e9:.2f}G",
            f"**Activations**: {float(total.get('activations', 0)) / 1e6:.2f}M",
        ]
    )

    # Per-stage table
    md.new_header(level=2, title="Per-Stage Breakdown")
    stages = report.get("stages", {})

    if stages:
        table_data = ["Stage", "Params (M)", "FLOPs (G)", "Acts (M)", "Top Ops"]
        for stage_name in ["sam", "clip", "projector", "prefill", "decode"]:
            if stage_name not in stages:
                continue
            stage_data = stages[stage_name]
            params_m = float(stage_data.get("params", 0)) / 1e6
            flops_g = float(stage_data.get("flops", 0)) / 1e9
            acts_m = float(stage_data.get("activations", 0)) / 1e6

            # Top 3 operators
            ops = stage_data.get("operators", {})
            if isinstance(ops, dict):
                top_ops_list = sorted(ops.items(), key=lambda x: float(x[1]), reverse=True)[:3]
                top_ops_str = ", ".join(f"{op}" for op, _ in top_ops_list) if top_ops_list else "N/A"
            else:
                top_ops_str = "N/A"

            table_data.extend([
                stage_name,
                f"{params_m:.2f}",
                f"{flops_g:.2f}",
                f"{acts_m:.2f}",
                top_ops_str,
            ])

        md.new_table(columns=5, rows=len(stages) + 1, text=table_data, text_align="left")
    else:
        md.new_paragraph("No stage data available.")

    # Notes
    md.new_header(level=2, title="Notes")
    notes_list = report.get("notes", [])
    if notes_list:
        md.new_list(items=notes_list)

    # Per-stage details
    md.new_header(level=2, title="Per-Stage Details")
    for stage_name in ["sam", "clip", "projector", "prefill", "decode"]:
        if stage_name not in stages:
            continue
        stage_data = stages[stage_name]

        md.new_header(level=3, title=f"Stage: {stage_name}")

        details = []
        details.append(f"**Parameters**: {float(stage_data.get('params', 0)) / 1e6:.2f}M")
        details.append(f"**FLOPs (extracted)**: {float(stage_data.get('flops', 0)) / 1e9:.2f}G")

        if stage_data.get("flops_isolated") is not None:
            details.append(f"**FLOPs (isolated)**: {float(stage_data['flops_isolated']) / 1e9:.2f}G")

        if stage_data.get("flops_analytic") is not None:
            details.append(f"**FLOPs (analytic)**: {float(stage_data['flops_analytic']) / 1e9:.2f}G")

        details.append(f"**Activations**: {float(stage_data.get('activations', 0)) / 1e6:.2f}M")

        md.new_list(items=details)

        # Notes for this stage
        stage_notes = stage_data.get("notes", [])
        if stage_notes:
            md.new_paragraph("**Notes**: " + "; ".join(stage_notes))

    md.create_md_file()


def write_static_compute_fvcore_table(
    model: Any,
    inputs: tuple,
    output_path: Path,
    max_depth: int = 3,
) -> None:
    """Use fvcore's built-in table formatter.

    Parameters
    ----------
    model : Any
        PyTorch model to analyze
    inputs : tuple
        Input tensors for the model
    output_path : Path
        Destination file path for output
    max_depth : int, default=3
        Module hierarchy depth for table
    """
    try:
        from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis, flop_count_table

        flops = FlopCountAnalysis(model, inputs)
        acts = ActivationCountAnalysis(model, inputs)

        table_str = flop_count_table(
            flops,
            max_depth=max_depth,
            activations=acts,
            show_param_shapes=True,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Static Compute Analysis (fvcore)\n\n")
            f.write("```\n")
            f.write(table_str)
            f.write("\n```\n")

    except Exception as e:
        # Fallback: write error message
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Static Compute Analysis (fvcore)\n\n")
            f.write(f"Error generating fvcore table: {e}\n")
