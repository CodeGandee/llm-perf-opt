#!/usr/bin/env python3
"""
NCU kernel analysis: parse per-kernel Nsight Compute exports, summarize metrics, classify
memory/compute bound kernels, and generate roofline and histogram visualizations.

What this script does
- Scans an Nsight Compute export directory that contains one subdirectory per kernel
  invocation and extracts commonly used metrics from the SpeedOfLight, Memory Workload
  Analysis, and Occupancy sections.
- Produces tidy, cross-kernel aggregates (CSV/PKL), per-metric histograms, and a
  roofline scatter plot.
- Classifies each kernel as memory_bound, compute_bound, or balanced by comparing
  Compute (SM) Throughput (%) vs DRAM Throughput (%).
- Optionally constructs a flat Thicket object (root → one node per kernel) and writes
  it to JSON for downstream analysis with Thicket/Hatchet tools.

Expected input layout (example)
- <input_dir>/
  - command.yaml (optional)
  - kernel_0001_<hash>/
    - ncu.section_SpeedOfLight.csv
    - ncu.section_MemoryWorkloadAnalysis.csv
    - ncu.section_Occupancy.csv
    - ncu.section_SchedulerStats.csv (optional)
    - ncu.section_SpeedOfLight_RooflineChart.csv (optional; may be empty)
    - ncu.details.csv (optional)
  - kernel_0002_<hash>/
  - ...

Outputs (written under <input_dir>/analysis/)
- metrics_summary.csv / metrics_summary.pkl
  One row per kernel with selected, normalized metrics extracted from NCU sections. These
  include, if available: Compute (SM) Throughput (%), DRAM/Memory Throughput (%), L1/TEX
  and L2 throughput/hit rates, duration (us), Mem Busy (%), Max Bandwidth (%), Achieved
  and Theoretical Occupancy (%). Also includes the simple bound classification.
- classification_summary.csv
  Kernel id → {memory_bound, compute_bound, balanced, unknown}.
- all_metrics_long.csv
  Long-form table of all metrics we aggregated per kernel (section, metric_name, value).
- histograms/<metric>.png
  Histograms across kernels for selected metrics (e.g., sm_throughput_pct, dram_throughput_pct,
  achieved_occupancy_pct, memory_throughput_gbs, etc.).
- roofline_scatter_physical.png (if available)
  Physical roofline plot: Arithmetic Intensity (FLOPs/Byte) on x-axis vs Performance (FLOPs/s)
  on y-axis (both on log scales), generated when NCU roofline data are present and non-empty.
- roofline_scatter_normalized.png
  Normalized roofline plot: DRAM Throughput (%) vs Compute (SM) Throughput (%), generated
  whenever throughput metrics are available.
- roofline_per_kernel/<kernel>_physical.png (if available)
  Per-kernel physical roofline plots.
- roofline_per_kernel/<kernel>_normalized.png
  Per-kernel normalized roofline plots.
- thicket.json (optional)
  A JSON-serialized Thicket object representing a flat tree (root with one child per
  kernel). This is convenient for EDA with Thicket; it is not a call-tree mapping. For
  full call-tree integration, see thicket.ncu.add_ncu with Caliper profiles.

Classification rule (simple heuristic)
- compute_bound if SM% − DRAM% > 5.0
- memory_bound if SM% − DRAM% < −5.0
- balanced otherwise (within ±5%).

Roofline
- Physical (preferred): Nsight Compute's SpeedOfLight_RooflineChart.csv provides Arithmetic
  Intensity (FLOPs/Byte) and Performance (FLOPs/s). We parse any variant columns containing
  those units, normalizing GFLOP/s or TFLOP/s to FLOP/s. Generated when data is available.
- Normalized (always): When SM Throughput (%) and DRAM Throughput (%) metrics are available,
  we generate a normalized scatter plot for quick directional insight. This is generated
  alongside the physical roofline, not as a replacement.

Thicket output
- When Thicket+Hatchet are importable in the environment, we build a flat Thicket object
  with (node, profile) indexing from the per-kernel table and write it to thicket.json.
  This enables downstream Thicket stats/plotting without requiring NCU at analysis time.

Usage examples
- Pixi (recommended):
  - CSV directory input:
    pixi run -e rtx5090 python scripts/ncu/analysis/analyze_ncu_dir.py tmp/ncu-profile/<run_id>
  - NCU report input (.ncu-rep via ncu_report API):
    pixi run -e rtx5090 python scripts/ncu/analysis/analyze_ncu_dir.py tmp/ncu-profile/<run_id>/profile.ncu-rep

Dependencies
- Required: pandas, matplotlib
- Optional: llnl-thicket, llnl-hatchet (for writing thicket.json). Script otherwise runs
  without them; Thicket output is skipped if not available.
- Optional (preferred): NVIDIA Nsight Compute Python Report Interface (ncu_report) available
  on PYTHONPATH, typically under /opt/nvidia/nsight-compute/<ver>/extras/python.

Notes & limitations
- Nsight Compute versions can change metric names/sections; this script targets common
  fields in SpeedOfLight, Memory Workload Analysis, and Occupancy sections.
- Physical roofline plots require SpeedOfLight_RooflineChart.csv data, which is often empty
  unless explicitly requested during NCU export. Normalized roofline plots are always generated
  when throughput metrics are available, providing complementary insights.
- We do not infer FLOP rates or arithmetic intensity if NCU did not export them.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
try:
    from thicket.thicket import Thicket
    from thicket import helpers as th_helpers
    from hatchet.graph import Graph
    from hatchet.node import Node
    from hatchet.graphframe import GraphFrame as HatchetGraphFrame
except Exception:
    Thicket = None  # type: ignore[assignment]
    th_helpers = None  # type: ignore[assignment]
    Graph = None  # type: ignore[assignment]
    Node = None  # type: ignore[assignment]
    HatchetGraphFrame = None  # type: ignore[assignment]

# Optional: NVIDIA Nsight Compute Python Report Interface (preferred)
try:
    import ncu_report  # type: ignore
except Exception:
    ncu_report = None  # type: ignore


NUMERIC_COL = "Metric Value"
SECTION_COL = "Section Name"
METRIC_COL = "Metric Name"
KERNEL_NAME_COL = "Kernel Name"
GRID_SIZE_COL = "Grid Size"
BLOCK_SIZE_COL = "Block Size"
ID_COL = "ID"


@dataclasses.dataclass
class KernelPaths:
    root: Path
    speed_of_light: Optional[Path]
    mem_workload: Optional[Path]
    occupancy: Optional[Path]
    scheduler: Optional[Path]
    details: Optional[Path]
    roofline_csv: Optional[Path]
    metrics_csv: Optional[Path]


def find_kernel_dirs(input_dir: Path) -> List[KernelPaths]:
    kernels: List[KernelPaths] = []
    for sub in sorted(input_dir.iterdir()):
        if not sub.is_dir():
            continue
        if not sub.name.startswith("kernel_"):
            continue
        def f(name: str) -> Optional[Path]:
            p = sub / name
            return p if p.exists() else None

        kernels.append(
            KernelPaths(
                root=sub,
                speed_of_light=f("ncu.section_SpeedOfLight.csv"),
                mem_workload=f("ncu.section_MemoryWorkloadAnalysis.csv"),
                occupancy=f("ncu.section_Occupancy.csv"),
                scheduler=f("ncu.section_SchedulerStats.csv"),
                details=f("ncu.details.csv"),
                roofline_csv=f("ncu.section_SpeedOfLight_RooflineChart.csv"),
                metrics_csv=f("ncu.metrics.csv"),
            )
        )
    return kernels


def read_csv_safe(path: Path) -> pd.DataFrame:
    # Let pandas handle thousands separators where present.
    df = pd.read_csv(path, thousands=",")
    # Normalize column names by stripping quotes if any (pandas typically handles this already).
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_numeric(val: object) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    # Remove thousands separators, percent signs (if any), and whitespace
    s = s.replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def pivot_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Reduce the long-form section CSV to a wide form by metric, averaging across IDs
    if multiple entries exist for the same metric.

    Returns
    - metrics_wide: columns are metric names; values are numeric (float) means
    - info: basic fields like kernel name, grid/block sizes (from first row)
    """
    if df.empty:
        return pd.DataFrame(), {}

    # Extract common identifying info from first row
    info: Dict[str, object] = {}
    for col in [KERNEL_NAME_COL, GRID_SIZE_COL, BLOCK_SIZE_COL, "Device", "CC", "Process Name", "Process ID"]:
        if col in df.columns and len(df[col]) > 0:
            info[col] = df[col].iloc[0]

    # Clean numeric values (vectorized)
    if NUMERIC_COL in df.columns:
        df = df.copy()
        df[NUMERIC_COL] = pd.to_numeric(df[NUMERIC_COL], errors="coerce")

    # Average across IDs per metric name
    if METRIC_COL not in df.columns:
        return pd.DataFrame(), info
    grouped = df.groupby(METRIC_COL, dropna=False)[NUMERIC_COL].mean(numeric_only=True)
    wide = grouped.to_frame().T
    # Flatten column names for consistent access
    wide.columns = [str(c) for c in wide.columns]
    return wide, info


def extract_selected_metrics(
    sol_wide: pd.DataFrame, mwa_wide: pd.DataFrame, occ_wide: pd.DataFrame
) -> Dict[str, Optional[float]]:
    """Map known metric names to normalized keys; return dictionary of floats or None."""
    def g(df: pd.DataFrame, metric: str) -> Optional[float]:
        if df is None or df.empty:
            return None
        if metric in df.columns:
            v = df.iloc[0][metric]
            return None if pd.isna(v) else float(v)
        # Try relaxed lookup (case-insensitive exact)
        for c in df.columns:
            if c.lower() == metric.lower():
                v = df.iloc[0][c]
                return None if pd.isna(v) else float(v)
        return None

    out: Dict[str, Optional[float]] = {
        # SpeedOfLight
        "sm_throughput_pct": g(sol_wide, "Compute (SM) Throughput"),
        "dram_throughput_pct": g(sol_wide, "DRAM Throughput"),
        "mem_throughput_pct": g(sol_wide, "Memory Throughput"),
        "l1tex_throughput_pct": g(sol_wide, "L1/TEX Cache Throughput"),
        "l2_throughput_pct": g(sol_wide, "L2 Cache Throughput"),
        "duration_us": g(sol_wide, "Duration"),
        # Memory Workload Analysis
        "mem_busy_pct": g(mwa_wide, "Mem Busy"),
        "max_bandwidth_pct": g(mwa_wide, "Max Bandwidth"),
        "memory_throughput_gbs": g(mwa_wide, "Memory Throughput"),
        "l1tex_hit_rate_pct": g(mwa_wide, "L1/TEX Hit Rate"),
        "l2_hit_rate_pct": g(mwa_wide, "L2 Hit Rate"),
        # Occupancy
        "achieved_occupancy_pct": g(occ_wide, "Achieved Occupancy"),
        "theoretical_occupancy_pct": g(occ_wide, "Theoretical Occupancy"),
    }
    return out


def classify_bound(sm_pct: Optional[float], dram_pct: Optional[float], margin: float = 5.0) -> str:
    if sm_pct is None or dram_pct is None:
        return "unknown"
    diff = float(sm_pct) - float(dram_pct)
    if diff > margin:
        return "compute_bound"
    if diff < -margin:
        return "memory_bound"
    return "balanced"


def sanitize_filename(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("._")
    return s or "unnamed"


def build_all_metrics_long(
    kernel_id: str, dfs: Iterable[Tuple[str, pd.DataFrame]]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for section, df in dfs:
        if df is None or df.empty:
            continue
        for col in df.columns:
            val = df.iloc[0][col]
            if pd.isna(val):
                continue
            rows.append({
                "kernel_id": kernel_id,
                "section": section,
                "metric_name": col,
                "value": float(val),
            })
    return pd.DataFrame(rows)


def parse_roofline_points(path: Path) -> Optional[pd.DataFrame]:
    """
    Parse the Roofline Chart CSV if available to extract points with x=FLOPs/Byte,
    y=FLOPs/s. Returns a DataFrame with columns [ai_flops_per_byte, flops_per_s].

    The exported CSV format can vary across ncu versions. We attempt a heuristic:
    - Load CSV; search columns containing 'intensity' for x; 'flop' and '/s' or
      'gflop'/'tflop' for y.
    - If y is in GFLOP/s or TFLOP/s, normalize to FLOP/s.
    - If empty or missing expected columns, return None.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    # Two possible export formats:
    # 1) Header-only CSV with columns like 'Arithmetic Intensity (FLOP/Byte)' and 'Performance (GFLOP/s)'
    # 2) print-details=all long-form rows with 'Metric Name' and 'Metric Value'
    cols_lower = {c.lower(): c for c in df.columns}

    # Case 1: direct numeric columns
    x_col: Optional[str] = None
    y_col: Optional[str] = None
    for lc, orig in cols_lower.items():
        if ("intensity" in lc) and ("flop" in lc) and ("byte" in lc):
            x_col = orig
        if ("flop" in lc) and ("/s" in lc):
            y_col = orig
    if x_col and y_col:
        out = pd.DataFrame({
            "ai_flops_per_byte": pd.to_numeric(df[x_col], errors="coerce"),
            "flops_per_s_raw": df[y_col],
        }).dropna()
        y_name_l = y_col.lower()
        if "tflop" in y_name_l:
            out["flops_per_s"] = pd.to_numeric(out["flops_per_s_raw"], errors="coerce") * 1e12
        elif "gflop" in y_name_l:
            out["flops_per_s"] = pd.to_numeric(out["flops_per_s_raw"], errors="coerce") * 1e9
        else:
            out["flops_per_s"] = pd.to_numeric(out["flops_per_s_raw"], errors="coerce")
        return out[["ai_flops_per_byte", "flops_per_s"]].dropna()

    # Case 2: long-form rows with Metric Name/Value
    if METRIC_COL in df.columns and NUMERIC_COL in df.columns:
        # Look for achieved AI and FLOP/s entries
        # Match substrings robustly; prefer rows labeled 'Achieved Value'
        mnames = df[METRIC_COL].astype(str).str.lower()
        units = df.get("Metric Unit", pd.Series([None] * len(df))).astype(str).str.lower()
        values = pd.to_numeric(df[NUMERIC_COL], errors="coerce")

        # arithmetic intensity (FLOP/Byte)
        mask_ai = mnames.str.contains("intensity") & mnames.str.contains("flop") & units.str.contains("byte")
        # achieved performance (FLOP/s)
        mask_perf = (mnames.str.contains("flop") & units.str.contains("/s")) | units.str.contains("flop/s")

        if mask_ai.any() and mask_perf.any():
            ai_vals = values[mask_ai]
            perf_vals = values[mask_perf]
            # Normalize perf to FLOP/s by inspecting units
            perf_units = units[mask_perf]
            perf_norm = []
            for v, u in zip(perf_vals, perf_units):
                if pd.isna(v):
                    perf_norm.append(None)
                elif "tflop" in u:
                    perf_norm.append(float(v) * 1e12)
                elif "gflop" in u:
                    perf_norm.append(float(v) * 1e9)
                else:
                    perf_norm.append(float(v))
            out = pd.DataFrame({
                "ai_flops_per_byte": ai_vals.reset_index(drop=True),
                "flops_per_s": pd.Series(perf_norm, dtype=float),
            }).dropna()
            if not out.empty:
                return out
    return None


def compute_roofline_from_metrics(
    metrics_df: Optional[pd.DataFrame],
    mem_throughput_gbs: Optional[float],
    duration_us: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute (AI, FLOPs/s) from explicit metrics when available.

    - FLOPs: sum available flop_count_* counters (hp/sp/dp/tensor if present).
    - Time: prefer gpu__time_duration.sum (ns/us), else fall back to duration_us (SpeedOfLight).
    - Bytes: use Memory Throughput (Gbyte/s) * time.
    """
    if metrics_df is None or metrics_df.empty:
        return None, None
    df = metrics_df
    # Expect columns: Metric Name, Metric Unit, Metric Value
    if METRIC_COL not in df.columns or NUMERIC_COL not in df.columns:
        return None, None
    name = df[METRIC_COL].astype(str)
    unit = df.get("Metric Unit", pd.Series([None] * len(df))).astype(str)
    val = pd.to_numeric(df[NUMERIC_COL], errors="coerce")

    # Sum FLOP counters
    flop_mask = name.str.contains(r"^flop_count_", case=False, regex=True)
    flops_total = val[flop_mask].sum(min_count=1)

    # Duration preference: gpu__time_duration.sum
    time_mask = name.str.fullmatch("gpu__time_duration.sum", case=False)
    if time_mask.any():
        v = val[time_mask].iloc[0]
        u = unit[time_mask].iloc[0].lower() if isinstance(unit[time_mask].iloc[0], str) else "ns"
        if "ms" in u:
            duration_s = float(v) * 1e-3
        elif "us" in u or "µs" in u:
            duration_s = float(v) * 1e-6
        elif "ns" in u:
            duration_s = float(v) * 1e-9
        else:
            duration_s = float(v)  # assume seconds
    elif duration_us is not None:
        duration_s = float(duration_us) * 1e-6
    else:
        duration_s = None

    if pd.isna(flops_total) or flops_total is None or duration_s is None or duration_s <= 0:
        return None, None

    flops_per_s = float(flops_total) / float(duration_s)

    # Compute bytes using measured memory throughput if available
    if mem_throughput_gbs is None or mem_throughput_gbs <= 0:
        return None, flops_per_s
    bytes_moved = float(mem_throughput_gbs) * 1e9 * float(duration_s)
    if bytes_moved <= 0:
        return None, flops_per_s
    ai = float(flops_total) / bytes_moved
    return ai, flops_per_s


# ------------------------------
# ncu_report (API) path support
# ------------------------------

def _safe_action_metric_value(action, name: str) -> Optional[float]:
    try:
        m = action.metric_by_name(name)
        return None if m is None else m.value()
    except Exception:
        return None


def analyze_ncu_report(report_path: Path, margin: float = 5.0) -> None:
    if ncu_report is None:
        raise SystemExit("ncu_report module not available; ensure Nsight Compute extras/python is on PYTHONPATH")
    ctx = ncu_report.load_report(str(report_path))

    analysis_dir = report_path.parent / "analysis"
    hist_dir = analysis_dir / "histograms"
    per_kernel_roofline_dir = analysis_dir / "roofline_per_kernel"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)
    per_kernel_roofline_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    long_rows: List[pd.DataFrame] = []

    for range_idx in range(len(ctx)):
        r = ctx[range_idx]
        for action_idx in range(len(r)):
            a = r[action_idx]
            kernel_id = f"kernel_{range_idx:04d}_{action_idx:04d}"

            # Names
            try:
                kernel_name = a.name()
            except Exception:
                kernel_name = None
            try:
                kernel_name_demangled = a.name(demangle=True)
            except Exception:
                kernel_name_demangled = kernel_name

            # Launch dimensions (may be strings or tuples)
            grid_size = _safe_action_metric_value(a, "launch__grid_size")
            block_size = _safe_action_metric_value(a, "launch__block_size")

            # Core metrics (names may vary by version; these are common)
            sm_pct = _safe_action_metric_value(a, "smsp__throughput.avg.pct_of_peak_sustained_elapsed")
            dram_pct = _safe_action_metric_value(a, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
            mem_pct = _safe_action_metric_value(a, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
            l1tex_pct = _safe_action_metric_value(a, "l1tex__throughput.avg.pct_of_peak_sustained_elapsed")
            l2_pct = _safe_action_metric_value(a, "l2_throughput.avg.pct_of_peak_sustained_elapsed")
            duration_ns = _safe_action_metric_value(a, "gpu__time_duration.sum")
            duration_us = (duration_ns / 1000.0) if (duration_ns is not None) else None
            l1tex_hit = _safe_action_metric_value(a, "l1tex__t_sector_hit_rate.pct")
            l2_hit = _safe_action_metric_value(a, "lts__t_sector_hit_rate.pct")
            occ_achieved = _safe_action_metric_value(a, "sm__warps_active.avg.pct_of_peak_sustained_active")
            occ_theoretical = _safe_action_metric_value(a, "sm__maximum_warps_per_active_cycle_pct")

            metrics = {
                "sm_throughput_pct": sm_pct,
                "dram_throughput_pct": dram_pct,
                "mem_throughput_pct": mem_pct,
                "l1tex_throughput_pct": l1tex_pct,
                "l2_throughput_pct": l2_pct,
                "duration_us": duration_us,
                "l1tex_hit_rate_pct": l1tex_hit,
                "l2_hit_rate_pct": l2_hit,
                "achieved_occupancy_pct": occ_achieved,
                "theoretical_occupancy_pct": occ_theoretical,
            }

            bound = classify_bound(sm_pct, dram_pct, margin=margin)
            row: Dict[str, object] = {
                "kernel_id": kernel_id,
                "kernel_name": kernel_name_demangled or kernel_name,
                "kernel_name_raw": kernel_name,
                "range_idx": range_idx,
                "action_idx": action_idx,
                "grid_size": grid_size,
                "block_size": block_size,
                **metrics,
                "classification": bound,
            }
            summary_rows.append(row)

            # Long form (API section)
            rows = []
            for k, v in metrics.items():
                if v is None:
                    continue
                rows.append({"kernel_id": kernel_id, "section": "API", "metric_name": k, "value": float(v)})
            if rows:
                long_rows.append(pd.DataFrame(rows))

            # Per-kernel roofline: no physical AI/FLOPs/s available via this path (without extra metric names)
            # Generate normalized roofline only
            if sm_pct is not None and dram_pct is not None:
                fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
                ax.scatter([dram_pct], [sm_pct], c=["C0"], label=kernel_id)
                ax.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1, label="x=y")
                ax.set_xlabel("DRAM Throughput (%)")
                ax.set_ylabel("Compute (SM) Throughput (%)")
                ax.set_title(f"Roofline (normalized) • {kernel_id}")
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                ax.grid(True, linestyle=":", alpha=0.5)
                ax.legend(loc="lower right", fontsize=8)
                fig.tight_layout()
                fig.savefig(per_kernel_roofline_dir / f"{sanitize_filename(kernel_id)}_normalized.png")
                plt.close(fig)

    # Summaries
    summary_df = pd.DataFrame(summary_rows)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(analysis_dir / "metrics_summary.csv", index=False)
    summary_df.to_pickle(analysis_dir / "metrics_summary.pkl")
    (summary_df[["kernel_id", "classification"]]
     .to_csv(analysis_dir / "classification_summary.csv", index=False))
    all_long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    if not all_long_df.empty:
        all_long_df.to_csv(analysis_dir / "all_metrics_long.csv", index=False)

    # Aggregate normalized plot
    if not summary_df.empty and {"dram_throughput_pct", "sm_throughput_pct"}.issubset(summary_df.columns):
        plot_df = summary_df.dropna(subset=["dram_throughput_pct", "sm_throughput_pct"])  # type: ignore[arg-type]
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
            ax.scatter(plot_df["dram_throughput_pct"], plot_df["sm_throughput_pct"], c="C1")
            for _, r in plot_df.iterrows():
                label = str(r["kernel_id"]) if pd.notna(r["kernel_id"]) else ""
                ax.annotate(label, (r["dram_throughput_pct"], r["sm_throughput_pct"]), fontsize=7, alpha=0.7)
            ax.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1)
            ax.set_xlabel("DRAM Throughput (%)")
            ax.set_ylabel("Compute (SM) Throughput (%)")
            ax.set_title("Roofline (normalized throughput) across kernels")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.grid(True, linestyle=":", alpha=0.5)
            fig.tight_layout()
            fig.savefig(analysis_dir / "roofline_scatter_normalized.png")
            plt.close(fig)

    # Thicket JSON (flat tree)
    if (
        Thicket is not None
        and Graph is not None
        and Node is not None
        and th_helpers is not None
        and HatchetGraphFrame is not None
        and not summary_df.empty
    ):
        perf_cols = [c for c in summary_df.columns if c not in {
            "kernel_id", "kernel_name", "kernel_name_raw", "range_idx", "action_idx",
            "grid_size", "block_size", "classification"
        } and pd.to_numeric(summary_df[c], errors="coerce").notna().any()]

        if perf_cols:
            kernel_ids = list(summary_df["kernel_id"].astype(str))
            root = Node.from_lists(["NCU Kernels", *kernel_ids])
            graph = Graph([root])
            child_nodes = {child.frame["name"]: child for child in root.children}
            profile_label = report_path.name
            rows_th = []
            for _, sr in summary_df.iterrows():
                node_obj = child_nodes.get(str(sr["kernel_id"]))
                if node_obj is None:
                    continue
                row = {"node": node_obj, "profile": profile_label}
                for c in perf_cols:
                    row[c] = _parse_numeric(sr[c])
                rows_th.append(row)
            if rows_th:
                df_perf = pd.DataFrame(rows_th).set_index(["node", "profile"])
                stats_df = th_helpers._new_statsframe_df(df_perf)
                stats_gf = HatchetGraphFrame(graph=graph, dataframe=stats_df, exc_metrics=[], inc_metrics=[], default_metric=perf_cols[0])
                th = Thicket(graph=graph, dataframe=df_perf, exc_metrics=[], inc_metrics=[], default_metric=perf_cols[0], performance_cols=perf_cols, statsframe=stats_gf)
                try:
                    json_str = th.to_json()
                    (analysis_dir / "thicket.json").write_text(json_str)
                except Exception:
                    print("Warning: failed to serialize Thicket JSON for ncu_report path")


def analyze_dir(input_dir: Path, margin: float = 5.0) -> None:
    # Support both .ncu-rep file (preferred API) and legacy CSV directory
    if input_dir.is_file() and input_dir.suffix == ".ncu-rep":
        return analyze_ncu_report(input_dir, margin=margin)

    kernels = find_kernel_dirs(input_dir)
    if not kernels:
        raise SystemExit(f"No kernel_* directories found in {input_dir}")

    analysis_dir = input_dir / "analysis"
    hist_dir = analysis_dir / "histograms"
    per_kernel_roofline_dir = analysis_dir / "roofline_per_kernel"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)
    per_kernel_roofline_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    long_rows: List[pd.DataFrame] = []

    for kp in kernels:
        kernel_id = kp.root.name

        sol_wide: pd.DataFrame = pd.DataFrame()
        sol_info: Dict[str, object] = {}
        if kp.speed_of_light is not None:
            sol_df = read_csv_safe(kp.speed_of_light)
            sol_wide, sol_info = pivot_metrics(sol_df)

        mwa_wide: pd.DataFrame = pd.DataFrame()
        if kp.mem_workload is not None:
            mwa_df = read_csv_safe(kp.mem_workload)
            mwa_wide, _ = pivot_metrics(mwa_df)

        occ_wide: pd.DataFrame = pd.DataFrame()
        if kp.occupancy is not None:
            occ_df = read_csv_safe(kp.occupancy)
            occ_wide, _ = pivot_metrics(occ_df)

        metrics = extract_selected_metrics(sol_wide, mwa_wide, occ_wide)
        bound = classify_bound(metrics.get("sm_throughput_pct"), metrics.get("dram_throughput_pct"), margin=margin)

        row: Dict[str, object] = {
            "kernel_id": kernel_id,
            "kernel_name": sol_info.get(KERNEL_NAME_COL),
            "grid_size": sol_info.get(GRID_SIZE_COL),
            "block_size": sol_info.get(BLOCK_SIZE_COL),
            "device": sol_info.get("Device"),
            "cc": sol_info.get("CC"),
            "process": sol_info.get("Process Name"),
            "process_id": sol_info.get("Process ID"),
            **metrics,
            "classification": bound,
        }
        summary_rows.append(row)

        # Long-form rows for histogramming all available metrics
        long_df = build_all_metrics_long(kernel_id, [
            ("SpeedOfLight", sol_wide),
            ("MemoryWorkloadAnalysis", mwa_wide),
            ("Occupancy", occ_wide),
        ])
        if not long_df.empty:
            long_rows.append(long_df)

        # Per-kernel roofline plots: generate both physical and normalized when data is available

        # Physical roofline (FLOPs/Byte vs FLOPs/s) - preferred
        ai_val: Optional[float] = None
        perf_val: Optional[float] = None

        # Path A: parse roofline chart CSV if it contains numeric points
        rl_df = parse_roofline_points(kp.roofline_csv) if kp.roofline_csv else None
        if rl_df is not None and not rl_df.empty:
            ai_val = float(rl_df["ai_flops_per_byte"].median())
            perf_val = float(rl_df["flops_per_s"].median())
        else:
            # Path B: compute from explicit metrics (ncu.metrics.csv) + memory throughput + duration
            metrics_df = read_csv_safe(kp.metrics_csv) if kp.metrics_csv and kp.metrics_csv.exists() else None
            ai_c, perf_c = compute_roofline_from_metrics(
                metrics_df=metrics_df,
                mem_throughput_gbs=metrics.get("memory_throughput_gbs"),
                duration_us=metrics.get("duration_us"),
            )
            ai_val = ai_c if ai_c is not None else ai_val
            perf_val = perf_c if perf_c is not None else perf_val

        if ai_val is not None and perf_val is not None:
            # Persist into summary row for aggregate plotting
            row["roofline_ai_flops_per_byte"] = ai_val
            row["roofline_flops_per_s"] = perf_val
            # Per-kernel physical plot
            fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
            ax.scatter([ai_val], [perf_val], c=["C0"], s=18)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)")
            ax.set_ylabel("Performance (FLOPs/s)")
            ax.set_title(f"Roofline (physical) • {kernel_id}")
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            fig.tight_layout()
            fig.savefig(per_kernel_roofline_dir / f"{sanitize_filename(kernel_id)}_physical.png")
            plt.close(fig)

        # Normalized roofline (% vs %) - always generate when throughput data is available
        sm_pct = metrics.get("sm_throughput_pct")
        dram_pct = metrics.get("dram_throughput_pct")
        if sm_pct is not None and dram_pct is not None:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
            ax.scatter([dram_pct], [sm_pct], c=["C0"], label=kernel_id)
            ax.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1, label="x=y")
            ax.set_xlabel("DRAM Throughput (%)")
            ax.set_ylabel("Compute (SM) Throughput (%)")
            ax.set_title(f"Roofline (normalized) • {kernel_id}")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="lower right", fontsize=8)
            out_path = per_kernel_roofline_dir / f"{sanitize_filename(kernel_id)}_normalized.png"
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)

    # Build summary DataFrames
    summary_df = pd.DataFrame(summary_rows)
    # Save summary
    summary_df.to_csv(analysis_dir / "metrics_summary.csv", index=False)
    summary_df.to_pickle(analysis_dir / "metrics_summary.pkl")

    # Classification summary
    (summary_df[["kernel_id", "classification"]]
     .to_csv(analysis_dir / "classification_summary.csv", index=False))

    # Long-form metrics across kernels
    all_long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    if not all_long_df.empty:
        all_long_df.to_csv(analysis_dir / "all_metrics_long.csv", index=False)

    # Construct a Thicket object (flat tree: root -> kernel nodes) when available
    if (
        Thicket is not None
        and Graph is not None
        and Node is not None
        and th_helpers is not None
        and HatchetGraphFrame is not None
        and not summary_df.empty
    ):
        # Choose numeric performance columns
        perf_cols: List[str] = []
        skip_cols = {
            "kernel_id",
            "kernel_name",
            "grid_size",
            "block_size",
            "device",
            "cc",
            "process",
            "process_id",
            "classification",
        }
        for c in summary_df.columns:
            if c in skip_cols:
                continue
            s = pd.to_numeric(summary_df[c], errors="coerce")
            if s.notna().any():
                perf_cols.append(c)

        if perf_cols:
            # Build root with one child per kernel_id (use kernel_id as node name)
            kernel_ids = list(summary_df["kernel_id"].astype(str))
            root = Node.from_lists(["NCU Kernels", *kernel_ids])
            graph = Graph([root])
            child_nodes = {child.frame["name"]: child for child in root.children}

            profile_label = input_dir.name
            rows_th: List[Dict[str, object]] = []
            for _, sr in summary_df.iterrows():
                kid = str(sr["kernel_id"])
                node_obj = child_nodes.get(kid)
                if node_obj is None:
                    continue
                row: Dict[str, object] = {"node": node_obj, "profile": profile_label}
                for c in perf_cols:
                    row[c] = _parse_numeric(sr[c])
                rows_th.append(row)

            if rows_th:
                df_perf = pd.DataFrame(rows_th).set_index(["node", "profile"])
                stats_df = th_helpers._new_statsframe_df(df_perf)
                stats_gf = HatchetGraphFrame(
                    graph=graph,
                    dataframe=stats_df,
                    exc_metrics=[],
                    inc_metrics=[],
                    default_metric=perf_cols[0],
                )
                th = Thicket(
                    graph=graph,
                    dataframe=df_perf,
                    exc_metrics=[],
                    inc_metrics=[],
                    default_metric=perf_cols[0],
                    performance_cols=perf_cols,
                    statsframe=stats_gf,
                )
                # Persist: JSON only (no pickle)
                import json
                try:
                    json_str = th.to_json()
                    (analysis_dir / "thicket.json").write_text(json_str)
                except Exception:
                    pass

    # Aggregate roofline scatter across kernels: generate both physical and normalized

    # Physical roofline aggregate (FLOPS/byte vs FLOPS/s) when data is available
    have_physical = {"roofline_ai_flops_per_byte", "roofline_flops_per_s"}.issubset(summary_df.columns)
    if have_physical:
        plot_df = summary_df.dropna(subset=["roofline_ai_flops_per_byte", "roofline_flops_per_s"])  # type: ignore[arg-type]
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
            ax.scatter(plot_df["roofline_ai_flops_per_byte"], plot_df["roofline_flops_per_s"], c="C3")
            for _, r in plot_df.iterrows():
                label = str(r["kernel_id"]) if pd.notna(r["kernel_id"]) else ""
                ax.annotate(label, (r["roofline_ai_flops_per_byte"], r["roofline_flops_per_s"]), fontsize=7, alpha=0.7)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)")
            ax.set_ylabel("Performance (FLOPs/s)")
            ax.set_title("Roofline (physical) across kernels")
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            fig.tight_layout()
            fig.savefig(analysis_dir / "roofline_scatter_physical.png")
            plt.close(fig)

    # Normalized roofline aggregate (% vs %) - always generate when throughput data is available
    if not summary_df.empty and {"dram_throughput_pct", "sm_throughput_pct"}.issubset(summary_df.columns):
        plot_df = summary_df.dropna(subset=["dram_throughput_pct", "sm_throughput_pct"])  # type: ignore[arg-type]
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
            ax.scatter(plot_df["dram_throughput_pct"], plot_df["sm_throughput_pct"], c="C1")
            for _, r in plot_df.iterrows():
                label = str(r["kernel_id"]) if pd.notna(r["kernel_id"]) else ""
                ax.annotate(label, (r["dram_throughput_pct"], r["sm_throughput_pct"]), fontsize=7, alpha=0.7)
            ax.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1)
            ax.set_xlabel("DRAM Throughput (%)")
            ax.set_ylabel("Compute (SM) Throughput (%)")
            ax.set_title("Roofline (normalized throughput) across kernels")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.grid(True, linestyle=":", alpha=0.5)
            fig.tight_layout()
            fig.savefig(analysis_dir / "roofline_scatter_normalized.png")
            plt.close(fig)

    # Histograms for selected metrics
    histogram_metrics = [
        "sm_throughput_pct",
        "dram_throughput_pct",
        "mem_throughput_pct",
        "mem_busy_pct",
        "achieved_occupancy_pct",
        "theoretical_occupancy_pct",
        "max_bandwidth_pct",
        "memory_throughput_gbs",
        "duration_us",
        "l1tex_hit_rate_pct",
        "l2_hit_rate_pct",
        "l1tex_throughput_pct",
        "l2_throughput_pct",
    ]

    for m in histogram_metrics:
        if m not in summary_df.columns:
            continue
        series = pd.to_numeric(summary_df[m], errors="coerce").dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
        ax.hist(series, bins=10, color="C2", edgecolor="black", alpha=0.8)
        ax.set_title(f"Histogram: {m}")
        ax.set_xlabel(m)
        ax.set_ylabel("Count")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(hist_dir / f"{sanitize_filename(m)}.png")
        plt.close(fig)

    print(f"Analysis complete. Outputs in: {analysis_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Nsight Compute (ncu) data from a CSV directory or an .ncu-rep file")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to NCU CSV directory (kernel_* subdirs) or a .ncu-rep report file",
    )
    parser.add_argument(
        "--classify-margin",
        type=float,
        default=5.0,
        help="Margin (percentage points) for memory/compute-bound classification (default: 5.0)",
    )
    args = parser.parse_args()
    input_path = Path(args.input_path).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")
    analyze_dir(input_path, margin=args.classify_margin)


if __name__ == "__main__":
    main()
