"""Domain data models for Stage 1 profiling.

This module defines `attrs`-based data models for the profiling domain. These
are internal representations and may be converted to/from public contract
schemas using `cattrs` hooks (see `llm_perf_opt.contracts.convert`).

Classes
-------
StageTiming
    Per-stage timing and throughput fields.
OperatorSummary
    Top-K operator summary record (domain mirror).
Stats
    Mean/std pair used for aggregates.
LLMProfileReport
    Full internal report artifact for a run.
"""

from __future__ import annotations

from typing import Dict, Literal

from attrs import define, field
from attrs.validators import instance_of


@define(kw_only=True)
class KernelRecord:
    """Kernel-level metric record (Stage 2, Nsight Compute parsed).

    Minimal attribution for top-kernel tables. Values are expected to be
    non-negative; callers should sanitize raw tool outputs accordingly.
    """

    kernel_name: str = field(validator=[instance_of(str)])
    device: str = field(validator=[instance_of(str)])
    total_ms: float = field(validator=[instance_of(float)])
    calls: int = field(validator=[instance_of(int)])
    mean_ms: float = field(validator=[instance_of(float)])


@define(kw_only=True)
class StageTiming:
    """Timing information for a single stage.

    Parameters
    ----------
    stage : {'prefill', 'decode'}
        Stage name.
    elapsed_ms : float
        Elapsed time in milliseconds.
    tokens : int, default=0
        Tokens generated (decode) or proxy (prefill) if used.
    throughput_toks_per_s : float, default=0.0
        Computed throughput in tokens per second.
    """

    stage: Literal["prefill", "decode"] = field()
    elapsed_ms: float = field(validator=[instance_of(float)])
    tokens: int = field(default=0, validator=[instance_of(int)])
    throughput_toks_per_s: float = field(default=0.0, validator=[instance_of(float)])


@define(kw_only=True)
class OperatorSummary:
    """Aggregated operator record (domain)."""

    op_name: str = field(validator=[instance_of(str)])
    total_time_ms: float = field(validator=[instance_of(float)])
    cuda_time_ms: float = field(validator=[instance_of(float)])
    calls: int = field(validator=[instance_of(int)])


@define(kw_only=True)
class Stats:
    """Mean and standard deviation."""

    mean: float = field(validator=[instance_of(float)])
    std: float = field(validator=[instance_of(float)])


@define(kw_only=True)
class LLMProfileReport:
    """Internal artifact representation of a profiling run.

    Parameters
    ----------
    run_id : str
        Unique identifier for this run.
    timings : list[StageTiming]
        List of stage timing records.
    operators_topk : list[OperatorSummary]
        Top-K operators by total time.
    mfu_model_level : float
        Overall MFU value.
    mfu_per_stage : dict[str, float]
        Per-stage MFU mapping.
    aggregates : dict[str, Stats]
        Aggregated metrics (e.g., means/stds).
    notes : str
        Stakeholder-oriented notes.
    """

    run_id: str = field(validator=[instance_of(str)])
    timings: list[StageTiming] = field(factory=list)
    operators_topk: list[OperatorSummary] = field(factory=list)
    # Optional: populated in Stage 2 when Nsight Compute results are available
    kernels_topk: list[KernelRecord] = field(factory=list)
    mfu_model_level: float = field(validator=[instance_of(float)])
    mfu_per_stage: Dict[str, float] = field(factory=dict)
    aggregates: Dict[str, Stats] = field(factory=dict)
    notes: str = field(validator=[instance_of(str)])
