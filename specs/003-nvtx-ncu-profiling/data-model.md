# Data Model: NVTX Range Reporting (reusing existing models)

This feature reuses existing `attrs` data classes where applicable and introduces only the minimal new types needed for range‑scoped outputs. In particular, kernel-level records reuse `llm_perf_opt.data.models.KernelRecord`.

## Entities

- KernelRecord (reuse)
  - Source: `src/llm_perf_opt/data/models.py`
  - Fields: `kernel_name: str`, `device: str`, `total_ms: float`, `calls: int`, `mean_ms: float`
  - Notes: Selection (include/exclude) is handled externally; we do not add a `selected` flag to the model to preserve compatibility. Selection can be represented in report metadata (see NCUProfileRegion below).

- NCUProfileRegion (new)
  - name: string (NVTX range label)
  - parent: string | null (derived from label hierarchy, e.g., `A::B` → parent `A`)
  - device: string (e.g., `cuda:0`)
  - process_id: int (when available)
  - metrics: object (key → number per selected metric)
  - sections: object (section_name → arbitrary JSON from NCU export)
  - kernels: KernelRecord[] (after selection rules applied)
  - selection:
    - include_patterns: list[string]
    - exclude_patterns: list[string]

- NCUProfileRegionReport (new)
  - scope: string ("aggregate" | "process:<pid>" | "device:<id>")
  - regions: NCUProfileRegion[]
  - totals: object (roll-up metrics across regions; parents inclusive by default)
  - generated_at: ISO 8601 string
  - config_fingerprint: string (hash of relevant config)

## Validation Rules

- NCUProfileRegion.name MUST be non-empty; illegal filesystem characters are sanitized for directory names, original preserved in JSON.
- KernelRecord.mean_ms MUST be ≥ 0; if absent, compute as total_ms / max(1, calls) before serialization.
- Parent/child relationships inferred by `::` separator; parent totals are inclusive by default (exclusive values may be derived when needed).

## State Transitions

1. Discovered ranges → Captured per‑range NCU outputs → Parsed sections/metrics → JSON assembled (NCUProfileRegionReport) → Markdown rendered.
