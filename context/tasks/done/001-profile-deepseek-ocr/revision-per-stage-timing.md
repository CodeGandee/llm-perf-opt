# Revision Plan: Per‑Stage Timings in Aggregates and Stakeholder Summary

Goal: Include time for each NVTX stage in the Aggregates section and Stakeholder Summary, with clear tables and non‑ambiguous attribution notes.

## Current State (Inventory)
- NVTX stages
  - Top‑level ranges: `prefill`, `decode` (via `nvtx_utils.py`)
  - Vision sub‑stages: `sam`, `clip`, `projector` (forward hooks in `dsocr_session.py` accumulating `m_stage_time_ms`)
- Runner outputs (`run_inference` in `dsocr_session.py`)
  - Returns: `prefill_ms`, `decode_ms`, `tokens`, `prefill_len`, `vision_ms` (sum of `sam+clip+projector`)
- Aggregation (`_summarize_runs` in `llm_profile_runner.py`)
  - Aggregates: `prefill_ms`, `decode_ms`, `tokens`, `tokens_per_s`
  - MFU uses average of `vision_ms` but timing for `vision` not surfaced in `aggregates`
- Stakeholder Summary (`write_stakeholder_summary` in `export.py`)
  - Tables: Environment, Aggregates (currently prefill/decode/tokens/tokens/s), MFU
  - Narrative: Stage takeaways, recommendations

## Requirements
- Add per‑stage timing stats (mean/std) for all available stages to Aggregates:
  - Required: `prefill`, `decode`
  - Preferred: `vision` (sum over `sam+clip+projector`), and sub‑stages `sam`, `clip`, `projector` if available
- Present these in the Stakeholder Summary as a dedicated table (e.g., “Per‑Stage Timings”) and keep the short narrative afterward
- Clarify attribution: Vision sub‑stages run inside `prefill`; do not double‑sum `prefill + vision` as total runtime
- Backward compatible metrics.json (do not remove existing keys)

## Design
- Data model changes
  - Extend `ImageRun` (runner) with optional sub‑stage fields: `sam_ms`, `clip_ms`, `projector_ms` (float)
  - Alternatively, add a dict `stage_times: dict[str, float]` (but keep top‑level `vision_ms` for compatibility)
- Capture changes (session)
  - In `DeepSeekOCRSession.run_inference`, return `sam_ms`, `clip_ms`, `projector_ms` along with existing fields
- Aggregation changes
  - Add `stage_ms` to `summary['aggregates']`:
    - `stage_ms = { 'prefill': {mean,std}, 'decode': {mean,std}, 'vision': {mean,std}, 'sam': {mean,std}, 'clip': {mean,std}, 'projector': {mean,std} }`
    - Only include keys with at least one non‑zero measurement
- Export changes (stakeholder summary)
  - Add a new table “Per‑Stage Timings (ms)” with columns: Stage | Mean | Std
  - Optionally add “Stage Share (%)” tables:
    - Run‑level: prefill vs decode share of (prefill_mean + decode_mean)
    - Prefill‑level: sam/clip/projector share of prefill_mean (note that their sum can be ≤ prefill due to non‑instrumented work)
  - Keep narrative “Stage Takeaways” after tables
- Metrics JSON changes
  - Keep top‑level: `prefill_ms`, `decode_ms` aggregates as today
  - Add nested `aggregates.stage_ms` as above
- Robustness
  - If hooks or NVTX are disabled, omit missing rows gracefully

## Implementation Steps
1) Session outputs
   - File: `src/llm_perf_opt/runners/dsocr_session.py`
   - Action: Add to result dict of `run_inference`:
     - `sam_ms`, `clip_ms`, `projector_ms` from `self.m_stage_time_ms`
2) Runner data model and aggregation
   - File: `src/llm_perf_opt/runners/llm_profile_runner.py`
   - Action:
     - Extend `ImageRun` to include sub‑stage fields (default 0.0)
     - When collecting runs, populate `sam_ms`, `clip_ms`, `projector_ms` from session result
     - In `_summarize_runs`, compute mean/std for `vision_ms` and sub‑stages and set `summary['aggregates']['stage_ms'] = {...}`
3) Stakeholder summary export
   - File: `src/llm_perf_opt/profiling/export.py`
   - Action:
     - Accept `stats['stage_ms']` in `write_stakeholder_summary`
     - Render a new table: “Per‑Stage Timings (ms)” with rows for present stages (prefill, decode, vision, sam, clip, projector)
     - Optionally compute and render shares (run‑level and prefill‑level) when sufficient data is present
4) Metrics JSON structure
   - File: `src/llm_perf_opt/runners/llm_profile_runner.py`
   - Action:
     - Include `aggregates.stage_ms` in the `summary` dict written to `metrics.json`
5) Documentation
   - Files:
     - `specs/001-profile-deepseek-ocr/quickstart.md`: add note about “Per‑Stage Timings (ms)” and share tables
     - `context/tasks/001-profile-deepseek-ocr/impl-phase-4-us2-summary.md`: mention the new tables under Stakeholder Summary
6) Validation
   - Run `pixi run stage1-run`
   - Verify `metrics.json` contains `aggregates.stage_ms`
   - Verify stakeholder summary includes “Per‑Stage Timings (ms)” table and accurate values (sanity: decode ≫ prefill on current setup)

## Data Shape (Example)
```json
{
  "aggregates": {
    "prefill_ms": {"mean": 240.706, "std": 1.073},
    "decode_ms": {"mean": 2297.598, "std": 8.626},
    "tokens": {"mean": 64.0, "std": 0.0},
    "tokens_per_s": {"mean": 27.856, "std": 0.104},
    "stage_ms": {
      "vision": {"mean": 180.0, "std": 5.0},
      "sam": {"mean": 100.0, "std": 3.0},
      "clip": {"mean": 60.0, "std": 2.0},
      "projector": {"mean": 20.0, "std": 1.0},
      "prefill": {"mean": 240.706, "std": 1.073},
      "decode": {"mean": 2297.598, "std": 8.626}
    }
  }
}
```

## Notes & Caveats
- `vision` sub‑stages are nested under `prefill`; do not sum `prefill + vision` for total runtime
- Report shares clearly: decode vs prefill (run‑level), and sam/clip/projector vs prefill (prefill‑level)
- Keep mdutils for Markdown table generation
- Maintain backward compatibility by preserving existing aggregate keys

## Rollback Plan
- If sub‑stage hooks are unstable on some environments, hide sub‑stage rows and only show prefill/decode/vision
- Feature gate via a small flag (e.g., `outputs.include_stage_ms`) if needed

