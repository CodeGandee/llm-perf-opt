# Phase 3 Revision Plan — MFU Computation Accuracy

Feature: Basic Profiling for DeepSeek-OCR (Stage 1)
Location: /data2/huangzhe/code/llm-perf-opt/context/tasks/001-profile-deepseek-ocr/revision-phase-3-mfu-compute.md
Date: 2025-10-29

## Summary

MFU in the current Stage 1 report is conservatively low because we (a) divide by an optimistic peak TFLOPs estimate, (b) use a simplified FLOPs/token formula with a fixed context length, and (c) do not attribute vision/prefill compute to per-stage MFU. This plan revises MFU computation to be configurable, data-driven, and per-stage, while keeping the rest of Phase 3 unchanged (no throughput optimization work).

## Goals

- Add configuration to control decoder context length used in MFU math (fixed/auto/max).
- Compute a static computational breakdown (params and FLOPs) using `fvcore` and export a report.
- Provide per-stage MFU (vision, prefill, decode) using static compute + measured timings.
- Add warmup rounds for profiling to stabilize measurements; optionally use synthetic inputs.
- Maintain compatibility with existing outputs and keep ruff/mypy clean.

## Non-Goals

- No throughput improvements (e.g., batching, kernel fusion, runtime swaps) in this revision.
- No Nsight integrations beyond current scope.

---

## Configuration Additions

All new keys are additive and default to existing behavior when not provided.

1) Decoder Context Length Control (inference group)
- File: conf/model/<model>/infer/<model>.<variant>.yaml
- Keys:
  - `context_len_mode: {auto|max|fixed}`
    - `auto` (default): use actual measured lengths from a run (prefill length Lp; for decode, use either per-step or average context length).
    - `max`: use model window size (e.g., `config.max_position_embeddings` or derived) as worst-case context.
    - `fixed`: use the fixed value below.
  - `context_len_fixed: <int>` (used only when `fixed`)

Example (Default):
```
# conf/model/deepseek_ocr/infer/deepseek_ocr.default.yaml
context_len_mode: auto
context_len_fixed: 1024
```

2) Torch Profiler Warmup (profiling group)
- File: conf/profiling/torch/torch-profiler.{min|default|max}.yaml
- Keys:
  - `warmup_rounds: <int>` (default 0)
  - `warmup_synthetic: <bool>` (default true) — if true, warm up with synthetic random inputs matching shapes; else use first real sample.

Example (default preset):
```
# conf/profiling/torch/torch-profiler.default.yaml
warmup_rounds: 2
warmup_synthetic: true
```

---

## API Additions (src)

1) Static compute report (fvcore) — DeepSeekOCRSession
- File: src/llm_perf_opt/runners/dsocr_session.py
- New method:
  - `def estimate_static_compute(self, image_h: int = 1024, image_w: int = 1024, seq_len: int = 1024) -> dict:`
    - Uses `fvcore.nn.FlopCountAnalysis` on a minimal forward graph to estimate FLOPs for:
      - Vision stack (SAM/CLIP/projector) for one image (global view + typical crops)
      - LLM prefill (seq_len tokens)
      - LLM decode (1 token at representative context length)
    - Returns a dictionary with params and FLOPs per block/stage.

- New writer utility (re-uses mdutils):
  - `write_static_compute_report(data: dict, artifacts_dir: Path) -> None`
    - Writes `static_compute.json` and `static_compute.md` under `tmp/stage1/<run_id>/`.

2) MFU utilities (profiling/mfu.py)
- Extend existing helper to accept:
  - `context_len_mode`, `context_len_fixed`, measured `prefill_len`.
  - Return both model-level and per-stage MFU given stage FLOPs and stage times.

3) Warmup controls (runner)
- File: src/llm_perf_opt/runners/llm_profile_runner.py
- Before representative profiling:
  - If `cfg.profiling.warmup_rounds > 0`:
    - If `warmup_synthetic`, synthesize one random image (H=W=base_size) and a minimal prompt; call session.run_inference with short `max_new_tokens`.
    - Else, run warm ups on the first dataset image with short `max_new_tokens`.

---

## MFU Formulation (Proposed)

Symbols
- `T`: new tokens generated
- `Lp`: measured prefill length (tokens after prompt assembly)
- `ctx_len_mode`: {auto|max|fixed}
- `Lctx`: context length used for decode MFU
- `F_decode(Lctx)`: decode FLOPs per token at context `Lctx`
- `F_prefill(Lp)`: total prefill FLOPs for sequence length `Lp` (approx. sum over i=1..Lp of per-token cost)
- `F_vision`: vision FLOPs per image (from fvcore analysis)
- `P_peak`: calibrated peak TFLOPs (we keep the same lookup for now)
- `t_prefill`, `t_decode`, `t_vision`: measured stage times

Context length
- `auto`: `Lctx = Lp + 0.5*T` (average attention length across the T decode steps)
- `fixed`: `Lctx = context_len_fixed`
- `max`: `Lctx = model_window`

FLOPs
- Decode FLOPs per token (single layer):
  - `F_attn ≈ 2·d_model·d_k·h + 2·Lctx·d_model` (QKV projections + attention matmul; constants adjusted to impl)
  - `F_mlp ≈ 2·d_model·d_ff + 2·d_ff·d_model`
  - Sum across layers; include `lm_head` projection.
- Prefill FLOPs: accumulate with `Lctx = 1..Lp` or approximate with integral.
- Vision FLOPs: from fvcore pass over the vision encoder (global view + representative number of crops).

MFU
- Stage MFU:
  - `mfu_decode = (T · F_decode(Lctx)) / (P_peak · t_decode)`
  - `mfu_prefill = F_prefill(Lp) / (P_peak · t_prefill)`
  - `mfu_vision = F_vision / (P_peak · t_vision)`
- Model-level MFU: weighted by wall-clock using stage FLOPs:
  - `mfu_model = (F_vision + F_prefill(Lp) + T · F_decode(Lctx)) / (P_peak · (t_vision + t_prefill + t_decode))`

Notes
- Constants (2×, etc.) should be verified against the current DeepSeek-OCR implementation and HF modules.
- For Stage 1, it is acceptable to use (a) fvcore analysis + (b) analytic decode FLOPs.

---

## Per-Stage Attribution

We will use existing NVTX hooks and logger timings to bound stage durations:
- Vision: wrapper around SAM/CLIP/projector combined — measured via forward hooks (`dsocr_session._install_nvtx_stage_hooks`) and/or wall-clock around preprocessing + first part of forward.
- Prefill: we already time the prefill forward.
- Decode: we already time the generate() call.

For Stage 1, combining hook timings and wall-clock is adequate to compute MFU per stage. Later phases can refine vision/prefill segmentation if needed.

---

## Deliverables

- `tmp/stage1/<run_id>/static_compute.json` and `static_compute.md` (fvcore static report)
- Report updates:
  - `report.md` MFU section includes vision/prefill/decode MFU
  - Context length section with mode and values used
- Config additions and docs updated in `docs/configuration.md`

---

## Task Breakdown

- T060 Add inference context length controls (auto/max/fixed) to `conf/model/deepseek_ocr/infer/deepseek_ocr.default.yaml`; read in runner.
- T061 Implement decode context length selection in `profiling/mfu.py` and switch runner to use it.
- T062 Add `estimate_static_compute()` to `src/llm_perf_opt/runners/dsocr_session.py` using fvcore; expose writer for `static_compute.{json,md}`.
- T063 Compute `F_vision`, `F_prefill`, and `F_decode` using fvcore + analytic helpers; plumb outputs to runner.
- T064 Add per-stage MFU in the summarization step and include in report.md.
- T065 Add warmup configuration in `conf/profiling/torch/torch-profiler.*.yaml` (`warmup_rounds`, `warmup_synthetic`) and implement in runner.
- T066 Update docs: configuration and internals to describe MFU math and configs.
- T067 Validate on 3–5 images; compare decode MFU across modes (auto vs fixed vs max) and ensure stable ranges.

---

## Acceptance Criteria

- Report shows `mfu_model` and `mfu_per_stage` with non-zero prefill/vision values on images that exercise those paths.
- Changing `context_len_mode` affects decode MFU in the expected direction (max ≤ fixed ≤ auto for typical runs).
- Static compute report is generated without errors and numbers are consistent order-of-magnitude with analytic estimates.
- Warmup rounds reduce variance of stage timings compared to cold runs.
- ruff/mypy: PASS on src/.

---

## Risks & Notes

- fvcore FLOP analysis for custom modules may need small adapters; where unsupported, fall back to analytic formulas and document assumptions.
- Time attribution for “vision” may include some I/O or PIL transforms; we will keep it documented and focus on major compute blocks.
- Peak TFLOPs is still a lookup; an optional override can be introduced later for calibrated values.
