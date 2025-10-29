# Analyzer & MFU

This page explains how the static analyzer estimates FLOPs for DeepSeek‑OCR and how those estimates combine with measured timings to compute MFU (Model FLOPs Utilization).

- Static analyzer: `src/llm_perf_opt/runners/dsocr_analyzer.py`
- MFU helpers: `src/llm_perf_opt/profiling/mfu.py`
- Peak TFLOPs lookup: `src/llm_perf_opt/profiling/hw.py`

## Goals
- Derive consistent per‑stage FLOPs using fvcore with analytical fallbacks.
- Combine FLOPs with measured stage timings to compute per‑stage MFU and a model‑level MFU.
- Keep “vision” cost transparent (SAM + CLIP + projector) without double‑counting it as a separate stage in the model‑level MFU.

## Workflow
1) Build representative inputs
- Pads a global view to `base_size` and optional dynamic local crops at `image_size`.
- Mirrors vendor image token span sizing (`patch_size=16`, `downsample_ratio=4`).
- See `DeepseekOCRStaticAnalyzer.prepare_inputs(...)` in `src/llm_perf_opt/runners/dsocr_analyzer.py`.

2) Full‑model static pass (fvcore)
- Wraps the model so fvcore can trace a single forward with positional args.
- Collects totals and by‑module/by‑operator FLOPs and activation counts.
- See `analyze_full_model(...)` in `src/llm_perf_opt/runners/dsocr_analyzer.py`.

3) Per‑stage analyses and fallbacks
- Isolated fvcore passes when feasible: SAM, CLIP, projector.
- Analytical formulas for LLM: prefill (total), decode (per‑token).
- Stage mapping uses name prefixes to extract module‑scoped FLOPs when full pass succeeds.
- See `analyze_all_stages(...)` and helpers in `src/llm_perf_opt/runners/dsocr_analyzer.py`.

4) Merge results into a report
- Stage entries include: params, flops, activations, operator mix, and notes.
- Report includes metadata (shapes, crop mode, fvcore version) and high‑level notes.
- See `generate_report(...)` in `src/llm_perf_opt/runners/dsocr_analyzer.py`.

## Key Inputs & Assumptions
- Shapes: `AnalysisConfig(image_h, image_w, base_size, image_size, seq_len, crop_mode)` drive FLOP counts.
- Dtypes/devices: analysis uses `bfloat16` tensors on the target device.
- Vision features: CLIP/projector isolated passes may use real SAM outputs (preferred) or mock shapes if tracing fails.
- LLM formulas approximate dense transformer math; FlashAttention or fused kernels change constants but preserve scaling.
- Vision stage = SAM + CLIP + projector. We show its cost on a separate line for transparency; it is nested within prefill and is not a standalone stage for model‑level MFU aggregation.

## FLOP Models (LLM)
- Prefill total FLOPs: `estimate_prefill_flops_total(d_model, d_ff, n_layers, seq_len)`.
  - Closed‑form of growing attention context from 1..L.
- Decode per‑token FLOPs: `estimate_decode_flops_per_token(d_model, d_ff, n_layers, Lctx)`.
  - Effective context `Lctx` chosen via `select_decode_context_len(mode)`: auto (Lp + 0.5·T), fixed, or max.
- See `src/llm_perf_opt/profiling/mfu.py`.

## Timing and NVTX
- Measured stage times come from the runner with NVTX ranges for prefill/decode and optional submodule hooks for SAM/CLIP/projector.
- PyTorch Profiler often attributes “CUDA time” primarily to kernel entries (operator rows may show 0). We synchronize after runs, and we surface kernel‑level timing in operator/kernels tables; see troubleshooting for details.
- See `docs/troubleshooting.md` and summary `context/summaries/issue-pytorch-profiler-zero-cuda-time.md`.

## MFU Computation
- Per‑stage MFU uses analyzer FLOPs divided by (peak TFLOPs × measured seconds) for that stage.
- Model‑level MFU aggregates prefill + decode FLOPs over combined wall time. Vision is not double‑counted (nested within prefill).
- Helper: `compute_stage_mfu(...)` in `src/llm_perf_opt/profiling/mfu.py`.

Formula sketch
- `MFU = Achieved_FLOPs_per_s / Peak_FLOPs_per_s`.
- Example (decode): `MFU_decode = (T * FLOPs_per_token) / (Peak_TFLOPs×1e12 × t_decode_seconds)`.

## Peak TFLOPs Handling
- Lookup uses a small device table with environment override: `MFU_PEAK_TFLOPS`.
- If a GPU is unknown, default is `100.0`; set an override or extend the table.
- See `get_peak_tflops(...)` in `src/llm_perf_opt/profiling/hw.py` and references in `context/hints/nv-profile-kb/peak-tflops-reference.md`.

## Repro and Outputs
- Runner writes static analysis artifacts alongside benchmarks: `static_compute.json` and `static_compute.md`, plus `metrics.json` and summary pages containing MFU tables.
- To run a representative pipeline:
  - `pixi run stage1-run`
  - Optionally override peak TFLOPs: `MFU_PEAK_TFLOPS=142 pixi run stage1-run`
- See `docs/running.md` and `docs/artifacts.md` for artifact locations and formats.

## Limitations & Caveats
- fvcore tracing can miss custom ops; we fall back to analytical estimators and note gaps.
- FLOPs are shape‑dependent; change `base_size`, crops, or `seq_len` to reflect your workload.
- Mixed precision and fused kernels alter real compute vs nominal FLOPs; MFU is a heuristic indicator, not a guaranteed hardware utilization metric.

## References
- Peak TFLOPs notes and links: `context/hints/nv-profile-kb/peak-tflops-reference.md`
- Operator and kernel timing guidance: `docs/visualization.md`, `docs/troubleshooting.md`
- Static analyzer implementation walkthrough: `STATIC_ANALYZER_IMPLEMENTATION.md`
