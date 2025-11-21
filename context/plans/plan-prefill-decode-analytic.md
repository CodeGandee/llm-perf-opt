# Plan: Prefill/Decode Analytic Cost vs Sequence Length

**Feature**  
Analytically estimate and validate DeepSeek‑OCR prefill/decode FLOPs/IO/memory as a function of sequence length using the ModelMeter analytic stack and Stage‑1/Stage‑2 profiling.

**Location**  
`context/plans/plan-prefill-decode-analytic.md`

**Date**  
2025-11-21

**Related**  
- `context/tasks/task-anything.md` (high‑level task: estimate prefill/decode cost vs seq_len, verify against measurements)  
- `extern/modelmeter/models/deepseek_ocr/HOLISTIC_ANALYSIS.md` (stateful analytic usage)  
- `context/tasks/refactor/task-refactor-dsocr-stateful-analytic-layers.md` (design for stateful analytic root + KV cache)  
- `src/llm_perf_opt/runners/dsocr_analyzer.py` (static analytic report generator)  

---

## 1. Executive Summary

We want a **quantitative, end‑to‑end view** of how DeepSeek‑OCR prefill and decode costs scale with **sequence length** and how well the **stateful analytic model** matches **empirical measurements**.

This plan describes how to:

1. Use the analytic stack (`DeepseekOCRModel` + `DeepseekV2DecoderLayer` + `SyntheticKVCache`) to compute **prefill** and **per‑token decode** FLOPs/IO/memory across a configurable grid of sequence lengths.  
2. Collect **measured** prefill/decode FLOPs and timings from real runs (Stage‑1/Stage‑2 style, using `DeepSeekOCRSession` and FLOP counters).  
3. Compare the analytic curves to empirical results and export artifacts (JSON/plots) for downstream MFU and capacity‑planning analysis.

The focus is on **single‑batch (B=1) OCR workloads** with varying text sequence lengths, but the design leaves room for batch and KV‑head sensitivity studies later.

---

## 2. Background and Current Building Blocks

### 2.1 Analytic model (ModelMeter)

We already have:

- Per‑layer analytic implementations for DeepSeek‑OCR under `extern/modelmeter/models/deepseek_ocr/layers/`, including:
  - Vision: `ImageEncoderViT`, `VitModel`, `MlpProjector`, `NoTP*`  
  - Decoder: `DeepseekV2DecoderLayer`, `DeepseekV2MLP`, `DeepseekV2MoE`, `DeepseekV2RMSNorm`, `MoEGate`  
  - LLaMA primitives: `LlamaFlashAttention2`, `LlamaRotaryEmbedding`
- A **stateful** analytic root model:
  - `DeepseekOCRModel` aggregates a vision stack (`_CompositeLayer`) and a repeated decoder layer.
  - It now tracks stage and KV meta state:
    - `operation_mode` → `"prefill"` / `"decode"`.
    - `start_prefill(context_len, batch_size, kv_cache=None) -> SyntheticKVCache`.
    - `start_decode(kv_cache=None) -> SyntheticKVCache`.
    - `decode_one_token() -> SyntheticKVCache`.
  - In `"prefill"` mode, `forward_*` stats answer “cost of prefill at `(context_len, batch_size)`”.
  - In `"decode"` mode, `forward_*` stats answer “cost of decoding **one more token** from the current KV state”.

`HOLISTIC_ANALYSIS.md` has already been updated to describe how to use these APIs for simulated prefill + decode runs.

### 2.2 Static analyzer

`DeepseekOCRStaticAnalyzer` in `src/llm_perf_opt/runners/dsocr_analyzer.py`:

- Instantiates analytic vision and decoder layers and wraps them in `DeepseekOCRModel.from_layers(...)`.
- Configures a representative prefill workload with `AnalysisConfig.seq_len` and calls `start_prefill(...)` on the root model.
- Uses `forward_*` on the configured model to populate an `AnalyticModelReport` (FLOPs, I/O, memory per module).

This gives us a **single‑point** analytic snapshot for a chosen `seq_len`, but not yet a **scaling curve** or explicit decode‑mode sweep.

### 2.3 Dynamic profiling / MFU

From existing code and docs:

- `DeepSeekOCRSession` and Stage‑1/Stage‑2 pipelines already:
  - run DeepSeek‑OCR with NVTX segmentation (`sam`, `clip`, `projector`, `prefill`, `decode`),  
  - collect timing and, in some flows, FLOP counts using `torch.utils.flop_counter.FlopCounterMode`.
- `docs/analyzer-mfu.md` and `context/plans/plan-per-stage-static-analysis.md` describe how to combine static/dynamic metrics into MFU estimates.

We will re‑use this machinery to get **measured prefill/decode FLOPs and timings** at selected sequence lengths.

---

## 3. Goals and Scope

### 3.1 Goals

1. **Analytic scaling curves**  
   - For a grid of `S_prefill` values (e.g., 256, 512, 1024, 2048, 4096):
     - Compute model‑level prefill FLOPs/IO/memory using `DeepseekOCRModel.start_prefill(...)`.
     - Compute per‑token decode FLOPs/IO/memory using `start_decode(...)` + `decode_one_token()`.
   - Export curves (JSON/CSV + optional plots) in a reproducible location (e.g., `tmp/profile-output/<run_id>/analytic-scaling/`).

2. **Empirical validation**  
   - For the same grid of `S_prefill` values:
     - Run real DeepSeek‑OCR inference (prefill + decode) using `DeepSeekOCRSession` with synthetic inputs.
     - Measure:
       - prefill FLOPs (and time),
       - per‑token decode FLOPs (and time) for a small, fixed number of tokens `K` (e.g., 16).  
   - Compare analytic vs measured FLOPs and annotate relative differences.

3. **Artifacts for downstream analysis**  
   - Produce machine‑readable artifacts:
     - `analytic_scaling.json` – analytic and measured metrics vs `S_prefill`.  
     - `analytic_scaling_plots.png` (optional) – prefill/decode FLOPs vs sequence length.  
   - Document a short “how to re‑run” section for the scaling experiment.

### 3.2 Non‑goals (for now)

- Modeling multi‑batch behavior (`B > 1`) beyond a fixed scaling factor.  
- Exploring MoE sparsity patterns beyond the existing analytic assumptions.  
- Generalizing to non‑DeepSeek‑OCR LLMs (we can factor out the code later, but this plan focuses on DeepSeek‑OCR).

---

## 4. Analytic Scaling Implementation

### 4.1 New helper: analytic scaling script

Add a small script (or CLI entry) that sweeps sequence length and uses the stateful analytic model:

- **Location (proposed)**:  
  `extern/modelmeter/models/deepseek_ocr/scripts/run_prefill_decode_scaling.py`

- **Inputs**:
  - HF config or a small `DeepseekOCRModelSpec` (hidden size, heads, layers, MoE config).  
  - Sequence length grid: `--seq-lens 256 512 1024 2048 4096`.  
  - Decode steps per point: `--decode-steps K` (default e.g. 16).  
  - Batch size (default 1).

- **Core flow (per `S_prefill`)**:

  ```python
  # 1) Instantiate analytic model from spec/config.
  vision_stack = _CompositeLayer([...])         # reuse pattern from run_verify_core
  decoder_layer = DeepseekV2DecoderLayer(...)
  analytic_model = DeepseekOCRModel.from_layers(
      vision_stack,
      decoder_layer,
      num_decoder_layers=num_layers,
  )

  # 2) Prefill analytics.
  kv_cache = analytic_model.start_prefill(context_len=S_prefill, batch_size=B, kv_cache=None)
  assert analytic_model.operation_mode == "prefill"
  F_prefill = (
      (analytic_model.forward_tensor_core_flops() or 0.0)
      + (analytic_model.forward_cuda_core_flops() or 0.0)
  )
  IO_prefill = analytic_model.forward_cal_io() or 0.0
  W_prefill = analytic_model.forward_memory_weight() or 0.0
  ACT_prefill = analytic_model.forward_memory_activation() or 0.0
  KV_prefill = analytic_model.forward_memory_kvcache() or 0.0

  # 3) Decode analytics for K steps.
  analytic_model.start_decode(kv_cache=None)  # reuse internal KV cache
  assert analytic_model.operation_mode == "decode"
  F_decode_total = 0.0
  IO_decode_total = 0.0
  ACT_decode_total = 0.0
  for _ in range(K):
      F_step = (
          (analytic_model.forward_tensor_core_flops() or 0.0)
          + (analytic_model.forward_cuda_core_flops() or 0.0)
      )
      IO_step = analytic_model.forward_cal_io() or 0.0
      ACT_step = analytic_model.forward_memory_activation() or 0.0

      F_decode_total += F_step
      IO_decode_total += IO_step
      ACT_decode_total += ACT_step

      analytic_model.decode_one_token()

  F_decode_per_token = F_decode_total / max(K, 1)
  IO_decode_per_token = IO_decode_total / max(K, 1)
  ACT_decode_per_token = ACT_decode_total / max(K, 1)
  KV_final = analytic_model.forward_memory_kvcache() or 0.0
  ```

- **Output schema** (`analytic_scaling.json`):

  ```json
  {
    "model": { "hidden_size": 1280, "num_layers": 40, ... },
    "batch_size": 1,
    "decode_steps": 16,
    "points": [
      {
        "seq_len_prefill": 256,
        "prefill": {
          "flops_tflops": 123.4,
          "io_tb": 0.56,
          "weights_gb": 4.2,
          "activations_gb": 1.1,
          "kv_gb": 0.3
        },
        "decode": {
          "flops_tflops_per_token": 0.45,
          "io_tb_per_token": 0.002,
          "activations_gb_per_token": 0.01,
          "kv_gb_final": 0.34
        }
      },
      ...
    ]
  }
  ```

### 4.2 Wiring into existing analytic reports (optional)

If useful, we can:

- Add a **mode** to `DeepseekOCRStaticAnalyzer` (e.g., `run_analytic_scaling(...)`) that:
  - reads a scaling config (seq_len grid, decode steps),  
  - internally calls the new analytic scaling helper, and  
  - writes results alongside the existing `AnalyticModelReport` artifacts.

This keeps the scaling workflow discoverable via a single CLI (`dsocr_analyzer`) without overloading the existing `run_analytic` path.

---

## 5. Empirical Verification Plan

### 5.1 Measurement strategy

For each `S_prefill` in the grid:

1. Build synthetic inputs compatible with `DeepSeekOCRSession.prepare_inputs(...)`:
   - Single synthetic image (as already done in `prepare_inputs`).
   - Text prompt truncated or padded to yield the target `seq_len` in the decoder input (including image tokens).
2. Run:
   - Prefill once (no cache),  
   - Decode `K` tokens (using `prepare_inputs_for_generation` and `use_cache=True`).

3. Wrap the relevant calls in a FLOP counter (where feasible):

   - **Preferred**: use `torch.utils.flop_counter.FlopCounterMode` on the HF modules (`DeepseekOCRForCausalLM` or `DeepseekV2Model`) to get measured FLOPs per stage, similar to `verify_by_impl` helpers in the analytic layers.
   - **Fallback**: rely on existing Stage‑1/Stage‑2 profiles (time‑only) and compare analytic FLOPs against measured times via MFU instead of FLOP‑to‑FLOP.

### 5.2 Implementation sketch

- **Location (proposed)**:  
  `src/llm_perf_opt/runners/dsocr_prefill_decode_scaling.py`

- **Responsibilities**:
  - Initialize `DeepSeekOCRSession.from_local(...)` with a configurable model path and device.  
  - For each `S_prefill`:
    - Construct or adapt inputs so that the decoder sees the desired sequence length.  
    - Measure prefill FLOPs and time.  
    - Run a decode loop for `K` tokens, measuring FLOPs/time per token (or averaged over K).  
  - Emit a `measured_scaling.json` with the same `points` structure as `analytic_scaling.json` plus timings.

- **Measured schema (per point)**:

  ```json
  {
    "seq_len_prefill": 1024,
    "prefill": {
      "flops": 1.23e13,
      "time_ms": 45.6
    },
    "decode": {
      "flops_per_token": 4.5e11,
      "time_ms_per_token": 1.2
    }
  }
  ```

---

## 6. Comparison and Reporting

### 6.1 Comparison logic

Implement a small comparison helper (could live in `scripts/` or under `src/llm_perf_opt/visualize/`) that:

- Loads `analytic_scaling.json` and `measured_scaling.json`.  
- Aligns points by `seq_len_prefill`.  
- Computes:
  - Relative FLOP differences:
    - `|analytic_prefill_flops - measured_prefill_flops| / measured_prefill_flops`.  
    - Same for decode flops_per_token.  
  - MFU‑style ratios:
    - `analytic_prefill_flops / (peak_tflops * measured_prefill_time_s)`.  
    - Same for decode.

Outputs:

- A small text/markdown summary with:
  - Per‑`S_prefill` relative differences (prefill & decode).  
  - Aggregate statistics (mean/median/max diff).  
  - Optional recommendation if discrepancies exceed a threshold (e.g., 5–10%).

### 6.2 Visualization (optional)

- Use matplotlib or a simple plotting helper to generate:
  - `F_prefill(S)` and `F_decode_per_token(S)` curves (analytic vs measured).  
  - `time_prefill(S)` and `time_decode_per_token(S)` curves.  
- Store plots alongside JSON artifacts in a run‑specific directory under `tmp/profile-output/<run_id>/prefill-decode-scaling/`.

---

## 7. Validation and Guardrails

1. **Unit tests (analytic side)**:
   - Add tests to verify that:
     - `start_prefill` followed by `start_decode` and `decode_one_token` produce monotonic KV‑cache growth via `SyntheticKVCache`.  
     - Prefill FLOPs and per‑token decode FLOPs scale monotonically with `S_prefill` for a fixed config.
2. **Sanity checks (empirical side)**:
   - For a single `S_prefill`, manually compare:
     - Analytic prefill FLOPs vs `DeepseekOCRModel`’s `verify_by_impl` style FLOP measurements (where available).  
     - Analytic decode FLOPs for the first few tokens vs a direct FLOP‑counted single‑step decode.
3. **Performance guardrails**:
   - Keep decode steps `K` small (e.g., 16–32) for profiling, to avoid very long runs.  
   - If FLOP counting proves too heavy on large configs, fall back to time‑only comparison and MFU‑style validation.

---

## 8. Risks and Open Questions

- **FLOP counter limitations**: `FlopCounterMode` may not support all custom operators or KV‑cache paths cleanly; we may need to rely on time‑only comparison for some segments.  
- **Sequence length control**: precisely shaping `S_prefill` at the HF level (especially with vision tokens) may require careful prompt and mask construction; we should document the mapping from `AnalysisConfig.seq_len` to actual decoder sequence length.  
- **Model variants**: DeepSeek‑OCR config variations (different MoE settings, context windows) may require multiple analytic scaling runs; the plan assumes one canonical config to start.

Despite these risks, the work is incremental on top of the existing stateful analytic model and profiling infrastructure, and can be developed in stages (analytic sweep first, then empirical validation, then visualization/reporting).

