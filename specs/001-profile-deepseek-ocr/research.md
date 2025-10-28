# Phase 0 Research: DeepSeek‑OCR Stage 1 Profiling

This document resolves all clarifications and establishes best‑practice choices for Stage 1.

## Unknowns and Decisions

1) Decision: MFU estimation methodology and target accuracy
- Rationale: Provide directional MFU using measured throughput and analytical FLOPs/token. Stage 1 must be stable and actionable without Nsight.
- Decision: Compute MFU = (Achieved FLOPs) / (Theoretical Peak FLOPs). Achieved FLOPs derived from tokens/sec × FLOPs/token (decode) and analytical estimate for prefill (vision encoder). Target accuracy: MFU within ±15% of a later Nsight‑based baseline; variance across repeats within ±10% (per SC‑002).
- Alternatives considered: (a) Nsight Systems/Compute for MFU — heavier tooling, deferred to Stage 2; (b) GPU Active Cycles proxy — less interpretable and hardware‑specific.

2) Decision: FLOPs per token model (decode, causal transformer)
- Rationale: Early estimate needed without deep kernel analysis.
- Decision: Use model config to approximate per‑token FLOPs for decode as a dense transformer:
  FLOPs/token ≈ N_layers × [4·d_model·d_ff + 2·d_model² + 2·d_model·(d_k+d_v) + AttnMatMulCost], with AttnMatMulCost simplified to 2·d_model·S for Stage 1 where S is effective KV length. For Stage 1, default S to a small constant window (e.g., 512) unless explicitly provided; recommend logging actual decode context length to refine.
- Alternatives considered: (a) Exact kernel‑level FLOP summation — complex for Stage 1; (b) Pure parameter‑count heuristics — too coarse for decode growth with S.

3) Decision: Prefill FLOPs model (vision encoder)
- Rationale: Prefill includes image embedding/encoder forward pass.
- Decision: If encoder config is available (ViT‑like), approximate FLOPs ≈ N_layers × [2·d_model² + 2·d_model·d_mlp] × N_patches plus attention terms (2·d_model·N_patches) simplified. If config unavailable, perform a one‑time calibration run with PyTorch Profiler to sum matmul‑like ops under the `prefill` NVTX range and cache the resulting GFLOPs constant for the selected model resolution. Use that constant for subsequent runs.
- Alternatives considered: (a) Nsight Compute per‑kernel FLOP metrics — deferred; (b) Blind time‑based MFU without compute — not analytically grounded.

4) Decision: Operator‑level summary tool
- Rationale: Lightweight, scriptable, and Python‑native.
- Decision: Use PyTorch Profiler (CPU+CUDA activities) over the full run with NVTX range annotations; export aggregated operator table (self/total time, cuda time, calls, estimated memory where available) to Markdown.
- Alternatives considered: Nsight Systems operator summaries — heavier setup; TB profiler — less flexible for CLI export.

5) Decision: NVTX segmentation pattern
- Rationale: Clear prefill vs decode attribution and readable timelines.
- Decision: Wrap the first forward pass after tensors are on‑device with `nvtx.range_push("prefill")`/`range_pop()`. Wrap the token‑generation loop with `nvtx.range_push("decode")`/`range_pop()`. Optionally nest per‑step ranges `decode_step[i]` for fine granularity.
- Alternatives considered: Timestamp‑only wall‑clock sections — not visible in Nsight; profiler steps without NVTX — harder to segment.

6) Decision: Repetitions and aggregation
- Rationale: Improve stability and meet SC‑002/SC‑005.
- Decision: Default repeats = 3 for a given input set. Report mean and standard deviation for stage timings, tokens/sec, and MFU. Allow override via CLI/env.
- Alternatives considered: Single pass — higher variance; >3 passes — diminishing returns for Stage 1.

7) Decision: Dataset composition and location
- Rationale: Reproducibility and coverage.
- Decision: Use 10–20 images spanning text‑heavy, mixed‑layout, and image‑rich documents under /data2/huangzhe/code/llm-perf-opt/data/samples. Allow override via `DSOCR_IMAGE` (file or directory). Document the exact list used in the report metadata.
- Alternatives considered: Synthetic images — less representative.

8) Decision: Batch size and decoding settings
- Rationale: Stable, comparable results.
- Decision: Batch size = 1, greedy decoding, `max_new_tokens` fixed (e.g., 64 for fallback generate). For DeepSeek‑OCR `.infer`, use default arguments unless explicitly set. Expose overrides via env (e.g., `DSOCR_MAX_NEW_TOKENS`, `DSOCR_USE_FLASH_ATTN`).
- Alternatives considered: Beam or sampling — increases variance.

9) Decision: Hardware detection and theoretical peak
- Rationale: Needed for MFU denominator.
- Decision: Detect GPU via `torch.cuda.get_device_properties` and NVML. Map device name to theoretical BF16/FP16 peak TFLOPs using an internal table (A100/H100, 4090, etc.) with an environment variable override `MFU_PEAK_TFLOPS` for unlisted devices. Log the selected precision path (bf16 preferred) and corresponding peak.
- Alternatives considered: Derive from clocks/SM count dynamically — error‑prone without low‑level telemetry.

## Consolidated Choices (TL;DR)

- MFU: tokens/sec × FLOPs/token ÷ peak TFLOPs; ±15% target vs Nsight baseline.
- Prefill compute: analytic ViT‑like approximation; fallback one‑time profiler calibration per model/resolution.
- Profiler: PyTorch Profiler (CPU+CUDA), NVTX ranges for prefill/decode segmentation; export operator table.
- Repeats: 3 by default; mean/std aggregation.
- Dataset: 10–20 images under `/data/samples` (absolute path resolved under repo root) with overrides.
- Settings: bs=1, greedy decoding; flash‑attn optional with fallback.
- Hardware: torch+NVML detection; internal TFLOPs table with env override.

## Next Steps

Use these decisions to drive Phase 1 artifacts: data‑model schemas, OpenAPI contracts for a future service/CLI, and a quickstart that binds environment, dataset, and manual scripts to the Stage 1 workflow.

