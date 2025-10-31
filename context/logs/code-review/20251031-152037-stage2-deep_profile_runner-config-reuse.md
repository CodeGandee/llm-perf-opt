# Stage‑2 deep_profile_runner — Config Reuse & NSYS/NCU Gating Review

**Scope**
- File under review: `src/llm_perf_opt/runners/deep_profile_runner.py`
- Goal: Verify compatibility with unified `conf/config.yaml` and propose changes so Stage‑2 reuses Stage‑1 configs where applicable. Also validate NSYS/NCU gating and artifacts layout.

**Summary**
- The runner largely aligns with the unified config (uses `conf/config.yaml` and `pipeline.*`) and enforces correct NSYS NVTX gating semantics. A few issues remain:
  1) Hydra override `dataset/sampling@dataset.sampling=default` should be removed (or prefixed with `+`). Prefer direct `dataset.sampling.*` keys and rely on defaults.
  2) Hard-coded `infer.max_new_tokens=64` should be replaced by Stage‑1’s value or a dedicated `run.representative_max_new_tokens` knob.
  3) `nsys_summary_base` is referenced unconditionally; predeclare and guard existence.
  4) Permit NCU runs without NSYS by handling missing NSYS summary (use `kernel_regex=None`).
  5) Use the shared Stage‑1 inference engine explicitly (either `llm_profile_runner` with profiling disabled, or `direct_inference_runner`).
  6) Reuse dataset sampling and model/infer knobs from Stage‑1; override only when necessary for Nsight stability.

**Details & Rationale**

- 1) Hydra overrides for dataset sampling
  - Current: pushes `dataset/sampling@dataset.sampling=default` (line ~128); without leading `+` it errors.
  - Recommendation: drop this override (preset already mounted) or use `+dataset/sampling@dataset.sampling=default`. Keep direct keys: `dataset.sampling.num_epochs=1`, `dataset.sampling.num_samples_per_epoch=<N>`, `dataset.sampling.randomize=false`.

- 2) Reuse `max_new_tokens`
  - Current: hard-coded `infer.max_new_tokens=64`.
  - Recommendation: reuse `cfg.infer.max_new_tokens`, or add `run.representative_max_new_tokens` and prefer it if set. Avoid silent divergence that impacts decode length and capture windows.

- 3) NSYS summary scope
  - Issue: `nsys_summary_base` only defined when report exists; later referenced unconditionally.
  - Recommendation: `nsys_summary_base = artifacts.out_dir("nsys") / "summary"` early; only compute kernels if the CSV exists. Enables NCU even when NSYS is disabled.

- 4) NCU without NSYS
  - Issue: assumes NSYS CSV for top kernels.
  - Recommendation: if no NSYS CSV, set `kernel_regex=None` and proceed. Keep rerun-on-"No kernels were profiled" logic.

- 5) Workload path clarity
  - Option A: `llm_profile_runner` with `pipeline.torch_profiler.enable=false` (current behavior, fine since Stage‑1 uses shared engine).
  - Option B: `direct_inference_runner` to emphasize “no profiling” and reduce I/O by disabling visualization by default during captures.

- 6) Config reuse across stages
  - Dataset: pass through `dataset.root`, `dataset.subset_filelist`, `dataset.fallback_patterns`.
  - Sampling: reuse `dataset.sampling.*`; only set explicit epochs/samples when needed.
  - Model/infer: reuse `model.*` and `infer.*`; only change what Nsight requires.
  - Outputs: keep workload outputs in `tmp/workload/torch_profiler/` to avoid collisions.

**NSYS Gating Semantics**
- Correct behaviors implemented:
  - Error when `capture_range=nvtx` and `nvtx_capture` omitted/empty.
  - `capture_range_end` supported; omitted when null/empty.
  - `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` passed to match dynamic strings.
- Enhancement (optional): expose domain include/exclude flags (`--nvtx-domain-include/--nvtx-domain-exclude`).

**Concrete Suggested Changes**
- Remove or prefix with `+`: `"dataset/sampling@dataset.sampling=default"`.
- Replace: hard-coded `"infer.max_new_tokens=64"` with pass-through from `cfg.infer.max_new_tokens`, or honor `run.representative_max_new_tokens` if present.
- Predeclare: `nsys_summary_base = artifacts.out_dir("nsys") / "summary"`; guard usage with CSV existence.
- Kernel selection: derive `kernel_regex` only when NSYS summary CSV exists; else use `None`.
- Optional: Switch workload to `llm_perf_opt.runners.direct_inference_runner` and disable visualization by default for capture runs.

**Compatibility With Unified Config**
- Uses `@hydra.main(..., config_name="config")` — correct.
- Reads `pipeline.nsys.*` and `pipeline.ncu.*` — correct/future‑proof.
- Deprecate `run.stage1_repeats` in favor of `dataset.sampling.*` (mapping exists; update docs accordingly).

**Tests To Add**
- NSYS error path: `capture_range=nvtx` with empty/omitted `nvtx_capture` raises.
- NSYS `capture_range_end` passed only when set.
- Workload argv builder receives device, subset filelist, and sampling keys as expected.
- NCU runs both with/without NSYS summary; rerun path triggers on “No kernels were profiled”.

**References**
- File: `src/llm_perf_opt/runners/deep_profile_runner.py`
- Nsight Systems CLI User Guide (capture-range, nvtx-capture, repeat)
- Config entrypoint/presets: `conf/config.yaml`, `conf/profiling/nsys/nsys.default.yaml`
