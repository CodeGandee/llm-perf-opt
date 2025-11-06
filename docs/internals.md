# Internals

Session wrapper (`dsocr_session.py`)
- Loads tokenizer/model once (`from_local`) and installs optional NVTX hooks for SAM, CLIP, and projector submodules.
- Inference methods:
  - `run_inference_generate(...)` — Legacy vendor-parity method using `generate()` for both prefill and decode in one call. Suitable for quick validation but less precise for profiling.
  - `run_inference(...)` — Explicit prefill + decode split with NVTX ranges:
    - Preprocess: load PIL image, pad to `base_size`, optional dynamic crops to `image_size` grid (3×2 etc.).
    - Token assembly: mirrors vendor image token span sizing using `patch_size=16` / `downsample_ratio=4` with base and crop grids.
    - Prefill: forward once with `images`, `images_seq_mask`, `images_spatial_crop` (builds KV cache).
    - Decode: token-by-token loop using KV cache; controllable via `infer` config.
  - Both methods log timings and tokens; return dictionary (prefill_ms, decode_ms, tokens, optional text).

Runner (`llm_profile_runner.py`)
- Discovers images (`dataset.root` + `fallback_patterns` or `subset_filelist`).
- Profiles a representative image with PyTorch profiler; collects operator records.
- Repeats runs over dataset; aggregates timings and MFU.
- Writes `torch_profiler/report.md`, `torch_profiler/operators.md`, `torch_profiler/metrics.json`, and optional predictions/viz/`llm_profile_runner.log`. Writes `static_analysis/static_compute.{json,md}` when enabled.
- Writes reproducibility artifacts: `env.json`, `inputs.yaml`, `assumptions.md`.
- Can run a static analyzer (configurable) to estimate per‑stage FLOPs; uses those with measured times to compute improved MFU (prefill total FLOPs; decode per‑token FLOPs × tokens; vision from sam+clip+projector).

Configuration structure
- `conf/model/<name>/arch/<name>.default.yaml` — architecture + preprocessing
- `conf/model/<name>/infer/<name>.default.yaml` — inference knobs
- `conf/config.yaml` composes all pipelines from one entry:
  - mounts `profiling/torch@pipeline.torch_profiler`, `profiling/nsys@pipeline.nsys`, `profiling/ncu@pipeline.ncu`
  - toggles are under `pipeline.*.enable`

NVTX ranges
- High-level ranges for prefill/decode (`profiling/nvtx_utils.py`).
- Optional submodule hooks for SAM/CLIP/projector in the session (forward pre/post hooks).
- These enable clear stage timing and help correlate with Nsight tools.

MFU estimation
- Analyzer‑based FLOPs: `DeepseekOCRStaticAnalyzer` (fvcore + analytic fallbacks) provides stage FLOPs.
- Per‑stage MFU uses analyzer FLOPs with measured stage times; model‑level MFU sums prefill + decode FLOPs over their combined time (vision nested within prefill, not double counted).
