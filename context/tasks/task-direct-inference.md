# Design: `pipeline.direct_inference` stage (inference without profiling)

## What to Add
- A new pipeline stage `pipeline.direct_inference` that runs model inference over the dataset without any profiling overhead (no PyTorch profiler, Nsight Systems, or Nsight Compute).
- The stage appears before `pipeline.static_analysis` in our recommended execution order and follows the existing per‑pipeline config pattern.
- Output controls mirror other stages using the unified schema:
  - `pipeline.direct_inference.enable: bool`
  - `pipeline.direct_inference.output.prediction.{enable, save_dir}` (default `enable=false`, `save_dir=pred`)
  - `pipeline.direct_inference.output.visualization.{enable, save_dir}` (default `enable=true`, `save_dir=viz`)
  - `pipeline.direct_inference.output.extra.<model-name>.*` for model‑specific knobs (e.g., DeepSeek‑OCR visualization controls)

## Why Add This Stage
- Fast, zero‑overhead runs to validate data/model wiring, produce baseline predictions, and generate visualizations without the runtime cost of profilers.
- Clear separation of concerns: “do the task” (inference) vs. “measure the task” (profilers). This avoids conflating timing/overhead and keeps outputs easier to reason about.
- Consistent UX: follows the unified `pipeline.*` layout, per‑stage `output.*` controls, and the dataset sampling semantics already in place.

## How to Integrate

### 1) Config schema and defaults
- Extend `conf/config.yaml` Hydra defaults to mount stage outputs and model‑specific extras:

```yaml
defaults:
  # … existing defaults …
  - output/direct@pipeline.direct_inference.output: default
  - model/deepseek_ocr/output/direct@pipeline.direct_inference.output.extra.deepseek_ocr: default
  - _self_
```

- Add the stage block under `pipeline:` alongside others:

```yaml
pipeline:
  direct_inference:
    enable: false
    # output.* is mounted via defaults -> pipeline.direct_inference.output
  static_analysis:
    enable: true
    # … existing keys …
  torch_profiler:
    enable: ${pipeline.torch_profiler.enabled}
  nsys:
    enable: false
  ncu:
    enable: false
```

- New presets (symmetry with torch stage outputs):
  - `conf/output/direct/default.yaml` (model‑agnostic output controls)
  - `conf/model/deepseek_ocr/output/direct/default.yaml` (model‑specific extras)

Example contents:

```yaml
# conf/output/direct/default.yaml
prediction:
  enable: false
  save_dir: pred    # relative to this stage’s output dir
visualization:
  enable: true
  save_dir: viz     # relative to this stage’s output dir

# conf/model/deepseek_ocr/output/direct/default.yaml
prediction:
  strip_special_tokens: false
visualization:
  max_images: 16
  thumbnail_width: 480
```

Notes:
- Relative `save_dir` resolves under `Artifacts.out_dir("direct_inference")`.
- Defaults align with other stages: if `enable=true` and `save_dir` is omitted/null, default to `pred`/`viz` respectively.

### 2) Orchestration and ordering
- Recommended execution order (when enabled):
  1. `direct_inference` (fast predictions + visualizations)
  2. `static_analysis`
  3. `torch_profiler`
  4. `nsys`
  5. `ncu`

- The orchestrator respects `pipeline.<stage>.enable` flags. `direct_inference` neither starts nor configures any profilers. It strictly reuses the dataset loop and model session.

### 3) Runner implementation approach (Option B: shared engine)
- Minimize duplication by extracting a shared inference engine from Stage‑1 (`llm_profile_runner`) and reuse it for both `torch_profiler` and `direct_inference`.

Refactor steps:
1. Extract an orchestrator from `llm_profile_runner` that owns the dataset loop and session calls, e.g. a class or function:
   - `InferenceEngine.run(dataset_cfg, model_cfg, infer_cfg, outputs_cfg, artifacts, hooks)`
   - Hooks provide optional profiler contexts and stage‑specific side‑effects.
2. Define `ProfilingHooks` with no‑op defaults:
   - Callbacks: `on_epoch_start`, `on_iter_start`, `on_iter_end`, `on_epoch_end`.
   - `context_provider()` yields a context manager; default is a no‑op context.
   - Torch profiler supplies a context provider wrapping `torch.profiler.profile(...)` (reuse existing logic behind this hook).
   - `direct_inference` uses default no‑ops (hence no profiling), while keeping NVTX tags within `dsocr_session`.
3. Parameterize stage I/O:
   - Pass `stage_name` to resolve `Artifacts.out_dir(stage_name)` and `Artifacts.tmp_dir(stage_name)`.
   - Feed `pipeline.<stage>.output.*` into the engine for prediction/visualization.
4. Enforce integer‑only `infer.max_new_tokens` centrally in the engine and fail fast on invalid values.
5. Provide a thin entrypoint for `direct_inference` (can be a small module) that loads config, checks `pipeline.direct_inference.enable`, builds default hooks, and invokes the engine with `stage_name="direct_inference"`.

Notes:
- NVTX markers in `dsocr_session` remain; they do not activate external profilers and are negligible overhead.
- This guarantees identical inference/tokenization/viz paths across stages, differing only by the profiling hooks.

### 4) Outputs and layout
- Root: `${hydra:run.dir}` (unchanged)
- Stage output: `${run.dir}/direct_inference/`
  - `pred/` — when `prediction.enable=true` (format mirrors torch stage; optionally add an aggregated `predictions.tsv`)
  - `viz/` — when `visualization.enable=true` (vendor‑compatible: per‑image hash dir with `images/`, `result_with_boxes.jpg`, `result.mmd`, `info.json`)
- Stage tmp: `${run.dir}/tmp/direct_inference/` — intermediate artifacts only

### 5) CLI examples
- Direct inference only:
```
pixi run python -m llm_perf_opt.runners.direct_inference_runner \
  pipeline.direct_inference.enable=true \
  pipeline.static_analysis.enable=false \
  pipeline.torch_profiler.enable=false \
  infer.max_new_tokens=8192 \
  dataset.sampling.num_epochs=1 dataset.sampling.num_samples_per_epoch=null
```

- Include in “run all (no ncu)” flows by enabling `direct_inference` and leaving others as needed.

## Impact Analysis
- Back‑compat: no breaking changes. New stage is disabled by default; existing stages continue to function.
- Performance: identical to Stage‑1 inference minus profiler overhead — faster wall‑clock time.
- Storage: additional stage folder `direct_inference/` under the run dir; consistent with our stage layout contract.
- Risks: duplication if users enable both `direct_inference` and `torch_profiler` predictions/viz. Docs should clarify intended uses (direct_inference for fast checks, torch_profiler for measured runs).
- Nsight gating: no effect in this stage; the runner must not emit `nsys`/`ncu` commands regardless of their configs.

## Expected Outcome
- Users can quickly validate datasets and models, produce predictions and visualizations, and share outputs without profiling overhead.
- Consistent config experience across stages (`pipeline.<stage>.output.*`).
- Clean artifact structure: `${run.dir}/direct_inference/{pred,viz}` and `${run.dir}/tmp/direct_inference/`.

## References
- Config entrypoint and pipelines: `conf/config.yaml`
- Output presets to mirror: `conf/output/torch/default.yaml`, `conf/model/deepseek_ocr/output/torch/default.yaml`
- Session and Stage‑1 reference implementation: `src/llm_perf_opt/runners/llm_profile_runner.py`, `src/llm_perf_opt/runners/dsocr_session.py`
- Dataset sampling presets: `conf/dataset/sampling/*.yaml`
- Coding/style conventions: `magic-context/general/python-coding-guide.md`

## Before / After Snippets

### Hydra defaults (new mounts)
Before:
```yaml
defaults:
  - output/torch@pipeline.torch_profiler.output: default
  - model/deepseek_ocr/output/torch@pipeline.torch_profiler.output.extra.deepseek_ocr: default
```

After:
```yaml
defaults:
  - output/torch@pipeline.torch_profiler.output: default
  - model/deepseek_ocr/output/torch@pipeline.torch_profiler.output.extra.deepseek_ocr: default
  - output/direct@pipeline.direct_inference.output: default
  - model/deepseek_ocr/output/direct@pipeline.direct_inference.output.extra.deepseek_ocr: default
```

### Pipeline block
Before:
```yaml
pipeline:
  static_analysis:
    enable: true
  torch_profiler:
    enable: ${pipeline.torch_profiler.enabled}
```

After:
```yaml
pipeline:
  direct_inference:
    enable: false
  static_analysis:
    enable: true
  torch_profiler:
    enable: ${pipeline.torch_profiler.enabled}
```
