# Refactor Plan: Toggle Static Model Analysis via Runner Config (Stage 1)

## What to Refactor
- Stage 1 runner always performs static model analysis at the end of a run. The logic in `src/llm_perf_opt/runners/llm_profile_runner.py` unconditionally constructs `DeepseekOCRStaticAnalyzer` and writes static compute reports.
- The Pixi task `stage1-run` in `pyproject.toml` invokes the runner without a way to disable static analysis.
- For now, we will control this behavior via a runner configuration file: `conf/runners/stage1.no-static.yaml`, and make the Stage 1 runner use this runner config when requested.

## Why Refactor
- Control runtime overhead: static analysis can be expensive and not always needed when users only want operator profiling and MFU from measured runs.
- Improve ergonomics: provide a single Hydra boolean flag to enable/disable static analysis and (optionally) writing static analysis artifacts.
- Future-proofing: centralize analysis knobs (e.g., `use_analytic_fallback`, `use_synthetic_inputs`) under a dedicated config node to avoid hard‑coded defaults in code.

## How to Refactor
1. Add runner config group (runners)
   - Create `conf/runners/` group with at least:
     - `conf/runners/stage1.default.yaml` (baseline behavior)
     - `conf/runners/stage1.no-static.yaml` (disables static analysis)
   - In these files, set the toggle under an analysis node (used by the runner code):
     - stage1.default.yaml → `analysis: { static: { enabled: true, write_reports: true } }`
     - stage1.no-static.yaml → `analysis: { static: { enabled: false, write_reports: false } }`

2. Wire defaults
   - Update `conf/config.yaml` defaults to mount the new group with the default option:
     - Add `- runners: stage1.default`
   - Keep all current defaults intact to avoid behavior changes.

3. Guard static analysis in runner
   - Wrap static analysis block in `llm_profile_runner.py` with the new toggle:
     - Only run static analysis when `cfg.analysis.static.enabled` is true.
     - Drive `AnalysisConfig` from `cfg.analysis.static` (e.g., `use_analytic_fallback`, `use_synthetic_inputs`).
     - Only write `static_compute.*` when `cfg.analysis.static.write_reports` is true.

4. CLI and Pixi task surface
   - Keep existing `stage1-run` task unchanged (defaults to enabled via `runners=stage1.default`).
   - Add convenience task `stage1-run-no-static` selecting the runner config:
     - Append Hydra override: `runners=stage1.no-static`
   - Users can also pass this override manually: `pixi run stage1-run runners=stage1.no-static`.

5. Documentation updates
   - In Stage 1 quickstart/README, add a section “Runner Configs” showing how to pick `stage1.no-static`.

## Impact Analysis
- Backward compatibility: default remains enabled, so current flows and tests are unaffected.
- Risk: If the new config group isn’t loaded, direct attribute access may fail. Mitigation:
  - Use defensive access in runner: treat missing node as enabled by default.
  - Ensure Hydra defaults include the group.
- Performance: Disabling static analysis reduces end‑of‑run time and skips writing two files.

## Expected Outcome
- Users can opt out of static analysis for faster Stage 1 runs by setting `analysis.static.enabled=false`.
- Static analysis knobs are centralized under `cfg.analysis.static`, making future tuning straightforward.

## Snippets (Before → After)

### Runner guard
Before (`src/llm_perf_opt/runners/llm_profile_runner.py`):
```python
# Compute improved MFU using static analyzer
try:
    pre_cfg = getattr(getattr(cfg, "model", {}), "preprocess", {})
    # ...
    analyzer = DeepseekOCRStaticAnalyzer(session)
    aconf = AnalysisConfig(..., use_analytic_fallback=True, use_synthetic_inputs=True)
    static_report = analyzer.generate_report(aconf)
    # write static_compute.json/md and update MFUs
except Exception:
    # fallback
    ...
```

After (guarded + driven by runner config):
```python
analysis = getattr(cfg, "analysis", {})
static_cfg = getattr(analysis, "static", {})
enabled = bool(getattr(static_cfg, "enabled", True))
if enabled:
    try:
        pre_cfg = getattr(getattr(cfg, "model", {}), "preprocess", {})
        analyzer = DeepseekOCRStaticAnalyzer(session)
        aconf = AnalysisConfig(
            # ...
            use_analytic_fallback=bool(getattr(static_cfg, "use_analytic_fallback", True)),
            use_synthetic_inputs=bool(getattr(static_cfg, "use_synthetic_inputs", True)),
        )
        static_report = analyzer.generate_report(aconf)
        if bool(getattr(static_cfg, "write_reports", True)):
            write_static_compute_json(static_report, artifacts_dir / "static_compute.json")
            write_static_compute_markdown(static_report, artifacts_dir / "static_compute.md")
        # update MFUs from static_report
    except Exception:
        # fallback path remains unchanged
        ...
```

### Hydra defaults
Before (`conf/config.yaml`):
```yaml
defaults:
  - dataset: omnidocbench
  - model/deepseek_ocr/arch@model: deepseek_ocr.default
  - model/deepseek_ocr/infer@infer: deepseek_ocr.default
  - profiling/torch@torch_profiler: torch-profiler.default
  - _self_
```

After (mount runner group with default):
```yaml
defaults:
  - dataset: omnidocbench
  - model/deepseek_ocr/arch@model: deepseek_ocr.default
  - model/deepseek_ocr/infer@infer: deepseek_ocr.default
  - profiling/torch@torch_profiler: torch-profiler.default
  - runners: stage1.default
  - _self_
```

### New runner configs
`conf/runners/stage1.default.yaml`:
```yaml
# Runner defaults for Stage 1
analysis:
  static:
    enabled: true
    write_reports: true
    use_analytic_fallback: true
    use_synthetic_inputs: true
```

`conf/runners/stage1.no-static.yaml`:
```yaml
# Disable static model analysis for faster runs
analysis:
  static:
    enabled: false
    write_reports: false
    use_analytic_fallback: true
    use_synthetic_inputs: true
```

### New config group (`conf/analysis/static.yaml`)
```yaml
static:
  enabled: true
  write_reports: true
  use_analytic_fallback: true
  use_synthetic_inputs: true
```

### Pixi tasks
Before (`pyproject.toml`):
```toml
[tool.pixi.tasks]
stage1-run = { cmd = "python -m llm_perf_opt.runners.llm_profile_runner ... 'torch_profiler.activities=[cpu,cuda]'" }
```

After (add convenience task using runner config):
```toml
[tool.pixi.tasks]
stage1-run = { cmd = "python -m llm_perf_opt.runners.llm_profile_runner ... 'torch_profiler.activities=[cpu,cuda]'" }
stage1-run-no-static = { cmd = "python -m llm_perf_opt.runners.llm_profile_runner ... 'torch_profiler.activities=[cpu,cuda]' runners=stage1.no-static" }
```

## References
- Runner static analysis hot‑path: `src/llm_perf_opt/runners/llm_profile_runner.py:872`
- Runner config (new): `conf/runners/stage1.no-static.yaml`
- Pixi Stage 1 task: `pyproject.toml:52`
- Hydra base config: `conf/config.yaml:1`
- PyTorch profiler defaults (for context): `conf/profiling/torch/torch-profiler.default.yaml:1`
- Libraries: hydra-core (Context7 id suggestion: `/facebookresearch/hydra`)
