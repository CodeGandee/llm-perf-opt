# Refactor Plan: Hydra-Driven Model & Dataset Paths

## What to Refactor
- `src/llm_perf_opt/runners/deep_profile_runner.py` lines 107‑118 currently hard-code repo-relative paths for the DeepSeek-OCR model and OmniDocBench dataset before launching the Stage‑1 workload.
- Any other runners that perform similar overrides (e.g., `src/llm_perf_opt/runners/dsocr_session.py` uses the same `repo_root / "models" / "deepseek-ocr"` convention for helper scripts).
- Configuration docs/tests that assume the hard-coded overrides.

## Why Refactor
- **Config correctness**: Hydra configs already define `model.path` and `dataset.root` (see `conf/model/deepseek_ocr/arch/deepseek_ocr.default.yaml` and `conf/dataset/omnidocbench.yaml`). The hard-coded overrides ignore user-supplied overrides and break custom profiles.
- **Maintainability**: Every additional model/dataset combination would require code edits; centralizing path resolution keeps the runner agnostic to specific assets.
- **Reusability**: Upcoming dummy models/tests (e.g., `src/llm_perf_opt/dnn_models`) need the runner to honor Hydra config paths.
- **Consistency**: Stage‑1 and other tooling rely on Hydra-only paths. Aligning Stage‑2 removes divergent behaviors.

## How to Refactor
1. **Audit path overrides**
   - Search for `repo_root / "models"` and similar constructs in runners and utilities.
2. **Introduce resolver helper**
   - Add a small utility in a new module `src/llm_perf_opt/utils/paths.py`:
     - `resolve_hydra_path(value: str | None, cwd: Path) -> str | None` converts config-provided paths to absolute paths (resolving against `HydraConfig.get().runtime.cwd`).
     - Keep it side-effect free (no I/O) so other components can reuse it.
3. **Update Stage‑2 runner**
   - Replace the hard-coded block with logic that reads `cfg.model.path` and `cfg.dataset.root`, resolves them via the helper, and only appends overrides when the config provides a non-empty value.
   - Preserve existing behavior when paths are unset (fall back to Hydra defaults). Ensure provenance (cmd.txt) still reflects the final absolute paths.
4. **Extend other runners (if applicable)**
   - Apply the same helper to `dsocr_session.py` (see below) and any other runner that constructs repo-rooted paths.
5. **Adjust docs/tests**
   - Update docs under `docs/running.md` if they mention hard-coded paths.
   - Ensure manual tests (e.g., `tests/manual/ncu/manual_nvtx_regions.py`) pass explicit overrides when needed.
6. **Validation**
   - Run `pixi run pytest tests/unit/` (if quick) and execute a dry-run of Stage‑2 with custom overrides (e.g., a dummy model path) to confirm the runner respects config settings.

### Code Snippet (Before)
```python
# src/llm_perf_opt/runners/deep_profile_runner.py
repo_root = Path(HydraConfig.get().runtime.cwd)
model_path_abs = str(repo_root / "models" / "deepseek-ocr")
ds_root_abs = str(repo_root / "datasets" / "omnidocbench" / "source-data")
overrides.append(f"model.path={model_path_abs}")
overrides.append(f"dataset.root={ds_root_abs}")
```

### Code Snippet (After)
```python
# Pseudocode illustrating the desired resolver
repo_root = Path(HydraConfig.get().runtime.cwd)

def resolve_path(config_value: str | None) -> str | None:
    if not config_value:
        return None
    p = Path(config_value)
    return str((repo_root / p).resolve()) if not p.is_absolute() else str(p.resolve())

for key, value in (("model.path", getattr(cfg.model, "path", None)),
                   ("dataset.root", getattr(cfg.dataset, "root", None))):
    resolved = resolve_path(value)
    if resolved:
        overrides.append(f"{key}={resolved}")
```

## Impacted Code Files & Handling

- src/llm_perf_opt/runners/deep_profile_runner.py
  - Replace hard-coded overrides (lines ~110–116) with resolved values from `cfg.model.path` and `cfg.dataset.root` using `utils.paths.resolve_hydra_path`.
  - Continue to append other overrides (e.g., dataset sampling) unchanged.
  - Keep writing constructed commands to `ncu/cmd.txt` and `nsys/cmd.txt` for provenance.

- src/llm_perf_opt/runners/dsocr_session.py
  - `_build_dsocr_prompt` currently loads `conversation.py` via `repo_root / "models" / "deepseek-ocr"`.
  - Change to derive the conversation module path from the configured model path:
    - Add an optional `conv_module_path: str | None` parameter to `from_local(...)` and store it on the instance.
    - In `_build_dsocr_prompt`, if `conv_module_path` is provided (or if `Path(model_path)/"conversation.py"` exists), import from there; otherwise fall back to raw prompt.
  - This keeps behavior for DeepSeek-OCR while removing repository coupling.

- src/llm_perf_opt/runners/llm_profile_runner.py
  - Already uses `cfg.model.path` and `cfg.dataset.root`. No change required.
  - Optional: reuse `utils.paths.resolve_hydra_path` to normalize absolute paths for any filesystem writes in provenance (non-blocking).

- src/llm_perf_opt/runners/direct_inference_runner.py
  - Already uses `cfg.model.path` and `cfg.dataset.root`. No change required.
  - Optional: normalize paths via `utils.paths.resolve_hydra_path` for consistency.

- src/llm_perf_opt/runners/inference_engine.py
  - Reads dataset paths from config; no direct overrides. No change required.

- src/llm_perf_opt/profiling/vendor/launch.py
  - No change. Continues to accept an argv list with overrides from callers.

- conf/model/**, conf/dataset/**
  - No change. These are the ground truth for defaults and will be respected by the refactor.

Note: Search results show the only hard-coded model/dataset path constructions live in `deep_profile_runner.py` and `dsocr_session.py`. Other runners already rely on Hydra config values.

## Impact Analysis
- **Positive**: Users can supply alternate models/datasets via Hydra overrides or config defaults without modifying code.
- **Risk**: Hydrated configs may reference relative paths that rely on current working directory; resolving them incorrectly could break Stage‑1 startup. Mitigation: resolve relative paths against `HydraConfig.get().runtime.cwd`, matching how Hydra composes `${hydra:runtime.cwd}`.
- **Testing**: Need to re-run Stage‑2 manual workflows to ensure no regressions. Potential addition of integration test to cover custom path overrides.

## Expected Outcome
- Stage‑2 runner (and any dependent component) honors Hydra configuration for assets, enabling flexible profiling setups.
- Simplified maintenance when adding new models/datasets for profiling.
- Documentation/tests reflect configuration-driven paths rather than repo assumptions.

## References
- Config defaults: `conf/model/deepseek_ocr/arch/deepseek_ocr.default.yaml`, `conf/dataset/omnidocbench.yaml`
- Runner implementation: `src/llm_perf_opt/runners/deep_profile_runner.py#L100-L150`
- Related runner using similar pattern: `src/llm_perf_opt/runners/dsocr_session.py#L120-L130`
- Hydra documentation (uses `omegaconf`/Hydra core): context7 library ID `/facebook/hydra`

## Plan Review
- Scope: Correct and minimal. Only two code files require mandatory changes; other runners already consume Hydra config.
- Risk: Low to moderate. Biggest behavior change is removing repo-coupled defaults in Stage‑2; mitigated by resolving config paths and preserving explicit overrides when provided.
- Testing strategy: Adequate. Manual runs cover primary flows; can add a lightweight integration check ensuring a non-default `model.path` is respected by Stage‑2.
- Consistency: Aligns with project guidelines and existing config design; avoids introducing new configuration keys; adds a single reuseable path utility.
