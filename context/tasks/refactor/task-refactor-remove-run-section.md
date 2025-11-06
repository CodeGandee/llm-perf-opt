# Refactor Plan: Remove `run` Section from `conf/config.yaml`

## What to Refactor
- Remove the top‑level `run` section from `conf/config.yaml` and migrate its usages to existing, canonical config locations.
- Keys currently present/used:
  - `run.mode` (not actively used by Stage‑2; vestigial)
  - `run.stage1_repeats` (legacy; mapped to dataset sampling in Stage‑2)
  - `run.dataset_subset_filelist` (subset filelist for Stage‑1 workload)
  - `run.top_n_kernels` (Stage‑2 NCU convenience for top‑K selection)
  - `run.representative_max_new_tokens` (rarely used; duplicates `infer.max_new_tokens`)

Impacted files (direct and docs):
- `conf/config.yaml`
- `src/llm_perf_opt/runners/deep_profile_runner.py`
- `src/llm_perf_opt/profiling/vendor/launch.py` (minor cleanup: `+run.mode` injection helper can be retired later)
- Documentation: `docs/configuration.md`, `docs/running.md`

## Why Refactor
- The `run` section duplicates knobs that already exist in the model/dataset/pipeline configs and creates confusion.
- Centralizing behavior under canonical groups (`dataset.*`, `infer.*`, `pipeline.*`) improves clarity and removes special cases.
- Stage‑2 already maps legacy values (e.g., `stage1_repeats`) to dataset sampling; making this explicit removes hidden behavior.

## How to Refactor
1) Introduce/confirm canonical config keys (no shim)
- Dataset subset: keep using `dataset.subset_filelist`
- Sampling: `dataset.sampling.num_epochs`, `dataset.sampling.num_samples_per_epoch`, `dataset.sampling.randomize`
- Tokens: `infer.max_new_tokens`
- NCU kernel selection: use existing `ncu_cli.kernel_name` and `kernel_name_base` (maps to `--kernel-name` and `--kernel-name-base`) and NVTX include filters. No top‑K control in pipeline config.

2) Remove `run` from `conf/config.yaml`
- Delete the `run:` block entirely.
- Do not reintroduce defaults under a new name; rely on existing dataset/infer/pipeline groups.

3) Update Stage‑2 runner logic (hard remove `run.*`)
- Before (reads legacy values):
```python
# deep_profile_runner.py (before)
stage1_repeats = int(getattr(getattr(cfg, "run", {}), "stage1_repeats", 1))
subset_filelist = getattr(getattr(cfg, "run", {}), "dataset_subset_filelist", None)
rep_mnt = getattr(getattr(cfg, "run", {}), "representative_max_new_tokens", None)
```
- After (use canonical keys, with optional legacy shim):
```python
# deep_profile_runner.py (after, no back-compat)
# Canonical only
sam = getattr(getattr(cfg, "dataset", {}), "sampling", {})
stage1_repeats = int(sam.get("num_samples_per_epoch", 1))
subset_filelist = getattr(getattr(cfg, "dataset", {}), "subset_filelist", None)
rep_mnt = int(getattr(getattr(cfg, "infer", {}), "max_new_tokens", 64))
# Remove NSYS-derived top‑K kernel regex inference. Rely on
# `pipeline.ncu.ncu_cli.kernel_name` and `kernel_name_base` when users
# want to restrict kernels, optionally combined with NVTX includes.
```

4) Adjust docs & examples (no back-compat messaging)
- Remove all `run.*` references from docs and examples.
- Show overrides using `dataset.subset_filelist`, `dataset.sampling.*`, `infer.max_new_tokens`, and `pipeline.ncu.ncu_cli.kernel_name`/`kernel_name_base`.

5) Cleanup supporting utilities
- `src/llm_perf_opt/profiling/vendor/launch.py`: remove `run_mode` parameter and `+run.mode` injection entirely.

## Impact Analysis
- Functional behavior retained with canonical keys only:
  - Repeats → via `dataset.sampling.num_samples_per_epoch`
  - Subset filelist → `dataset.subset_filelist`
  - Representative tokens → `infer.max_new_tokens`
  - Kernel selection → via `pipeline.ncu.ncu_cli.kernel_name` (exact or `regex:`) and `kernel_name_base`; combine with NVTX `ncu_cli.nvtx.include` as needed.
- Risks: Existing users relying on `+run.*` overrides will fail fast (Hydra unknown key). Mitigation: clear release notes and docs; examples updated.
- Code impact: localized to `deep_profile_runner.py`, `vendor/launch.py`, plus docs and `conf/config.yaml`.

## Expected Outcome
- No `run` section in `conf/config.yaml`.
- Stage‑2 runner honors canonical config keys only (no `run.*`).
- Documentation and examples consistently show canonical overrides.
- Simpler configuration surface and fewer special cases.

## References
- Hydra config (project): `conf/config.yaml`, `conf/dataset/omnidocbench.yaml`, `conf/profiling/ncu/*.yaml`
- Stage‑2 runner: `src/llm_perf_opt/runners/deep_profile_runner.py`
- Path resolver: `src/llm_perf_opt/utils/paths.py`
- Hydration engine: context7 library ID `/facebook/hydra`
