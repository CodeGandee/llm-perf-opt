# Refactor Plan: Replace `repeats` with `dataset.sampling.*`

## What to Refactor
- Replace the ambiguous top‑level key `repeats` with an explicit dataset sampling schema:
  - `dataset.sampling.num_samples_per_epoch: int | null` — number of samples to process per epoch (null → all samples).
  - `dataset.sampling.num_epochs: int` — how many epochs to run.
  - `dataset.sampling.randomize: bool` — whether to randomize the sample order each epoch.
  - Optional: `dataset.sampling.seed: int | null` — deterministic shuffling when `randomize=true`.
- Update Stage‑1 runner (`llm_perf_opt/runners/llm_profile_runner.py`) to drive the inference loop using the sampling schema.
- Update Stage‑2 runner (`deep_profile_runner.py`) to pass sampling overrides to the Stage‑1 workload instead of the legacy `run.stage1_repeats`.
- Deprecate `repeats` (and `run.stage1_repeats`) with a migration path and deprecation warnings.

Scope
- Config: `conf/config.yaml` (add `dataset.sampling.*` with defaults), docs/examples.
- Code: `src/llm_perf_opt/runners/llm_profile_runner.py`, `src/llm_perf_opt/runners/deep_profile_runner.py` (override wiring), and report/log messages.
- Tasks: update Pixi tasks where `repeats=` is passed.

## Why Refactor
- Clarity: `repeats` suggests time/iteration count without context; actual behavior is dataset sampling. The new schema is explicit and familiar to ML practitioners.
- Control: cleanly express full‑dataset sweeps, partial sampling, and randomized vs. sequential order.
- Extensibility: adding seeds, stratified sampling, or weighted sampling is straightforward under a `dataset.sampling` namespace.

## How to Refactor

1) Config schema and defaults (config group)
- Create a dedicated Hydra config group under `conf/dataset/sampling/` to hold sampling presets:

  - `conf/dataset/sampling/default.yaml`
  - `conf/dataset/sampling/random.yaml` (optional convenience preset)
  - (room for future variants: `full.yaml`, `fixed.yaml`, etc.)

- Mount the group into the resolved config under `dataset.sampling` from `conf/config.yaml` defaults:

```yaml
defaults:
  - dataset: omnidocbench
  - dataset/sampling@dataset.sampling: default  # NEW: sampling preset
  - ...
```

Example preset: `conf/dataset/sampling/default.yaml`

```yaml
# Dataset sampling controls (per Stage‑1 run and Stage‑1-as-workload)
num_samples_per_epoch: null   # null → process all available samples each epoch
num_epochs: 1                 # how many epochs to run
randomize: false              # whether to shuffle order per epoch
seed: null                    # optional RNG seed for deterministic shuffles
```

Optional preset: `conf/dataset/sampling/random.yaml`

```yaml
num_samples_per_epoch: null
num_epochs: 1
randomize: true
seed: null
```

- Deprecation bridge: when `repeats` is set, map it to the new fields if `dataset.sampling` is unspecified:
  - If `repeats <= dataset_size`: `num_samples_per_epoch = repeats`, `num_epochs = 1`, `randomize = false`.
  - If `repeats > dataset_size`: `num_samples_per_epoch = dataset_size`, `num_epochs = ceil(repeats / dataset_size)`, and use a remainder for the last epoch. Log a WARNING about computed mapping.

2) Stage‑1 runner (sampling loop)
- Before (simplified):

```python
# current
repeats = int(cfg.repeats)
images_iter = iter(images)
for i in range(repeats):
    try:
        img = next(images_iter)
    except StopIteration:
        images_iter = iter(images)
        img = next(images_iter)
    run_one(img)
```

- After (proposed):

```python
sam = getattr(cfg.dataset, "sampling", {})  # resolved from conf/dataset/sampling/*
N = sam.get("num_samples_per_epoch", None)  # None → all
E = int(sam.get("num_epochs", 1))
rand = bool(sam.get("randomize", False))
seed = sam.get("seed", None)

images_all = list(images)  # discovered files
rng = random.Random(int(seed)) if seed is not None else random.Random()

for epoch in range(E):
    order = images_all[:]
    if rand:
        rng.shuffle(order)
    if N is None:
        selected = order
    else:
        if not rand:
            # deterministic contiguous chunks across epochs
            start = epoch * int(N)
            selected = order[start % len(order):] + order[:start % len(order)]
            selected = selected[: int(N)]
        else:
            # random sample without replacement per epoch
            selected = order[: int(min(int(N), len(order)))]
    for img in selected:
        run_one(img)
```

- Notes:
  - Warmup and representative profiled run remain separate from sampling.
  - Include sampling settings and effective counts in logs and the Stage‑1 report.md.

3) Stage‑2 runner (workload overrides)
- Before: passes `repeats=stage1_repeats`.

```python
stage1_repeats = int(getattr(cfg.run, "stage1_repeats", 1))
overrides += [f"repeats={stage1_repeats}"]
```

- After: pass new sampling overrides.

```python
N = int(getattr(cfg.run, "stage1_repeats", 1))  # legacy cfg
overrides += [
    "dataset/sampling@dataset.sampling=default",  # ensure group mounted
    "dataset.sampling.num_epochs=1",
    f"dataset.sampling.num_samples_per_epoch={N}",
    "dataset.sampling.randomize=false",
]
```

- Deprecate `run.stage1_repeats` in favor of `dataset.sampling.*` and log a WARNING if used.

4) Pixi tasks and docs
- Replace `repeats=...` overrides with either a preset or explicit keys:
  - Use a preset: `dataset/sampling@dataset.sampling=random`
  - Or set keys: `dataset.sampling.num_epochs=...`, `dataset.sampling.num_samples_per_epoch=...`, `dataset.sampling.randomize=true|false`
- Update quickstarts and configuration docs to use the new API and describe semantics.

5) Backward compatibility
- Support `repeats` and `run.stage1_repeats` for one release window:
  - If `dataset.sampling.*` is not provided, compute a sampling plan from legacy fields and log a WARNING explaining the mapping.
  - Prefer the new keys when both are present.

## Impact Analysis
- Behavior change: explicit epochs/samples semantics replace the implicit cycling behavior of `repeats`. The compatibility layer ensures prior overrides keep producing similar totals.
- Risk: low; changes local to dataset iteration logic and workload overrides. Mitigations:
  - WARNING logs when using legacy keys or computed mappings.
  - Unit tests for sampling selection (randomized and deterministic modes), and for legacy→new mappings.
- Performance: unaffected; only iteration order/selection changes.

## Expected Outcome
- Clear, expressive dataset sampling controls: epochs, per‑epoch sample counts, and randomization.
- Users can reproduce full sweeps, fixed‑size subsets per epoch, or randomized subsets deterministically via a seed.
- Legacy `repeats` and `run.stage1_repeats` are gracefully deprecated without breaking existing runs.

## References
- Hydra configuration (package mounting): /facebookresearch/hydra
- Code references
  - Stage‑1 iteration: `src/llm_perf_opt/runners/llm_profile_runner.py` (current repeats handling at ~lines 781–860)
  - Stage‑2 overrides: `src/llm_perf_opt/runners/deep_profile_runner.py` (current `run.stage1_repeats` handling)
- Sampling best practices: deterministic shuffling via seeds (Python `random.Random`) and sequential chunking for non‑random epochs.
