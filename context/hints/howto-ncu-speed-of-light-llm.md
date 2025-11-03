# How to run a light Nsight Compute (ncu) profile for an LLM kernel (SpeedOfLight section)

This hint explains why ncu can appear to “hang” before model load and how to run a faster, light profile focused on a single known kernel using the Speed Of Light (SOL) section.

## Why it can be slow
- Nsight Compute often requires multiple replay passes to collect many metrics. In kernel replay mode, it saves/restores device memory between passes, which is expensive for large LLM workloads.
- Solution: minimize collection scope (use one section like SpeedOfLight) and prefer application replay to avoid per‑kernel memory save/restore.

References:
- Nsight Compute Profiling Guide (replay overhead): https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

## Verify the section name
Speed Of Light’s section identifier is `SpeedOfLight`.

- List sections to confirm availability:
```bash
ncu --list-sections | grep -i speed
```
- Note: If using a regex for sections, you must prefix with `regex:`. For an exact match, just use `SpeedOfLight`.

References:
- OLCF slide deck shows the identifier: `SpeedOfLight` (DisplayName: "GPU Speed Of Light"): https://www.olcf.ornl.gov/wp-content/uploads/2020/02/OLCF-Webinar-Nsight-Compute.pdf
- Forum example using `--section SpeedOfLight`: https://forums.developer.nvidia.com/t/how-to-get-speed-of-light-with-ncu-cli/279483

## Light profiling command (paste-ready)
Profile only one kernel by name using the SOL section and application replay. Replace the kernel pattern and workload args as needed.

```bash
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name '.*gemvx.*' \
  --section SpeedOfLight \
  --replay-mode application \
  -o tmp/ncu/top1_gemv \
  pixi run -e rtx5090 python -m llm_perf_opt.runners.llm_profile_runner \
    'hydra.run.dir=tmp/ncu-work/${now:%Y%m%d-%H%M%S}' \
    dataset.subset_filelist=datasets/omnidocbench/subsets/dev-3.txt \
    device=cuda:0 \
    pipeline.torch_profiler.enable=false pipeline.static_analysis.enable=false \
    dataset.sampling.num_epochs=1 dataset.sampling.num_samples_per_epoch=1 dataset.sampling.randomize=false \
    infer.max_new_tokens=64
```

- Kernel regex tips:
  - Use a short, distinctive substring from your NSYS kernel CSV (e.g., `gemvx`, `flash_fwd_splitkv_kernel`, `sdpa`).
  - If many launches exist, narrow with: `--launch-skip N --launch-count 1`.

## When to use sets or more sections
- `--set roofline` provides a broader view (slower than a single section). Prefer `SpeedOfLight` first; expand only if needed.
- To add another small section (e.g., occupancy):
```bash
ncu --section SpeedOfLight --section '.*Occupancy.*' ...
```
If the regex form doesn’t match, use the exact identifier from `ncu --list-sections` and prefix with `regex:` only when matching by pattern.

## Troubleshooting
- "Option '--section .*SpeedOfLight.*' did not match": Use `--section SpeedOfLight` (exact) or `--section 'regex:.*SpeedOfLight.*'`.
- Slow start: switch to `--replay-mode application`; keep the dataset small and `infer.max_new_tokens` low.
- No report: widen the kernel regex or use `--launch-skip/--launch-count` to hit a specific occurrence.
- Open results: `ncu-ui tmp/ncu/top1_gemv.ncu-rep`.

## Related docs
- Nsight Compute CLI: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- Nsight Compute CLI (PDF): https://docs.nvidia.com/nsight-compute/pdf/NsightComputeCli.pdf
- SOL explainer video: https://www.youtube.com/watch?v=uHN5fpfu8As

