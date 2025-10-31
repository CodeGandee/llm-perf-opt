#!/usr/bin/env bash
set -euo pipefail

RUN_ID=$(date +%Y%m%d-%H%M%S)

# Stage-1: Torch profiler + metrics (no NSYS/NCU)
python -m llm_perf_opt.runners.llm_profile_runner \
  hydra.run.dir=tmp/profile-output/$RUN_ID \
  dataset.subset_filelist=datasets/omnidocbench/subsets/dev-20.txt \
  device=cuda:0 infer.max_new_tokens=64 \
  'pipeline.torch_profiler.activities=[cpu,cuda]' \
  pipeline.torch_profiler.output.prediction.enable=true \
  dataset/sampling@dataset.sampling=default \
  dataset.sampling.num_epochs=1 \
  dataset.sampling.num_samples_per_epoch=3 \
  dataset.sampling.randomize=false \
  pipeline.nsys.enable=false pipeline.ncu.enable=false

# Stage-2: Deep profiling with Nsight Systems (no NCU)
python -m llm_perf_opt.runners.deep_profile_runner \
  hydra.run.dir=tmp/profile-output/$RUN_ID \
  pipeline.nsys.enable=true pipeline.ncu.enable=false \
  pipeline.static_analysis.enable=false
