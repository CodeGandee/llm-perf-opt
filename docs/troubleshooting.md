# Troubleshooting

No outputs created
- Ensure dataset root exists: `conf/dataset/omnidocbench.yaml` → `datasets/omnidocbench/source-data`
- Run with Pixi (not system Python): `pixi run stage1-run`

Boxes misaligned in visualization
- Verify that predictions are saved with specials intact (`outputs.save_predictions=true`).
- Compare our `viz/<stem>/result_with_boxes.jpg` with vendor output from `scripts/deepseek-ocr-infer-one.py`.

Flash-Attn warnings
- Model is moved to GPU; dtype defaults to bf16. Warnings are benign for inference; set `use_flash_attn=false` to compare.

Hydra override errors
- When selecting grouped configs, include targets: `model/deepseek_ocr/arch@model=...` and `model/deepseek_ocr/infer@infer=...`.

Where are logs?
- Each run writes `llm_profile_runner.log` under `tmp/profile-output/<run_id>/torch_profiler/`.

PyTorch profiler shows 0 CUDA time for many ops
- Expected behavior: device time is attributed to kernel entries (e.g., `cudaLaunchKernel`, `flash_attn::*`), not necessarily to high‑level `aten::*` rows.
- Mitigation: we call `torch.cuda.synchronize()` at the end of the profiled block to flush GPU work. Consider adding a “Top GPU Kernels” view or use Nsight for kernel‑level timing.
- See: `context/summaries/issue-pytorch-profiler-zero-cuda-time.md`.
