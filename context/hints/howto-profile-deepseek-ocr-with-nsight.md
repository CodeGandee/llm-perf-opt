# How to Profile DeepSeek‑OCR with NVIDIA Nsight

This hint shows how to instrument and profile the DeepSeek‑OCR Hugging Face model with NVIDIA Nsight tools. It follows our install preference order: uv (PyPI) > pixi (conda‑forge) > apt > direct .deb.

## What You’ll Use

- Nsight Systems (`nsys`) for end‑to‑end timelines (CPU/GPU, CUDA API, NVTX)
- Nsight Compute (`ncu`) for per‑kernel analysis and metrics
- NVTX markers (Python) to annotate model stages in the timeline
- Optional: PyTorch Profiler for additional traces

## Install: Preferred Order

1) PyPI (uv) — instrumentation utilities

```
uv pip install -U nvtx nvidia-ml-py3 torch-tb-profiler
# nvtx: add timeline ranges; nvidia-ml-py3: GPU info/telemetry; tb-profiler: optional
```

2) Pixi/Conda — Nsight Compute (ncu)

Conda‑forge provides `nsight-compute` (cross‑platform):

```
pixi add nsight-compute -c conda-forge
# Or via conda/mamba directly: conda install -c conda-forge nsight-compute
```

3) APT (Ubuntu/Debian) — Nsight Systems (nsys)

Nsight Systems is best obtained via Ubuntu repos or NVIDIA dev tools repo. Example (Ubuntu 22.04+):

```
sudo apt-get update
sudo apt-get install -y nsight-systems nsight-systems-target
# If packages aren’t present, add NVIDIA CUDA repo keyring first, then apt update.
# See: https://forums.developer.nvidia.com/t/installing-nsys-with-conda/246439
```

4) Direct download (.deb) — Nsight Systems or Graphics

As a last resort, download from NVIDIA Developer (choose your distro/version):
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- Docs (Systems): https://docs.nvidia.com/nsight-systems/user-guide/
- Nsight Compute docs: https://docs.nvidia.com/nsight-compute/

## Add NVTX Markers (Python)

Annotate key phases to make Nsight timelines readable:

```python
import nvtx

with nvtx.annotate("load_tokenizer", color="blue"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)

with nvtx.annotate("load_model", color="green"):
    model = AutoModel.from_pretrained(
        MODEL_ID, _attn_implementation="flash_attention_2", trust_remote_code=True,
        use_safetensors=True, local_files_only=True,
    ).to("cuda").to(torch.bfloat16).eval()

with nvtx.annotate("infer_batch", color="red"):
    res = model.infer(tokenizer, prompt=prompt, image_file=str(img), output_path=str(out_dir),
                      base_size=1024, image_size=640, crop_mode=True, save_results=True, test_compress=True)
```

PyTorch also exposes NVTX helpers:

```python
torch.cuda.nvtx.range_push("preprocess")
# ... work ...
torch.cuda.nvtx.range_pop()
```

## Profile Commands

Nsight Systems (timeline):

```
# Capture CUDA + NVTX + OS runtime; write a .qdrep report
nsys profile \
  --sample=cpu --trace=cuda,nvtx,osrt \
  --capture-range=nvtx --capture-range-end=stop \
  -o tmp/nsys/deepseek \
  pixi run python tests/manual/deepseek_ocr_hf_manual.py

# Optionally, start/stop capture via NVTX marks around your hot loop
```

Nsight Compute (kernels):

```
# Collect detailed kernel metrics (may slow execution)
ncu --target-processes all --set full \
  --section Regex:.* --import-source no \
  pixi run python tests/manual/deepseek_ocr_hf_manual.py

# Narrow to a specific kernel by name regex
ncu --kernel-name Regex:.*attention.* --set speed-of-light \
  pixi run python tests/manual/deepseek_ocr_hf_manual.py
```

Tips:
- Use smaller inputs or fewer iterations when running with `ncu` to keep runs manageable.
- Ensure you run on the same GPU/driver stack used for deployment; Nsight relies on CUPTI/driver compatibility.

## Minimal Torch Profiler (Optional)

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("infer"):
        res = model.infer(...)
print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("tmp/trace.json")
```

## References
- Nsight Compute (conda‑forge feedstock): https://github.com/conda-forge/nsight-compute-feedstock
- NVIDIA Nsight Systems — download: https://developer.nvidia.com/nsight-systems
- Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/user-guide/
- Nsight Compute User Guide: https://docs.nvidia.com/nsight-compute/
- NVTX (PyPI): https://pypi.org/project/nvtx/
- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html

