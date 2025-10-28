**Question**
If I setup a profiling project, targeted to sysmetically profile different LLMs (they are symlinked into project subdir), for python, using pixi as env management tool, with pyproject.toml as pixi manifest file, using hydra+omegaconf to manage configurations, how to structure this project?

**Answer**
Perfect—here’s a clean, opinionated scaffold you can drop in and extend. It uses **Pixi** (with `pyproject.toml` as the manifest), **Hydra+OmegaConf** for configs/sweeps, and slots in the profiling harness we discussed (PyTorch profiler + NVTX + NVML + NSYS/NCU helpers).

---

# Project layout

```
llm-profiler/
├── pyproject.toml                 # Pixi env + tasks + project metadata
├── conf/                          # Hydra config tree (config groups)
│   ├── config.yaml                # top-level defaults list
│   ├── hydra/
│   │   └── default.yaml           # run dir/job name/chdir pattern
│   ├── model/                     # LLM choices (point to local symlinks)
│   │   ├── qwen2_5_7b.yaml
│   │   ├── llama3_70b.yaml
│   │   └── …                      # add more here
│   ├── runtime/                   # framework/runtime layer
│   │   ├── pytorch.yaml
│   │   ├── vllm.yaml
│   │   └── tensorrtllm.yaml
│   ├── hardware/                  # GPU affinity & caps
│   │   └── single_gpu.yaml
│   └── profiling/                 # what to collect
│       ├── minimal.yaml
│       └── full.yaml
├── models/                        # symlinks to weight dirs live here
│   ├── qwen2_5_7b -> /data/weights/qwen2_5_7b/
│   └── llama3_70b -> /data/weights/llama3_70b/
├── src/
│   └── llmprof/
│       ├── __init__.py
│       ├── cli.py                 # @hydra.main entry, route to runners & profiling
│       ├── profiling/
│       │   ├── harness.py         # NVTX, PyTorch profiler, NVML sampler, NSYS/NCU helpers
│       │   ├── nsight_nsys.py     # thin wrappers (optional)
│       │   ├── nsight_ncu.py      # thin wrappers (optional)
│       │   └── parsers/
│       │       └── cublas_log.py  # GEMM M/N/K→FLOPs (optional)
│       └── runners/
│           ├── base.py            # Runner interface
│           ├── pytorch_runner.py  # vanilla torch/vLLM glue (select one)
│           └── tensorrtllm_runner.py
├── scripts/
│   ├── make_symlinks.sh           # convenience: link external weight dirs into models/
│   └── sanity.sh                  # quick smoke runs
└── README.md
```

Why this shape?

* **Hydra** wants grouped configs under `conf/<group>/<option>.yaml`, with a top-level `defaults` list; it also lets you **template the output/run directory** (e.g., `runs/${experiment}/${model.name}/…`) so every run is self-contained and reproducible. ([Hydra][1])
* **Pixi** can use **`pyproject.toml`** as the single manifest and supports **tasks** (handy shortcuts like `pixi run profile`). ([Prefix Dev][2])

---

# `pyproject.toml` (Pixi as manifest + tasks)

```toml
[project]
name = "llm-profiler"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "torch",                # pick the CUDA build you need
  "omegaconf",
  "hydra-core",
  "pynvml",
  "rich",
  "pandas",
]

[tool.pixi.workspace]
channels  = ["conda-forge"]      # add "nvidia" channel if you want CUDA toolkit via conda
platforms = ["linux-64"]         # add others as needed

# If you prefer conda packages for some tools:
[tool.pixi.dependencies]
python = ">=3.10"

# Handy shortcuts: `pixi run <task>`
[tool.pixi.tasks]
# pure Python run (no external profilers) with full profiling config
profile = "python -m llmprof.cli experiment=baseline profiling=full"

# Nsight Systems (NVTX-gated capture). Open .qdrep in the GUI.
nsys = """
nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx --capture-range-end=nvtx \
  -o runs/nsys python -m llmprof.cli experiment=baseline profiling.nsys.enable=true
"""

# Nsight Compute deep-dive (adjust metrics/kernels later)
ncu = """
ncu --target-processes all --set full \
  --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct \
  -o runs/ncu python -m llmprof.cli experiment=baseline profiling.ncu.enable=true
"""
```

Pixi will happily append its sections into `pyproject.toml` (`[tool.pixi.workspace]`, `[tool.pixi.tasks]`, etc.). ([Prefix Dev][2])

---

# Hydra configs

## `conf/config.yaml` (top-level “defaults”)

```yaml
# Compose the final config from groups:
defaults:
  - hydra: default
  - model: qwen2_5_7b
  - runtime: pytorch
  - hardware: single_gpu
  - profiling: minimal
  - _self_
experiment: baseline
```

Hydra’s **defaults list** composes configs from groups; you can swap any piece via CLI overrides (e.g., `model=llama3_70b profiling=full`). ([Hydra][3])

## `conf/hydra/default.yaml` (run dir & behavior)

```yaml
hydra:
  run:
    # one directory per run (microseconds avoids collisions on fast launchers)
    dir: runs/${experiment}/${model.name}/${now:%Y-%m-%d_%H-%M-%S-%f}
  job:
    name: ${experiment}
    chdir: true
```

These fields control where Hydra writes `.hydra/` configs, logs, and where your code runs (CWD). ([Hydra][4])

## `conf/model/qwen2_5_7b.yaml`

```yaml
name: qwen2_5_7b
# Point to a symlink under ./models (models/qwen2_5_7b -> /abs/path/on/disk)
path: ${oc.env:PROJECT_ROOT,${hydra:runtime.cwd}}/models/qwen2_5_7b
dtype: bf16
max_seq_len: 4096
tokenizer: ${model.path}
```

## `conf/runtime/pytorch.yaml`

```yaml
type: pytorch
impl: "vanilla"   # or "vllm"
batch_size: 1
num_new_tokens: 64
nvtx_ranges: true
torch_profiler:
  enabled: true
  warmup: 5
  steps: 50
```

## `conf/hardware/single_gpu.yaml`

```yaml
device_index: 0
power_limit_w: null   # set to e.g. 250 for power-capped runs
```

## `conf/profiling/minimal.yaml`

```yaml
nsys: { enable: false }
ncu:  { enable: false }
nvml: { enable: true, interval_s: 0.5 }
outputs:
  write_ops_csv: true
```

## `conf/profiling/full.yaml`

```yaml
nsys:
  enable: true
  args: "--trace=cuda,nvtx,osrt --capture-range=nvtx --capture-range-end=nvtx"
ncu:
  enable: true
  metrics: >
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,
    sm__throughput.avg.pct_of_peak_sustained_elapsed,
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,
    dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct
nvml:
  enable: true
  interval_s: 0.2
outputs:
  write_ops_csv: true
```

---

# Minimal code you need

## `src/llmprof/profiling/harness.py`

Drop in the harness we built earlier (NVTX context manager, `run_torch_profiler`, `NVMLSampler`, and the “advise NSYS/NCU command” helpers). PyTorch Profiler & NVTX usage matches the official docs/workflows. ([PyTorch Docs][5])

## `src/llmprof/runners/base.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class Runner(ABC):
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    @abstractmethod
    def prefill(self, prompt_ids): ...
    @abstractmethod
    def decode(self, num_new_tokens: int): ...
```

## `src/llmprof/runners/pytorch_runner.py` (sketch)

```python
import torch
from .base import Runner

class PyTorchRunner(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)
        # TODO: load your model from cfg.model.path
        self.device = f"cuda:{cfg.hardware.device_index}"
        # self.model = ...
        # self.tokenizer = ...

    def prefill(self, prompt_ids):
        # with torch.inference_mode(): ...
        pass

    def decode(self, n_new):
        for _ in range(n_new):
            # single-step decode; use NVTX in caller
            pass
```

## `src/llmprof/cli.py` (Hydra entry + profiling glue)

```python
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from llmprof.profiling.harness import (
    nvtx_range, NVMLSampler, run_torch_profiler, prof_to_csv
)

def build_runner(cfg):
    if cfg.runtime.type == "pytorch":
        from llmprof.runners.pytorch_runner import PyTorchRunner
        return PyTorchRunner(cfg)
    # elif cfg.runtime.type == "tensorrtllm": ...
    raise ValueError(f"Unknown runtime: {cfg.runtime.type}")

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.environ.setdefault("PROJECT_ROOT", os.getcwd())

    runner = build_runner(cfg)

    nvml = None
    if cfg.profiling.nvml.enable:
        nvml = NVMLSampler(device_index=cfg.hardware.device_index,
                           interval_s=cfg.profiling.nvml.interval_s,
                           out_csv="nvml.csv")
        nvml.start()

    def one_step(_):
        with nvtx_range("prefill"): runner.prefill(prompt_ids=None)
        with nvtx_range("decode"):  runner.decode(cfg.runtime.num_new_tokens)

    if getattr(cfg.runtime, "torch_profiler", {}).get("enabled", False):
        prof = run_torch_profiler(
            one_step,
            steps=cfg.runtime.torch_profiler.steps,
            warmup=cfg.runtime.torch_profiler.warmup,
            out_dir="tb"
        )
        if cfg.profiling.outputs.write_ops_csv:
            prof_to_csv(prof, "ops.csv")
    else:
        # plain run (still places NVTX ranges for NSYS)
        one_step(0)

    if nvml:
        nvml.stop(); nvml.join()

if __name__ == "__main__":
    main()
```

This keeps your **end-to-end phases NVTX-annotated** for Nsight Systems (`--capture-range=nvtx`) and dumps a **PyTorch Profiler** Chrome trace + `ops.csv` when enabled. ([NVIDIA Docs][6])

---

# Typical flows

* **Plain run (minimal stats):**

  ```
  pixi run profile
  ```

* **Nsight Systems timeline** (NVTX-gated):

  ```
  pixi run nsys
  ```

  (Uses `--capture-range=nvtx` so only your labeled region is captured.) ([NVIDIA Docs][6])

* **Nsight Compute deep-dive** (kernel metrics for roofline/MFU):

  ```
  pixi run ncu
  ```

  Tweak metrics/filters as you focus on attention/GEMMs. ([NVIDIA Docs][7])

* **Swap models/runtimes via Hydra overrides:**

  ```
  pixi run profile model=llama3_70b runtime=vllm profiling=full
  ```

---

# Notes & tips

* **Config groups** scale well: add `profiling/{minimal,full,roofline-only}`; `hardware/{single_gpu,multi_gpu}`; `runtime/{pytorch,vllm,tensorrtllm}`; `model/*`. Hydra’s defaults list is the switchboard; you get reproducible runs and easy sweeps. ([Hydra][1])
* **Run directory pattern** is fully yours (`hydra.run.dir`), so all artifacts—NVML CSV, `ops.csv`, TensorBoard trace, NSYS `.qdrep`, NCU reports—land under `runs/<exp>/<model>/<timestamp>`. ([Hydra][4])
* If you monitor nodes continuously, pair this with **DCGM Exporter** (Prometheus `/metrics`) for fleet-level GPU telemetry; it complements per-run files. ([NVIDIA Docs][8])
* Keep **`models/` as symlinks**; your YAML just points to `${model.path}`—no code changes when you rotate weight stores.

---

If you want, I can fill in a **ready-to-run `harness.py`** file in `src/llmprof/profiling/` (about ~150 lines) and a **`pytorch_runner.py`** that exercises a tiny HF model so you can smoke-test the pipeline end-to-end.

[1]: https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/?utm_source=chatgpt.com "Grouping config files"
[2]: https://prefix-dev.github.io/pixi/dev/python/pyproject_toml/ "pyproject.toml - Pixi by prefix.dev"
[3]: https://hydra.cc/docs/advanced/defaults_list/?utm_source=chatgpt.com "The Defaults List"
[4]: https://hydra.cc/docs/configure_hydra/workdir/?utm_source=chatgpt.com "Customizing working directory pattern"
[5]: https://docs.pytorch.org/docs/stable/profiler.html?utm_source=chatgpt.com "torch.profiler — PyTorch 2.9 documentation"
[6]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html?utm_source=chatgpt.com "User Guide — nsight-systems"
[7]: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html?utm_source=chatgpt.com "4. Nsight Compute CLI"
[8]: https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html?utm_source=chatgpt.com "DCGM Exporter — NVIDIA GPU Telemetry 1.0.0 ..."

---

**Question**
we also need a place to put the model source code (from github/hugging face), as implementation reference, we may symlink or checkout official source as submodules

**Answer**
Great call. Add a **vendor-style area** to your repo and treat upstream model repos as **reference code** that’s pinned and reproducible. Below is an opinionated structure + ready-to-use commands that support both **Git submodules** and **alternatives** (subtree, sparse-checkout, HF Hub snapshots). I’ll also show where it plugs into your Pixi/Hydra setup.

---

# Folder layout (extended)

```
llm-profiler/
├── third_party/                 # reference sources live here (read-only in practice)
│   ├── github/                  # GitHub repos (Transformers, vLLM, TRT-LLM, Flash-Attn…)
│   │   ├── transformers/        # submodule / subtree / sparse clone
│   │   └── vllm/
│   └── hf/                      # Hugging Face repos (git+LFS) or hub snapshots
│       ├── meta-llama/Llama-3/  # submodule (git LFS!) OR snapshot dir
│       └── Qwen/Qwen2.5/        # idem
├── models/                      # your weights symlinks (unchanged)
├── conf/                        # Hydra configs (unchanged)
├── src/llmprof/…                # code (unchanged)
└── pyproject.toml               # Pixi tasks below will manage third_party/
```

**Why a dedicated `third_party/`?** Keeps a clean boundary, makes it obvious what’s vendored, and lets you wire pre-commit rules to **discourage edits** there (see hook idea below).

---

# Option 1 — Git submodules (recommended for source you want to pin)

**Add & pin (example: Transformers + vLLM):**

```bash
git submodule add https://github.com/huggingface/transformers third_party/github/transformers
git submodule add https://github.com/vllm-project/vllm      third_party/github/vllm
git submodule update --init --recursive
# (Optionally pin to a specific commit for reproducibility)
git -C third_party/github/transformers checkout <commit-sha>
git -C third_party/github/vllm checkout <commit-sha>
git add .gitmodules third_party/github
git commit -m "Add reference sources as submodules"
```

Submodules make the superproject track an **exact commit** of each dependency and are first-class in Git. (Pro Git book; official submodule docs.) ([Git][1])

**Don’t show local edits in `git status` (reference-only behavior):**

```bash
# Hide “dirty” submodule worktrees in status (commit your .gitmodules change)
git config -f .gitmodules submodule.third_party/github/transformers.ignore dirty
git config -f .gitmodules submodule.third_party/github/vllm.ignore dirty
git add .gitmodules && git commit -m "Ignore dirty submodule worktrees"
```

`ignore=dirty` (or `all`) prevents submodule worktree changes from cluttering `git status`—you’ll still see when the **submodule HEAD** changes, which is what you care about. (Documented in `git status` and discussion around `--ignore-submodules`.) ([Git][2])

**Shallow/fast clones (big repos):**

```bash
git submodule update --init --depth 1 -- third_party/github/transformers
```

(Depth works for speed; bump or remove depth when you need history.) Submodule fundamentals and knobs are in the official docs. ([Git][3])

---

# Option 2 — Git subtree (if you want a normal folder, no submodule UX)

```bash
git subtree add --prefix third_party/github/transformers \
  https://github.com/huggingface/transformers main --squash
# Later update:
git subtree pull --prefix third_party/github/transformers \
  https://github.com/huggingface/transformers main --squash
```

Subtrees **behave like regular dirs** (no `.gitmodules`, no submodule commands), at the cost of a bigger superproject history. Good if you dislike submodule ergonomics. (man page + tutorials). ([Debian Manpages][4])

---

# Option 3 — Sparse-checkout (for *parts* of giant monorepos)

If you only need, say, `src/transformers/models/llama/`:

```bash
git clone https://github.com/huggingface/transformers third_party/github/transformers
cd third_party/github/transformers
git sparse-checkout init --cone
git sparse-checkout set src/transformers/models/llama
```

This keeps your working tree tiny while the repo remains complete under the hood. (Official sparse-checkout docs + GitHub write-up.) ([Git][5])

---

# Option 4 — Hugging Face Hub snapshots (no git history, read-only)

For model repos on the Hub (many include **reference implementation files** alongside weights), you can snapshot specific files/dirs into `third_party/hf/**`:

```python
# scripts/snapshot_hf.py
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5",
    local_dir="third_party/hf/Qwen/Qwen2.5",
    allow_patterns=["*.py", "LICENSE*"],  # keep only source, not big weights
    revision="main"                        # or specific commit/revision
)
```

This uses HF’s **content-addressed cache** and is simple for *reference code only*. (Docs for `huggingface_hub` download & cache behavior.) ([Hugging Face][6])

> If you submodule an HF repo instead, remember **Git LFS** is required to fetch large files/weights; snapshots avoid LFS entirely if you filter to `.py` only. (Git LFS overview.) ([GitHub Docs][7])

---

## Pixi tasks to manage externals

Add to your existing `pyproject.toml`:

```toml
[tool.pixi.tasks]
# Initialize / update submodules
externals:init = "git submodule update --init --recursive"
externals:update = "git submodule update --remote --merge --recursive"

# Optional: pull/refresh HF snapshots of source-only files
externals:snapshot = "python scripts/snapshot_hf.py"

# For giant repos you want sparsified
externals:sparse-transformers = """
bash -lc '
cd third_party/github/transformers && \
git sparse-checkout init --cone && \
git sparse-checkout set src/transformers/models/llama
'"""
```

Now you can do:

```
pixi run externals:init
pixi run externals:snapshot
```

---

## Hydra config pointers

Reference these sources inside your model configs to make links explicit:

```yaml
# conf/model/llama3_70b.yaml
name: llama3_70b
impl_src:
  transformers_dir: ${oc.env:PROJECT_ROOT,${hydra:runtime.cwd}}/third_party/github/transformers
  notes: "Reference implementation; do not modify. Pinned via submodule."
hf_src:
  repo_dir: ${oc.env:PROJECT_ROOT,${hydra:runtime.cwd}}/third_party/hf/meta-llama/Llama-3
```

This keeps your **experiment** and **reference code** traceable in the run artifacts Hydra writes.

---

## Guard rails (optional but handy)

* **Pre-commit**: add hooks so contributors don’t accidentally edit `third_party/**` or add new submodules without review.
  Install pre-commit and use a simple “forbid changes under path” hook. (pre-commit docs + patterns discussion.) ([pre-commit.com][8])
* **.gitmodules defaults**: you can set `submodule.<name>.update = none` if you truly want them frozen by default, and `submodule.<name>.ignore = dirty` for quiet status. (gitmodules reference.) ([Git][9])
* **Licenses**: copy upstream `LICENSE` files alongside the referenced code to keep compliance obvious—HF snapshots example above includes `LICENSE*`.

---

## Which option when?

* **You want history + easy pinning** → **Submodules**. (Default.) ([Git][3])
* **You want a normal folder, no special Git UX** → **Subtree**. ([Debian Manpages][4])
* **You only need a slice of a big monorepo** → **Sparse-checkout**. ([Git][5])
* **You just want reference `.py` files from HF, no weights** → **Hub snapshot**. ([Hugging Face][6])

All four play nicely with your **Pixi + Hydra** scaffold from earlier; they only affect how `third_party/**` is populated and pinned.

If you tell me the exact repos you want (e.g., `transformers`, `flash-attention`, `tensorrt-llm`, `vllm`, specific HF model repos), I can drop in the **exact submodule adds**, a **sparse set** for Transformers’ LLaMA/Qwen code paths, and a **`snapshot_hf.py`** that pulls only the implementation files you care about.

[1]: https://git-scm.com/book/en/v2/Git-Tools-Submodules?utm_source=chatgpt.com "Submodules"
[2]: https://git-scm.com/docs/git-status/2.11.4?utm_source=chatgpt.com "Git - git-status Documentation"
[3]: https://git-scm.com/docs/git-submodule?utm_source=chatgpt.com "Git - git-submodule Documentation"
[4]: https://manpages.debian.org/testing/git-man/git-subtree.1.en.html?utm_source=chatgpt.com "git-subtree(1) — git-man — Debian testing"
[5]: https://git-scm.com/docs/git-sparse-checkout?utm_source=chatgpt.com "git-sparse-checkout Documentation"
[6]: https://huggingface.co/docs/huggingface_hub/en/guides/download?utm_source=chatgpt.com "Download files from the Hub"
[7]: https://docs.github.com/repositories/working-with-files/managing-large-files/about-git-large-file-storage?utm_source=chatgpt.com "About Git Large File Storage"
[8]: https://pre-commit.com/?utm_source=chatgpt.com "pre-commit"
[9]: https://git-scm.com/docs/gitmodules?utm_source=chatgpt.com "Git - gitmodules Documentation"
