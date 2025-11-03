# How to Install CUDA Toolkit with Pixi

This guide explains how to install NVIDIA CUDA Toolkit using Pixi, both for project environments and globally for system-wide access.

## Overview

Pixi supports installing CUDA toolkit packages through conda-forge and NVIDIA channels. You can install CUDA toolkit in three ways:

1. **Project-level installation** - CUDA toolkit as a project dependency
2. **Global installation** - System-wide CUDA toolkit accessible from anywhere
3. **Feature-based installation** - Separate environments for GPU/CPU configurations

## Prerequisites

- Pixi installed ([installation guide](https://prefix.dev/docs/pixi/installation))
- NVIDIA GPU with compatible driver installed
- Check your CUDA driver version: `nvidia-smi`

## Method 1: Global Installation

Global installation makes CUDA toolkit available system-wide, useful when you need CUDA tools across multiple projects.

### Install Specific Version

To install CUDA toolkit matching your driver version (e.g., CUDA 12.8):

```bash
# Install CUDA toolkit with version constraint
pixi global install 'cuda-toolkit<12.9' --channel nvidia --channel conda-forge

# Or specify exact version
pixi global install 'cuda-toolkit==12.8.*' --channel nvidia --channel conda-forge
```

### Verify Installation

```bash
# List globally installed packages
pixi global list

# Check nvcc version
~/.pixi/envs/cuda-toolkit/bin/nvcc --version
```

### Add to PATH

To make CUDA tools accessible from any terminal, add to your `~/.bashrc`:

```bash
# Add Pixi CUDA toolkit to PATH
export PATH="$HOME/.pixi/envs/cuda-toolkit/bin:$PATH"
```

Apply changes:

```bash
source ~/.bashrc
nvcc --version
```

**Source:** Based on [Pixi global install documentation](https://prefix.dev/docs/pixi/cli#global)

## Method 2: Project-Level Installation

Install CUDA toolkit as a dependency in your `pixi.toml` or `pyproject.toml` for project-specific needs.

### Using pixi.toml

```toml
[workspace]
channels = ["nvidia", "conda-forge"]
platforms = ["linux-64"]

[system-requirements]
cuda = "12"  # Inform Pixi that CUDA 12 is available

[dependencies]
cuda-toolkit = "12.8.*"
cuda-version = "12.8.*"  # Lock CUDA version
```

### Using pyproject.toml

```toml
[project]
name = "my-cuda-project"

[tool.pixi.project]
channels = ["nvidia", "conda-forge"]
platforms = ["linux-64"]

[tool.pixi.system-requirements]
cuda = "12"

[tool.pixi.dependencies]
cuda-toolkit = "12.8.*"
cuda-version = "12.8.*"
```

**Source:** [Pixi System Requirements](https://prefix-dev.github.io/pixi/dev/workspace/system_requirements/)

### Install and Verify

```bash
# Install dependencies
pixi install

# Run nvcc in the project environment
pixi run nvcc --version
```

## Method 3: Multi-Environment Setup (GPU/CPU)

Create separate environments for machines with and without CUDA support.

### Using pixi.toml

```toml
[workspace]
channels = ["nvidia", "conda-forge"]
platforms = ["linux-64"]

[feature.cuda.system-requirements]
cuda = "12"

[feature.cuda.dependencies]
cuda-toolkit = "12.8.*"
cuda-version = "12.8.*"

[feature.cpu]
# CPU-only dependencies

[environments]
gpu = ["cuda"]
cpu = ["cpu"]
default = ["cuda"]  # Default to GPU environment
```

### Using pyproject.toml

```toml
[project]
name = "my-cuda-project"

[tool.pixi.project]
channels = ["nvidia", "conda-forge"]
platforms = ["linux-64"]

[tool.pixi.feature.cuda.system-requirements]
cuda = "12"

[tool.pixi.feature.cuda.dependencies]
cuda-toolkit = "12.8.*"
cuda-version = "12.8.*"

[tool.pixi.feature.cpu]
# CPU-only dependencies

[tool.pixi.environments]
gpu = ["cuda"]
cpu = ["cpu"]
default = ["cuda"]
```

**Source:** [Pixi Multi-Environment](https://prefix-dev.github.io/pixi/v0.21.1/features/multi_environment/)

### Run Commands in Specific Environments

```bash
# Run in GPU environment
pixi run -e gpu nvcc --version

# Run in CPU environment
pixi run -e cpu python my_script.py
```

## Understanding System Requirements

The `[system-requirements]` table tells Pixi what system capabilities are available:

```toml
[system-requirements]
cuda = "12"  # CUDA 12.x is available on this system
```

This enables Pixi to:
- Resolve packages that depend on `__cuda >= 12` virtual package
- Install GPU-enabled versions of libraries (e.g., PyTorch, TensorFlow)
- Lock appropriate CUDA-dependent packages in the lock file

**Important:** System requirements specify the *available* system version, not a minimum or maximum. Packages then declare their own requirements (e.g., `__cuda >= 12`), and Pixi resolves compatible versions.

**Source:** [Pixi System Requirements](https://prefix-dev.github.io/pixi/dev/workspace/system_requirements/)

## Using cuda-version Package

The `cuda-version` package constrains both the `__cuda` virtual package and `cudatoolkit` package versions:

```toml
[dependencies]
cuda-version = "12.8.*"  # Ensures CUDA 12.8.x is used
```

This guarantees that all CUDA-dependent packages resolve against the specified CUDA version.

**Source:** [Pixi PyTorch Installation - CUDA Version](https://prefix-dev.github.io/pixi/dev/python/pytorch/#installing-from-conda-forge)

## Available CUDA Packages

Key CUDA-related packages available through Pixi:

- `cuda-toolkit` - Full CUDA development toolkit (compiler, debugger, profiler, libraries)
- `cuda-version` - Version constraint package for locking CUDA versions
- `cudatoolkit` - CUDA runtime libraries (legacy, prefer `cuda-toolkit`)
- `cuda-compiler` - CUDA compiler (nvcc)
- `cuda-libraries` - CUDA runtime libraries
- `cuda-libraries-dev` - CUDA development libraries
- `cuda-nvml-dev` - NVIDIA Management Library development files
- `cuda-tools` - CUDA profiling and debugging tools

## Troubleshooting

### Check Detected CUDA Version

```bash
pixi info
```

Look for the `__cuda` virtual package in the output:

```
Virtual packages: __unix=0=0
                : __linux=6.5.9=0
                : __cuda=12.5=0
```

If `__cuda` is missing, verify your driver installation with `nvidia-smi`.

**Source:** [Pixi PyTorch Troubleshooting](https://prefix-dev.github.io/pixi/dev/python/pytorch/#checking-the-cuda-version-of-your-machine)

### Override System CUDA Version

If Pixi detects the wrong CUDA version, override it with an environment variable:

```bash
CONDA_OVERRIDE_CUDA=12.8 pixi install
```

**Source:** [Pixi System Requirements - Override Options](https://prefix-dev.github.io/pixi/dev/workspace/system_requirements/#available-override-options)

### Search for Available Versions

```bash
# Search for CUDA toolkit packages
pixi search cuda-toolkit --channel nvidia --channel conda-forge

# Search for specific CUDA version
pixi search cuda-version --channel nvidia --channel conda-forge | grep "12.8"
```

## Complete Example

Here's a complete example for a PyTorch project with CUDA 12.8:

```toml
[project]
name = "pytorch-cuda-project"
requires-python = ">=3.11,<3.13"

[tool.pixi.project]
channels = ["nvidia", "conda-forge"]
platforms = ["linux-64"]

[tool.pixi.system-requirements]
cuda = "12"

[tool.pixi.dependencies]
python = ">=3.11,<3.13"
cuda-toolkit = "12.8.*"
cuda-version = "12.8.*"
pytorch-gpu = "*"
torchvision = "*"

[tool.pixi.feature.cpu.dependencies]
pytorch-cpu = "*"

[tool.pixi.environments]
gpu = { features = [] }
cpu = { features = ["cpu"] }
default = { features = [] }
```

Install and test:

```bash
pixi install
pixi run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
pixi run nvcc --version
```

## References

- [Pixi PyTorch Installation Guide](https://prefix-dev.github.io/pixi/dev/python/pytorch/)
- [Pixi System Requirements](https://prefix-dev.github.io/pixi/dev/workspace/system_requirements/)
- [Pixi Multi-Environment](https://prefix-dev.github.io/pixi/v0.21.1/features/multi_environment/)
- [Conda Virtual Packages](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
