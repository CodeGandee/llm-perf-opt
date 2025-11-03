# Installing CUDA 12.8 on Ubuntu via Official NVIDIA Repository

## Problem
When installing CUDA via Pixi globally, the CUDA headers and libraries are not properly accessible to system compilers, causing compilation failures for packages like flash-attn.

## Solution: Use Official NVIDIA APT Repository

### For Ubuntu 20.04/22.04/24.04

#### 1. Remove Outdated Signing Key (if exists)
```bash
sudo apt-key del 7fa2af80
```

#### 2. Download and Install CUDA Keyring Package

**For Ubuntu 20.04:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```

**For Ubuntu 22.04:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```

**For Ubuntu 24.04:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```

#### 3. Update APT Repository Cache
```bash
sudo apt-get update
```

#### 4. Install CUDA Toolkit 12.8

**Option A: Install only the toolkit (no driver, recommended if driver already installed):**
```bash
sudo apt-get install cuda-toolkit-12-8
```

**Option B: Install everything including driver:**
```bash
sudo apt-get install cuda-12-8
```

#### 5. Set Environment Variables

Add to your `~/.bashrc` or `~/.profile`:

```bash
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.8
```

Then source it:
```bash
source ~/.bashrc
```

#### 6. Verify Installation
```bash
nvcc --version
nvidia-smi
```

### For RHEL/Rocky Linux 8/9

#### 1. Enable Optional Repositories (RHEL 9):
```bash
sudo subscription-manager repos --enable codeready-builder-for-rhel-9-$(arch)-rpms
```

#### 2. Install Network Repository:
```bash
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
```

#### 3. Clean and Install:
```bash
sudo dnf clean all
sudo dnf install cuda-toolkit-12-8
```

## After CUDA Installation: Flash-Attention Build

### Reduce Memory Usage During Compilation

Set `MAX_JOBS` to limit parallel compilation (avoids OOM):

```bash
# For systems with 16GB RAM:
MAX_JOBS=2 pip install flash-attn --no-build-isolation

# For systems with 32GB RAM:
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# For systems with 64GB+ RAM:
MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

### Alternative: Try Installing Older Flash-Attn with Prebuilt Wheels

If you still encounter issues, consider using a stable PyTorch version with prebuilt flash-attn:

```bash
# Check your current PyTorch/CUDA version
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

# For PyTorch 2.5.x + CUDA 12.4 (more stable):
pip install flash-attn --no-build-isolation
```

## Cleanup Pixi CUDA (Optional)

If you want to remove the Pixi CUDA toolkit:

```bash
pixi global remove cuda-toolkit
```

## References
- [CUDA 12.8 Installation Guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/)
- [CUDA 12.8 Downloads](https://developer.nvidia.com/cuda-12-8-0-download-archive)
- [Flash-Attention Compilation Issues](https://github.com/Dao-AILab/flash-attention/issues/1038)
