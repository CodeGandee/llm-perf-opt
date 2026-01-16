# How to Use the RTX 5090 Environment

This guide explains how to set up and use the dedicated RTX 5090 environment for NVIDIA's Blackwell architecture with CUDA compute capability sm_120.

## Background

The RTX 5090 uses NVIDIA's Blackwell architecture with CUDA compute capability **sm_120**, which is not supported by stable PyTorch releases (up to v2.6.x). The default environment uses PyTorch 2.6.0 with CUDA 12.6, which only supports up to sm_90.

### Requirements for RTX 5090

- **PyTorch**: Nightly builds with CUDA 12.8+ (includes sm_120 support)
- **CUDA Toolkit**: Version 12.8 or higher
- **Flash Attention**: Must be built from source with `TORCH_CUDA_ARCH_LIST="12.0"`
- **Triton**: Latest main branch with Blackwell support

## Installation

### Automated Installation (Recommended)

The easiest way to set up the RTX 5090 environment is to use the automated installation script:

```bash
# Run the automated setup script
./scripts/install-deps-rtx5090.sh
```

This script will:
1. Install the base rtx5090 environment
2. Install PyTorch nightly with CUDA 12.8
3. Build Triton from source with sm_120 support
4. Build Flash Attention from source with sm_120 support
5. Verify the installation

**Total time**: 20-40 minutes (mostly building Flash Attention)

After installation, verify everything works:

```bash
./scripts/verify-rtx5090.sh
```

### Manual Installation (Alternative)

If you prefer to install step-by-step:

#### Step 1: Install the RTX 5090 Environment

```bash
# Install the rtx5090 environment dependencies
pixi install --environment rtx5090
```

This will create a separate environment with:
- Python 3.11/3.12
- pip, ninja, setuptools, wheel
- All project dependencies

#### Step 2: Install PyTorch Nightly

```bash
# Install PyTorch nightly with CUDA 12.8
pixi run --environment rtx5090 install-pytorch-nightly
```

#### Step 3: Build Flash Attention and Triton

After installing the base dependencies, you need to build Flash Attention and Triton from source with sm_120 support:

```bash
# Build Triton (5-10 minutes)
pixi run --environment rtx5090 build-triton

# Build Flash Attention (15-30 minutes)
pixi run --environment rtx5090 build-flash-attn

# Verify installation
pixi run --environment rtx5090 verify-rtx5090
```

**Note**: Building Flash Attention can take 10-30 minutes depending on your system.

### Alternative: Manual Build

If you prefer to build manually:

```bash
# Activate the environment
pixi shell --environment rtx5090

# Build Triton
pip install git+https://github.com/triton-lang/triton.git@main

# Build Flash Attention with sm_120 support
TORCH_CUDA_ARCH_LIST="12.0" pip install flash-attn --no-build-isolation

# Verify
python -c "import torch; print(torch.zeros(3).cuda())"
```

## Usage

### Running Commands in RTX 5090 Environment

Use `pixi run --environment rtx5090` to execute commands in the RTX 5090 environment:

```bash
# Run Python script
pixi run --environment rtx5090 python your_script.py

# Run profiling
pixi run --environment rtx5090 stage1-run

# Run direct inference
pixi run --environment rtx5090 direct-infer-dev20
```

### Activating the Environment Shell

To work interactively in the RTX 5090 environment:

```bash
pixi shell --environment rtx5090
```

Once inside the shell, all commands use the RTX 5090 environment automatically.

### Verifying Installation

Check that PyTorch recognizes your RTX 5090:

```bash
pixi run --environment rtx5090 verify-rtx5090
```

Expected output:
```
PyTorch: 2.7.0+cu128 (or later)
CUDA available: True
CUDA version: 12.8
Device: NVIDIA GeForce RTX 5090
Test tensor on CUDA: tensor([0., 0., 0.], device='cuda:0')
```

## Troubleshooting

### Issue: "sm_120 is not compatible with the current PyTorch installation"

**Solution**: Ensure you're using the `rtx5090` environment, not the default:
```bash
# Wrong - uses default environment
pixi run python script.py

# Correct - uses rtx5090 environment  
pixi run --environment rtx5090 python script.py
```

### Issue: Flash Attention Build Fails

**Solution**: Make sure you have the CUDA 12.8 toolkit installed and accessible:
```bash
# Check CUDA version
pixi shell --environment rtx5090
nvcc --version  # Should show 12.8 or higher
```

If nvcc is not found, the CUDA toolkit may not be properly installed in the environment.

### Issue: "CUDA error: no kernel image is available for execution"

**Causes**:
1. Using wrong environment (default instead of rtx5090)
2. Flash Attention or Triton not built with sm_120 support
3. Library cache issues

**Solutions**:
```bash
# Rebuild Flash Attention with proper architecture
pixi shell --environment rtx5090
pip uninstall flash-attn -y
TORCH_CUDA_ARCH_LIST="12.0" pip install flash-attn --no-build-isolation --force-reinstall
```

## Switching Between Environments

### Default Environment (CUDA 12.6, up to sm_90)
```bash
pixi run python script.py                    # Auto-selects default
pixi shell                                    # Opens default shell
```

### RTX 5090 Environment (CUDA 12.8, sm_120)
```bash
pixi run --environment rtx5090 python script.py
pixi shell --environment rtx5090
```

## Known Limitations

1. **Binary Incompatibility**: The rtx5090 environment uses nightly PyTorch builds which may have API changes or instabilities
2. **Separate Environments**: The default and rtx5090 environments are completely independent; packages installed in one won't appear in the other
3. **Build Time**: First-time setup requires compiling Flash Attention and Triton from source (20-40 minutes total)
4. **Disk Space**: Having two separate environments roughly doubles the disk space usage (~10-15 GB total)

## Reference Links

- [PyTorch RTX 5090 Support Discussion](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099)
- [PyTorch GitHub Issue #159207](https://github.com/pytorch/pytorch/issues/159207)
- [Flash Attention Blackwell Support](https://github.com/Dao-AILab/flash-attention/issues/1683)
- [vLLM RTX 5090 Support](https://github.com/vllm-project/vllm/issues/13306)

## Related Files

- `pyproject.toml`: Environment configuration
- `context/hints/howto-install-cuda-toolkit-pixi.md`: General CUDA toolkit setup
- `context/hints/howto-profile-deepseek-ocr-with-nsight.md`: Profiling guide
