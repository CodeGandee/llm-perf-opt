#!/usr/bin/env bash
# install-deps-rtx5090.sh
# Automated installation script for RTX 5090 (Blackwell/sm_120) dependencies
# This script sets up PyTorch nightly with CUDA 12.8, Triton, and Flash Attention

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "RTX 5090 Environment Setup"
echo "=========================================="
echo ""

# Check if we're in the project root
cd "$PROJECT_ROOT"

# Step 1: Install base environment
echo "Step 1/5: Installing base rtx5090 environment..."
pixi install --environment rtx5090

# Step 2: Install PyTorch nightly with CUDA 12.8
echo ""
echo "Step 2/5: Installing PyTorch nightly with CUDA 12.8 (sm_120 support)..."
echo "This will take a few minutes to download (~2-3 GB)..."
pixi run --environment rtx5090 install-pytorch-nightly

# Step 3: Verify PyTorch installation
echo ""
echo "Step 3/5: Verifying PyTorch installation..."
pixi run --environment rtx5090 python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')
"

# Step 4: Build Triton from source
echo ""
echo "Step 4/5: Building Triton from source with sm_120 support..."
echo "This will take 5-10 minutes..."
pixi run --environment rtx5090 build-triton

# Step 5: Build Flash Attention from source
echo ""
echo "Step 5/5: Building Flash Attention from source with sm_120 support..."
echo "This will take 15-30 minutes depending on your CPU..."
echo "Using MAX_JOBS=4 to avoid overwhelming the system..."
pixi run --environment rtx5090 build-flash-attn

# Final verification
echo ""
echo "=========================================="
echo "Final Verification"
echo "=========================================="
pixi run --environment rtx5090 verify-rtx5090

echo ""
echo "=========================================="
echo "✅ RTX 5090 Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  # Run commands in rtx5090 environment:"
echo "  pixi run --environment rtx5090 python your_script.py"
echo ""
echo "  # Or activate the environment shell:"
echo "  pixi shell --environment rtx5090"
echo ""
echo "  # Run profiling tasks:"
echo "  pixi run --environment rtx5090 stage1-run"
echo "  pixi run --environment rtx5090 direct-infer-dev20"
echo ""
echo "For more information, see:"
echo "  context/hints/howto-use-rtx5090-environment.md"
echo ""
