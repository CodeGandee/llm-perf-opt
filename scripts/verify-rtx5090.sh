#!/usr/bin/env bash
# Quick verification script for RTX 5090 environment
# Use this to check if your RTX 5090 setup is working correctly

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "RTX 5090 Environment Verification"
echo "=========================================="
echo ""

# Check if environment exists
if ! pixi info --environment rtx5090 &>/dev/null; then
    echo "❌ RTX 5090 environment not found!"
    echo ""
    echo "Run the installation script first:"
    echo "  ./scripts/install-deps-rtx5090.sh"
    exit 1
fi

echo "✅ RTX 5090 environment exists"
echo ""

# Run comprehensive verification
echo "Checking PyTorch and CUDA setup..."
pixi run --environment rtx5090 python -c "
import sys
import torch

print('Python version:', sys.version.split()[0])
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('cuDNN version:', torch.backends.cudnn.version())
    print('Number of GPUs:', torch.cuda.device_count())
    print('Current GPU:', torch.cuda.current_device())
    print('GPU name:', torch.cuda.get_device_name(0))
    
    capability = torch.cuda.get_device_capability(0)
    print(f'CUDA capability: sm_{capability[0]}{capability[1]}')
    
    if capability == (12, 0):
        print('✅ RTX 5090 (sm_120) detected!')
    else:
        print(f'⚠️  Expected sm_120, got sm_{capability[0]}{capability[1]}')
    
    # Test tensor operation
    print('')
    print('Testing tensor operations on GPU...')
    t = torch.randn(1000, 1000).cuda()
    result = torch.matmul(t, t.T)
    print(f'✅ Matrix multiplication successful! Result shape: {result.shape}')
else:
    print('❌ CUDA not available!')
    sys.exit(1)
"

echo ""
echo "Checking Triton..."
pixi run --environment rtx5090 python -c "
try:
    import triton
    print(f'✅ Triton version: {triton.__version__}')
except ImportError as e:
    print(f'⚠️  Triton not installed: {e}')
    print('   Run: pixi run --environment rtx5090 build-triton')
" || true

echo ""
echo "Checking Flash Attention..."
pixi run --environment rtx5090 python -c "
try:
    import flash_attn
    print(f'✅ Flash Attention version: {flash_attn.__version__}')
except ImportError as e:
    print(f'⚠️  Flash Attention not installed: {e}')
    print('   Run: pixi run --environment rtx5090 build-flash-attn')
" || true

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
