#!/bin/bash
# Install CUDA Toolkit 12.8 on Ubuntu 24.04 from official NVIDIA repository
# This script installs CUDA toolkit system-wide so compilers can find headers
#
# Usage: ./scripts/install-cuda-toolkit-12-8.sh
# (Do NOT run with sudo - script will request sudo when needed)

set -e  # Exit on any error

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Error: Do not run this script with sudo"
    echo "The script will request sudo privileges when needed"
    echo "Usage: ./scripts/install-cuda-toolkit-12-8.sh"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}CUDA Toolkit 12.8 Installation${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check if running on Ubuntu
if ! command -v lsb_release &> /dev/null; then
    echo -e "${RED}Error: lsb_release not found. This script is for Ubuntu.${NC}"
    exit 1
fi

DISTRO=$(lsb_release -is)
VERSION=$(lsb_release -rs)
CODENAME=$(lsb_release -cs)

if [ "$DISTRO" != "Ubuntu" ]; then
    echo -e "${RED}Error: This script is for Ubuntu. Detected: $DISTRO${NC}"
    exit 1
fi

echo -e "${YELLOW}Detected: Ubuntu $VERSION ($CODENAME)${NC}"
echo ""

# Check if NVIDIA GPU exists
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. Make sure NVIDIA driver is installed.${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}NVIDIA driver detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo ""
fi

# Determine Ubuntu version for CUDA repo
case $VERSION in
    20.04)
        CUDA_REPO="ubuntu2004"
        ;;
    22.04)
        CUDA_REPO="ubuntu2204"
        ;;
    24.04)
        CUDA_REPO="ubuntu2404"
        ;;
    *)
        echo -e "${RED}Error: Unsupported Ubuntu version: $VERSION${NC}"
        echo "Supported versions: 20.04, 22.04, 24.04"
        exit 1
        ;;
esac

echo -e "${GREEN}Step 1: Removing outdated signing key (if exists)${NC}"
sudo apt-key del 7fa2af80 2>/dev/null || echo "No outdated key to remove"
echo ""

echo -e "${GREEN}Step 2: Downloading CUDA keyring package${NC}"
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
KEYRING_DEB="/tmp/cuda-keyring_1.1-1_all-$$.deb"

# Remove old file if exists
rm -f "$KEYRING_DEB"

wget -q --show-progress "$KEYRING_URL" -O "$KEYRING_DEB"
echo ""

echo -e "${GREEN}Step 3: Installing CUDA keyring${NC}"
sudo dpkg -i "$KEYRING_DEB"
echo ""

echo -e "${GREEN}Step 4: Updating APT repository cache${NC}"
sudo apt-get update
echo ""

echo -e "${GREEN}Step 5: Installing CUDA Toolkit 12.8${NC}"
echo -e "${YELLOW}Note: Installing toolkit only (no driver update)${NC}"
echo ""

# Check if cuda-toolkit is already installed
if dpkg -l | grep -q "cuda-toolkit-12-8"; then
    echo -e "${YELLOW}CUDA Toolkit 12.8 is already installed. Skipping installation.${NC}"
else
    sudo apt-get install -y cuda-toolkit-12-8
fi
echo ""

echo -e "${GREEN}Step 6: Setting up environment variables${NC}"

# Get the real user's home directory (not root if running with sudo)
USER_HOME=$(eval echo ~${SUDO_USER:-$USER})
USER_BASHRC="$USER_HOME/.bashrc"

# Backup bashrc if this is first time
if ! grep -q "CUDA 12.8 Environment" "$USER_BASHRC"; then
    cp "$USER_BASHRC" "$USER_BASHRC.backup-$(date +%Y%m%d-%H%M%S)"
    echo -e "${YELLOW}Created backup: $USER_BASHRC.backup-$(date +%Y%m%d-%H%M%S)${NC}"
    
    cat >> "$USER_BASHRC" << 'EOF'

# CUDA 12.8 Environment (added by install-cuda-toolkit-12-8.sh)
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.8
EOF
    echo -e "${GREEN}Added CUDA environment variables to $USER_BASHRC${NC}"
else
    echo -e "${YELLOW}CUDA environment variables already in $USER_BASHRC${NC}"
fi

# Also set for current session
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.8

echo ""

echo -e "${GREEN}Step 7: Verifying installation${NC}"
echo ""

if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✓ nvcc found:${NC}"
    nvcc --version
else
    echo -e "${RED}✗ nvcc not found in PATH${NC}"
    echo "Try running: source ~/.bashrc"
    exit 1
fi

echo ""

if [ -f "/usr/local/cuda-12.8/include/cuda_runtime_api.h" ]; then
    echo -e "${GREEN}✓ CUDA headers found at: /usr/local/cuda-12.8/include/${NC}"
else
    echo -e "${RED}✗ CUDA headers not found${NC}"
    exit 1
fi

if [ -f "/usr/local/cuda-12.8/lib64/libcudart.so" ]; then
    echo -e "${GREEN}✓ CUDA libraries found at: /usr/local/cuda-12.8/lib64/${NC}"
else
    echo -e "${RED}✗ CUDA libraries not found${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Source your shell configuration:"
echo "   ${GREEN}source ~/.bashrc${NC}"
echo ""
echo "2. Verify CUDA is accessible:"
echo "   ${GREEN}nvcc --version${NC}"
echo ""
echo "3. Install flash-attn with limited parallel jobs (to avoid OOM):"
echo "   ${GREEN}MAX_JOBS=2 pip install flash-attn --no-build-isolation -v${NC}"
echo ""
echo "   For systems with more RAM, you can increase MAX_JOBS:"
echo "   ${GREEN}MAX_JOBS=4  # for 32GB RAM${NC}"
echo "   ${GREEN}MAX_JOBS=8  # for 64GB+ RAM${NC}"
echo ""
echo "4. (Optional) Remove Pixi CUDA toolkit if no longer needed:"
echo "   ${GREEN}pixi global remove cuda-toolkit${NC}"
echo ""

# Cleanup
rm -f "$KEYRING_DEB"
