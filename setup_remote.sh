#!/bin/bash
#
# Weatherman-LoRA Remote GPU Environment Setup
# For H100/3090 GPU machines (Lambda Labs, RunPod, or home 3090)
#
# This script:
# 1. Verifies Python 3.10 and CUDA 12.1+ are installed
# 2. Creates a conda environment from environment-remote.yml
# 3. Runs GPU diagnostics
# 4. Verifies storage requirements
# 5. Displays setup summary
#
# Prerequisites:
# - Conda or Miniconda installed
# - CUDA 12.1+ drivers installed
# - NVIDIA GPU with 24GB+ VRAM
#
# Usage: ./setup_remote.sh

set -e  # Exit on error

echo "============================================================"
echo "Weatherman-LoRA Remote GPU Environment Setup"
echo "============================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is installed
echo "Step 1: Checking conda installation..."
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed${NC}"
    echo "Please install Miniconda or Anaconda:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

CONDA_VERSION=$(conda --version)
echo -e "${GREEN}✓${NC} Found: $CONDA_VERSION"
echo ""

# Check CUDA installation
echo "Step 2: Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}Warning: nvcc not found in PATH${NC}"
    echo "CUDA toolkit may not be installed or not in PATH"
    echo "The conda environment will install CUDA toolkit automatically"
    CUDA_VERSION="Not found (will be installed by conda)"
else
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d'.' -f2)

    if [ "$CUDA_MAJOR" -lt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 1 ]); then
        echo -e "${YELLOW}Warning: CUDA version $CUDA_VERSION is older than required 12.1${NC}"
        echo "The conda environment will install CUDA 12.1 toolkit"
    else
        echo -e "${GREEN}✓${NC} Found: CUDA $CUDA_VERSION"
    fi
fi
echo ""

# Check if environment already exists
echo "Step 3: Checking conda environment..."
if conda env list | grep -q "weatherman-lora"; then
    echo -e "${YELLOW}Environment 'weatherman-lora' already exists${NC}"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n weatherman-lora -y
        echo -e "${GREEN}✓${NC} Removed existing environment"
    else
        echo "Keeping existing environment. Skipping creation."
        echo ""
        echo "To activate the environment:"
        echo "  conda activate weatherman-lora"
        exit 0
    fi
fi
echo ""

# Create conda environment
echo "Step 4: Creating conda environment from environment-remote.yml..."
if [ ! -f "environment-remote.yml" ]; then
    echo -e "${RED}Error: environment-remote.yml not found${NC}"
    exit 1
fi

echo "This may take 10-15 minutes to download and install packages..."
conda env create -f environment-remote.yml
echo -e "${GREEN}✓${NC} Environment created: weatherman-lora"
echo ""

# Activate environment
echo "Step 5: Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate weatherman-lora
echo -e "${GREEN}✓${NC} Activated weatherman-lora environment"
echo ""

# Run GPU diagnostics
echo "Step 6: Running GPU diagnostics..."
if [ -f "scripts/check_gpu.py" ]; then
    python scripts/check_gpu.py
else
    echo -e "${YELLOW}Warning: scripts/check_gpu.py not found, skipping GPU check${NC}"
fi
echo ""

# Verify storage
echo "Step 7: Verifying storage requirements..."
if [ -f "scripts/check_storage.py" ]; then
    python scripts/check_storage.py
else
    echo -e "${YELLOW}Warning: scripts/check_storage.py not found, skipping storage check${NC}"
fi
echo ""

# Display installed packages summary
echo "Step 8: Installed packages summary..."
echo "Core ML libraries:"
conda list | grep -E "pytorch|transformers|peft|trl|accelerate|bitsandbytes|datasets" || true
echo ""

# Success message
echo "============================================================"
echo -e "${GREEN}Remote GPU Environment Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "Environment: weatherman-lora"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "To activate the environment:"
echo "  conda activate weatherman-lora"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
echo ""
echo "Next steps:"
echo "  1. Sync processed data from local machine"
echo "  2. Download base model (Llama 3.1 8B or Mistral 7B)"
echo "  3. Configure training parameters in configs/"
echo "  4. Run LoRA training (Roadmap Items 8, 10)"
echo ""
