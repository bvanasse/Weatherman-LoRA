#!/bin/bash
#
# Fix Corrupted Conda Environment for H100 Training
# This script rebuilds the weatherman-lora environment from scratch
#

set -e

echo "============================================================"
echo "Fixing Corrupted Conda Environment"
echo "============================================================"
echo ""

# Deactivate any active environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Deactivating current environment: $CONDA_DEFAULT_ENV"
    conda deactivate 2>/dev/null || true
fi

# Remove corrupted environment
echo "Removing corrupted environment..."
conda env remove -n weatherman-lora -y 2>/dev/null || echo "Environment doesn't exist yet"

# Create fresh environment with Python 3.10
echo ""
echo "Creating fresh conda environment..."
conda create -n weatherman-lora python=3.10 -y

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate weatherman-lora

# Verify activation
if [ "$CONDA_DEFAULT_ENV" != "weatherman-lora" ]; then
    echo "ERROR: Failed to activate environment"
    exit 1
fi

echo "✓ Environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Install pip properly
echo "Installing pip..."
conda install -y pip

# Verify pip works
python -m pip --version
echo "✓ pip working"
echo ""

# Install PyTorch with CUDA 12.1 support
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Install Axolotl and dependencies
echo "Installing Axolotl (this will take 10-15 minutes)..."
pip install packaging wheel setuptools

# Install Axolotl from git (more reliable than PyPI)
pip install git+https://github.com/axolotl-ai-cloud/axolotl.git

# Install deepspeed separately
echo ""
echo "Installing DeepSpeed..."
pip install deepspeed

# Install flash-attention (optional but recommended for H100)
echo ""
echo "Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import axolotl; print(f'✓ Axolotl {axolotl.__version__}')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import deepspeed; print(f'✓ DeepSpeed {deepspeed.__version__}')"

echo ""
echo "============================================================"
echo "✓ Environment Fixed Successfully!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate weatherman-lora"
echo "  2. Run training: ./train_with_axolotl_h100.sh"
echo ""
