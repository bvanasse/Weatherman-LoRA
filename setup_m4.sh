#!/bin/bash
#
# Weatherman-LoRA Mac M4 Training Environment Setup
# For Mac M4 with MPS backend and 32GB unified memory
#
# This script:
# 1. Verifies Python 3.10+ is installed
# 2. Creates a virtual environment in .venv-m4/
# 3. Installs MPS-compatible training dependencies
# 4. Validates MPS backend availability
# 5. Checks system memory (32GB recommended)
# 6. Displays setup summary and training guidance
#
# Usage: ./setup_m4.sh

set -e  # Exit on error

echo "============================================================"
echo "Weatherman-LoRA Mac M4 Training Environment Setup"
echo "============================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python 3.10+ is available
echo "Step 1: Checking Python version..."
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
        PYTHON_CMD="python3"
    else
        echo -e "${RED}Error: Python 3.10+ is required but found Python $PYTHON_VERSION${NC}"
        echo "Please install Python 3.10 or newer:"
        echo "  brew install python@3.10"
        exit 1
    fi
else
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.10 or newer:"
    echo "  brew install python@3.10"
    exit 1
fi

PYTHON_FULL_VERSION=$($PYTHON_CMD --version)
echo -e "${GREEN}✓${NC} Found: $PYTHON_FULL_VERSION"
echo ""

# Check if running on Apple Silicon
echo "Step 2: Verifying Apple Silicon (M1/M2/M3/M4)..."
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo -e "${GREEN}✓${NC} Running on Apple Silicon ($ARCH)"
else
    echo -e "${YELLOW}⚠${NC} Warning: Not running on Apple Silicon (detected: $ARCH)"
    echo "  This setup is optimized for M4 with MPS backend"
    echo "  Training may not work correctly on Intel Macs"
fi
echo ""

# Check system memory
echo "Step 3: Checking system memory..."
TOTAL_MEMORY=$(sysctl -n hw.memsize)
MEMORY_GB=$((TOTAL_MEMORY / 1024 / 1024 / 1024))
echo "Total unified memory: ${MEMORY_GB}GB"

if [ "$MEMORY_GB" -lt 16 ]; then
    echo -e "${RED}✗${NC} Insufficient memory: ${MEMORY_GB}GB < 32GB recommended"
    echo "  Training will likely fail due to OOM errors"
    echo "  Consider using H100 cloud GPU instead"
    exit 1
elif [ "$MEMORY_GB" -lt 32 ]; then
    echo -e "${YELLOW}⚠${NC} Limited memory: ${MEMORY_GB}GB (32GB recommended)"
    echo "  Training possible but may require aggressive memory optimization"
    echo "  Use batch_size=1, seq_length=1024, gradient_accumulation=16"
else
    echo -e "${GREEN}✓${NC} Sufficient memory: ${MEMORY_GB}GB"
    echo "  Suitable for M4 training with reduced batch size"
fi
echo ""

# Create virtual environment
echo "Step 4: Creating virtual environment..."
if [ -d ".venv-m4" ]; then
    echo -e "${YELLOW}Virtual environment already exists, skipping creation${NC}"
else
    $PYTHON_CMD -m venv .venv-m4
    echo -e "${GREEN}✓${NC} Created .venv-m4/"
fi
echo ""

# Activate virtual environment
echo "Step 5: Activating virtual environment..."
source .venv-m4/bin/activate
echo -e "${GREEN}✓${NC} Activated .venv-m4/"
echo ""

# Upgrade pip
echo "Step 6: Upgrading pip..."
pip install --upgrade pip -q
PIP_VERSION=$(pip --version | cut -d' ' -f2)
echo -e "${GREEN}✓${NC} pip upgraded to version $PIP_VERSION"
echo ""

# Install dependencies
echo "Step 7: Installing dependencies from requirements-m4.txt..."
if [ ! -f "requirements-m4.txt" ]; then
    echo -e "${RED}Error: requirements-m4.txt not found${NC}"
    exit 1
fi

echo "This may take 5-10 minutes (PyTorch, Transformers, etc.)..."
pip install -r requirements-m4.txt -q
echo -e "${GREEN}✓${NC} All dependencies installed"
echo ""

# Validate MPS backend
echo "Step 8: Validating MPS backend..."
MPS_CHECK=$(python3 -c "import torch; print(torch.backends.mps.is_available())" 2>&1)
if [ "$MPS_CHECK" = "True" ]; then
    echo -e "${GREEN}✓${NC} MPS (Metal Performance Shaders) backend is available"
    echo "  GPU acceleration enabled for Apple Silicon"

    # Test MPS computation
    echo "  Testing MPS computation..."
    MPS_TEST=$(python3 -c "import torch; x = torch.randn(100, 100, device='mps'); y = torch.matmul(x, x); print('success')" 2>&1)
    if [ "$MPS_TEST" = "success" ]; then
        echo -e "${GREEN}✓${NC} MPS computation test passed"
    else
        echo -e "${RED}✗${NC} MPS computation test failed:"
        echo "  $MPS_TEST"
    fi
else
    echo -e "${RED}✗${NC} MPS backend is not available"
    echo "  Possible issues:"
    echo "  1. Not running on Apple Silicon (M1/M2/M3/M4)"
    echo "  2. PyTorch version too old (need 2.0+)"
    echo "  3. macOS version too old (need 12.3+)"
    exit 1
fi
echo ""

# Display installed packages summary
echo "Step 9: Installed packages summary..."
echo "Core libraries:"
pip list | grep -E "torch|transformers|peft|trl|accelerate|bitsandbytes|datasets" || true
echo ""

# Run platform detection
echo "Step 10: Running platform detection..."
if [ -f "scripts/check_gpu.py" ]; then
    python3 scripts/check_gpu.py || true
else
    echo -e "${YELLOW}⚠${NC} Platform detection script not found (scripts/check_gpu.py)"
fi
echo ""

# Success message
echo "============================================================"
echo -e "${GREEN}Mac M4 Training Environment Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "Virtual environment: .venv-m4/"
echo "Python version: $PYTHON_FULL_VERSION"
echo "Unified memory: ${MEMORY_GB}GB"
echo "MPS backend: Available ✓"
echo ""
echo "To activate the environment:"
echo "  source .venv-m4/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "Recommended training configuration: configs/training_config_m4.yaml"
echo ""
echo "Expected training time: 8-12 hours for 3 epochs (15K examples)"
echo "  - Sequence length: 2048 tokens (vs 4096 on H100)"
echo "  - Batch size: 1-2 (vs 4-8 on H100)"
echo "  - No Flash Attention 2 (CUDA-only)"
echo "  - 2-3x slower than H100 due to MPS vs CUDA"
echo ""
echo "Memory optimization tips:"
echo "  - Close all unnecessary applications"
echo "  - Monitor Activity Monitor for memory pressure"
echo "  - Train overnight or during off-hours"
echo "  - Use wandb for remote monitoring"
echo ""
echo "Next steps:"
echo "  1. Validate environment: python scripts/validate_environment.py --env=m4"
echo "  2. Check platform: python scripts/check_gpu.py"
echo "  3. Validate config: python scripts/validate_training_config.py --config configs/training_config_m4.yaml"
echo "  4. Ensure training data exists: data/processed/train.jsonl"
echo "  5. Start training with M4 config"
echo ""
