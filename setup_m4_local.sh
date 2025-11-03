#!/bin/bash
#
# Weatherman-LoRA Mac M4 Local Training Environment Setup
# For Mac M4 with MPS backend and 32GB+ unified memory
#
# This script:
# 1. Verifies Mac M4 hardware and Python 3.10+
# 2. Checks unified memory (32GB recommended)
# 3. Creates virtual environment (.venv-local)
# 4. Installs MPS-compatible dependencies
# 5. Validates MPS backend availability
# 6. Runs M4-specific validation
# 7. Sets up data paths
# 8. Displays warnings and completion message
#
# Usage: ./setup_m4_local.sh

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions for structured output
success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗ ERROR:${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠ WARNING:${NC} $1"
}

info() {
    echo -e "$1"
}

echo "============================================================"
echo "Weatherman-LoRA Mac M4 Local Training Environment Setup"
echo "============================================================"
echo ""

info "[SETUP-M4] Starting Mac M4 local setup"
echo ""

# Step 3.2: Verify Mac M4 hardware and Python version
info "Step 1: Verifying Mac M4 hardware..."

# Check if running on Apple Silicon
CHIP_INFO=$(system_profiler SPHardwareDataType | grep "Chip")
if echo "$CHIP_INFO" | grep -q "Apple M"; then
    CHIP_TYPE=$(echo "$CHIP_INFO" | awk -F': ' '{print $2}')
    success "Found: $CHIP_TYPE"
else
    error "Not running on Apple Silicon"
    info "This setup is optimized for Mac M4 (Apple Silicon)"
    exit 1
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    error "Python 3 is not installed"
    info "Please install Python 3.10+:"
    info "  brew install python@3.10"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    error "Python version $PYTHON_VERSION is too old (require 3.10+)"
    info "Please install Python 3.10+:"
    info "  brew install python@3.10"
    exit 1
fi

success "Found: Python $PYTHON_VERSION"
echo ""

# Step 3.3: Check unified memory
info "Step 2: Checking unified memory..."

TOTAL_MEMORY=$(sysctl -n hw.memsize)
MEMORY_GB=$((TOTAL_MEMORY / 1024 / 1024 / 1024))

info "Unified Memory: ${MEMORY_GB}GB"

if [ "$MEMORY_GB" -lt 16 ]; then
    error "INSUFFICIENT: ${MEMORY_GB}GB < 16GB minimum"
    info "Mac M4 training requires at least 16GB unified memory"
    info "32GB+ strongly recommended for stable training"
    exit 1
elif [ "$MEMORY_GB" -lt 32 ]; then
    warning "LIMITED: ${MEMORY_GB}GB (32GB recommended)"
    info "Training is possible but may experience memory pressure"
    info "Recommendations:"
    info "  - Use batch_size=1, gradient_accumulation=16"
    info "  - Reduce sequence_length to 1024 or 2048"
    info "  - Close all unnecessary applications"
    success "Unified Memory: ${MEMORY_GB}GB"
else
    info "SUFFICIENT: ${MEMORY_GB}GB"
    success "Unified Memory: ${MEMORY_GB}GB"
fi

echo ""

# Step 3.4: Create virtual environment
info "Step 3: Creating virtual environment..."

if [ -d ".venv-local" ]; then
    warning "Virtual environment .venv-local already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv-local
        success "Removed existing environment"
    else
        info "Keeping existing environment"
        source .venv-local/bin/activate
        SKIP_ENV_CREATION=true
    fi
fi

if [ "$SKIP_ENV_CREATION" != "true" ]; then
    python3 -m venv .venv-local
    success "Virtual environment created"

    # Activate
    source .venv-local/bin/activate
    success "Activated .venv-local"

    # Upgrade pip
    pip install --upgrade pip -q
    PIP_VERSION=$(pip --version | cut -d' ' -f2)
    success "pip upgraded to $PIP_VERSION"
fi

echo ""

# Step 3.5: Install packages from requirements-local.txt
info "Step 4: Installing packages from requirements-local.txt..."

if [ ! -f "requirements-local.txt" ]; then
    error "requirements-local.txt not found"
    exit 1
fi

info "Installing dependencies (this may take 5-10 minutes)..."
pip install -r requirements-local.txt -q

# Verify PyTorch MPS support
PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
success "Packages installed"
info "PyTorch version: $PYTORCH_VERSION"

echo ""

# Step 3.6: Validate MPS backend availability
info "Step 5: Validating MPS backend..."

# Check MPS availability
MPS_AVAILABLE=$(python3 -c "import torch; print(torch.backends.mps.is_available())" 2>&1)

if [ "$MPS_AVAILABLE" != "True" ]; then
    error "MPS backend not available"
    info "Possible issues:"
    info "  1. Not running on Apple Silicon"
    info "  2. PyTorch version too old (need 2.1+)"
    info "  3. macOS version too old (need 12.3+)"
    exit 1
fi

success "MPS backend available"

# Test MPS computation
info "Testing MPS computation..."
MPS_TEST=$(python3 -c "import torch; t = torch.randn(100, 100, device='mps'); result = torch.matmul(t, t); print('success')" 2>&1)

if [ "$MPS_TEST" = "success" ]; then
    success "MPS computation test passed"
else
    error "MPS computation test failed:"
    info "$MPS_TEST"
    exit 1
fi

echo ""

# Step 3.7: Run M4-specific validation
info "Step 6: Running M4-specific validation..."

if [ -f "scripts/validate_environment.py" ]; then
    if python3 scripts/validate_environment.py --env=m4; then
        success "Environment validation passed"
    else
        error "Environment validation failed"
        exit 1
    fi
else
    warning "Skipping environment validation (script not found)"
fi

if [ -f "configs/training_config_m4.yaml" ]; then
    if [ -f "scripts/validate_training_config.py" ]; then
        if python3 scripts/validate_training_config.py --config configs/training_config_m4.yaml; then
            success "Training config validation passed"
        else
            error "Training config validation failed"
            exit 1
        fi
    else
        warning "Skipping config validation (script not found)"
    fi
else
    warning "Training config not found: configs/training_config_m4.yaml"
fi

if [ -f "scripts/check_storage.py" ]; then
    if python3 scripts/check_storage.py; then
        success "Storage check passed"
    else
        error "Storage check failed"
        exit 1
    fi
else
    warning "Skipping storage check (script not found)"
fi

# Create data symlinks
info "Setting up data paths..."

mkdir -p data/processed

if [ ! -f "data/processed/train.jsonl" ]; then
    if [ -f "data/synthetic/final_train.jsonl" ]; then
        ln -sf "$(pwd)/data/synthetic/final_train.jsonl" data/processed/train.jsonl
        success "Created symlink: data/processed/train.jsonl"
    else
        warning "data/synthetic/final_train.jsonl not found"
    fi
else
    success "Training data path exists"
fi

if [ ! -f "data/processed/val.jsonl" ]; then
    if [ -f "data/synthetic/final_validation.jsonl" ]; then
        ln -sf "$(pwd)/data/synthetic/final_validation.jsonl" data/processed/val.jsonl
        success "Created symlink: data/processed/val.jsonl"
    else
        warning "data/synthetic/final_validation.jsonl not found"
    fi
else
    success "Validation data path exists"
fi

echo ""

# Step 3.8: Display M4 training warnings and completion
echo "============================================================"
success "[SETUP-M4-COMPLETE] Ready for local training"
echo "============================================================"
echo ""

warning "WARNING: M4 training takes 12-18 hours (vs 3-4 hours on H100)"
warning "Close unnecessary applications to free unified memory"
echo ""

echo "Environment Details:"
echo "  Virtual Environment: .venv-local"
echo "  Python: $PYTHON_VERSION"
echo "  PyTorch: $PYTORCH_VERSION"
echo "  Chip: $CHIP_TYPE"
echo "  Unified Memory: ${MEMORY_GB}GB"
echo "  MPS Backend: Available ✓"
echo ""

echo "Training Configuration:"
echo "  Config file: configs/training_config_m4.yaml"
echo "  Estimated time: 12-18 hours"
echo "  Batch size: 1 (vs 4-8 on H100)"
echo "  Sequence length: 2048 tokens (vs 4096 on H100)"
echo "  Checkpoints: every 250 steps (vs 500 on H100)"
echo ""

echo "Memory Optimization Tips:"
echo "  - Close all browsers, IDEs, and unnecessary apps"
echo "  - Monitor Activity Monitor for memory pressure"
echo "  - Train overnight or during off-hours"
echo "  - Use wandb for remote monitoring"
echo ""

echo "Next Steps:"
echo "  1. Activate environment: source .venv-local/bin/activate"
echo "  2. Start training: ./train_m4_local.sh"
echo ""

info "[SETUP-M4] Setup completed successfully"
echo ""
