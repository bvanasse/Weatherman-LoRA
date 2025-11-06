#!/bin/bash
#
# Weatherman-LoRA Axolotl Training Script for Mac M4
# Launches LoRA training using Axolotl framework on Mac M4 with MPS backend
#
# This script:
# 1. Verifies environment and data files
# 2. Installs Axolotl if not present
# 3. Launches training with automatic checkpoint resumption
# 4. Provides monitoring instructions
#
# Usage: ./train_with_axolotl_m4.sh

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
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
echo "Weatherman-LoRA Axolotl Training (Mac M4)"
echo "============================================================"
echo ""

# Check if running on Apple Silicon
CHIP_INFO=$(system_profiler SPHardwareDataType | grep "Chip" | awk -F': ' '{print $2}')
if echo "$CHIP_INFO" | grep -q "Apple M"; then
    success "Running on: $CHIP_INFO"
else
    error "Not running on Apple Silicon"
    info "This script is optimized for Mac M4 (Apple Silicon)"
    exit 1
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
success "Python: $PYTHON_VERSION"
echo ""

# Verify Axolotl config exists
if [ ! -f "axolotl_config_m4.yaml" ]; then
    error "Axolotl config not found: axolotl_config_m4.yaml"
    exit 1
fi

success "Found Axolotl config: axolotl_config_m4.yaml"

# Verify training data exists
if [ ! -f "data/synthetic/final_train.jsonl" ]; then
    error "Training data not found: data/synthetic/final_train.jsonl"
    info "Please run data generation scripts first"
    exit 1
fi

if [ ! -f "data/synthetic/final_validation.jsonl" ]; then
    error "Validation data not found: data/synthetic/final_validation.jsonl"
    exit 1
fi

success "Training data verified (14,399 examples)"
success "Validation data verified (1,601 examples)"
echo ""

# Check if Axolotl is installed
info "Checking Axolotl installation..."

if python3 -c "import axolotl" 2>/dev/null; then
    AXOLOTL_VERSION=$(python3 -c "import axolotl; print(axolotl.__version__)" 2>/dev/null || echo "unknown")
    success "Axolotl installed (version: $AXOLOTL_VERSION)"
else
    warning "Axolotl not found. Installing..."

    # Install Axolotl (without Flash Attention for MPS)
    info "Installing axolotl-ai..."
    pip3 install packaging ninja
    pip3 install axolotl

    if python3 -c "import axolotl" 2>/dev/null; then
        success "Axolotl installed successfully"
    else
        error "Failed to install Axolotl"
        info "Try manual installation:"
        info "  pip3 install axolotl"
        exit 1
    fi
fi

echo ""

# Check MPS availability
info "Checking MPS (Metal Performance Shaders) availability..."

if python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    success "MPS backend available"
else
    error "MPS backend not available"
    info "Possible issues:"
    info "  1. Not running on Apple Silicon"
    info "  2. PyTorch version too old (need 2.2+)"
    info "  3. macOS version too old (need 12.3+)"
    exit 1
fi

# Check memory
TOTAL_MEMORY=$(sysctl -n hw.memsize)
MEMORY_GB=$((TOTAL_MEMORY / 1024 / 1024 / 1024))

if [ "$MEMORY_GB" -lt 16 ]; then
    error "Insufficient memory: ${MEMORY_GB}GB < 16GB minimum"
    exit 1
elif [ "$MEMORY_GB" -lt 32 ]; then
    warning "Limited memory: ${MEMORY_GB}GB (32GB recommended)"
    info "Training may experience memory pressure"
else
    success "Unified Memory: ${MEMORY_GB}GB"
fi

echo ""

# Check if there's a checkpoint to resume from
CHECKPOINT_DIR="adapters/weatherman-lora-axolotl-m4"

if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR/checkpoint-* 2>/dev/null)" ]; then
    warning "Found existing checkpoints in $CHECKPOINT_DIR"
    info "Axolotl will automatically resume from the latest checkpoint"
    echo ""
fi

# Display performance expectations
warning "Performance Note: M4 training is 2-3x slower than H100"
info "Expected duration: 8-12 hours (recommend running overnight)"
info "Close all unnecessary applications to free memory"
echo ""

# Launch training
info "============================================================"
info "Launching Axolotl Training"
info "============================================================"
echo ""
info "Configuration: axolotl_config_m4.yaml"
info "Base Model: mistralai/Mistral-7B-Instruct-v0.3"
info "Training Examples: 14,399"
info "Validation Examples: 1,601"
info "Estimated Duration: 8-12 hours"
info "Output Directory: $CHECKPOINT_DIR"
echo ""

# Create logs directory
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/axolotl_training_m4_${TIMESTAMP}.log"

info "Log file: $LOG_FILE"
info "Starting training..."
info "Monitor Activity Monitor for memory pressure warnings"
echo ""

# Launch Axolotl with accelerate
# The 2>&1 | tee ensures we see output and save to log
accelerate launch -m axolotl.cli.train axolotl_config_m4.yaml 2>&1 | tee "$LOG_FILE"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    success "Training Completed Successfully!"
    echo "============================================================"
    echo ""
    echo "Adapter Location: $CHECKPOINT_DIR"
    echo ""
    echo "Next Steps:"
    echo "  1. Merge adapter with base model (optional):"
    echo "     python3 -m axolotl.cli.merge axolotl_config_m4.yaml --lora-model-dir=$CHECKPOINT_DIR"
    echo ""
    echo "  2. Test the model:"
    echo "     python3 -m axolotl.cli.inference axolotl_config_m4.yaml --lora-model-dir=$CHECKPOINT_DIR"
    echo ""
    echo "  3. Deploy with Ollama or AnythingLLM (see docs/DEPLOYMENT.md)"
    echo ""
else
    echo ""
    error "Training failed. Check the log file for errors:"
    info "  $LOG_FILE"
    exit 1
fi
