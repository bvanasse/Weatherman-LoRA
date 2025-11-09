#!/bin/bash
#
# Weatherman-LoRA Axolotl Training Script for H100
# Launches LoRA training using Axolotl framework on RunPod H100 instance
#
# This script:
# 1. Verifies environment and data files
# 2. Installs/updates Axolotl and dependencies
# 3. Ensures compatible package versions
# 4. Launches training with automatic checkpoint resumption
# 5. Provides monitoring instructions
#
# Usage: ./train_with_axolotl_h100.sh

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    echo -e "${BLUE}ℹ${NC} $1"
}

echo "============================================================"
echo "Weatherman-LoRA Axolotl Training (H100)"
echo "============================================================"
echo ""

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "weatherman-lora" ]; then
    warning "weatherman-lora conda environment not activated"
    info "Attempting to activate..."

    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate weatherman-lora
        success "Activated weatherman-lora environment"
    else
        error "conda not found. Please activate environment manually:"
        info "  conda activate weatherman-lora"
        exit 1
    fi
fi

# Verify Axolotl config exists
CONFIG_FILE="axolotl_config_h100.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    error "Axolotl config not found: $CONFIG_FILE"
    exit 1
fi

success "Found Axolotl config: $CONFIG_FILE"

# Verify training data exists
TRAIN_DATA="data/synthetic/final_train_diverse.jsonl"
VAL_DATA="data/synthetic/final_validation_diverse.jsonl"

if [ ! -f "$TRAIN_DATA" ]; then
    error "Training data not found: $TRAIN_DATA"
    info "Please run data generation scripts first"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    error "Validation data not found: $VAL_DATA"
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$TRAIN_DATA" 2>/dev/null || echo "0")
VAL_COUNT=$(wc -l < "$VAL_DATA" 2>/dev/null || echo "0")

success "Training data verified ($TRAIN_COUNT examples)"
success "Validation data verified ($VAL_COUNT examples)"
echo ""

# Determine Python command (python3 or python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    error "Python not found. Please install Python 3.11+"
    exit 1
fi

info "Using Python: $PYTHON_CMD"
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
success "Python version: $PYTHON_VERSION"

# Check if Axolotl is installed
info "Checking Axolotl installation..."

if $PYTHON_CMD -c "import axolotl" 2>/dev/null; then
    AXOLOTL_VERSION=$($PYTHON_CMD -c "import axolotl; print(axolotl.__version__)" 2>/dev/null || echo "unknown")
    success "Axolotl installed (version: $AXOLOTL_VERSION)"
else
    warning "Axolotl not found. Installing..."
    
    # Install build dependencies first
    info "Installing build dependencies..."
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja

    # Step 1: Install PyTorch first (required for flash-attn compilation)
    info "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Verify PyTorch installation
    info "Verifying PyTorch installation..."
    if $PYTHON_CMD -c "import torch; print('PyTorch', torch.__version__)" 2>&1; then
        TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
        success "PyTorch installed (version: $TORCH_VERSION)"
        
        # Check CUDA availability (non-fatal)
        info "Checking CUDA availability..."
        if $PYTHON_CMD -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
            GPU_NAME=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
            success "CUDA is available: $GPU_NAME"
        else
            warning "CUDA check failed, but continuing (may be a detection issue)"
            info "PyTorch is installed. Training will proceed and verify CUDA at runtime."
        fi
    else
        error "Failed to verify PyTorch installation"
        error "Try manually: $PYTHON_CMD -c 'import torch; print(torch.__version__)'"
        exit 1
    fi

    # Step 2: Install flash-attn (now that torch is available)
    info "Installing Flash Attention (this may take 5-10 minutes)..."
    pip install flash-attn==2.8.2 --no-build-isolation || {
        warning "Flash Attention installation failed, continuing without it..."
        warning "Training will be slower but should still work"
    }

    # Step 3: Install Axolotl with DeepSpeed
    info "Installing Axolotl with DeepSpeed (this may take several minutes)..."
    
    # Install without flash-attn extra since we handle it separately
    pip install "axolotl[deepspeed]" || {
        warning "Failed to install with deepspeed extras, trying basic install..."
        pip install axolotl
    }

    # Verify installation
    if $PYTHON_CMD -c "import axolotl" 2>/dev/null; then
        AXOLOTL_VERSION=$($PYTHON_CMD -c "import axolotl; print(axolotl.__version__)" 2>/dev/null || echo "unknown")
        success "Axolotl $AXOLOTL_VERSION installed"
    else
        error "Failed to install Axolotl"
        error "Try manually: pip install axolotl[deepspeed]"
        exit 1
    fi
fi

echo ""

# Ensure compatible package versions
info "Checking package compatibility..."

# Check and upgrade PEFT if needed (common compatibility issue)
PEFT_VERSION=$(pip show peft 2>/dev/null | grep "Version:" | cut -d " " -f 2 || echo "not installed")
if [ -n "$PEFT_VERSION" ] && [ "$PEFT_VERSION" != "not installed" ]; then
    info "PEFT version: $PEFT_VERSION"
    # Upgrade to latest PEFT for compatibility
    pip install --upgrade peft
    success "PEFT upgraded to latest version"
else
    info "Installing PEFT..."
    pip install peft
    success "PEFT installed"
fi

# Ensure accelerate is up to date
info "Checking accelerate..."
pip install --upgrade accelerate
ACCEL_VERSION=$(pip show accelerate 2>/dev/null | grep "Version:" | cut -d " " -f 2 || echo "unknown")
success "Accelerate version: $ACCEL_VERSION"

# Ensure bitsandbytes is installed for QLoRA
info "Checking bitsandbytes..."
if ! $PYTHON_CMD -c "import bitsandbytes" 2>/dev/null; then
    warning "bitsandbytes not found, installing..."
    pip install bitsandbytes
    success "bitsandbytes installed"
else
    success "bitsandbytes already installed"
fi

echo ""

# Check GPU availability
info "Checking GPU availability..."

# Use nvidia-smi first as it's independent of Python
if command -v nvidia-smi &> /dev/null; then
    info "GPU hardware detected via nvidia-smi:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    success "NVIDIA GPU hardware is available"
else
    warning "nvidia-smi not found, but continuing anyway..."
fi

# Try to check via PyTorch
if $PYTHON_CMD -c "import torch; assert torch.cuda.is_available(); print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
    GPU_NAME=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown GPU")
    success "PyTorch CUDA check passed: $GPU_NAME"
else
    warning "PyTorch CUDA check failed (may be module cache issue)"
    info "GPU hardware confirmed via nvidia-smi. Training will use fresh Python process."
fi

echo ""

# Check if there's a checkpoint to resume from
CHECKPOINT_DIR="adapters/weatherman-lora-axolotl-h100"

if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR/checkpoint-* 2>/dev/null)" ]; then
    warning "Found existing checkpoints in $CHECKPOINT_DIR"
    LATEST_CHECKPOINT=$(ls -td $CHECKPOINT_DIR/checkpoint-* 2>/dev/null | head -1)
    info "Latest checkpoint: $LATEST_CHECKPOINT"
    info "Axolotl will automatically resume from the latest checkpoint"
    echo ""
fi

# Launch training
info "============================================================"
info "Launching Axolotl Training"
info "============================================================"
echo ""
info "Configuration: $CONFIG_FILE"
info "Base Model: mistralai/Mistral-7B-Instruct-v0.3"
info "Training Examples: $TRAIN_COUNT"
info "Validation Examples: $VAL_COUNT"
info "Estimated Duration: 3-4 hours"
info "Output Directory: $CHECKPOINT_DIR"
echo ""

# Create logs directory
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/axolotl_training_h100_${TIMESTAMP}.log"

info "Log file: $LOG_FILE"
info "Starting training..."
echo ""

# Use axolotl CLI directly (simpler than accelerate launch)
# This handles accelerate internally
if command -v axolotl &> /dev/null; then
    info "Using axolotl CLI command..."
    axolotl train "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
else
    # Fallback to accelerate launch with explicit Python
    info "Using accelerate launch (axolotl CLI not found)..."
    info "Command: accelerate launch -m axolotl.cli.train $CONFIG_FILE"
    accelerate launch -m axolotl.cli.train "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
fi

# Check if training completed successfully
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================"
    success "Training Completed Successfully!"
    echo "============================================================"
    echo ""
    echo "Adapter Location: $CHECKPOINT_DIR"
    echo ""
    echo "Next Steps:"
    echo "  1. Merge adapter with base model (optional):"
    echo "     axolotl merge-lora $CONFIG_FILE --lora-model-dir=$CHECKPOINT_DIR"
    echo ""
    echo "  2. Test the model:"
    echo "     axolotl inference $CONFIG_FILE --lora-model-dir=$CHECKPOINT_DIR --prompt=\"What's the weather in Boston?\""
    echo ""
    echo "  3. Deploy with Ollama or AnythingLLM (see docs/DEPLOYMENT.md)"
    echo ""
else
    echo ""
    error "Training failed with exit code: $TRAIN_EXIT_CODE"
    error "Check the log file for errors:"
    info "  $LOG_FILE"
    echo ""
    info "Common issues:"
    info "  - Out of memory: Reduce micro_batch_size or sequence_len in config"
    info "  - Dataset format: Verify data/synthetic/final_train_diverse.jsonl format"
    info "  - Package conflicts: Try 'pip install --upgrade peft accelerate bitsandbytes'"
    exit 1
fi
