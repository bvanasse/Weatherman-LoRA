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

# Determine Python command - prefer conda environment's Python
# This ensures we use the same Python that pip installs to
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    # Use conda's python if in conda environment
    PYTHON_CMD=$(which python)
    if [ -z "$PYTHON_CMD" ]; then
        PYTHON_CMD=$(which python3)
    fi
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    error "Python not found. Please install Python 3.11+"
    exit 1
fi

# Verify Python can be executed
if ! $PYTHON_CMD --version &>/dev/null; then
    error "Python command '$PYTHON_CMD' is not executable"
    exit 1
fi

info "Using Python: $PYTHON_CMD"
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
PYTHON_PATH=$(which $PYTHON_CMD 2>/dev/null || echo "unknown")
success "Python version: $PYTHON_VERSION"
info "Python path: $PYTHON_PATH"

# Determine pip command to match Python - ALWAYS use python -m pip
# This ensures we install to the correct Python environment
PIP_CMD="$PYTHON_CMD -m pip"

# Verify pip works - check both command and import
PIP_AVAILABLE=false
if $PIP_CMD --version &>/dev/null 2>&1; then
    PIP_AVAILABLE=true
elif $PYTHON_CMD -c "import pip" 2>/dev/null; then
    # pip module is importable even if command fails; proceed quietly with python -m pip
    PIP_AVAILABLE=true
fi

# Install pip if not available
if [ "$PIP_AVAILABLE" = false ]; then
    warning "pip not found for Python $PYTHON_CMD"
    info "Installing pip..."
    
    # Try to install pip using ensurepip
    if $PYTHON_CMD -m ensurepip --upgrade 2>&1 | grep -q "Requirement already satisfied\|Successfully installed"; then
        # ensurepip reports pip is installed/available
        success "pip should be available (ensurepip reports installed)"
        # Verify it works now
        if $PIP_CMD --version &>/dev/null 2>&1 || $PYTHON_CMD -c "import pip" 2>/dev/null; then
            PIP_AVAILABLE=true
        fi
    fi
    
    # If still not available, try conda (but only if really needed)
    if [ "$PIP_AVAILABLE" = false ] && command -v conda &> /dev/null; then
        warning "pip still not available, trying conda..."
        warning "Note: This may require accepting conda Terms of Service"
        conda install -y pip 2>/dev/null && PIP_AVAILABLE=true || {
            error "Failed to install pip via conda"
            error "You may need to accept conda TOS first:"
            error "  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main"
            error "Then run: conda install pip"
            exit 1
        }
    fi
    
    # Final check
    if [ "$PIP_AVAILABLE" = false ]; then
        error "pip installation failed. Please install manually:"
        error "  conda install pip"
        error "  OR: $PYTHON_CMD -m ensurepip --upgrade"
        exit 1
    fi
fi

PIP_VERSION=$($PIP_CMD --version 2>&1)
info "Using pip: $PIP_CMD"
success "Pip version: $PIP_VERSION"

# Check if Axolotl is installed
info "Checking Axolotl installation..."

if $PYTHON_CMD -c "import axolotl" 2>/dev/null; then
    AXOLOTL_VERSION=$($PYTHON_CMD -c "import axolotl; print(axolotl.__version__)" 2>/dev/null || echo "unknown")
    success "Axolotl installed (version: $AXOLOTL_VERSION)"
else
    warning "Axolotl not found. Installing..."
    
    # Install build dependencies first
    info "Installing build dependencies..."
    $PIP_CMD install -U packaging==23.2 setuptools==75.8.0 wheel ninja

    # Step 1: Install PyTorch first (required for flash-attn compilation)
    # Note: Axolotl will install its own torch version, so we install compatible versions
    info "Installing PyTorch with CUDA 12.1 support..."
    info "Note: Axolotl may install torch 2.6.0, so we'll let it handle torch installation"
    # Don't install torch here - let Axolotl install the version it needs
    # $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # PyTorch will be installed by Axolotl, so skip verification here
    info "PyTorch will be installed by Axolotl with compatible versions"

    # Step 2: Install Axolotl with DeepSpeed (this installs torch and all dependencies)
    info "Installing Axolotl with DeepSpeed (this may take several minutes)..."
    info "This will also install compatible versions of torch, transformers, etc."
    
    # Install Axolotl - it will handle all dependencies including torch
    if $PIP_CMD install "axolotl[deepspeed]"; then
        success "Axolotl installation completed"
    else
        warning "Installation had warnings, but continuing..."
    fi

    # Verify installation
    info "Verifying Axolotl installation..."
    if $PYTHON_CMD -c "import axolotl" 2>/dev/null; then
        AXOLOTL_VERSION=$($PYTHON_CMD -c "import axolotl; print(axolotl.__version__)" 2>/dev/null || echo "unknown")
        success "Axolotl $AXOLOTL_VERSION installed"
    else
        error "Failed to import Axolotl after installation"
        error "Try manually: $PIP_CMD install axolotl[deepspeed]"
        exit 1
    fi
    
    # Verify PyTorch is now available
    info "Verifying PyTorch installation..."
    if $PYTHON_CMD -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
        success "PyTorch installed (version: $TORCH_VERSION)"
        
        # Check CUDA availability (non-fatal)
        if $PYTHON_CMD -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
            GPU_NAME=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
            success "CUDA is available: $GPU_NAME"
        else
            warning "CUDA check failed, but continuing (may be a detection issue)"
            info "Training will proceed and verify CUDA at runtime."
        fi
    else
        warning "PyTorch import failed, but Axolotl is installed"
        info "Training may install PyTorch at runtime if needed"
    fi
    
    # Step 3: Try to install flash-attn if torch is available
    if $PYTHON_CMD -c "import torch" 2>/dev/null; then
        info "Installing Flash Attention (this may take 5-10 minutes)..."
        $PIP_CMD install flash-attn==2.8.2 --no-build-isolation || {
            warning "Flash Attention installation failed, continuing without it..."
            warning "Training will be slower but should still work"
        }
    else
        warning "Skipping Flash Attention installation (torch not available)"
        info "Flash Attention can be installed later if needed"
    fi
fi

echo ""

# Ensure compatible package versions
info "Checking package compatibility..."

# Check and upgrade PEFT if needed (common compatibility issue)
PEFT_VERSION=$($PIP_CMD show peft 2>/dev/null | grep "Version:" | cut -d " " -f 2 || echo "not installed")
if [ -n "$PEFT_VERSION" ] && [ "$PEFT_VERSION" != "not installed" ]; then
    info "PEFT version: $PEFT_VERSION"
    # Upgrade to latest PEFT for compatibility
    $PIP_CMD install --upgrade peft
    success "PEFT upgraded to latest version"
else
    info "Installing PEFT..."
    $PIP_CMD install peft
    success "PEFT installed"
fi

# Ensure accelerate is up to date
info "Checking accelerate..."
$PIP_CMD install --upgrade accelerate
ACCEL_VERSION=$($PIP_CMD show accelerate 2>/dev/null | grep "Version:" | cut -d " " -f 2 || echo "unknown")
success "Accelerate version: $ACCEL_VERSION"

# Ensure bitsandbytes is installed for QLoRA
info "Checking bitsandbytes..."
if ! $PYTHON_CMD -c "import bitsandbytes" 2>/dev/null; then
    warning "bitsandbytes not found, installing..."
    $PIP_CMD install bitsandbytes
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

# Try to check via PyTorch (if available)
if $PYTHON_CMD -c "import torch" 2>/dev/null; then
    if $PYTHON_CMD -c "import torch; assert torch.cuda.is_available(); print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
        GPU_NAME=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown GPU")
        success "PyTorch CUDA check passed: $GPU_NAME"
    else
        warning "PyTorch CUDA check failed (may be module cache issue)"
        info "GPU hardware confirmed via nvidia-smi. Training will use fresh Python process."
    fi
else
    info "PyTorch not yet imported (will be loaded during training)"
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
