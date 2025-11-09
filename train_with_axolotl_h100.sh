#!/bin/bash
#
# Weatherman-LoRA Axolotl Training Script for H100
# Launches LoRA training using Axolotl framework on RunPod H100 instance
#
# This script:
# 1. Verifies environment and data files
# 2. Installs Axolotl if not present
# 3. Launches training with automatic checkpoint resumption
# 4. Provides monitoring instructions
#
# Usage: ./train_with_axolotl_h100.sh

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
if [ ! -f "axolotl_config_h100.yaml" ]; then
    error "Axolotl config not found: axolotl_config_h100.yaml"
    exit 1
fi

success "Found Axolotl config: axolotl_config_h100.yaml"

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

if python -c "import axolotl" 2>/dev/null; then
    AXOLOTL_VERSION=$(python -c "import axolotl; print(axolotl.__version__)" 2>/dev/null || echo "unknown")
    success "Axolotl installed (version: $AXOLOTL_VERSION)"
else
    warning "Axolotl not found. Installing..."

    # Clean up any existing broken installations
    if python -c "import torch" 2>/dev/null; then
        warning "Removing existing PyTorch installation..."
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    fi

    # Install build dependencies first
    info "Installing build dependencies..."
    pip install packaging ninja

    # Step 1: Install PyTorch first (required for flash-attn compilation)
    info "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Check if pip reports successful installation
    if pip show torch >/dev/null 2>&1; then
        TORCH_VERSION=$(pip show torch | grep "Version:" | cut -d " " -f 2)
        success "PyTorch $TORCH_VERSION installed (verified via pip)"

        # Verify import works (use fresh Python process, no bytecode cache)
        info "Verifying PyTorch can be imported..."
        if python3 -B -c "import sys; import torch; print(f'Import successful: torch {torch.__version__}')" 2>&1; then
            success "PyTorch import successful"

            # Check CUDA availability (non-fatal)
            if python3 -B -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
                GPU_NAME=$(python3 -B -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
                success "CUDA is available: $GPU_NAME"
            else
                warning "CUDA check failed, but continuing (will verify during flash-attn build)..."
            fi
        else
            warning "PyTorch import had issues, but pip confirms it's installed - continuing..."
        fi
    else
        error "Failed to install PyTorch"
        exit 1
    fi

    # Step 2: Install flash-attn (now that torch is available)
    info "Installing Flash Attention (this may take 5-10 minutes)..."
    pip install flash-attn==2.8.2 --no-build-isolation

    if python -c "import flash_attn" 2>/dev/null; then
        success "Flash Attention installed successfully"
    else
        warning "Flash Attention installation may have failed, continuing anyway..."
    fi

    # Step 3: Install Axolotl with DeepSpeed (flash-attn already installed)
    info "Installing Axolotl with DeepSpeed..."
    pip install "axolotl[deepspeed]"

    # Check final torch version
    TORCH_VERSION=$(pip show torch | grep "Version:" | cut -d " " -f 2 2>/dev/null || echo "unknown")
    info "Final PyTorch version: $TORCH_VERSION"

    # Note: Axolotl may upgrade torch (e.g., 2.5.1 -> 2.6.0)
    # This is expected and we don't downgrade it as it would break Axolotl
    # Any version warnings from pip can be ignored - the packages are compatible

    # Verify installation
    if python3 -B -c "import axolotl; print(f'Axolotl version: {axolotl.__version__}')" 2>&1; then
        success "Axolotl installed successfully"
    else
        error "Failed to import Axolotl"
        info "Axolotl import failed. This might be due to missing dependencies."
        info "Try manual installation:"
        info "  pip install flash-attn --no-build-isolation"
        info "  pip install axolotl[deepspeed]"
        exit 1
    fi
fi

echo ""

# Check GPU availability
info "Checking GPU availability..."

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEMORY=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')")
    success "GPU: $GPU_NAME ($GPU_MEMORY)"
else
    error "CUDA GPU not available"
    info "This script requires a CUDA-capable GPU (H100)"
    exit 1
fi

echo ""

# Check if there's a checkpoint to resume from
CHECKPOINT_DIR="adapters/weatherman-lora-axolotl-h100"

if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR/checkpoint-* 2>/dev/null)" ]; then
    warning "Found existing checkpoints in $CHECKPOINT_DIR"
    info "Axolotl will automatically resume from the latest checkpoint"
    echo ""
fi

# Launch training
info "============================================================"
info "Launching Axolotl Training"
info "============================================================"
echo ""
info "Configuration: axolotl_config_h100.yaml"
info "Base Model: mistralai/Mistral-7B-Instruct-v0.3"
info "Training Examples: 14,399"
info "Validation Examples: 1,601"
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

# Launch Axolotl with accelerate
# The 2>&1 | tee ensures we see output and save to log
accelerate launch -m axolotl.cli.train axolotl_config_h100.yaml 2>&1 | tee "$LOG_FILE"

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
    echo "     python -m axolotl.cli.merge axolotl_config_h100.yaml --lora-model-dir=$CHECKPOINT_DIR"
    echo ""
    echo "  2. Test the model:"
    echo "     python -m axolotl.cli.inference axolotl_config_h100.yaml --lora-model-dir=$CHECKPOINT_DIR"
    echo ""
    echo "  3. Deploy with Ollama or AnythingLLM (see docs/DEPLOYMENT.md)"
    echo ""
else
    echo ""
    error "Training failed. Check the log file for errors:"
    info "  $LOG_FILE"
    exit 1
fi
