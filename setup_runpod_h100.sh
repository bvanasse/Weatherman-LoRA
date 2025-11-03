#!/bin/bash
#
# Weatherman-LoRA H100 RunPod Environment Setup
# Optimized for RunPod H100 instances with CUDA 12.1+
#
# This script:
# 1. Verifies RunPod H100 environment (CUDA 12.1+, H100 GPU with 80GB VRAM)
# 2. Creates conda environment from environment-remote.yml
# 3. Installs Flash Attention 2 for optimized training
# 4. Runs comprehensive validation checks
# 5. Creates data symlinks
# 6. Configures Weights & Biases authentication
#
# Prerequisites:
# - RunPod H100 instance
# - Conda or Miniconda installed
# - CUDA 12.1+ drivers installed
#
# Usage: ./setup_runpod_h100.sh

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
echo "Weatherman-LoRA H100 RunPod Environment Setup"
echo "============================================================"
echo ""

info "[SETUP-H100] Starting RunPod H100 environment setup"
echo ""

# Step 1.2: Implement RunPod environment verification
info "Step 1: Verifying RunPod H100 environment..."

# Check conda installation
if ! command -v conda &> /dev/null; then
    error "conda is not installed"
    info "Please install Miniconda:"
    info "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    info "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

CONDA_VERSION=$(conda --version)
success "Found: $CONDA_VERSION"

# Detect CUDA version
if ! command -v nvcc &> /dev/null; then
    error "CUDA toolkit (nvcc) not found in PATH"
    info "CUDA 12.1+ is required for H100"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d'.' -f2)

if [ "$CUDA_MAJOR" -lt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 1 ]); then
    error "CUDA version $CUDA_VERSION is older than required 12.1+"
    exit 1
fi

success "Found: CUDA $CUDA_VERSION"

# Verify H100 GPU
if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi not found - GPU driver not installed"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
if echo "$GPU_INFO" | grep -q "H100"; then
    GPU_VRAM=$(echo "$GPU_INFO" | awk -F',' '{print $2}' | tr -d ' ' | sed 's/MiB//')
    GPU_VRAM_GB=$((GPU_VRAM / 1024))

    if [ "$GPU_VRAM_GB" -ge 80 ]; then
        success "Found: H100 GPU with ${GPU_VRAM_GB}GB VRAM"
    else
        warning "H100 found but VRAM (${GPU_VRAM_GB}GB) is less than expected 80GB"
    fi
else
    error "H100 GPU not detected. Found: $GPU_INFO"
    info "This script is optimized for H100 GPUs"
    exit 1
fi

echo ""

# Step 1.3: Create conda environment
info "Step 2: Setting up conda environment..."

if conda env list | grep -q "weatherman-lora"; then
    warning "Environment 'weatherman-lora' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n weatherman-lora -y
        success "Removed existing environment"
    else
        info "Keeping existing environment. Activating..."
        eval "$(conda shell.bash hook)"
        conda activate weatherman-lora
        info "Environment activated. Skipping environment creation."
        # Jump to validation steps
        SKIP_ENV_CREATION=true
    fi
fi

if [ "$SKIP_ENV_CREATION" != "true" ]; then
    if [ ! -f "environment-remote.yml" ]; then
        error "environment-remote.yml not found"
        exit 1
    fi

    info "Creating conda environment (this may take 10-15 minutes)..."
    conda env create -f environment-remote.yml
    success "Conda environment created: weatherman-lora"

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate weatherman-lora
    success "Activated weatherman-lora environment"
fi

echo ""

# Step 1.4: Install Flash Attention 2
info "Step 3: Installing Flash Attention 2..."

if python -c "import flash_attn" 2>/dev/null; then
    success "Flash Attention 2 already installed"
else
    info "Installing Flash Attention 2 (this may take 5-10 minutes)..."

    if pip install flash-attn --no-build-isolation; then
        success "Flash Attention 2 installed"
    else
        error "Failed to install Flash Attention 2"
        info "Troubleshooting steps:"
        info "  1. Ensure CUDA 12.1+ is installed"
        info "  2. Ensure gcc/g++ compiler is available"
        info "  3. Try: pip install flash-attn --no-build-isolation --no-cache-dir"
        exit 1
    fi
fi

echo ""

# Step 1.5: Run validation scripts
info "Step 4: Running validation checks..."

if [ ! -f "scripts/validate_environment.py" ]; then
    warning "scripts/validate_environment.py not found, skipping environment validation"
else
    info "Validating H100 environment..."
    if python scripts/validate_environment.py --env=h100; then
        success "Environment validation passed"
    else
        error "Environment validation failed"
        exit 1
    fi
fi

if [ ! -f "scripts/check_gpu.py" ]; then
    warning "scripts/check_gpu.py not found, skipping GPU check"
else
    info "Running GPU diagnostics..."
    if python scripts/check_gpu.py; then
        success "GPU check passed"
    else
        error "GPU check failed"
        exit 1
    fi
fi

if [ ! -f "scripts/check_storage.py" ]; then
    warning "scripts/check_storage.py not found, skipping storage check"
else
    info "Checking storage (require 50GB+ free)..."
    if python scripts/check_storage.py; then
        success "Storage check passed"
    else
        error "Storage check failed - insufficient free space"
        exit 1
    fi
fi

if [ ! -f "configs/training_config_h100.yaml" ]; then
    warning "configs/training_config_h100.yaml not found, skipping config validation"
else
    if [ ! -f "scripts/validate_training_config.py" ]; then
        warning "scripts/validate_training_config.py not found, skipping config validation"
    else
        info "Validating training configuration..."
        if python scripts/validate_training_config.py --config configs/training_config_h100.yaml; then
            success "Training config validation passed"
        else
            error "Training config validation failed"
            exit 1
        fi
    fi
fi

echo ""

# Step 1.6: Create data symlinks
info "Step 5: Setting up data paths..."

mkdir -p data/processed

if [ ! -f "data/processed/train.jsonl" ]; then
    if [ -f "data/synthetic/final_train.jsonl" ]; then
        ln -sf "$(pwd)/data/synthetic/final_train.jsonl" data/processed/train.jsonl
        success "Created symlink: data/processed/train.jsonl -> data/synthetic/final_train.jsonl"
    else
        warning "data/synthetic/final_train.jsonl not found"
        info "You will need to create training data before training"
    fi
else
    success "Training data path already exists: data/processed/train.jsonl"
fi

if [ ! -f "data/processed/val.jsonl" ]; then
    if [ -f "data/synthetic/final_validation.jsonl" ]; then
        ln -sf "$(pwd)/data/synthetic/final_validation.jsonl" data/processed/val.jsonl
        success "Created symlink: data/processed/val.jsonl -> data/synthetic/final_validation.jsonl"
    else
        warning "data/synthetic/final_validation.jsonl not found"
    fi
else
    success "Validation data path already exists: data/processed/val.jsonl"
fi

success "Data paths verified"
echo ""

# Step 1.7: Configure Weights & Biases
info "Step 6: Configuring Weights & Biases..."

if [ -z "$WANDB_API_KEY" ]; then
    warning "WANDB_API_KEY environment variable not set"
    info "To enable W&B logging:"
    info "  1. Get your API key from: https://wandb.ai/authorize"
    info "  2. Run: wandb login"
    info "  3. Or set: export WANDB_API_KEY=your_key_here"
else
    info "Verifying W&B authentication..."
    if python -c "import wandb; wandb.login()" 2>/dev/null; then
        success "Weights & Biases authenticated"
    else
        warning "W&B authentication verification failed"
        info "You may need to run: wandb login"
    fi
fi

echo ""

# Step 1.8: Print setup completion summary
echo "============================================================"
success "[SETUP-H100-COMPLETE] Ready for training"
echo "============================================================"
echo ""
echo "Environment Details:"
echo "  Conda Environment: weatherman-lora"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  CUDA Version: $CUDA_VERSION"
echo "  Flash Attention 2: $(python -c 'import flash_attn; print("✓ Installed")' 2>/dev/null || echo "✗ Not installed")"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""
echo "Training Configuration:"
echo "  Estimated training time: 3-4 hours on H100"
echo "  Config file: configs/training_config_h100.yaml"
echo "  Training data: data/processed/train.jsonl"
echo "  Validation data: data/processed/val.jsonl"
echo ""
echo "Next Steps:"
echo "  1. Activate environment: conda activate weatherman-lora"
echo "  2. Start training: ./train_h100_runpod.sh"
echo ""
info "[SETUP-H100] Setup completed successfully"
echo ""
