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
set -o pipefail

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

# Fast-path environment settings (caches and logging)
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export WANDB_DISABLED=${WANDB_DISABLED:-true}
export BITSANDBYTES_NOWELCOME=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:true}
ulimit -n 4096 || true

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

# Get pip version (non-fatal)
PIP_VERSION=$($PIP_CMD --version 2>&1 || echo "unknown")
if echo "$PIP_VERSION" | grep -q "No module named pip.__main__"; then
    warning "pip is installed but broken (missing __main__)"
    info "Reinstalling pip..."
    $PYTHON_CMD -m ensurepip --upgrade 2>/dev/null || true
    conda install -y pip 2>/dev/null || {
        error "Could not fix pip. Please manually reinstall:"
        error "  conda install -c conda-forge pip"
        exit 1
    }
    PIP_VERSION=$($PIP_CMD --version 2>&1 || echo "reinstalled")
fi

info "Using pip: $PIP_CMD"
success "Pip version: $PIP_VERSION"
echo ""

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

    # Step 2: Install Axolotl with DeepSpeed and Flash-Attn (installs torch and deps)
    info "Installing Axolotl with DeepSpeed (this may take several minutes)..."
    info "This will also install compatible versions of torch, transformers, etc."
    
    # Install Axolotl - it will handle all dependencies including torch
    if $PIP_CMD install --no-build-isolation "axolotl[flash-attn,deepspeed]"; then
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
    
    # Verify PyTorch is now available (non-fatal)
    info "Verifying PyTorch installation..."
    if $PYTHON_CMD -c "import torch; import sys; sys.stdout.write(getattr(torch, '__version__', 'unknown'))" 2>/dev/null; then
        TORCH_VERSION=$($PYTHON_CMD -c "import torch; import sys; sys.stdout.write(getattr(torch, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        success "PyTorch installed (version: $TORCH_VERSION)"
    else
        warning "PyTorch import failed, continuing"
    fi
fi

echo ""

## Skip manual compatibility pinning; Axolotl managed dependencies are sufficient

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

# Launch training in persistent session
info "============================================================"
info "Launching Axolotl Training in Persistent Session"
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
echo ""

# Determine training command
if command -v axolotl &> /dev/null; then
    TRAIN_CMD="axolotl train $CONFIG_FILE"
    info "Using axolotl CLI command"
else
    TRAIN_CMD="accelerate launch -m axolotl.cli.train $CONFIG_FILE"
    info "Using accelerate launch (axolotl CLI not found)"
fi

# Launch in persistent session (tmux or nohup)
TMUX_SESSION="weatherman-axolotl-training"

# Try to use tmux first for best experience
if command -v tmux &> /dev/null; then
    # Test if tmux works (check for library issues)
    if tmux -V &> /dev/null; then
        info "Using tmux for session persistence..."

        # Check if tmux session already exists
        if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            warning "Tmux session '$TMUX_SESSION' already exists"
            info "To view existing session: tmux attach -t $TMUX_SESSION"
            info "To kill existing session: tmux kill-session -t $TMUX_SESSION"
            echo ""
            read -p "Kill existing session and create new one? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                tmux kill-session -t "$TMUX_SESSION"
                success "Killed existing session"
            else
                error "Cannot proceed with existing session active"
                exit 1
            fi
        fi

        # Create tmux session and launch training
        tmux new-session -d -s "$TMUX_SESSION" bash
        tmux send-keys -t "$TMUX_SESSION" "eval \"\$(conda shell.bash hook)\"" C-m
        tmux send-keys -t "$TMUX_SESSION" "conda activate weatherman-lora" C-m
        sleep 2
        tmux send-keys -t "$TMUX_SESSION" "cd $(pwd)" C-m
        tmux send-keys -t "$TMUX_SESSION" "$TRAIN_CMD 2>&1 | tee $LOG_FILE" C-m

        success "Training launched in tmux session: $TMUX_SESSION"
        USING_TMUX=true
    else
        warning "tmux has library compatibility issues, trying conda installation..."

        # Try to install tmux via conda
        if conda install -y -c conda-forge tmux &> /dev/null; then
            success "Installed tmux via conda"

            # Test if conda-installed tmux works
            if tmux -V &> /dev/null; then
                info "Conda tmux works, launching training..."

                tmux new-session -d -s "$TMUX_SESSION" bash
                tmux send-keys -t "$TMUX_SESSION" "eval \"\$(conda shell.bash hook)\"" C-m
                tmux send-keys -t "$TMUX_SESSION" "conda activate weatherman-lora" C-m
                sleep 2
                tmux send-keys -t "$TMUX_SESSION" "cd $(pwd)" C-m
                tmux send-keys -t "$TMUX_SESSION" "$TRAIN_CMD 2>&1 | tee $LOG_FILE" C-m

                success "Training launched in tmux session: $TMUX_SESSION"
                USING_TMUX=true
            else
                warning "Conda tmux still has issues, falling back to nohup..."
                USING_TMUX=false
            fi
        else
            warning "Could not install tmux via conda, falling back to nohup..."
            USING_TMUX=false
        fi
    fi
else
    warning "tmux not found, using nohup for background execution..."
    USING_TMUX=false
fi

# Fallback to nohup if tmux unavailable
if [ "$USING_TMUX" != "true" ]; then
    info "Using nohup for persistent background execution..."

    # Create wrapper script to activate conda and run training
    WRAPPER_SCRIPT="/tmp/weatherman_axolotl_training.sh"
    cat > "$WRAPPER_SCRIPT" <<WRAPPER_EOF
#!/bin/bash
eval "\$(conda shell.bash hook)"
conda activate weatherman-lora
cd $(pwd)
$TRAIN_CMD
WRAPPER_EOF
    chmod +x "$WRAPPER_SCRIPT"

    # Launch with nohup
    nohup bash "$WRAPPER_SCRIPT" > "$LOG_FILE" 2>&1 &
    TRAINING_PID=$!

    # Save PID for monitoring
    echo $TRAINING_PID > /tmp/weatherman_axolotl_training.pid

    success "Training launched with nohup (PID: $TRAINING_PID)"
fi

echo ""
info "Waiting for training to initialize (30 seconds)..."
sleep 30

# Check for immediate errors
if [ -f "$LOG_FILE" ]; then
    ERROR_COUNT=$(grep -i -E "ERROR|RuntimeError|CUDA out of memory|OutOfMemoryError|Traceback" "$LOG_FILE" | wc -l)

    if [ "$ERROR_COUNT" -gt 0 ]; then
        error "Detected $ERROR_COUNT error(s) in initial log output"
        info "Recent errors:"
        grep -i -E "ERROR|RuntimeError|CUDA out of memory|OutOfMemoryError|Traceback" "$LOG_FILE" | tail -10
        echo ""
        error "Training appears to have encountered errors during initialization"
        info "Full log: tail -f $LOG_FILE"
        if [ "$USING_TMUX" = "true" ]; then
            info "Attach to session: tmux attach -t $TMUX_SESSION"
        fi
        exit 1
    fi

    # Check if training started
    if grep -q -E "Starting training|Epoch|Step|Loading" "$LOG_FILE"; then
        success "Training initialized successfully"
    else
        warning "Training may still be initializing. Monitor the log file."
    fi
else
    warning "Log file not created yet. Training may still be starting."
fi

echo ""
echo "============================================================"
success "Training Session Active"
echo "============================================================"
echo ""

if [ "$USING_TMUX" = "true" ]; then
    echo "Session Type: tmux"
    echo "Tmux Session: $TMUX_SESSION"
    echo ""
    echo "Monitoring Commands:"
    echo "  Attach to session:     tmux attach -t $TMUX_SESSION"
    echo "  Detach from session:   Ctrl+B, then D"
    echo "  View log file:         tail -f $LOG_FILE"
    echo "  Check GPU usage:       nvidia-smi"
    echo "  List sessions:         tmux ls"
    echo ""
    echo "To Stop Training:"
    echo "  Kill session:          tmux kill-session -t $TMUX_SESSION"
else
    echo "Session Type: nohup (background process)"
    if [ -f "/tmp/weatherman_axolotl_training.pid" ]; then
        echo "Process PID: $(cat /tmp/weatherman_axolotl_training.pid)"
    fi
    echo ""
    echo "Monitoring Commands:"
    echo "  View log file:         tail -f $LOG_FILE"
    echo "  Check process:         ps -p \$(cat /tmp/weatherman_axolotl_training.pid)"
    echo "  Check GPU usage:       nvidia-smi"
    echo ""
    echo "To Stop Training:"
    echo "  Kill process:          kill \$(cat /tmp/weatherman_axolotl_training.pid)"
fi

echo ""
echo "Output Location: $CHECKPOINT_DIR"
echo "Estimated Duration: 3-4 hours"
echo ""
info "Training will continue even if you disconnect from SSH"
info "You can safely close this terminal"
echo ""
echo "After Training Completes:"
echo "  1. Adapter saved to: $CHECKPOINT_DIR/"
echo "  2. See deployment guide: docs/M4_DEPLOYMENT.md"
echo ""
