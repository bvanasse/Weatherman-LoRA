#!/bin/bash
#
# Weatherman-LoRA H100 Training Execution Script
# Launches LoRA training on RunPod H100 instance
#
# This script:
# 1. Runs pre-flight validation checks
# 2. Creates timestamped log files
# 3. Implements checkpoint resumption with crash loop detection
# 4. Launches training in tmux session for persistence
# 5. Monitors initial steps for errors
# 6. Provides reconnection instructions
#
# Usage: ./train_h100_runpod.sh [--config path/to/config.yaml]

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

# Parse command line arguments
CONFIG="configs/training_config_h100.yaml"
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            info "Usage: $0 [--config path/to/config.yaml]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Weatherman-LoRA H100 Training Execution"
echo "============================================================"
echo ""

info "[TRAINING-H100] Pre-flight checks starting"
echo ""

# Step 2.1: Pre-flight checks
info "Step 1: Running pre-flight validation..."

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "weatherman-lora" ]; then
    warning "weatherman-lora conda environment not activated"
    info "Attempting to activate..."

    # Try to activate
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

# Verify training config exists
if [ ! -f "$CONFIG" ]; then
    error "Training config not found: $CONFIG"
    info "Please ensure the config file exists or specify correct path with --config"
    exit 1
fi

success "Found training config: $CONFIG"

# Run validation scripts
if [ -f "scripts/validate_environment.py" ]; then
    if python scripts/validate_environment.py --env=h100; then
        success "Environment validation passed"
    else
        error "Environment validation failed"
        exit 1
    fi
else
    warning "Skipping environment validation (script not found)"
fi

if [ -f "scripts/check_gpu.py" ]; then
    if python scripts/check_gpu.py; then
        success "GPU check passed"
    else
        error "GPU check failed"
        exit 1
    fi
else
    warning "Skipping GPU check (script not found)"
fi

if [ -f "scripts/check_storage.py" ]; then
    if python scripts/check_storage.py; then
        success "Storage check passed"
    else
        error "Storage check failed"
        exit 1
    fi
else
    warning "Skipping storage check (script not found)"
fi

if [ -f "scripts/validate_training_config.py" ]; then
    if python scripts/validate_training_config.py --config "$CONFIG"; then
        success "Training config validation passed"
    else
        error "Training config validation failed"
        exit 1
    fi
else
    warning "Skipping config validation (script not found)"
fi

# Verify data files exist
if [ ! -f "data/processed/train.jsonl" ]; then
    error "Training data not found: data/processed/train.jsonl"
    info "Please run data preparation scripts first"
    exit 1
fi

success "Training data verified"
echo ""

# Step 2.2: Create logs directory and timestamped log file
info "Step 2: Setting up logging..."

mkdir -p logs
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

info "[TRAINING-H100] Log file: $LOG_FILE"
success "Log file created: $LOG_FILE"
echo ""

# Step 2.3: Implement checkpoint resumption logic
info "Step 3: Checking for existing checkpoints..."

CHECKPOINT_DIR="adapters/weatherman-lora-h100"
RESUME_METADATA="$CHECKPOINT_DIR/resume_metadata.json"
RESUME_FLAG=""

if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR)" ]; then
    warning "Found existing checkpoints in $CHECKPOINT_DIR"

    if [ -f "$RESUME_METADATA" ]; then
        LAST_STEP=$(python3 -c "import json; print(json.load(open('$RESUME_METADATA')).get('last_step', 0))" 2>/dev/null || echo "0")
        RESUME_COUNT=$(python3 -c "import json; print(json.load(open('$RESUME_METADATA')).get('resume_count', 0))" 2>/dev/null || echo "0")

        warning "RESUME: Found checkpoint at step $LAST_STEP (resume count: $RESUME_COUNT)"

        # Crash loop detection
        if [ "$RESUME_COUNT" -ge 3 ]; then
            error "Crash loop detected: Training has failed 3 times at step $LAST_STEP"
            info "This may indicate a persistent issue with the checkpoint or configuration"
            info "Suggestions:"
            info "  1. Check logs for error patterns: tail -100 $LOG_FILE"
            info "  2. Remove checkpoint directory to start fresh: rm -rf $CHECKPOINT_DIR"
            info "  3. Verify training configuration: cat $CONFIG"
            exit 2
        fi

        # Update resume metadata
        RESUME_COUNT=$((RESUME_COUNT + 1))
        python3 -c "import json; import datetime; data = {'last_step': $LAST_STEP, 'resume_count': $RESUME_COUNT, 'last_resume': datetime.datetime.now().isoformat()}; json.dump(data, open('$RESUME_METADATA', 'w'), indent=2)"

        success "Resuming training from step $LAST_STEP (attempt $RESUME_COUNT/3)"
        RESUME_FLAG="--resume_from_checkpoint $CHECKPOINT_DIR"
    else
        info "Checkpoint found but no resume metadata. Creating new metadata..."
        python3 -c "import json; import datetime; data = {'last_step': 0, 'resume_count': 1, 'last_resume': datetime.datetime.now().isoformat()}; json.dump(data, open('$RESUME_METADATA', 'w'), indent=2)"
        RESUME_FLAG="--resume_from_checkpoint $CHECKPOINT_DIR"
    fi
else
    info "No existing checkpoints found. Starting fresh training."
    # Create checkpoint directory
    mkdir -p "$CHECKPOINT_DIR"
    # Create initial metadata
    python3 -c "import json; import datetime; data = {'last_step': 0, 'resume_count': 0, 'started': datetime.datetime.now().isoformat()}; json.dump(data, open('$RESUME_METADATA', 'w'), indent=2)"
fi

echo ""

# Step 2.4: Launch training in tmux session
info "Step 4: Launching training in tmux session..."

TMUX_SESSION="weatherman-training"

# Check if tmux session already exists
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    warning "Tmux session '$TMUX_SESSION' already exists"
    read -p "Kill existing session and create new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$TMUX_SESSION"
        success "Killed existing session"
    else
        error "Cannot proceed with existing session active"
        info "To attach to existing session: tmux attach -t $TMUX_SESSION"
        info "To kill existing session: tmux kill-session -t $TMUX_SESSION"
        exit 1
    fi
fi

# Create training command
TRAINING_CMD="python scripts/train.py --config $CONFIG $RESUME_FLAG 2>&1 | tee $LOG_FILE"

# Launch in tmux
tmux new-session -d -s "$TMUX_SESSION" bash
tmux send-keys -t "$TMUX_SESSION" "conda activate weatherman-lora" C-m
sleep 1
tmux send-keys -t "$TMUX_SESSION" "$TRAINING_CMD" C-m

info "[TRAINING-H100] Starting training in tmux session: $TMUX_SESSION"
success "Training launched successfully"
echo ""

# Step 2.5: Monitor first 100 steps for errors
info "Step 5: Monitoring initial training steps..."

info "Waiting for training to initialize (30 seconds)..."
sleep 30

# Check for errors in log file
if [ -f "$LOG_FILE" ]; then
    ERROR_COUNT=$(grep -i -E "ERROR|RuntimeError|CUDA out of memory|OutOfMemoryError" "$LOG_FILE" | wc -l)

    if [ "$ERROR_COUNT" -gt 0 ]; then
        error "Detected $ERROR_COUNT error(s) in log file"
        info "Recent errors:"
        grep -i -E "ERROR|RuntimeError|CUDA out of memory|OutOfMemoryError" "$LOG_FILE" | tail -5
        info ""
        error "Training appears to have encountered errors"
        info "Check full log: tail -f $LOG_FILE"
        info "Attach to session: tmux attach -t $TMUX_SESSION"
        exit 1
    fi

    # Check if training started
    if grep -q "Starting training" "$LOG_FILE" || grep -q "Epoch" "$LOG_FILE" || grep -q "Step" "$LOG_FILE"; then
        success "Training started successfully (first 100 steps)"
    else
        warning "Training may not have started yet. Monitor the log file."
    fi
else
    warning "Log file not created yet. Training may still be initializing."
fi

echo ""

# Step 2.6: Print reconnection and monitoring instructions
echo "============================================================"
success "[TRAINING-H100-STARTED] Training in progress"
echo "============================================================"
echo ""
echo "Training Details:"
echo "  Tmux Session: $TMUX_SESSION"
echo "  Config: $CONFIG"
echo "  Log File: $LOG_FILE"
echo "  Checkpoint Dir: $CHECKPOINT_DIR"
echo ""
info "[TRAINING-H100] Estimated duration: 3-4 hours"
echo ""
echo "Monitoring Commands:"
echo "  Reconnect to session:  tmux attach -t $TMUX_SESSION"
echo "  Detach from session:   Ctrl+B, then D"
echo "  View log file:         tail -f $LOG_FILE"
echo "  Check GPU usage:       nvidia-smi"
echo ""

if [ -n "$WANDB_API_KEY" ] || grep -q "wandb" "$CONFIG" 2>/dev/null; then
    echo "Remote Monitoring:"
    echo "  Weights & Biases:      https://wandb.ai"
    info "[TRAINING-H100] Monitor remotely via W&B dashboard"
    echo ""
fi

echo "Useful Commands:"
echo "  List tmux sessions:    tmux ls"
echo "  Kill training:         tmux kill-session -t $TMUX_SESSION"
echo "  Resume after crash:    ./train_h100_runpod.sh"
echo ""
echo "After Training Completes:"
echo "  1. Model will be saved to: $CHECKPOINT_DIR/"
echo "  2. Download to local machine for deployment"
echo "  3. See docs/DEPLOYMENT.md for usage instructions"
echo ""
info "[TRAINING-H100] Training session active. Safe to disconnect from SSH."
echo ""
