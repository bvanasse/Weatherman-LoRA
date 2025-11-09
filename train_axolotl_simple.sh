#!/bin/bash
#
# Simplified Axolotl Training Script for H100
# For use with RunPod instances that have Axolotl pre-installed
#
# Usage: ./train_axolotl_simple.sh
#

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

success() { echo -e "${GREEN}✓${NC} $1"; }
error() { echo -e "${RED}✗ ERROR:${NC} $1"; }
warning() { echo -e "${YELLOW}⚠ WARNING:${NC} $1"; }
info() { echo -e "${BLUE}ℹ${NC} $1"; }

echo "============================================================"
echo "Weatherman-LoRA Axolotl Training (H100)"
echo "============================================================"
echo ""

# Verify config exists
CONFIG_FILE="axolotl_config_h100.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    error "Config not found: $CONFIG_FILE"
    exit 1
fi
success "Found config: $CONFIG_FILE"

# Verify training data
TRAIN_DATA="data/synthetic/final_train_diverse_fixed.jsonl"
VAL_DATA="data/synthetic/final_validation_diverse_fixed.jsonl"

if [ ! -f "$TRAIN_DATA" ]; then
    error "Training data not found: $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    error "Validation data not found: $VAL_DATA"
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$TRAIN_DATA")
VAL_COUNT=$(wc -l < "$VAL_DATA")

success "Training data: $TRAIN_COUNT examples"
success "Validation data: $VAL_COUNT examples"
echo ""

# Check for existing checkpoints
CHECKPOINT_DIR="adapters/weatherman-lora-axolotl-h100"
if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR/checkpoint-* 2>/dev/null)" ]; then
    warning "Found existing checkpoints in $CHECKPOINT_DIR"
    LATEST_CHECKPOINT=$(ls -td $CHECKPOINT_DIR/checkpoint-* 2>/dev/null | head -1)
    info "Latest checkpoint: $LATEST_CHECKPOINT"
    info "Axolotl will automatically resume from latest checkpoint"
    echo ""
fi

# Create logs directory
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/axolotl_${TIMESTAMP}.log"

info "Log file: $LOG_FILE"
echo ""

# Training command
TRAIN_CMD="accelerate launch -m axolotl.cli.train $CONFIG_FILE"

info "============================================================"
info "Launching Training in Persistent Session"
info "============================================================"
echo ""
info "Base Model: mistralai/Mistral-7B-Instruct-v0.3"
info "Training Examples: $TRAIN_COUNT"
info "Validation Examples: $VAL_COUNT"
info "Output: $CHECKPOINT_DIR"
info "Estimated Duration: 3-4 hours"
echo ""

# Launch in tmux session
TMUX_SESSION="weatherman-training"

if command -v tmux &> /dev/null; then
    # Check if session exists
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        warning "Session '$TMUX_SESSION' already exists"
        info "To view: tmux attach -t $TMUX_SESSION"
        info "To kill: tmux kill-session -t $TMUX_SESSION"
        echo ""
        read -p "Kill existing session and create new? (y/N): " -n 1 -r
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
    tmux new-session -d -s "$TMUX_SESSION"
    tmux send-keys -t "$TMUX_SESSION" "cd $(pwd)" C-m
    sleep 1
    tmux send-keys -t "$TMUX_SESSION" "$TRAIN_CMD 2>&1 | tee $LOG_FILE" C-m

    success "Training launched in tmux session: $TMUX_SESSION"
    USING_TMUX=true
else
    # Fallback to nohup
    warning "tmux not found, using nohup"

    nohup bash -c "cd $(pwd) && $TRAIN_CMD" > "$LOG_FILE" 2>&1 &
    TRAINING_PID=$!
    echo $TRAINING_PID > /tmp/weatherman_training.pid

    success "Training launched with nohup (PID: $TRAINING_PID)"
    USING_TMUX=false
fi

echo ""
info "Waiting for training to initialize (30 seconds)..."
sleep 30

# Check for immediate errors
if [ -f "$LOG_FILE" ]; then
    ERROR_COUNT=$(grep -i -E "ERROR|RuntimeError|CUDA out of memory|Traceback" "$LOG_FILE" 2>/dev/null | wc -l)

    if [ "$ERROR_COUNT" -gt 0 ]; then
        error "Detected errors in log output"
        info "Recent errors:"
        grep -i -E "ERROR|RuntimeError|CUDA out of memory|Traceback" "$LOG_FILE" | tail -10
        echo ""
        error "Training failed during initialization"
        info "Check full log: tail -f $LOG_FILE"
        if [ "$USING_TMUX" = "true" ]; then
            info "Attach to session: tmux attach -t $TMUX_SESSION"
        fi
        exit 1
    fi

    if grep -q -E "Starting training|Epoch|Step|Loading" "$LOG_FILE"; then
        success "Training initialized successfully"
    else
        warning "Training may still be initializing"
    fi
else
    warning "Log file not created yet"
fi

echo ""
echo "============================================================"
success "Training Session Active"
echo "============================================================"
echo ""

if [ "$USING_TMUX" = "true" ]; then
    echo "Session Type: tmux"
    echo "Session Name: $TMUX_SESSION"
    echo ""
    echo "Monitor Training:"
    echo "  Attach to session:  tmux attach -t $TMUX_SESSION"
    echo "  Detach from session: Ctrl+B, then D"
    echo "  View log:           tail -f $LOG_FILE"
    echo "  Check GPU:          nvidia-smi"
    echo ""
    echo "Stop Training:"
    echo "  tmux kill-session -t $TMUX_SESSION"
else
    echo "Session Type: nohup"
    if [ -f "/tmp/weatherman_training.pid" ]; then
        echo "PID: $(cat /tmp/weatherman_training.pid)"
    fi
    echo ""
    echo "Monitor Training:"
    echo "  View log:      tail -f $LOG_FILE"
    echo "  Check process: ps -p \$(cat /tmp/weatherman_training.pid)"
    echo "  Check GPU:     nvidia-smi"
    echo ""
    echo "Stop Training:"
    echo "  kill \$(cat /tmp/weatherman_training.pid)"
fi

echo ""
echo "Output: $CHECKPOINT_DIR"
echo "Log: $LOG_FILE"
echo ""
info "Training will continue even if you disconnect from SSH"
info "Estimated completion: 3-4 hours"
echo ""
echo "After training completes:"
echo "  1. Model saved to: $CHECKPOINT_DIR/"
echo "  2. Download to local machine"
echo "  3. See docs/M4_DEPLOYMENT.md for deployment"
echo ""
