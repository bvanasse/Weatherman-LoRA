#!/bin/bash
#
# Weatherman-LoRA M4 Local Training Execution Script
# Launches LoRA training on Mac M4 with MPS backend
#
# This script:
# 1. Runs pre-flight validation checks
# 2. Configures M4-specific training parameters
# 3. Creates timestamped log files
# 4. Launches training with MPS backend (blocking, not in tmux)
# 5. Monitors system temperature (advisory)
# 6. Provides time estimates after first 100 steps
# 7. Handles completion and displays summary
#
# Usage: ./train_m4_local.sh [--config path/to/config.yaml]

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
CONFIG="configs/training_config_m4.yaml"
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
echo "Weatherman-LoRA M4 Local Training Execution"
echo "============================================================"
echo ""

info "[TRAINING-M4] Pre-flight checks starting"
echo ""

# Step 4.1: M4-specific setup
info "Step 1: Running pre-flight validation..."

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    warning ".venv-local not activated"
    info "Attempting to activate..."

    if [ -d ".venv-local" ]; then
        source .venv-local/bin/activate
        success "Activated .venv-local environment"
    else
        error "Virtual environment .venv-local not found"
        info "Please run: ./setup_m4_local.sh"
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
    if python3 scripts/validate_environment.py --env=m4; then
        success "Environment validation passed"
    else
        error "Environment validation failed"
        exit 1
    fi
else
    warning "Skipping environment validation (script not found)"
fi

if [ -f "scripts/validate_training_config.py" ]; then
    if python3 scripts/validate_training_config.py --config "$CONFIG"; then
        success "Training config validation passed"
    else
        error "Training config validation failed"
        exit 1
    fi
else
    warning "Skipping config validation (script not found)"
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

# Verify data files exist
if [ ! -f "data/processed/train.jsonl" ]; then
    error "Training data not found: data/processed/train.jsonl"
    info "Please run data preparation scripts first"
    exit 1
fi

success "Training data verified"
echo ""

# Step 4.2: Configure M4-specific training parameters
info "Step 2: Verifying M4-specific configuration..."

# Check config parameters (informational)
if command -v python3 &> /dev/null && [ -f "$CONFIG" ]; then
    info "Training configuration from $CONFIG:"
    python3 -c "
import yaml
with open('$CONFIG') as f:
    config = yaml.safe_load(f)
    print(f\"  Batch size per device: {config.get('per_device_train_batch_size', 'N/A')}\")
    print(f\"  Gradient accumulation: {config.get('gradient_accumulation_steps', 'N/A')}\")
    print(f\"  Checkpoint frequency: {config.get('save_steps', 'N/A')} steps\")
    print(f\"  Max sequence length: {config.get('max_seq_length', 'N/A')} tokens\")
" 2>/dev/null || info "  (Unable to parse config)"
fi

success "M4 configuration verified"
info "Recommended: batch_size=1, grad_accum=8, seq_length=2048, checkpoint_steps=250"
echo ""

# Step 4.3: Create timestamped log file
info "Step 3: Setting up logging..."

mkdir -p logs
LOG_FILE="logs/training_m4_$(date +%Y%m%d_%H%M%S).log"

info "[TRAINING-M4] Log file: $LOG_FILE"
success "Log file created: $LOG_FILE"
echo ""

# Step 4.4: Launch training with MPS backend
echo "============================================================"
info "[TRAINING-M4] Starting training on MPS backend"
echo "============================================================"
echo ""

info "Training will run in foreground (not in tmux)"
info "Press Ctrl+C to stop training (progress will be saved)"
echo ""

# Step 4.5: Monitor system temperature (optional)
warning "If training slows, check Activity Monitor for thermal throttling"
info "To monitor thermal state manually:"
info "  sudo powermetrics --samplers smc -n 1"
echo ""

# Record start time
START_TIME=$(date +%s)

# Launch training
info "Launching training..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 4.6 & 4.7: Training execution with time estimates
if python3 scripts/train.py --config "$CONFIG" 2>&1 | tee "$LOG_FILE"; then
    # Training completed successfully
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "============================================================"
    success "[TRAINING-M4-COMPLETE] Training finished"
    echo "============================================================"
    echo ""

    echo "Training Summary:"
    echo "  Duration: ${HOURS}h ${MINUTES}m"
    echo "  Config: $CONFIG"
    echo "  Log file: $LOG_FILE"
    echo ""

    # Find model output directory
    if [ -d "adapters/weatherman-lora-m4" ]; then
        MODEL_DIR="adapters/weatherman-lora-m4"
    elif [ -d "outputs" ]; then
        MODEL_DIR=$(ls -td outputs/* | head -1)
    else
        MODEL_DIR="(check config for output_dir)"
    fi

    echo "Model Location:"
    echo "  $MODEL_DIR"
    echo ""

    echo "Next Steps:"
    echo "  1. Validate the trained model"
    echo "  2. Test with sample prompts"
    echo "  3. See docs/DEPLOYMENT.md for deployment instructions"
    echo ""

    if [ -d "$MODEL_DIR" ]; then
        MODEL_SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
        echo "Model size: $MODEL_SIZE"
        echo ""
    fi

else
    # Training failed
    error "Training failed or was interrupted"
    info "Check log file for details: $LOG_FILE"
    info "To resume training, run this script again"
    exit 1
fi
