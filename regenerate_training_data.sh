#!/bin/bash
#
# Regenerate Training Data with Diverse Responses
#
# This script regenerates training data to remove repetitive templates
# and create more diverse, creative responses using Claude API.
#
# Usage: ./regenerate_training_data.sh

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

success() { echo -e "${GREEN}✓${NC} $1"; }
error() { echo -e "${RED}✗ ERROR:${NC} $1"; }
warning() { echo -e "${YELLOW}⚠ WARNING:${NC} $1"; }
info() { echo -e "$1"; }

echo "============================================================"
echo "Training Data Regeneration with Claude API"
echo "============================================================"
echo ""

# Check for ANTHROPIC_API_KEY
if [ -z "$ANTHROPIC_API_KEY" ]; then
    error "ANTHROPIC_API_KEY environment variable not set"
    info "Get your API key from: https://console.anthropic.com/"
    info "Set it with: export ANTHROPIC_API_KEY='your-api-key'"
    exit 1
fi

success "Found ANTHROPIC_API_KEY"

# Check if anthropic package is installed
if ! python3 -c "import anthropic" 2>/dev/null; then
    warning "anthropic package not installed"
    info "Installing anthropic..."
    pip3 install anthropic
    success "Installed anthropic package"
fi

# Verify input files exist
if [ ! -f "data/synthetic/final_train.jsonl" ]; then
    error "Input file not found: data/synthetic/final_train.jsonl"
    exit 1
fi

success "Found input training data"

# Create backup
BACKUP_DIR="data/synthetic/backups"
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/final_train_backup_$TIMESTAMP.jsonl"

info "Creating backup..."
cp data/synthetic/final_train.jsonl "$BACKUP_FILE"
success "Backup created: $BACKUP_FILE"

echo ""
info "Regeneration Settings:"
info "  Input: data/synthetic/final_train.jsonl"
info "  Output: data/synthetic/final_train_diverse.jsonl"
info "  Model: claude-sonnet-4-20250514 (latest)"
info "  Max templates per pattern: 100"
info "  Literary corpus: data/processed/"
echo ""

warning "This will:"
warning "  - Identify ~8,500 templated responses"
warning "  - Regenerate them with Claude API (~$20-40 in API costs)"
warning "  - Take approximately 3-4 hours to complete"
warning "  - Support checkpoint recovery if interrupted"
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    info "Aborted by user"
    exit 0
fi

echo ""
info "Starting regeneration..."
echo ""

# Run regeneration script
python3 scripts/regenerate_diverse_responses.py \
    --input data/synthetic/final_train.jsonl \
    --output data/synthetic/final_train_diverse.jsonl \
    --corpus-path data/processed \
    --max-templates 100 \
    --checkpoint-interval 100 \
    --model claude-sonnet-4-20250514

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    success "Regeneration Complete!"
    echo "============================================================"
    echo ""
    echo "New training file: data/synthetic/final_train_diverse.jsonl"
    echo "Original backup: $BACKUP_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Verify the new data:"
    echo "     python3 scripts/analyze_data_diversity.py data/synthetic/final_train_diverse.jsonl"
    echo ""
    echo "  2. Update validation data (if needed):"
    echo "     ./regenerate_validation_data.sh"
    echo ""
    echo "  3. Create new training symlink:"
    echo "     ln -sf \$(pwd)/data/synthetic/final_train_diverse.jsonl data/processed/train.jsonl"
    echo ""
    echo "  4. Train with Axolotl:"
    echo "     ./train_with_axolotl_h100.sh"
    echo ""
else
    error "Regeneration failed"
    info "Check the error messages above"
    info "The checkpoint file will allow you to resume"
    exit 1
fi
