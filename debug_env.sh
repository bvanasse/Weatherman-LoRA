#!/bin/bash
#
# Quick environment debug script
# Run this on RunPod to diagnose the issue
#

echo "=== Environment Debug ==="
echo ""

# Check conda environment
echo "Current conda env: $CONDA_DEFAULT_ENV"
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "No conda env active, attempting to activate weatherman-lora..."
    eval "$(conda shell.bash hook)"
    conda activate weatherman-lora
    echo "After activation: $CONDA_DEFAULT_ENV"
fi

echo ""

# Check Python
echo "Python command:"
PYTHON_CMD=$(which python)
echo "  Path: $PYTHON_CMD"
$PYTHON_CMD --version

echo ""

# Check pip
echo "Pip check:"
PIP_CMD="$PYTHON_CMD -m pip"
echo "  Command: $PIP_CMD"

if $PIP_CMD --version &>/dev/null; then
    echo "  ✓ pip works via python -m pip"
    $PIP_CMD --version
else
    echo "  ✗ pip FAILED via python -m pip"
    echo "  Trying direct import..."
    if $PYTHON_CMD -c "import pip; print(pip.__version__)" 2>/dev/null; then
        echo "  ✓ pip module importable"
    else
        echo "  ✗ pip module import FAILED"
    fi
fi

echo ""

# Check Axolotl
echo "Axolotl check:"
if $PYTHON_CMD -c "import axolotl" 2>/dev/null; then
    echo "  ✓ Axolotl installed"
    $PYTHON_CMD -c "import axolotl; print(f'Version: {axolotl.__version__}')" 2>/dev/null || echo "  (version unknown)"
else
    echo "  ✗ Axolotl NOT installed"
    echo ""
    echo "Need to install Axolotl. This will take 10-15 minutes."
    echo "Run: pip install axolotl[deepspeed,flash-attn]"
fi

echo ""

# Check for accelerate and axolotl CLI
echo "CLI tools:"
if command -v accelerate &>/dev/null; then
    echo "  ✓ accelerate CLI available"
else
    echo "  ✗ accelerate CLI not found"
fi

if command -v axolotl &>/dev/null; then
    echo "  ✓ axolotl CLI available"
else
    echo "  ✗ axolotl CLI not found"
fi

echo ""

# Check training data
echo "Training data:"
if [ -f "data/synthetic/final_train_diverse.jsonl" ]; then
    COUNT=$(wc -l < data/synthetic/final_train_diverse.jsonl)
    echo "  ✓ Training data exists ($COUNT examples)"
else
    echo "  ✗ Training data NOT found"
fi

if [ -f "data/synthetic/final_validation_diverse.jsonl" ]; then
    COUNT=$(wc -l < data/synthetic/final_validation_diverse.jsonl)
    echo "  ✓ Validation data exists ($COUNT examples)"
else
    echo "  ✗ Validation data NOT found"
fi

echo ""

# Check config
echo "Axolotl config:"
if [ -f "axolotl_config_h100.yaml" ]; then
    echo "  ✓ Config file exists"
else
    echo "  ✗ Config file NOT found"
fi

echo ""
echo "=== Debug Complete ==="
