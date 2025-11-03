#!/bin/bash
#
# Weatherman-LoRA Local Environment Setup
# For Mac M4 data processing (no CUDA required)
#
# This script:
# 1. Verifies Python 3.10 is installed
# 2. Creates a virtual environment in .venv-local/
# 3. Installs data processing dependencies
# 4. Verifies storage requirements
# 5. Displays setup summary
#
# Usage: ./setup_local.sh

set -e  # Exit on error

echo "============================================================"
echo "Weatherman-LoRA Local Environment Setup"
echo "============================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python 3.10 is available
echo "Step 1: Checking Python version..."
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ "$PYTHON_VERSION" = "3.10" ]; then
        PYTHON_CMD="python3"
    else
        echo -e "${RED}Error: Python 3.10 is required but found Python $PYTHON_VERSION${NC}"
        echo "Please install Python 3.10:"
        echo "  brew install python@3.10"
        exit 1
    fi
else
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.10:"
    echo "  brew install python@3.10"
    exit 1
fi

PYTHON_FULL_VERSION=$($PYTHON_CMD --version)
echo -e "${GREEN}✓${NC} Found: $PYTHON_FULL_VERSION"
echo ""

# Create virtual environment
echo "Step 2: Creating virtual environment..."
if [ -d ".venv-local" ]; then
    echo -e "${YELLOW}Virtual environment already exists, skipping creation${NC}"
else
    $PYTHON_CMD -m venv .venv-local
    echo -e "${GREEN}✓${NC} Created .venv-local/"
fi
echo ""

# Activate virtual environment
echo "Step 3: Activating virtual environment..."
source .venv-local/bin/activate
echo -e "${GREEN}✓${NC} Activated .venv-local/"
echo ""

# Upgrade pip
echo "Step 4: Upgrading pip..."
pip install --upgrade pip -q
PIP_VERSION=$(pip --version | cut -d' ' -f2)
echo -e "${GREEN}✓${NC} pip upgraded to version $PIP_VERSION"
echo ""

# Install dependencies
echo "Step 5: Installing dependencies from requirements-local.txt..."
if [ ! -f "requirements-local.txt" ]; then
    echo -e "${RED}Error: requirements-local.txt not found${NC}"
    exit 1
fi

pip install -r requirements-local.txt -q
echo -e "${GREEN}✓${NC} All dependencies installed"
echo ""

# Verify storage
echo "Step 6: Verifying storage requirements..."
python3 scripts/check_storage.py
echo ""

# Display installed packages summary
echo "Step 7: Installed packages summary..."
echo "Core libraries:"
pip list | grep -E "pandas|beautifulsoup4|trafilatura|datasets|datasketch|jsonlines|nltk|langdetect|requests" || true
echo ""

# Success message
echo "============================================================"
echo -e "${GREEN}Local Environment Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "Virtual environment: .venv-local/"
echo "Python version: $PYTHON_FULL_VERSION"
echo ""
echo "To activate the environment:"
echo "  source .venv-local/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. Download Project Gutenberg texts (Roadmap Item 2)"
echo "  2. Process Reddit humor data (Roadmap Item 3)"
echo "  3. Run data cleaning pipeline (Roadmap Item 4)"
echo ""
