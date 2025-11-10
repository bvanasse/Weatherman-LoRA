#!/bin/bash
# Automated setup script for running Weatherman-LoRA with Ollama

set -e  # Exit on error

echo "======================================"
echo "Weatherman-LoRA Ollama Setup"
echo "======================================"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed"
    echo "Please install from: https://ollama.ai"
    echo "Or run: brew install ollama"
    exit 1
fi

echo "✓ Ollama is installed"

# Step 1: Create virtual environment for merging
echo ""
echo "Step 1: Setting up Python environment for model merging..."
if [ ! -d "venv-merge" ]; then
    python3 -m venv venv-merge
fi
source venv-merge/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q torch transformers peft accelerate

# Step 2: Merge LoRA adapter
echo ""
echo "Step 2: Merging LoRA adapter with base model..."
echo "This will download ~14GB and may take 10-15 minutes..."
python merge_for_ollama.py

# Step 3: Check if llama.cpp exists
echo ""
echo "Step 3: Converting to GGUF format..."
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make
    cd ..
fi

# Convert to GGUF
echo "Converting merged model to GGUF (this may take 5-10 minutes)..."
python llama.cpp/convert.py merged_model --outfile weatherman-lora.gguf --outtype q4_K_M

# Step 4: Create Ollama model
echo ""
echo "Step 4: Creating Ollama model..."
ollama create weatherman -f Modelfile

echo ""
echo "======================================"
echo "✓ Setup Complete!"
echo "======================================"
echo ""
echo "Test your model with:"
echo "  ollama run weatherman \"What's the weather in San Francisco?\""
echo ""
echo "Or use the API:"
echo "  curl http://localhost:11434/api/generate -d '{\"model\": \"weatherman\", \"prompt\": \"What's the weather in Tokyo?\"}'"
echo ""
