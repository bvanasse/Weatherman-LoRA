#!/bin/bash
# Automated setup script for running Weatherman-LoRA with Ollama on Apple Silicon
# Optimized for M1/M2/M3 Macs

set -e  # Exit on error

echo "======================================"
echo "Weatherman-LoRA Ollama Setup"
echo "Apple Silicon (M1/M2/M3) Optimized"
echo "======================================"

# Detect architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "Warning: This script is optimized for Apple Silicon (arm64)"
    echo "Detected architecture: $ARCH"
    echo "Continuing anyway..."
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "Ollama is not installed!"
    echo "Please install it first:"
    echo "  Option 1: brew install ollama"
    echo "  Option 2: Download from https://ollama.ai"
    echo ""
    read -p "Would you like to install via brew now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "Homebrew not found. Please install Ollama manually from https://ollama.ai"
            exit 1
        fi
    else
        exit 1
    fi
fi

echo "✓ Ollama is installed"

# Check if Ollama service is running
echo ""
echo "Checking if Ollama service is running..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 3
fi
echo "✓ Ollama service is running"

# Step 1: Create virtual environment for merging
echo ""
echo "Step 1: Setting up Python environment for model merging..."
if [ ! -d "venv-merge" ]; then
    python3 -m venv venv-merge
fi
source venv-merge/bin/activate

# Install dependencies (no bitsandbytes for Mac)
echo "Installing dependencies (this may take a few minutes)..."
pip install -q --upgrade pip
pip install -q torch transformers peft accelerate

# Step 2: Merge LoRA adapter
echo ""
echo "Step 2: Merging LoRA adapter with base model..."
echo "⏳ This will download ~14GB and may take 10-15 minutes..."
echo ""
python merge_for_ollama.py

# Step 3: Check if llama.cpp exists
echo ""
echo "Step 3: Converting to GGUF format for Ollama..."
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp

    # Build for Apple Silicon with Metal support
    echo "Building llama.cpp with Metal acceleration..."
    make clean
    LLAMA_METAL=1 make
    cd ..
else
    echo "✓ llama.cpp already exists"
fi

# Convert to GGUF with optimal quantization for Apple Silicon
echo ""
echo "Converting merged model to GGUF format..."
echo "⏳ Using Q4_K_M quantization (optimal for Apple Silicon)"
echo "This may take 5-10 minutes..."
python llama.cpp/convert.py merged_model \
    --outfile weatherman-lora.gguf \
    --outtype q4_K_M

# Get file size
GGUF_SIZE=$(du -h weatherman-lora.gguf | cut -f1)
echo "✓ GGUF model created: weatherman-lora.gguf ($GGUF_SIZE)"

# Step 4: Create Ollama model
echo ""
echo "Step 4: Creating Ollama model..."
ollama create weatherman -f Modelfile

echo ""
echo "======================================"
echo "✓ Setup Complete!"
echo "======================================"
echo ""
echo "Your Weatherman-LoRA model is ready!"
echo ""
echo "📊 Model Info:"
echo "  - Size: $GGUF_SIZE (quantized for Apple Silicon)"
echo "  - Quantization: Q4_K_M (4-bit, optimal quality/speed)"
echo "  - Metal acceleration: Enabled"
echo ""
echo "🚀 Quick Test:"
echo "  ollama run weatherman \"What's the weather in San Francisco?\""
echo ""
echo "💬 Interactive Chat:"
echo "  ollama run weatherman"
echo ""
echo "🔌 API Usage:"
echo "  curl http://localhost:11434/api/generate -d '{"
echo "    \"model\": \"weatherman\","
echo "    \"prompt\": \"What's the weather in Tokyo?\""
echo "  }'"
echo ""
echo "📚 More Commands:"
echo "  ollama list              # List installed models"
echo "  ollama rm weatherman     # Remove model"
echo "  ollama show weatherman   # Show model details"
echo ""
