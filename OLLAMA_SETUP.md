# Running Weatherman-LoRA with Ollama

Ollama provides the easiest way to run your trained model locally with a simple CLI and API.

## Quick Start (Automated)

```bash
# 1. Install Ollama
brew install ollama

# 2. Run the automated setup script
./setup_ollama.sh
```

The script will:
- Merge your LoRA adapter with the base model (~15 min, downloads ~14GB)
- Convert to GGUF format for Ollama (~5-10 min)
- Create the Ollama model

## Manual Setup

### Step 1: Install Ollama

```bash
brew install ollama
# Or download from: https://ollama.ai
```

### Step 2: Merge LoRA with Base Model

```bash
python3 -m venv venv-merge
source venv-merge/bin/activate
pip install torch transformers peft accelerate
python merge_for_ollama.py
```

### Step 3: Convert to GGUF

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make && cd ..
python llama.cpp/convert.py merged_model --outfile weatherman-lora.gguf --outtype q4_K_M
```

### Step 4: Create Ollama Model

```bash
ollama create weatherman -f Modelfile
```

## Usage

### CLI
```bash
ollama run weatherman "What's the weather in San Francisco?"
```

### API
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "weatherman",
  "prompt": "What's the weather in Tokyo?"
}'
```

### Chat Interface
```bash
ollama run weatherman
# Interactive chat mode
```

## Quick Commands

```bash
# List models
ollama list

# Remove model
ollama rm weatherman

# Show model info
ollama show weatherman
```

## Alternative: Use Pre-converted Model

If you have access to a model conversion service or want to skip the GGUF conversion, you can:

1. Use the Python inference script instead
2. Export to HuggingFace and use their conversion tools
3. Use LM Studio (GUI alternative to Ollama that supports direct adapter loading)
