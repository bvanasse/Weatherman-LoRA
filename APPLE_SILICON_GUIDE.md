# Weatherman-LoRA on Apple Silicon (M1/M2/M3)

Complete guide for running your trained Weatherman-LoRA model on Apple Silicon Macs using Ollama.

## Why Ollama for Apple Silicon?

- ✅ **Native Metal acceleration** - Uses Apple's GPU
- ✅ **Optimized quantization** - 4-bit models run smoothly
- ✅ **Low memory usage** - ~4-6GB RAM for 7B models
- ✅ **Fast inference** - 15-30 tokens/second on M1/M2
- ✅ **Simple API** - Easy CLI and HTTP interface
- ✅ **No Python hassles** - No dependency conflicts

## Quick Start (Automated)

```bash
# 1. Install Ollama
brew install ollama

# 2. Run the setup script
chmod +x setup_ollama_apple_silicon.sh
./setup_ollama_apple_silicon.sh
```

**Time**: 20-30 minutes (includes downloads)

## What the Script Does

1. **Downloads base model** (~14GB) - Mistral-7B-Instruct-v0.3
2. **Merges your LoRA adapter** - Combines training with base model
3. **Converts to GGUF** - Optimizes for Apple Silicon Metal
4. **Quantizes to Q4_K_M** - 4-bit precision (best quality/speed)
5. **Creates Ollama model** - Ready to use

## Manual Setup (Step-by-Step)

### Step 1: Install Ollama

```bash
# Via Homebrew
brew install ollama

# Or download from
open https://ollama.ai
```

### Step 2: Start Ollama Service

```bash
# Start in background
ollama serve &

# Or let it start automatically
```

### Step 3: Merge LoRA Adapter

```bash
# Create environment
python3 -m venv venv-merge
source venv-merge/bin/activate

# Install dependencies (Mac-compatible)
pip install torch transformers peft accelerate

# Merge adapter with base model
python merge_for_ollama.py
```

This creates `merged_model/` directory (~14GB).

### Step 4: Convert to GGUF

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with Metal support
LLAMA_METAL=1 make

# Convert model
python convert.py ../merged_model \
    --outfile ../weatherman-lora.gguf \
    --outtype q4_K_M

cd ..
```

Creates `weatherman-lora.gguf` (~4GB quantized).

### Step 5: Import to Ollama

```bash
# Create model from GGUF
ollama create weatherman -f Modelfile
```

## Usage

### Command Line

```bash
# Single query
ollama run weatherman "What's the weather in San Francisco?"

# Interactive chat
ollama run weatherman
>>> What's the weather in Tokyo?
>>> Compare weather in NYC and LA
>>> exit
```

### API (HTTP)

```bash
# Generate response
curl http://localhost:11434/api/generate -d '{
  "model": "weatherman",
  "prompt": "What'\''s the weather in London?",
  "stream": false
}'

# Chat format
curl http://localhost:11434/api/chat -d '{
  "model": "weatherman",
  "messages": [
    {"role": "user", "content": "What'\''s the weather in Paris?"}
  ]
}'
```

### Python SDK

```bash
pip install ollama

python
```

```python
import ollama

response = ollama.chat(model='weatherman', messages=[
    {'role': 'user', 'content': 'What's the weather in Seattle?'}
])

print(response['message']['content'])
```

## Model Management

```bash
# List models
ollama list

# Show model details
ollama show weatherman

# Copy model
ollama cp weatherman weatherman-backup

# Remove model
ollama rm weatherman

# Pull updates (if pushed to registry)
ollama pull weatherman
```

## Performance Expectations

### M1 Mac (8GB RAM)
- **Speed**: ~15-20 tokens/second
- **Memory**: ~4-5GB
- **Quality**: Excellent with Q4_K_M

### M2/M3 Mac (16GB+ RAM)
- **Speed**: ~25-35 tokens/second
- **Memory**: ~5-6GB
- **Quality**: Excellent with Q4_K_M

### M3 Max/Ultra
- **Speed**: ~40-50 tokens/second
- **Memory**: ~6-8GB
- **Quality**: Can use Q5_K_M or Q8 for even better quality

## Quantization Options

| Format | Size | Quality | Speed | RAM |
|--------|------|---------|-------|-----|
| Q4_K_M | ~4GB | Good | Fast | 5-6GB |
| Q5_K_M | ~5GB | Better | Medium | 6-7GB |
| Q8_0   | ~7GB | Best | Slower | 8-10GB |

**Recommended**: Q4_K_M (default in setup script)

To use different quantization:
```bash
python llama.cpp/convert.py merged_model \
    --outfile weatherman-lora-q5.gguf \
    --outtype q5_K_M
```

## Troubleshooting

### Ollama Service Not Starting
```bash
# Check if already running
pgrep ollama

# Kill and restart
pkill ollama
ollama serve &
```

### Model Not Found
```bash
# List models
ollama list

# Recreate if missing
ollama create weatherman -f Modelfile
```

### Slow Performance
1. Close other applications
2. Check Activity Monitor for CPU/Memory usage
3. Try lowering context window in Modelfile
4. Ensure Metal acceleration is enabled

### Out of Memory
1. Use Q4_K_M quantization (not Q8)
2. Reduce `num_ctx` in Modelfile (4096 → 2048)
3. Close other apps
4. Restart Mac

## Advanced Configuration

Edit the `Modelfile` to customize:

```Modelfile
FROM ./weatherman-lora.gguf

# Adjust context window (default: 4096)
PARAMETER num_ctx 2048

# Temperature (0.0-1.0, higher = more creative)
PARAMETER temperature 0.7

# Top-p sampling
PARAMETER top_p 0.9

# Custom system prompt
SYSTEM """You are Weatherman, specialized in weather queries..."""
```

Apply changes:
```bash
ollama create weatherman -f Modelfile
```

## Integration Examples

### Shell Script
```bash
#!/bin/bash
WEATHER=$(ollama run weatherman "What's the weather in $1?" --format json)
echo $WEATHER | jq -r '.response'
```

### Alfred Workflow
Create Alfred workflow that calls:
```bash
ollama run weatherman "{query}"
```

### Raycast Extension
Use Ollama API: `http://localhost:11434/api/generate`

## Next Steps

1. **Test with real weather data** - Integrate OpenWeatherMap API
2. **Create web interface** - Use Flask/FastAPI + Ollama
3. **Deploy to server** - Run Ollama on Mac Mini/Studio
4. **Fine-tune further** - Add more training data
5. **Share your model** - Push to Ollama registry

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [llama.cpp on Apple Silicon](https://github.com/ggerganov/llama.cpp#metal-build)
- [Mistral Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

## Support

Issues? Check:
1. Ollama service is running: `ollama list`
2. Model exists: `ollama show weatherman`
3. Logs: `tail -f ~/.ollama/logs/server.log`
