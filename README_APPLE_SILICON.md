# 🌤️ Weatherman-LoRA for Apple Silicon

Run your trained Weatherman weather assistant locally on your M1/M2/M3 Mac!

## TL;DR - Get Started in 2 Commands

```bash
brew install ollama
./setup_ollama_apple_silicon.sh
```

Wait 20-30 minutes, then:

```bash
ollama run weatherman "What's the weather in San Francisco?"
```

## What Is This?

This is a **fine-tuned Mistral-7B model** trained to:
- Understand natural weather queries
- Use tool calling to fetch weather data
- Provide helpful weather-related responses

**Trained on**: H100 GPU with 14K+ examples
**Optimized for**: Apple Silicon Macs with Metal acceleration
**Model size**: ~4GB (quantized from 14GB)

## Requirements

- **Mac**: M1, M2, or M3 (any variant)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: ~20GB free space (temporary, ~6GB after cleanup)
- **macOS**: Monterey (12.0) or later

## Installation

### Option 1: Automated (Recommended)

```bash
# Install Ollama
brew install ollama

# Run setup script
./setup_ollama_apple_silicon.sh
```

The script will:
1. Download Mistral-7B base model (~14GB)
2. Merge your trained LoRA adapter
3. Convert to GGUF format for Apple Silicon
4. Quantize to 4-bit for efficiency
5. Create Ollama model

**Time**: 20-30 minutes depending on internet speed

### Option 2: Manual

See [APPLE_SILICON_GUIDE.md](./APPLE_SILICON_GUIDE.md) for detailed steps.

## Usage

### Command Line

```bash
# Ask about weather
ollama run weatherman "What's the weather in Tokyo?"

# Interactive chat
ollama run weatherman
>>> What's the weather like in London?
>>> Should I bring an umbrella to Seattle?
>>> Compare NYC and LA weather
>>> exit
```

### API

```bash
# HTTP API
curl http://localhost:11434/api/generate -d '{
  "model": "weatherman",
  "prompt": "What'\''s the weather in Paris?"
}'
```

### Python

```python
import ollama

response = ollama.chat(model='weatherman', messages=[
    {'role': 'user', 'content': 'What's the weather in Seattle?'}
])

print(response['message']['content'])
```

## Performance

| Mac Model | Speed | Memory | Quality |
|-----------|-------|--------|---------|
| M1 (8GB) | ~15-20 tok/s | ~5GB | Excellent |
| M2 (16GB) | ~25-30 tok/s | ~5GB | Excellent |
| M3 Pro/Max | ~30-40 tok/s | ~6GB | Excellent |
| M3 Ultra | ~40-50 tok/s | ~6GB | Excellent |

## Example Queries

```bash
# Current weather
ollama run weatherman "What's the weather in San Francisco?"

# Forecast
ollama run weatherman "Weather forecast for Tokyo this week"

# Comparisons
ollama run weatherman "Compare weather in NYC and Miami"

# Travel advice
ollama run weatherman "I'm visiting London next week, what should I pack?"

# Activity planning
ollama run weatherman "Is it good weather for hiking in Colorado?"
```

## Files Structure

```
Weatherman-LoRA/
├── setup_ollama_apple_silicon.sh   # 🚀 Automated setup
├── APPLE_SILICON_GUIDE.md          # 📖 Detailed guide
├── Modelfile                       # ⚙️  Ollama configuration
├── merge_for_ollama.py             # 🔧 Merge script
├── adapters/
│   └── weatherman-lora-axolotl-h100/  # Your trained model
└── README_APPLE_SILICON.md         # 👋 This file
```

## Troubleshooting

### "Ollama not found"
```bash
brew install ollama
```

### "Model not found"
```bash
ollama list  # Check if weatherman exists
ollama create weatherman -f Modelfile  # Recreate if missing
```

### Slow performance
- Close other applications
- Check Activity Monitor for memory pressure
- Ensure Ollama is using Metal: `ollama show weatherman`

### Out of memory
- Use Q4_K_M quantization (default)
- Reduce context window in Modelfile
- Upgrade RAM or close apps

## Advanced Usage

### Custom System Prompt

Edit `Modelfile`:
```Modelfile
SYSTEM """Your custom instructions here..."""
```

Apply:
```bash
ollama create weatherman -f Modelfile
```

### Adjust Parameters

```Modelfile
PARAMETER temperature 0.8      # Creativity (0.0-1.0)
PARAMETER num_ctx 2048         # Context window
PARAMETER top_p 0.9            # Sampling threshold
```

### Export Model

```bash
# Share as Ollama model
ollama push username/weatherman

# Export GGUF file
cp weatherman-lora.gguf ~/Desktop/
```

## What's Next?

1. **Integrate real weather API** - Connect to OpenWeatherMap
2. **Build web interface** - Create a chat UI
3. **Deploy as service** - Run on always-on Mac
4. **Train on more data** - Improve responses
5. **Add more tools** - Expand beyond weather

## Training Details

- **Base Model**: Mistral-7B-Instruct-v0.3
- **Method**: QLoRA (4-bit quantization)
- **Hardware**: RunPod H100 GPU
- **Dataset**: 14,399 synthetic tool-calling examples
- **Epochs**: 3
- **Training Time**: ~3-4 hours
- **Final Loss**: ~0.48

See [axolotl_config_h100.yaml](./axolotl_config_h100.yaml) for full training configuration.

## Resources

- 📖 [Full Apple Silicon Guide](./APPLE_SILICON_GUIDE.md)
- 🦙 [Ollama Documentation](https://github.com/ollama/ollama)
- 🤖 [Mistral Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- ⚡ [llama.cpp Metal Guide](https://github.com/ggerganov/llama.cpp#metal-build)

## Contributing

Found issues or want to improve the model?
- Report bugs in GitHub Issues
- Share better prompts or training data
- Contribute to the training pipeline

## License

Model based on Mistral-7B-Instruct-v0.3 (Apache 2.0 License)

---

**Ready to try it?**

```bash
brew install ollama && ./setup_ollama_apple_silicon.sh
```

Happy forecasting! 🌤️
