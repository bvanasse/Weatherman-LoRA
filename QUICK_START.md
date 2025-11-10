# Weatherman-LoRA Quick Start Guide

Your model has been successfully trained on H100! Here's how to run it locally.

## Option 1: Ollama (Recommended - Easiest)

**Best for**: Quick testing, CLI usage, API endpoints

```bash
# Install Ollama
brew install ollama

# Run automated setup
./setup_ollama.sh

# Test it
ollama run weatherman "What's the weather in San Francisco?"
```

**Time**: ~20-25 minutes for first-time setup
**Docs**: See `OLLAMA_SETUP.md`

## Option 2: Python Script (More Control)

**Best for**: Custom inference, integration into Python apps

```bash
# Setup
python3 -m venv venv-inference
source venv-inference/bin/activate
pip install -r requirements-inference.txt

# Run
python test_local_inference.py
```

**Time**: ~5-10 minutes for first run (downloads model)
**Docs**: See `LOCAL_INFERENCE_GUIDE.md`

## What You Have

```
adapters/weatherman-lora-axolotl-h100/
├── adapter_model.safetensors    # Your trained LoRA weights (160MB)
├── adapter_config.json          # LoRA configuration
├── tokenizer files              # Mistral tokenizer
└── checkpoints/                 # Training checkpoints
```

## Training Summary

- **Base Model**: Mistral-7B-Instruct-v0.3
- **Training**: 3 epochs on H100 GPU
- **Dataset**: 14,399 tool-calling examples
- **Time**: ~3-4 hours with optimized config
- **Method**: QLoRA (4-bit quantization)

## Next Steps

1. **Test locally** with either Ollama or Python
2. **Check WandB** for training metrics: `wandb/`
3. **Integrate with weather API** to make it functional
4. **Deploy** to a server or create a web interface

## Files Overview

- `OLLAMA_SETUP.md` - Full Ollama setup guide
- `LOCAL_INFERENCE_GUIDE.md` - Python inference guide
- `test_local_inference.py` - Python test script
- `merge_for_ollama.py` - Merge LoRA for GGUF conversion
- `setup_ollama.sh` - Automated Ollama setup
- `Modelfile` - Ollama configuration

## Troubleshooting

### Out of Memory
- Use Ollama (more memory efficient)
- Close other apps
- Reduce batch size in Python script

### Slow Generation
- Normal on Mac (2-5 tokens/sec)
- Use Ollama for better performance
- Consider cloud deployment for production

### Model Quality Issues
- Check WandB metrics in `wandb/` folder
- Verify training completed all 3 epochs
- Test with diverse queries

## Questions?

- Training issues: Check `adapters/weatherman-lora-axolotl-h100/debug.log`
- Inference issues: See troubleshooting in respective guides
- Config questions: See `axolotl_config_h100.yaml`
