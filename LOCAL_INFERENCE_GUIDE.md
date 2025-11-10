# Local Inference Guide - Weatherman-LoRA

## Setup Instructions

### 1. Create a virtual environment
```bash
cd /Users/benjaminvanasse/Apps/Weatherman-LoRA
python3 -m venv venv-inference
source venv-inference/bin/activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements-inference.txt
```

**Note for Apple Silicon Macs**: If you get errors with bitsandbytes, you may need to use the MPS backend instead. See Alternative Setup below.

### 3. Run the test script
```bash
python test_local_inference.py
```

## What to Expect

- **First run**: Will download the Mistral-7B base model (~14GB) - this takes time
- **Memory usage**: ~8-10GB RAM with 4-bit quantization
- **Speed**: Slower than H100, but should work on modern Macs

## Alternative Setup (if bitsandbytes fails on Mac)

If you encounter issues with bitsandbytes on Mac, you can use MPS (Metal Performance Shaders) instead:

```bash
# Install without bitsandbytes
pip install torch transformers peft accelerate safetensors sentencepiece

# Use the MPS-optimized script (coming soon)
python test_local_inference_mps.py
```

## Testing Your Model

The `test_local_inference.py` script will:
1. Load your trained LoRA adapter
2. Run 3 test weather queries
3. Show if the model is using tool calling correctly

Look for responses that include:
- `<tool_call>` tags
- `get_weather` function calls
- Structured JSON-like tool invocations

## Next Steps

Once inference works locally, you can:
1. Create a simple API endpoint (FastAPI/Flask)
2. Build a chat interface
3. Integrate with actual weather APIs
4. Deploy to a server

## Troubleshooting

### Out of Memory
- Close other applications
- Reduce `max_new_tokens` in the script (default: 512)
- Use a smaller quantization if available

### Slow Generation
- Normal on Mac - expect 2-5 tokens/second
- Use batch processing for multiple queries
- Consider using a cloud API for production

### Import Errors
```bash
pip install --upgrade transformers peft accelerate
```
