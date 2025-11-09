# M4 Mac Local Deployment Guide

This guide covers deploying your Axolotl-trained LoRA adapter on Apple Silicon M4 Mac for local inference.

## Prerequisites

- Trained LoRA adapter from H100 (in `adapters/weatherman-lora-axolotl-h100/`)
- Apple Silicon Mac (M4, M3, M2, or M1)
- macOS 14+ (for best Metal Performance Shaders support)
- 16GB+ unified memory recommended

---

## Step 1: Download Trained Adapter from RunPod

After training completes on RunPod H100, download the adapter to your M4 Mac:

### Using SCP

```bash
# From your M4 Mac terminal
cd /path/to/Weatherman-LoRA

# Download entire adapter directory
scp -r root@runpod-instance-ip:/workspace/Weatherman-LoRA/adapters/weatherman-lora-axolotl-h100 ./adapters/

# Or use specific RunPod SSH command (found in RunPod dashboard)
ssh -p 12345 root@123.456.78.90 "cd /workspace/Weatherman-LoRA && tar -czf adapter.tar.gz adapters/weatherman-lora-axolotl-h100"
scp -P 12345 root@123.456.78.90:/workspace/Weatherman-LoRA/adapter.tar.gz ./
tar -xzf adapter.tar.gz
```

### Using rsync (Recommended for Large Files)

```bash
# Faster for large files, shows progress, resumes interrupted transfers
rsync -avz --progress -e "ssh -p 12345" \
  root@runpod-ip:/workspace/Weatherman-LoRA/adapters/weatherman-lora-axolotl-h100/ \
  ./adapters/weatherman-lora-axolotl-h100/
```

### Expected Adapter Files

Your adapter directory should contain:

```
adapters/weatherman-lora-axolotl-h100/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # LoRA weights (preferred format)
├── adapter_model.bin            # Alternative weights format
├── tokenizer.json               # Tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── README.md                    # Axolotl-generated info
└── checkpoint-*/                # Training checkpoints (optional, can delete)
```

File Size: Expect 500MB-2GB depending on LoRA rank.

---

## Step 2: Install Ollama for M4

Ollama provides the best performance on Apple Silicon with native Metal support.

### Installation

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Or download from: https://ollama.ai/download

# Verify installation
ollama --version
# Expected: Ollama version 0.x.x
```

### Start Ollama Service

```bash
# Start Ollama server (runs in background)
ollama serve

# Or run in foreground for debugging
ollama serve --verbose
```

---

## Step 3: Merge LoRA Adapter with Base Model

You have two options for using the adapter:

### Option A: Use LoRA Adapter Directly (Recommended)

Most frameworks support loading LoRA adapters separately. This saves disk space.

```bash
# The adapter in adapters/weatherman-lora-axolotl-h100/ can be used as-is
# with inference frameworks that support LoRA
```

### Option B: Merge LoRA into Base Model

Create a single merged model for easier deployment:

```bash
# Using Axolotl (if you have it installed locally)
# First, install Axolotl on M4 (optional)
pip install axolotl[inference]

# Merge adapter with base model
axolotl merge-lora axolotl_config_h100.yaml \
  --lora-model-dir=adapters/weatherman-lora-axolotl-h100 \
  --merged-model-dir=models/weatherman-merged

# Expected output: Merged model in models/weatherman-merged/ (~14GB for Mistral 7B)
```

### Option C: Use Python Script for Merging

If Axolotl isn't available, use this Python script:

```python
# scripts/merge_lora_adapter.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
print(f"Loading base model: {base_model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
adapter_path = "adapters/weatherman-lora-axolotl-h100"
print(f"Loading LoRA adapter from: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge adapter weights into base model
print("Merging LoRA weights...")
model = model.merge_and_unload()

# Save merged model
output_path = "models/weatherman-merged"
print(f"Saving merged model to: {output_path}")
model.save_pretrained(output_path)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_path)

print("✓ Merge complete!")
print(f"Merged model saved to: {output_path}")
```

Run the script:

```bash
# Install dependencies
pip install transformers peft accelerate torch

# Run merge
python scripts/merge_lora_adapter.py

# Wait 5-10 minutes for merge to complete
# Output: models/weatherman-merged/ (~14GB)
```

---

## Step 4: Convert to GGUF for Ollama

Ollama uses GGUF format optimized for Apple Silicon.

### Install llama.cpp

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with Metal support (optimized for M4)
make clean
LLAMA_METAL=1 make -j

# Verify Metal compilation
./main --version
# Should mention "Metal" in build info
```

### Convert Model to GGUF

```bash
# Install Python dependencies for conversion
pip install -r requirements.txt

# Convert merged model to GGUF
python convert.py \
  --model ../models/weatherman-merged \
  --outtype f16 \
  --outfile ../models/weatherman-f16.gguf

# Expected output: ../models/weatherman-f16.gguf (~14GB)
```

### Quantize for Better Performance (Optional)

Quantization reduces model size and improves inference speed on M4:

```bash
# Q4_K_M: Best balance of speed and quality (recommended for M4)
./quantize ../models/weatherman-f16.gguf \
  ../models/weatherman-q4.gguf Q4_K_M

# Q5_K_M: Better quality, slightly slower
./quantize ../models/weatherman-f16.gguf \
  ../models/weatherman-q5.gguf Q5_K_M

# Q8_0: Highest quality, slower
./quantize ../models/weatherman-f16.gguf \
  ../models/weatherman-q8.gguf Q8_0
```

**Quantization Trade-offs for M4:**

| Quantization | Size  | Quality | Speed on M4 | Recommended |
|--------------|-------|---------|-------------|-------------|
| F16 (full)   | 14GB  | Best    | Slow        | 32GB+ RAM   |
| Q8_0         | 7.5GB | Great   | Medium      | 16GB+ RAM   |
| Q5_K_M       | 5GB   | Good    | Fast        | 16GB RAM    |
| Q4_K_M       | 4GB   | Fair    | Fastest     | ✅ Best for M4 |

---

## Step 5: Create Ollama Modelfile

Create `Modelfile` in project root:

```dockerfile
# Modelfile for Weatherman LoRA
FROM ./models/weatherman-q4.gguf

# Model parameters optimized for M4
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER num_gpu 1

# System prompt
SYSTEM """You are a weather assistant with Mark Twain's wit and style. After using weather tools, respond with Twain's characteristic humor, irony, and folksy wisdom.

You have access to weather tools to answer user queries. Always use the appropriate tool before providing your response."""

# Chat template for Mistral
TEMPLATE """{{ if .System }}<s>[INST] {{ .System }}

{{ .Prompt }} [/INST]{{ else }}<s>[INST] {{ .Prompt }} [/INST]{{ end }}{{ .Response }}</s>"""
```

---

## Step 6: Import Model to Ollama

```bash
# Create Ollama model from Modelfile
ollama create weatherman -f Modelfile

# Verify model was created
ollama list

# Expected output:
# NAME           ID              SIZE    MODIFIED
# weatherman     abc123def456    4.2GB   2 minutes ago
```

---

## Step 7: Test Inference on M4

### Interactive Chat

```bash
# Start interactive chat
ollama run weatherman

# Try these prompts:
>>> What's the weather in Boston?
>>> Give me a 7-day forecast for Seattle
>>> Is it going to rain in London tomorrow?

# Exit with /bye or Ctrl+D
```

### API Testing

```bash
# Test via Ollama API
curl http://localhost:11434/api/generate -d '{
  "model": "weatherman",
  "prompt": "What is the weather forecast for San Francisco this week?",
  "stream": false
}'
```

### Python Client

```python
# test_ollama.py
from ollama import Client

client = Client()

# Test basic query
response = client.generate(
    model='weatherman',
    prompt='What is the weather in New York City right now?'
)

print(response['response'])
```

---

## Performance Optimization for M4

### 1. Enable Metal GPU Acceleration

Ollama automatically uses Metal on Apple Silicon. Verify with:

```bash
# Check GPU usage during inference
sudo powermetrics --samplers gpu_power -n 1

# You should see GPU activity when running inference
```

### 2. Adjust Model Parameters

For faster inference on M4, tune these parameters in Modelfile:

```dockerfile
# Smaller context for faster inference
PARAMETER num_ctx 2048    # Default: 4096

# Reduce batch size
PARAMETER num_batch 256   # Default: 512

# Optimize thread count for M4 (adjust based on M4 variant)
PARAMETER num_thread 8    # M4 Pro/Max: 8-12, M4: 4-8
```

### 3. Monitor Performance

```bash
# Check inference speed
time ollama run weatherman "What's the weather?"

# Expected on M4:
# - Q4 model: 30-50 tokens/second
# - Q5 model: 25-40 tokens/second
# - Q8 model: 15-25 tokens/second
```

### 4. Memory Management

```bash
# Check memory usage
ollama ps

# Unload model when not in use
ollama stop weatherman

# Or set auto-unload timeout (in Modelfile)
PARAMETER keep_alive 5m   # Auto-unload after 5 minutes idle
```

---

## Integration Examples

### Use with Continue.dev (VS Code)

```json
// .continue/config.json
{
  "models": [
    {
      "title": "Weatherman LoRA",
      "provider": "ollama",
      "model": "weatherman",
      "apiBase": "http://localhost:11434"
    }
  ]
}
```

### Use with LangChain

```python
from langchain.llms import Ollama

llm = Ollama(
    model="weatherman",
    temperature=0.7,
    base_url="http://localhost:11434"
)

response = llm("What's the weather forecast for Tokyo?")
print(response)
```

### Use with llama.cpp Server

```bash
# Alternative to Ollama: Run llama.cpp server directly
cd llama.cpp
./server -m ../models/weatherman-q4.gguf \
  -c 4096 \
  --port 8080 \
  --n-gpu-layers 35

# API endpoint: http://localhost:8080/v1/chat/completions
```

---

## Troubleshooting

### "Model too large for memory"

- Use smaller quantization (Q4_K_M instead of Q8_0)
- Reduce `num_ctx` parameter
- Close other applications
- Consider upgrading to 32GB unified memory for full F16 model

### "Slow inference speed"

- Use Q4_K_M quantization (fastest)
- Reduce context length: `PARAMETER num_ctx 2048`
- Ensure Metal GPU is being used (check with `powermetrics`)
- Close background apps to free up memory bandwidth

### "LoRA adapter not loading"

- Ensure `adapter_model.safetensors` or `adapter_model.bin` exists
- Check adapter was trained with compatible base model
- Verify adapter_config.json has correct `base_model_name_or_path`

### "Responses don't match training style"

- Ensure you merged the LoRA adapter (Step 3)
- Check system prompt is set correctly in Modelfile
- Try Q5 or Q8 quantization (Q4 may lose some style nuances)

---

## M4-Specific Performance Tips

1. **Use Metal GPU**: Ollama automatically uses Metal. Don't override with CPU.

2. **Optimal Quantization**: Q4_K_M is the sweet spot for M4 (4GB, fast inference).

3. **Memory Bandwidth**: M4's unified memory provides ~100-200GB/s bandwidth. Keep total model + context + overhead under 80% of available RAM.

4. **Context Length**: For most queries, 2048 tokens is sufficient and 2x faster than 4096.

5. **Model Size vs RAM**:
   - 8GB M4: Use Q4 models only
   - 16GB M4: Q4 or Q5 models
   - 24GB M4: Q5 or Q8 models
   - 32GB M4: F16 full precision possible

6. **Power Management**: Plug in to AC power for consistent performance. Battery mode throttles GPU.

---

## Next Steps

1. Test with sample weather queries (see `docs/DEPLOYMENT.md` Section 4)
2. Validate tool calling works correctly
3. Test all three personas (Twain, Franklin, Onion)
4. Integrate into your application
5. Monitor inference latency and memory usage

For general deployment options (AnythingLLM, production API, etc.), see `docs/DEPLOYMENT.md`.

---

## Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [llama.cpp Metal Performance](https://github.com/ggerganov/llama.cpp/discussions/2407)
- [Apple Silicon ML Performance Guide](https://developer.apple.com/metal/pytorch/)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

For questions or issues, see main project README.
