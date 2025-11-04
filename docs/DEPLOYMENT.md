# Weatherman-LoRA Deployment Guide

This guide covers deploying your trained LoRA adapter for inference with AnythingLLM, Ollama, or other LLM platforms.

## Prerequisites

- Trained LoRA adapter (from H100 or M4 training)
- Base model: Mistral 7B Instruct v0.3 or Llama 3.1 8B
- Deployment platform: AnythingLLM, Ollama, or compatible LLM framework

---

## Section 1: Download Trained Adapter

### From RunPod H100 to Local Machine

After training completes on RunPod, download the adapter to your local machine:

```bash
# Using SCP (replace with your RunPod instance details)
scp -r user@runpod-instance:/workspace/Weatherman-LoRA/adapters/weatherman-lora-h100 ./adapters/

# Or using rsync for better performance
rsync -avz --progress user@runpod-instance:/workspace/Weatherman-LoRA/adapters/weatherman-lora-h100/ ./adapters/weatherman-lora-h100/
```

### From M4 Local Training

If you trained locally on M4, the adapter is already at:
```
./adapters/weatherman-lora-m4/
```

### Expected Files

Your adapter directory should contain:

```
adapters/weatherman-lora-h100/  (or weatherman-lora-m4/)
├── adapter_config.json         # LoRA configuration
├── adapter_model.bin            # LoRA weights (~500MB-2GB)
├── adapter_model.safetensors    # Alternative format
├── tokenizer.json               # Tokenizer configuration
├── tokenizer_config.json        # Tokenizer settings
├── special_tokens_map.json      # Special tokens
├── training_args.bin            # Training configuration
└── checkpoint-*/                # Intermediate checkpoints (optional)
```

**File Size:** Expect 500MB-2GB depending on LoRA rank and base model size.

---

## Section 2: Using with AnythingLLM

[AnythingLLM](https://anythingllm.com/) provides a user-friendly interface for running LLMs with LoRA adapters.

### Installation

```bash
# Linux/Mac
curl -sSL https://install.anythingllm.com | bash

# Or download desktop app from:
# https://anythingllm.com/download
```

### Setup Steps

1. **Launch AnythingLLM**
   ```bash
   anythingllm
   ```

2. **Load Base Model**
   - Navigate to Settings → LLM Provider
   - Select "Local LLM" or "Ollama"
   - Choose base model: `mistralai/Mistral-7B-Instruct-v0.3`
   - Wait for model download (7-14GB)

3. **Add LoRA Adapter**
   - Go to Settings → Advanced → LoRA Adapters
   - Click "Add LoRA Adapter"
   - Point to: `./adapters/weatherman-lora-h100/`
   - Name: "Weatherman LoRA"
   - Click "Load Adapter"

4. **Configure Workspace**
   - Create new workspace: "Weather Assistant"
   - Select "Weatherman LoRA" as active adapter
   - Set system prompt (optional):
     ```
     You are a weather assistant with a humorous personality.
     Use weather tools to provide accurate forecasts with wit.
     ```

5. **Test Query**
   ```
   User: What's the weather in Boston?

   Expected: Tool call to get_current_weather with proper parameters,
            followed by response in trained style (Twain/Franklin/Onion)
   ```

### Troubleshooting

- **Adapter not loading:** Ensure `adapter_model.bin` or `adapter_model.safetensors` exists
- **Out of memory:** Reduce context length or use quantization (4-bit/8-bit)
- **Slow inference:** Enable GPU acceleration in settings

---

## Section 3: Using with Ollama

[Ollama](https://ollama.ai/) is a lightweight LLM runtime optimized for local inference.

### Installation

```bash
# Mac
curl https://ollama.ai/install.sh | sh

# Linux
curl https://ollama.ai/install.sh | sh

# Or download from: https://ollama.ai/download
```

### GGUF Conversion

Ollama requires GGUF format. Convert your LoRA adapter:

```bash
# Install llama.cpp (if not already installed)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert LoRA to GGUF
python convert.py \
  --model-path ../adapters/weatherman-lora-h100 \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --output-type gguf \
  --output-file ../models/weatherman-lora.gguf
```

### Create Modelfile

Create `Modelfile` in project root:

```dockerfile
FROM models/weatherman-lora.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

SYSTEM """
You are a weather assistant with a humorous personality.
Use weather tools to provide accurate forecasts with wit.
"""

TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ end }}{{ .Response }}<|end|>"""
```

### Import to Ollama

```bash
# Create Ollama model
ollama create weatherman -f Modelfile

# Verify creation
ollama list

# Expected output:
# NAME                TAGS    SIZE     MODIFIED
# weatherman          latest  4.2GB    2 minutes ago
```

### Test Inference

```bash
# Interactive mode
ollama run weatherman "What's the weather in Seattle?"

# API mode (programmatic)
curl http://localhost:11434/api/generate -d '{
  "model": "weatherman",
  "prompt": "What is the weather forecast for New York this week?",
  "stream": false
}'
```

### Quantization Options

For faster inference or lower memory usage:

```bash
# 4-bit quantization (smallest, fastest)
python convert.py --quant q4_0 --output weatherman-q4.gguf

# 5-bit quantization (balanced)
python convert.py --quant q5_1 --output weatherman-q5.gguf

# 8-bit quantization (best quality)
python convert.py --quant q8_0 --output weatherman-q8.gguf
```

**Trade-offs:**
- Q4: 4GB, fastest, slight quality loss
- Q5: 5GB, balanced
- Q8: 7GB, best quality, slower

---

## Section 4: Sample Prompts

Test your deployed model with these sample prompts:

### Basic Weather Queries

```
1. What's the weather in Boston?
   → Should call get_current_weather(location="Boston, MA")

2. Give me a 7-day forecast for Seattle
   → Should call get_forecast(location="Seattle, WA", days=7)

3. Is it going to rain in London tomorrow?
   → Should call get_forecast(location="London, UK", days=1)

4. Compare weather in Miami and Denver
   → Should call get_current_weather twice
```

### Multi-Turn Conversations

```
User: What's the weather like in Chicago?
AI: [Calls tool, responds with current weather]

User: How about the forecast for this weekend?
AI: [Calls forecast tool for Chicago, responds]

User: Should I bring an umbrella?
AI: [Uses previous forecast data to advise]
```

### Error Handling

```
1. What's the weather on Mars?
   → Should gracefully explain data unavailable

2. Give me weather for XYZ123
   → Should handle invalid location

3. Show me the forecast for next month
   → Should explain 14-day limit
```

### Persona-Specific Responses

#### Mark Twain Style
```
User: Is it hot in Phoenix today?
AI: "Reports of Phoenix's heat have not been greatly exaggerated.
    Currently sitting at a biblical 108°F - hot enough to make
    the devil himself consider relocating."
```

#### Benjamin Franklin Style
```
User: Will it rain tomorrow?
AI: "Early to bed and early to rise makes a man healthy, wealthy,
    and dry - which you shall be, for the forecast shows naught
    but sunshine for the morrow."
```

#### The Onion Style
```
User: What's the weather in NYC?
AI: "BREAKING: New York City Weather Still Exists,
    Continues To Have Temperature (Currently 72°F)"
```

### Tool Use Validation

```
1. "Get me weather data for coordinates 40.7128, -74.0060"
   → Should call get_current_weather(latitude=40.7128, longitude=-74.0060)

2. "3-day forecast for Tokyo"
   → Should call get_forecast(location="Tokyo, Japan", days=3)

3. "Historical weather trends"
   → Should explain no historical data available
```

---

## Performance Optimization

### For AnythingLLM

```python
# Recommended settings for smooth inference
{
  "context_length": 4096,
  "batch_size": 8,
  "gpu_layers": 35,      # Adjust based on VRAM
  "use_mlock": true,     # Prevent swapping
  "threads": 8           # CPU threads
}
```

### For Ollama

```bash
# Environment variables
export OLLAMA_NUM_GPU=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1

# Launch with optimizations
ollama serve
```

### Memory Requirements

| Configuration | VRAM/RAM | Batch Size | Speed |
|--------------|----------|------------|-------|
| FP16 (full)  | 16GB     | 8-16       | Fast  |
| INT8         | 10GB     | 8-12       | Med   |
| INT4         | 6GB      | 4-8        | Slow  |

---

## Troubleshooting

### Common Issues

**1. "Adapter not compatible with base model"**
- Verify base model matches training config
- Check `adapter_config.json` for `base_model_name_or_path`

**2. "Out of memory during inference"**
- Reduce batch size
- Use quantized model (INT8/INT4)
- Decrease context length

**3. "Tool calls not working"**
- Ensure model was trained with tool-use examples
- Check prompt format matches training format
- Verify tool definitions are provided to LLM

**4. "Responses lack trained style"**
- LoRA adapter may not be loaded
- Check adapter weight in inference (should be > 0)
- Verify `adapter_model.bin` is present

**5. "Slow inference on M4"**
- MPS backend is 2-3x slower than CUDA
- Consider using quantized GGUF format
- Close background applications

### Validation Checklist

Before deploying to production:

- [ ] Test basic weather queries
- [ ] Verify tool calls work correctly
- [ ] Check response style matches training
- [ ] Test error handling
- [ ] Measure inference latency (<2s for short prompts)
- [ ] Validate memory usage is acceptable
- [ ] Test multi-turn conversations
- [ ] Verify all three personas (Twain/Franklin/Onion)

---

## Production Deployment

For production use:

1. **API Wrapper**
   ```python
   # Example FastAPI wrapper
   from fastapi import FastAPI
   from ollama import Client

   app = FastAPI()
   client = Client()

   @app.post("/weather")
   async def get_weather(query: str):
       response = client.generate(
           model="weatherman",
           prompt=query
       )
       return {"response": response["response"]}
   ```

2. **Load Balancing**
   - Use multiple Ollama instances
   - Round-robin requests
   - Monitor response times

3. **Monitoring**
   - Track inference latency
   - Monitor memory usage
   - Log tool call accuracy
   - Capture user feedback

4. **Scaling**
   - Start with single instance
   - Add instances based on load
   - Consider GPU inference servers for high throughput

---

## Additional Resources

- [AnythingLLM Documentation](https://docs.anythingllm.com/)
- [Ollama Documentation](https://github.com/ollama/ollama/tree/main/docs)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Mistral Documentation](https://docs.mistral.ai/)

For questions or issues, see project repository: [Your GitHub Link]
