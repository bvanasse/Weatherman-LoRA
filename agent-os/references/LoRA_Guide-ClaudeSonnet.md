# LoRA Fine-Tuning Training Plan: Humor-Enhanced Weather Bot
## A 3-Day Weekend Project for AI-Powered Humorist Tool Calling

---

## TLDR

**Goal**: Fine-tune a small language model (3-8B parameters) using LoRA to create a specialized bot that speaks like Mark Twain & Benjamin Franklin, tells weather jokes, and expertly uses weather APIs.

**Timeline**: 3 days (weekend project)

**Hardware**: Up to 8x H100 GPUs available (though 1-2 will suffice for this project)

**Recommended Model**: **Llama 3.2 3B** or **Phi-4 (16B)** over IBM Granite 4
- Llama 3.2: Best balance of performance, community support, and efficiency
- Phi-4: Superior reasoning if you want to push to 16B parameters
- Both have excellent fine-tuning documentation and LoRA support

**Training Framework**: Unsloth + LLaMA-Factory (optimal for speed and efficiency)

**Expected Outcome**: A 3-8B parameter model that combines classical American humor with modern tool-calling capabilities, demonstrating the impact of specialized training data on model behavior.

---

## Overview

This project demonstrates how focused dataset curation and LoRA fine-tuning can imbue a small language model with both stylistic characteristics (humorous, witty writing in the style of 19th-century American authors) and functional capabilities (weather API tool calling). The result is a unique "personality + utility" bot that showcases data management's impact on specialized model training.

### Why This Approach Works

1. **LoRA Efficiency**: Parameter-efficient fine-tuning allows training large models on limited hardware
2. **Dual-Focus Dataset**: Combining literary style (Mark Twain, Ben Franklin, The Onion) with functional data (weather API calls) creates a unique hybrid capability
3. **Small Model Advantage**: 3-8B models train faster, deploy easily, and with proper fine-tuning can rival much larger models on specialized tasks
4. **Practical Demonstration**: Shows how data composition directly influences model outputs

---

## Phase 1: Environment Setup & Data Collection (Day 1 - Morning)
**Duration**: 4-6 hours

### 1.1 Development Environment Setup

#### GPU & Framework Configuration
```bash
# Install core dependencies
pip install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --break-system-packages unsloth transformers accelerate peft trl
pip install --break-system-packages datasets wandb

# Install LLaMA-Factory for advanced training
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install --break-system-packages -e .

# Verify GPU availability
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

#### Model Selection & Download

**Primary Recommendation: Llama 3.2 3B**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 2048,
    dtype = None,  # Auto-detect
    load_in_4bit = True,  # Use 4-bit quantization for efficiency
)
```

**Alternative: Phi-4 (16B) for Higher Performance**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "microsoft/phi-4",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
```

**Why Not Granite 4?**
- While IBM Granite 4 is capable, Llama 3.2 and Phi-4 have:
  - Better community support and documentation
  - More extensive fine-tuning examples
  - Proven function-calling capabilities
  - Superior benchmark performance on humor/reasoning tasks

### 1.2 Data Collection Strategy

#### Dataset 1: The Onion (Satirical Humor)

**Source**: Hugging Face Dataset + Web Scraping
```python
from datasets import load_dataset

# Pre-existing Onion dataset
onion_dataset = load_dataset("Biddls/Onion_News")
# Contains ~33,880 articles with headers and body text

# Additional scraping script (if needed for fresh content)
"""
Use the provided scraper from the Biddls dataset repo:
- Approximately 30 minutes on modern hardware
- Respects robots.txt and rate limits
"""
```

**Processing Steps**:
1. Filter for articles with clear satirical structure
2. Extract headline + first 2-3 paragraphs (optimal for training)
3. Tag with "humor" label
4. Target: **5,000-10,000 high-quality examples**

**Data Format**:
```json
{
  "instruction": "Write a humorous news headline and opening paragraph about weather:",
  "input": "",
  "output": "Area Man Reports Moderate Enthusiasm for Partly Cloudy Skies\n\nLOCAL SOURCES—Citing feelings of cautious optimism mixed with profound indifference, area man Dale Hendricks, 34, reported to friends and family Thursday that he is moderately enthusiastic about this weekend's partly cloudy skies..."
}
```

#### Dataset 2: Mark Twain Writings

**Source**: Project Gutenberg
```python
from gutenbergr import gutenberg_works, gutenberg_download

# Mark Twain's complete works (164 texts available)
twain_books = gutenberg_works(author == "Twain, Mark")

# Priority downloads:
priority_titles = [
    "The Adventures of Tom Sawyer",
    "Adventures of Huckleberry Finn",
    "A Connecticut Yankee in King Arthur's Court",
    "The Innocents Abroad",
    "Life on the Mississippi",
    "Following the Equator",
    "The Prince and the Pauper",
    "Mark Twain's Letters"
]
```

**Processing Steps**:
1. Download complete texts from Project Gutenberg
2. Chunk into 250-500 word segments
3. Filter for passages with humor, wit, or distinctive voice
4. Create instruction-completion pairs
5. Target: **3,000-5,000 examples**

**Data Format**:
```json
{
  "instruction": "Rewrite this statement in the style of Mark Twain:",
  "input": "The weather forecast predicts rain tomorrow.",
  "output": "The prophets of the Weather Bureau have decreed that tomorrow shall witness a deluge—though their past prophecies suggest this prediction enjoys about the same reliability as a politician's promise or a cat's affection."
}
```

#### Dataset 3: Benjamin Franklin - Poor Richard's Almanack

**Source**: Founders Online Archive + Project Gutenberg
```python
# Available at: https://founders.archives.gov/documents/Franklin/
# All 25 years of Poor Richard's Almanack (1733-1758)
```

**Processing Steps**:
1. Extract proverbs, maxims, and weather-related content
2. Extract witty observations and advice
3. Create style-matching tasks
4. Target: **1,000-2,000 examples**

**Key Content Types**:
- Proverbs: "Early to bed and early to rise, makes a man healthy, wealthy and wise"
- Weather wisdom: Almanack predictions with wit
- Practical advice with humor

**Data Format**:
```json
{
  "instruction": "Provide advice about weather preparedness in the style of Poor Richard:",
  "input": "",
  "output": "He that waits for fair weather to travel may never begin his journey. Better to brave a storm with preparation than fear a cloud without one. As Poor Richard says, 'Lost time is never found again'—and neither is the man who checked the forecast but forgot his coat."
}
```

#### Dataset 4: Weather API Function Calling

**Source**: Synthetic Generation + Manual Curation
```python
# Create function calling examples for weather APIs
weather_functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g. New York, NY"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_weather_forecast",
        "description": "Get weather forecast for next N days",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 10}
            },
            "required": ["location", "days"]
        }
    }
]
```

**Processing Steps**:
1. Generate 500-1,000 diverse weather query examples
2. Mix formal and colloquial requests
3. Include edge cases (ambiguous locations, multiple queries)
4. Add humor-enhanced variations
5. Target: **1,000-1,500 examples**

**Data Format**:
```json
{
  "instruction": "You are an AI assistant with access to weather tools. Respond to the user's request.",
  "input": "What's the weather like in Boston? I need to know if I should bring an umbrella to the Common.",
  "output": "<function_call>\n{\n  \"name\": \"get_current_weather\",\n  \"parameters\": {\"location\": \"Boston, MA\", \"unit\": \"fahrenheit\"}\n}\n</function_call>",
  "system": "You have access to weather APIs. Use function calling when needed."
}
```

#### Dataset 5: Hybrid Humor + Weather Examples

**Source**: Manual Creation + GPT-4 Synthesis
```python
# Create examples that combine both capabilities
# Use GPT-4 to generate training data in the target style
```

**Processing Steps**:
1. Create 300-500 examples mixing humor with function calls
2. Show model how to inject personality into API responses
3. Demonstrate weather jokes based on API results

**Data Format**:
```json
{
  "instruction": "Respond to the weather query with personality and humor in Mark Twain's style:",
  "input": "What's the forecast for this weekend in St. Louis?",
  "output": "Allow me to consult the meteorological oracles...\n<function_call>{\"name\": \"get_weather_forecast\", \"parameters\": {\"location\": \"St. Louis, MO\", \"days\": 3}}</function_call>\n\nAh, the forecast suggests Saturday shall be blessed with sunshine and temperatures in the agreeable seventies—a rare gift from the Weather Bureau, whose predictions I trust about as much as I trust a riverboat gambler's smile. Sunday threatens rain, which means it will either pour buckets or remain stubbornly dry, as weather forecasts are more art than science and more fiction than art."
}
```

---

## Phase 2: Data Processing & Preparation (Day 1 - Afternoon)
**Duration**: 4-6 hours

### 2.1 Data Cleaning & Standardization

```python
import pandas as pd
from datasets import Dataset, concatenate_datasets

class DataProcessor:
    def __init__(self):
        self.onion_data = []
        self.twain_data = []
        self.franklin_data = []
        self.weather_api_data = []
        self.hybrid_data = []
    
    def clean_onion_articles(self, dataset):
        """Extract and format Onion articles"""
        cleaned = []
        for item in dataset:
            # Parse header and body (split by #~# token)
            parts = item['text'].split(' #~# ')
            if len(parts) == 2:
                header, body = parts
                cleaned.append({
                    'instruction': 'Write a satirical news article about:',
                    'input': self.extract_topic(header),
                    'output': f"{header}\n\n{body[:500]}..."  # Limit length
                })
        return cleaned
    
    def chunk_twain_text(self, text, chunk_size=400):
        """Split Twain texts into trainable chunks"""
        # Implement sentence-aware chunking
        # Preserve paragraph boundaries
        # Filter for humor/wit markers
        pass
    
    def format_weather_api_examples(self, examples):
        """Convert to unified function-calling format"""
        formatted = []
        for ex in examples:
            formatted.append({
                'instruction': ex['user_query'],
                'output': self.format_function_call(ex['function_name'], ex['parameters']),
                'system': 'You are a helpful assistant with weather API access.'
            })
        return formatted
    
    def combine_datasets(self):
        """Merge all datasets with proper balancing"""
        # Ratio: 40% humor (Onion/Twain/Franklin), 35% weather API, 25% hybrid
        return concatenate_datasets([
            Dataset.from_list(self.onion_data[:4000]),
            Dataset.from_list(self.twain_data[:2500]),
            Dataset.from_list(self.franklin_data[:1500]),
            Dataset.from_list(self.weather_api_data[:3500]),
            Dataset.from_list(self.hybrid_data[:2500])
        ])

# Execute processing
processor = DataProcessor()
final_dataset = processor.combine_datasets()

# Save processed dataset
final_dataset.save_to_disk("/mnt/training_data/humor_weather_dataset")
```

### 2.2 Data Quality Assurance

```python
# Quality checks
def validate_dataset(dataset):
    """Ensure data quality before training"""
    checks = {
        'total_examples': len(dataset),
        'avg_input_length': sum(len(ex['input']) for ex in dataset) / len(dataset),
        'avg_output_length': sum(len(ex['output']) for ex in dataset) / len(dataset),
        'function_call_ratio': sum('function_call' in ex['output'] for ex in dataset) / len(dataset),
        'humor_ratio': sum(any(keyword in ex['instruction'].lower() 
                               for keyword in ['humor', 'twain', 'satirical', 'joke']) 
                           for ex in dataset) / len(dataset)
    }
    
    print("Dataset Statistics:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    
    return checks

validate_dataset(final_dataset)
```

### 2.3 Train/Validation Split

```python
# 90/10 split with stratification
train_test = final_dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = train_test['train']
eval_dataset = train_test['test']

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(eval_dataset)}")
```

---

## Phase 3: LoRA Configuration & Training (Day 2 - Full Day)
**Duration**: 8-10 hours (mostly automated training time)

### 3.1 LoRA Configuration

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer

# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=32,                          # Rank: higher = more capacity (16-64 typical)
    lora_alpha=64,                 # Scaling factor (typically 2x rank)
    target_modules=[               # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,             # Regularization
    bias="none",                   # Don't adapt bias terms
    task_type="CAUSAL_LM"          # Language modeling task
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~1-2% of total parameters trainable
```

### 3.2 Training Configuration

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # Output & Logging
    output_dir="/mnt/training_output/humor_weather_lora",
    run_name="twain-weather-bot-v1",
    logging_dir="./logs",
    logging_steps=10,
    
    # Training Schedule
    num_train_epochs=3,            # 2-4 epochs typical for LoRA
    per_device_train_batch_size=4, # Adjust based on GPU memory
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, # Effective batch size = 4*4 = 16
    
    # Optimization
    learning_rate=2e-4,            # LoRA typically uses higher LR than full fine-tuning
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=0.3,
    
    # Efficiency
    fp16=False,
    bf16=True,                     # Use bfloat16 on H100s
    optim="paged_adamw_8bit",      # Memory-efficient optimizer
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Experiment Tracking
    report_to="wandb",             # Use Weights & Biases for monitoring
)
```

### 3.3 Training Execution

```python
from trl import SFTTrainer

# Format dataset for training
def formatting_func(example):
    """Convert examples to training format"""
    text = f"""<|system|>
{example.get('system', 'You are a helpful AI assistant with weather API access. You speak with wit and humor in the style of Mark Twain and Benjamin Franklin.')}
<|user|>
{example['instruction']}
{example.get('input', '')}
<|assistant|>
{example['output']}<|endoftext|>"""
    return text

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    formatting_func=formatting_func,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,  # Don't pack multiple examples per sequence
)

# Start training
print("🚀 Beginning training...")
trainer.train()

# Save final model
trainer.save_model("/mnt/training_output/humor_weather_lora/final_model")
print("✅ Training complete!")
```

### 3.4 Training Monitoring

**Key Metrics to Watch**:
- **Training Loss**: Should decrease steadily (target: <0.5 by end)
- **Eval Loss**: Should track training loss without diverging (watch for overfitting)
- **Perplexity**: Should decrease (good target: <10)
- **GPU Utilization**: Should be 80-95% during training

**Expected Timeline**:
- With 1x H100: ~6-8 hours for 3 epochs on 12-14k examples
- With 2x H100: ~3-4 hours (data parallelism)
- With 4x H100: ~2-3 hours

**Weights & Biases Dashboard**:
```bash
# View training progress
wandb login
# Navigate to wandb.ai and monitor:
# - Loss curves
# - Learning rate schedule
# - Gradient norms
# - Sample generations
```

---

## Phase 4: Evaluation & Testing (Day 3 - Morning)
**Duration**: 4-5 hours

### 4.1 Quantitative Evaluation

```python
# Evaluate on held-out test set
eval_results = trainer.evaluate(eval_dataset)

print("Evaluation Results:")
print(f"  Eval Loss: {eval_results['eval_loss']:.4f}")
print(f"  Perplexity: {eval_results['eval_perplexity']:.2f}")
```

### 4.2 Qualitative Testing

```python
from unsloth import FastLanguageModel

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/mnt/training_output/humor_weather_lora/final_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Test cases
test_prompts = [
    # Pure humor test
    {
        "prompt": "Write a humorous observation about weather forecasters:",
        "expected_style": "Twain/Franklin wit"
    },
    
    # Pure function calling test
    {
        "prompt": "What's the current weather in Seattle?",
        "expected_output": "function_call with get_current_weather"
    },
    
    # Hybrid test
    {
        "prompt": "I'm planning a picnic in Chicago this weekend. Can you check the forecast and give me advice?",
        "expected_behavior": "Function call + humorous response"
    },
    
    # Edge case
    {
        "prompt": "Tell me about the weather in an amusing way, but I don't need the actual forecast.",
        "expected_behavior": "Humor without function call"
    }
]

def test_model(prompt):
    """Generate and display model output"""
    inputs = tokenizer([f"<|user|>\n{prompt}\n<|assistant|>\n"], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Run tests
for i, test in enumerate(test_prompts):
    print(f"\n{'='*60}")
    print(f"Test {i+1}: {test['expected_style'] or test['expected_behavior']}")
    print(f"{'='*60}")
    print(f"Prompt: {test['prompt']}")
    print(f"\nModel Response:\n{test_model(test['prompt'])}")
```

### 4.3 A/B Comparison with Base Model

```python
# Load base model for comparison
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Run same prompts on base model
print("\n" + "="*60)
print("COMPARISON: Base Model vs Fine-Tuned Model")
print("="*60)

comparison_prompt = "What's the weather forecast for New York City this weekend?"

print(f"\n📝 Prompt: {comparison_prompt}")
print(f"\n🤖 Base Model Response:")
print(test_model_base(comparison_prompt, base_model, base_tokenizer))
print(f"\n✨ Fine-Tuned Model Response:")
print(test_model(comparison_prompt))
```

### 4.4 Function Calling Accuracy Test

```python
import json
import re

def extract_function_calls(text):
    """Parse function calls from model output"""
    pattern = r'<function_call>\s*({.*?})\s*</function_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    return [json.loads(match) for match in matches]

def validate_function_call(call, expected_function):
    """Check if function call is valid"""
    checks = {
        'has_name': 'name' in call,
        'correct_function': call.get('name') == expected_function,
        'has_parameters': 'parameters' in call,
        'valid_parameters': validate_parameters(call.get('parameters', {}))
    }
    return all(checks.values()), checks

# Test function calling accuracy
weather_queries = [
    ("What's the temperature in Miami?", "get_current_weather"),
    ("Give me a 5-day forecast for Denver", "get_weather_forecast"),
    ("Is it raining in Portland right now?", "get_current_weather"),
]

accuracy_results = []
for query, expected_func in weather_queries:
    response = test_model(query)
    calls = extract_function_calls(response)
    if calls:
        is_valid, details = validate_function_call(calls[0], expected_func)
        accuracy_results.append(is_valid)
        print(f"✓ Query: {query}")
        print(f"  Valid: {is_valid} | Function: {calls[0].get('name')}")
    else:
        accuracy_results.append(False)
        print(f"✗ Query: {query} | No function call generated")

print(f"\nFunction Calling Accuracy: {sum(accuracy_results)/len(accuracy_results)*100:.1f}%")
```

---

## Phase 5: Deployment & Iteration (Day 3 - Afternoon)
**Duration**: 3-4 hours

### 5.1 Model Export

```python
# Export merged model (base + LoRA adapters)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("/mnt/models/twain-weather-bot-merged")
tokenizer.save_pretrained("/mnt/models/twain-weather-bot-merged")

# Export to GGUF for llama.cpp deployment
merged_model.save_pretrained_gguf(
    "/mnt/models/twain-weather-bot-gguf",
    tokenizer,
    quantization_method="q4_k_m"  # 4-bit quantization
)

# Export LoRA adapters only (for flexible deployment)
model.save_pretrained("/mnt/models/twain-weather-lora-adapters")
```

### 5.2 Deployment Options

**Option A: Local API Server**
```python
# Using vLLM for high-performance inference
from vllm import LLM, SamplingParams

llm = LLM(
    model="/mnt/models/twain-weather-bot-merged",
    tensor_parallel_size=1,  # Use 1 GPU
    dtype="bfloat16"
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

def generate_response(prompt):
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

# Wrap in FastAPI for REST endpoint
from fastapi import FastAPI
app = FastAPI()

@app.post("/generate")
def generate(prompt: str):
    return {"response": generate_response(prompt)}
```

**Option B: Ollama Integration**
```bash
# Create Ollama modelfile
cat > Modelfile << EOF
FROM /mnt/models/twain-weather-bot-gguf/model-q4_k_m.gguf

SYSTEM """You are an AI assistant with weather API access. You speak with wit and humor in the style of Mark Twain and Benjamin Franklin. When asked about weather, use function calling to retrieve current data, then respond with entertaining commentary."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Create Ollama model
ollama create twain-weather-bot -f Modelfile

# Test
ollama run twain-weather-bot "What's the weather in Boston?"
```

### 5.3 Performance Benchmarking

```python
import time
from statistics import mean, stdev

# Latency test
prompts = [
    "What's the weather in {city}?".format(city=city)
    for city in ["Boston", "Seattle", "Miami", "Denver", "Austin"]
]

latencies = []
for prompt in prompts:
    start = time.time()
    response = generate_response(prompt)
    latency = time.time() - start
    latencies.append(latency)
    print(f"Prompt: {prompt}")
    print(f"Latency: {latency:.3f}s")
    print(f"Response length: {len(response)} chars\n")

print(f"\nAverage Latency: {mean(latencies):.3f}s ± {stdev(latencies):.3f}s")
print(f"Throughput: {1/mean(latencies):.2f} requests/second")
```

### 5.4 Documentation & Sharing

```markdown
# Twain Weather Bot - Model Card

## Model Description
A 3B parameter language model fine-tuned with LoRA to combine:
- Humorous writing style inspired by Mark Twain & Benjamin Franklin
- Weather API function calling capabilities
- Satirical news writing (The Onion style)

## Training Data
- 4,000 satirical articles (The Onion)
- 2,500 Mark Twain excerpts
- 1,500 Benjamin Franklin maxims
- 3,500 weather API examples
- 2,500 hybrid humor+weather examples

**Total**: ~14,000 training examples

## Architecture
- Base Model: Llama 3.2 3B Instruct
- Training Method: LoRA (rank=32, alpha=64)
- Trainable Parameters: 1.8% of total
- Training Time: 6 hours on 1x H100 GPU

## Performance
- Function Call Accuracy: 94.2%
- Humor Style Match: Qualitative assessment - strong
- Inference Speed: ~50 tokens/second on single GPU

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("username/twain-weather-bot")
tokenizer = AutoTokenizer.from_pretrained("username/twain-weather-bot")

prompt = "What's the weather forecast for San Francisco?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## Limitations
- Weather API knowledge limited to training examples
- May occasionally generate overly formal or anachronistic language
- Function calling requires specific prompt formats

## Future Improvements
- Expand to multi-turn conversations
- Add more diverse humor styles
- Include additional API types (news, stocks, etc.)
```

---

## Success Metrics & Evaluation Criteria

### Quantitative Metrics
1. **Training Loss**: Target final loss < 0.5
2. **Evaluation Loss**: Should be within 10% of training loss
3. **Function Call Accuracy**: Target > 90% on weather queries
4. **Inference Speed**: Target > 40 tokens/second on single GPU

### Qualitative Metrics
1. **Style Consistency**: Generated text matches Twain/Franklin wit
2. **Humor Quality**: Satirical elements present and entertaining
3. **Response Coherence**: Logical flow from API call to humorous commentary
4. **Edge Case Handling**: Graceful degradation on ambiguous queries

---

## Troubleshooting Guide

### Issue: High Training Loss (>1.0 after epoch 1)
**Solutions**:
- Reduce learning rate to 1e-4
- Increase warmup steps
- Check data quality (inspect examples manually)
- Verify tokenizer is handling special tokens correctly

### Issue: Model Doesn't Generate Function Calls
**Solutions**:
- Increase proportion of function-calling examples (target 40-50%)
- Add more explicit system prompts during training
- Use temperature=0.3 during inference for more deterministic outputs
- Verify function call format matches training data exactly

### Issue: Humor Style Not Coming Through
**Solutions**:
- Increase literary dataset proportion
- Use lower temperature (0.6-0.7) to reduce randomness
- Add more "style transfer" examples (rewrite X in Twain's style)
- Consider longer training (4-5 epochs)

### Issue: Out of Memory During Training
**Solutions**:
- Reduce batch size to 2 (increase gradient accumulation)
- Enable gradient checkpointing
- Use deeper quantization (4-bit or even 3-bit)
- Reduce max_seq_length to 1024

---

## Resource Requirements

### Hardware
- **Minimum**: 1x GPU with 24GB VRAM (A30, A100, H100)
- **Recommended**: 2x H100 GPUs for faster training
- **Storage**: 100GB for datasets, models, and outputs

### Software
- Python 3.10+
- PyTorch 2.0+
- CUDA 12.1+
- Unsloth (latest)
- Transformers 4.40+
- LLaMA-Factory (optional but recommended)

### Time Investment
- **Day 1 Morning**: Environment + data collection (4-6 hours)
- **Day 1 Afternoon**: Data processing (4-6 hours)
- **Day 2**: Training (8-10 hours, mostly automated)
- **Day 3 Morning**: Evaluation (4-5 hours)
- **Day 3 Afternoon**: Deployment (3-4 hours)

**Total Active Work**: ~20 hours
**Total Elapsed Time**: 3 days (weekend project)

---

## Alternative Approaches

### If Time is Limited (1-2 Days)
1. Use pre-processed datasets (Hugging Face)
2. Skip manual data curation
3. Use smaller base model (1B parameters)
4. Train for 1-2 epochs only
5. Focus on one capability (humor OR tools, not both)

### If More Resources Available (1 Week)
1. Expand dataset to 50k+ examples
2. Train multiple model sizes (3B, 7B, 14B)
3. Implement RLHF for response quality
4. Add multi-turn conversation support
5. Build comprehensive evaluation suite
6. Create interactive demo (Gradio/Streamlit)

---

## Expected Outcomes

By the end of this 3-day project, you will have:

1. ✅ **A Unique Fine-Tuned Model** that demonstrates personality + functionality
2. ✅ **Hands-On LoRA Experience** with modern efficient training techniques
3. ✅ **Comprehensive Dataset** of 10k+ high-quality examples
4. ✅ **Performance Benchmarks** showing training data impact
5. ✅ **Deployment-Ready Artifacts** (merged model, GGUF, LoRA adapters)
6. ✅ **Documentation** of methodology and lessons learned

### Demonstrable Skills
- Data curation and preprocessing
- LoRA configuration and hyperparameter tuning
- Training monitoring and debugging
- Model evaluation and comparison
- Deployment and inference optimization

---

## Appendix: Code Snippets & Templates

### A. Complete Training Script
```python
#!/usr/bin/env python3
"""
Complete LoRA fine-tuning script for Twain Weather Bot
Usage: python train.py --config config.yaml
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

# Initialize W&B
wandb.init(project="twain-weather-bot", name="training-run-1")

# Load model
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load dataset
print("Loading dataset...")
dataset = load_from_disk("/mnt/training_data/humor_weather_dataset")
train_test = dataset.train_test_split(test_size=0.1)

# Training arguments
training_args = TrainingArguments(
    output_dir="/mnt/training_output/humor_weather_lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    report_to="wandb",
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_test['train'],
    eval_dataset=train_test['test'],
    peft_config=lora_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
)

# Train!
print("🚀 Starting training...")
trainer.train()

# Save
print("💾 Saving model...")
trainer.save_model("/mnt/training_output/humor_weather_lora/final_model")
print("✅ Complete!")
```

### B. Dataset Preparation Template
```python
# dataset_builder.py
from datasets import Dataset
import json

class HumorWeatherDatasetBuilder:
    def __init__(self):
        self.data = []
    
    def add_onion_article(self, headline, body):
        self.data.append({
            'instruction': 'Write a satirical news article:',
            'input': '',
            'output': f"{headline}\n\n{body}"
        })
    
    def add_twain_passage(self, text, topic):
        self.data.append({
            'instruction': f'Rewrite this {topic} in Mark Twain\'s style:',
            'input': topic,
            'output': text
        })
    
    def add_franklin_maxim(self, maxim):
        self.data.append({
            'instruction': 'Provide wise advice in Benjamin Franklin\'s style:',
            'input': '',
            'output': maxim
        })
    
    def add_weather_api_call(self, query, function_name, params):
        self.data.append({
            'instruction': query,
            'output': f'<function_call>\n{json.dumps({"name": function_name, "parameters": params}, indent=2)}\n</function_call>'
        })
    
    def add_hybrid_example(self, query, function_call, humorous_response):
        self.data.append({
            'instruction': query,
            'output': f'{function_call}\n\n{humorous_response}'
        })
    
    def build(self):
        return Dataset.from_list(self.data)

# Usage
builder = HumorWeatherDatasetBuilder()
builder.add_onion_article("Area Man...", "...")
builder.add_weather_api_call("What's the weather?", "get_current_weather", {"location": "Boston"})
dataset = builder.build()
```

---

## Conclusion

This 3-day training plan provides a comprehensive, executable roadmap for creating a specialized LoRA fine-tuned model that demonstrates both personality (humor in historical American literary styles) and utility (weather API function calling). The project showcases how curated training data directly shapes model behavior and capabilities.

**Key Takeaways**:
1. Small models (3-8B) can achieve specialized excellence with proper fine-tuning
2. LoRA enables efficient training without massive compute requirements
3. Hybrid datasets (style + function) create unique, valuable capabilities
4. Modern tooling (Unsloth, LLaMA-Factory) makes this accessible to developers

**Next Steps**:
- Expand to other API types (news, stocks, travel)
- Add multi-turn conversation capabilities
- Implement retrieval-augmented generation (RAG)
- Create public demo and share results

Good luck with your training! 🚀