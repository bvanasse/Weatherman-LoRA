# TwainBot LoRA Training: Consolidated Implementation Guide
## Context and Reference for Roadmap Implementation

---

## Executive Summary

This guide consolidates insights from multiple AI models' training plans for building "TwainBot" - a specialized LoRA-adapted language model that combines Mark Twain's wit, Benjamin Franklin's wisdom, The Onion's satire, and weather API tool-calling capabilities.

**Core Achievement**: Transform a base 7B-8B instruction model into a witty weather assistant through focused data curation and efficient LoRA fine-tuning, achievable in a 3-day weekend.

---

## 1. Project Fundamentals

### 1.1 Goal Definition
Create a LoRA adapter that enables a base language model to:
- Speak with Mark Twain's sardonic wit and Benjamin Franklin's aphoristic wisdom
- Generate weather-related humor and satirical observations
- Execute weather API function calls using OpenAI-style tool calling
- Blend literary personality with practical functionality seamlessly

### 1.2 Success Metrics
- **Style Consistency**: 80-85%+ recognition of Twain/Franklin voice on validation set
- **Tool-Call Accuracy**: 90-95%+ correct API calls with valid JSON schema
- **Humor Quality**: Subjective but measurably improved over base model
- **Inference Speed**: 30-50 tokens/second on single GPU
- **Training Efficiency**: Complete in 6-10 hours on single H100 or 12-24 hours on consumer GPU

### 1.3 Technical Stack
- **Base Models**: Llama 3.1 8B Instruct, Mistral 7B Instruct v0.3, or Phi-4
- **Training Method**: QLoRA (4-bit quantization with LoRA adapters)
- **Framework**: Hugging Face Transformers + PEFT + TRL
- **Hardware**: Single GPU with 24GB+ VRAM or Mac M-series with 32GB+ RAM
- **Dataset Size**: 12,000-20,000 curated examples

---

## 2. Data Strategy: The Foundation of Success

### 2.1 Dataset Composition (Consensus Across All Sources)

**Target Distribution** (15,000-18,000 total examples):

| Category | Examples | Percentage | Purpose |
|----------|----------|------------|---------|
| **Literary Style** | 6,000-7,200 | 40% | Twain/Franklin voice, humor patterns |
| **Satire & Humor** | 4,500-5,400 | 30% | Modern satirical tone, weather jokes |
| **Tool-Use Patterns** | 3,000-3,600 | 20% | Weather API function calling |
| **Hybrid Examples** | 1,500-1,800 | 10% | Style + tool integration |

### 2.2 Primary Data Sources

#### A. Literary Corpus (Project Gutenberg - Public Domain)

**Mark Twain Priority Works**:
- *The Adventures of Tom Sawyer* (ID: 74)
- *Adventures of Huckleberry Finn* (ID: 76)
- *A Connecticut Yankee in King Arthur's Court* (ID: 86)
- *Life on the Mississippi* (ID: 245)
- *Roughing It* (ID: 3176)
- *Following the Equator* (ID: 3177)

**Benjamin Franklin Priority Works**:
- *Autobiography of Benjamin Franklin* (ID: 20203, 148)
- *Poor Richard's Almanack* (ID: 57795)
- Essays and Letters (ID: 40933)

**Extraction Strategy**:
- Focus on passages containing weather mentions, dialogue, humor, first-person narrative
- Chunk into 200-500 word segments preserving paragraph boundaries
- Prioritize humorous passages using keyword detection
- Target: 3,000-5,000 high-quality excerpts

#### B. Humor & Satire Corpus

**Existing Data** (Workspace Asset):
- `data_sources/reddit-theonion/data/TheOnion_181217_184244.csv`
- `data_sources/reddit-theonion/data/nottheonion_181217_184009.csv`

**Processing Approach**:
- Filter for weather-related keywords: weather, rain, snow, storm, climate, temperature, forecast, hurricane, tornado
- Extract headlines + first 2-3 paragraphs
- Label by type: satire (The Onion) vs. absurd reality (nottheonion)
- Target: 3,000-5,000 examples

**Supplementary Sources**:
- Weather joke databases (curated manually or from open datasets)
- Reddit r/dadjokes filtered for weather themes
- Target: 1,500-2,000 weather-specific jokes

#### C. Tool-Use Data (Synthetic Generation)

**Weather API Schema**:
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather conditions and forecast for a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "City and state/country, e.g., 'Boston, MA'"
        },
        "units": {
          "type": "string",
          "enum": ["fahrenheit", "celsius"],
          "default": "fahrenheit"
        },
        "days": {
          "type": "integer",
          "minimum": 1,
          "maximum": 7,
          "description": "Number of forecast days"
        }
      },
      "required": ["location"]
    }
  }
}
```

**Generation Strategy**:
- Create 1,000-1,500 examples per function (current weather, forecast, alerts)
- Vary natural language queries across cities and phrasing
- Include success cases, edge cases (missing location, API errors)
- Mix neutral, Twain-style, and Franklin-style responses
- Target: 3,000-3,500 tool-call transcripts

#### D. Hybrid Synthetic Examples

**Purpose**: Bridge style and functionality seamlessly

**Template Approach**:
```
User: {weather_query}
Assistant Thought: {style_specific_observation}
Tool Call: get_weather(location, units)
Tool Response: {weather_data}
Assistant: {styled_response_integrating_tool_data}
```

**Example Styles**:
- **Twain**: "Well, I reckon the meteorological oracles report {weather}. Their predictions are about as reliable as a politician's promise..."
- **Franklin**: "As Poor Richard says, 'Some are weather-wise, but most are otherwise.' The forecast shows {weather}. Best to prepare accordingly."

**Target**: 1,500-2,000 hybrid examples

### 2.3 Data Quality Pipeline

**Critical Processing Steps** (Consensus):

1. **Cleaning**:
   - Remove HTML, Gutenberg boilerplate, metadata artifacts
   - Normalize Unicode, quotes, whitespace
   - Strip offensive/NSFW content
   - Language detection (English only)

2. **Deduplication**:
   - MinHash/LSH with threshold 0.8 Jaccard similarity
   - TF-IDF cosine similarity for near-duplicates
   - Cross-source deduplication

3. **Normalization**:
   - Consistent tokenization
   - Standardized formatting (JSONL with messages structure)
   - Metadata tagging: persona (twain/franklin/neutral), tone, domain, source

4. **Validation**:
   - Length filtering: 10-600 words per example
   - Quality scoring: perplexity, coherence
   - Tool-call JSON schema validation
   - Manual spot-checking (100-300 samples)

5. **Splitting**:
   - 90/10 train/validation split
   - Stratified by category to maintain distribution
   - Separate test set for final evaluation (optional)

---

## 3. Training Configuration

### 3.1 LoRA Hyperparameters (Consensus)

**LoRA Configuration**:
```python
lora_config = LoraConfig(
    r=16,                    # Rank: 8-32 (start with 16)
    lora_alpha=32,           # Alpha: typically 2x rank
    lora_dropout=0.05,       # Regularization
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Quantization (QLoRA)**:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### 3.2 Training Arguments (Consensus)

```python
training_args = TrainingArguments(
    output_dir="./twainbot-lora",
    num_train_epochs=3,              # 2-4 epochs typical
    per_device_train_batch_size=4,   # Adjust for GPU
    gradient_accumulation_steps=4,   # Effective batch = 16
    learning_rate=2e-4,               # LoRA uses higher LR
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,                # 3-5% warmup
    weight_decay=0.01,
    max_grad_norm=0.3,
    bf16=True,                        # Use bfloat16 on modern GPUs
    optim="paged_adamw_32bit",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50-100,
    save_steps=100-500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)
```

### 3.3 Training Timeline

**Expected Duration** (Consensus):
- **1x H100 80GB**: 3-4 hours for 3 epochs
- **1x A100 40GB**: 4-6 hours for 3 epochs
- **Consumer GPU (RTX 4090)**: 12-18 hours for 3 epochs
- **Mac M2 Ultra**: 18-24 hours for 3 epochs

**Monitoring**:
- Training loss should decrease steadily (target: <0.5 final)
- Eval loss should track training loss (watch for divergence = overfitting)
- Perplexity should decrease (target: <10)
- GPU utilization: 80-95% during training

---

## 4. Implementation Phases (3-Day Plan)

### Day 1: Data Foundation (8-10 hours)

**Morning (4-5 hours)**:
- Environment setup (Python, CUDA/MPS, libraries)
- Download Twain/Franklin texts from Project Gutenberg
- Process existing Reddit data from workspace
- Initial cleaning and normalization

**Afternoon (4-5 hours)**:
- Implement deduplication pipeline
- Extract literary passages with weather/humor focus
- Filter and format satire/humor data
- Generate initial tool-use examples (500-1,000)
- Create train/validation splits

**Deliverable**: 8,000-12,000 raw examples, cleaned and formatted

### Day 2: Synthesis & Training Setup (8-10 hours)

**Morning (4 hours)**:
- Generate remaining synthetic tool-use data
- Create hybrid style+tool examples
- Final dataset balancing and quality checks
- Format to chat-style JSONL

**Afternoon (4-6 hours)**:
- Configure LoRA training script
- Set up base model with quantization
- Initialize training run
- Begin monitoring (can run overnight)

**Deliverable**: Complete dataset (15,000-18,000 examples), training initiated

### Day 3: Evaluation & Deployment (6-8 hours)

**Morning (3-4 hours)**:
- Evaluate trained adapter
- Run automated metrics (style classification, tool-call accuracy)
- Generate sample conversations
- Human/LLM-as-judge evaluation

**Afternoon (3-4 hours)**:
- Merge LoRA adapter (optional)
- Set up inference endpoint (vLLM or similar)
- Create demo with weather API integration
- Document results and usage

**Deliverable**: Production-ready LoRA adapter, evaluation report, demo

---

## 5. Evaluation Framework

### 5.1 Quantitative Metrics

**Style Consistency**:
- Classification accuracy on Twain/Franklin voice detection
- Target: 80-85%+ on validation set

**Tool-Call Accuracy**:
- JSON schema validity: 95%+ correct format
- Required parameters present: 95%+
- Appropriate function selection: 90%+

**Perplexity**:
- Validation set perplexity: <10 (lower is better)
- Compare to base model baseline

### 5.2 Qualitative Evaluation

**LLM-as-Judge** (50-100 samples):
- Rate Twain-ness on 1-5 scale
- Rate humor quality on 1-5 scale
- Verify groundedness in tool output

**Human Spot Checks** (20-30 samples):
- Voice authenticity
- Appropriateness of humor
- Correct tool integration
- No factual hallucinations

### 5.3 Test Scenarios

**Style Tests**:
- "Rewrite 'The weather forecast is uncertain' in Mark Twain's style"
- "Give me a Benjamin Franklin aphorism about weather preparation"

**Tool-Use Tests**:
- "What's the current weather in Seattle?"
- "Will it rain in Chicago this weekend?"

**Hybrid Tests**:
- "Tell me about the weather in San Francisco, but make it funny"
- "I'm planning a picnic in Boston tomorrow - should I bring an umbrella? Be witty about it."

---

## 6. Deployment Options

### 6.1 Model Export

**LoRA Adapter Only** (Lightweight):
- Save adapter weights separately (~100-500MB)
- Load dynamically on top of base model
- Flexible for swapping adapters

**Merged Model** (Standalone):
```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(adapter_path)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./twainbot-merged")
```

### 6.2 Serving Infrastructure

**Option A: vLLM** (Production-grade):
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="./twainbot-merged",
    tensor_parallel_size=1,
    dtype="bfloat16"
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)
```

**Option B: Text Generation Inference**:
- Hugging Face's optimized inference server
- OpenAI-compatible API endpoints
- Built-in batching and caching

**Option C: Ollama** (Local deployment):
```bash
# Create Ollama modelfile
ollama create twainbot -f Modelfile

# Run locally
ollama run twainbot "What's the weather in Boston?"
```

---

## 7. Critical Success Factors

### 7.1 Data Quality is Paramount

**Key Insights** (Consensus across all sources):
- 15,000 curated examples > 1,000,000 generic examples for specialized tasks
- Balance across categories prevents mode collapse
- Deduplication is non-negotiable for avoiding repetition
- Weather-specific examples ground the domain knowledge
- Style transfer examples bridge the gap between personality and function

### 7.2 Training Discipline

**Avoid Common Pitfalls**:
- **Overfitting**: Monitor eval loss; increase dropout if needed
- **Style Drift**: Maintain strict literary data proportion (40%)
- **Tool Hallucination**: Include explicit tool-call examples with correct schema
- **Catastrophic Forgetting**: Don't overtrain; 2-3 epochs sufficient

### 7.3 Iteration Strategy

**Phased Approach**:
1. Train style-only adapter first (validate voice)
2. Add tool-use examples incrementally
3. Fine-tune hybrid integration
4. Iterate based on evaluation results

---

## 8. Technical Troubleshooting

### 8.1 Common Issues & Solutions

**High Training Loss (>1.0 after epoch 1)**:
- Reduce learning rate to 1e-4
- Increase warmup steps
- Check data quality and formatting
- Verify tokenizer handling special tokens

**Model Doesn't Generate Function Calls**:
- Increase tool-use proportion to 40-50%
- Use temperature=0.3 for more deterministic outputs
- Verify function call format matches training exactly
- Add more explicit system prompts

**Humor Style Not Coming Through**:
- Increase literary dataset proportion
- Add more style-transfer examples
- Use lower temperature (0.6-0.7)
- Consider training for additional epoch

**Out of Memory**:
- Reduce batch size to 2
- Increase gradient accumulation
- Enable gradient checkpointing
- Use deeper quantization (3-bit)
- Reduce max_seq_length to 1024

---

## 9. Resources & References

### 9.1 Data Sources

- **Project Gutenberg**: https://www.gutenberg.org/
- **Gutenberg API**: https://gutendex.com/
- **Open-Meteo API**: https://open-meteo.com/ (free weather data)
- **OpenWeatherMap**: https://openweathermap.org/api
- **Weather.gov API**: https://www.weather.gov/documentation/services-web-api

### 9.2 Technical Libraries

- **Transformers**: https://huggingface.co/docs/transformers/
- **PEFT (LoRA)**: https://huggingface.co/docs/peft/
- **TRL (Training)**: https://huggingface.co/docs/trl/
- **bitsandbytes**: https://github.com/TimDettmers/bitsandbytes
- **vLLM**: https://github.com/vllm-project/vllm

### 9.3 Research Papers

- **LoRA**: "Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
  - https://arxiv.org/abs/2106.09685
- **QLoRA**: "Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
  - https://arxiv.org/abs/2305.14314
- **Instruction Tuning**: Survey paper - https://arxiv.org/abs/2308.10792

### 9.4 Community Resources

- **r/LocalLLaMA**: Active community for local LLM fine-tuning
- **Hugging Face Forums**: https://discuss.huggingface.co/
- **Weights & Biases**: Experiment tracking - https://wandb.ai/
- **Unsloth**: Optimized LoRA training - https://github.com/unslothai/unsloth

---

## 10. Roadmap Alignment

### Mapping to `agent-os/product/roadmap.md`

This guide directly supports the following roadmap items:

**1. Environment Setup & Data Infrastructure** (Item 1):
- Python environment with TRL/PEFT/transformers
- GPU dependencies (torch/CUDA)
- Data storage paths for raw/processed datasets

**2. Literary Corpus Collection** (Item 2):
- Project Gutenberg downloads (Twain, Franklin)
- 3,000-5,000 relevant passages with weather/humor mentions

**3. Reddit Humor Dataset Processing** (Item 3):
- Process existing `data_sources/reddit-theonion/` CSVs
- Extract weather-related posts
- Create 2,000-4,000 labeled humor examples

**4. Data Normalization & Deduplication Pipeline** (Item 4):
- MinHash/LSH deduplication
- Language filtering
- Safety filters
- Quality statistics

**5. Instructionalization & Tagging** (Item 5):
- Convert to chat-format JSONL
- Apply persona/tone/domain tags
- Create balanced train/validation splits

**6. Synthetic Tool-Use Data Generation** (Item 6):
- 1,000-3,000 OpenAI-style function calling examples
- Multi-turn conversations with tool calls
- Success cases and error handling

**7-12. Training & Evaluation** (Items 7-12):
- QLoRA configuration and training
- Style and tool-use evaluation
- Model serving setup

---

## 11. Key Takeaways

1. **Data is King**: The quality and composition of training data directly determines model behavior. Invest heavily in curation.

2. **Balance Matters**: 40% style, 30% humor, 20% tools, 10% hybrid creates the right mix for dual functionality.

3. **Efficiency Through LoRA**: Training only 1-2% of parameters makes this feasible on consumer hardware in a weekend.

4. **Hybrid Capability**: Synthetic examples bridging style and tools are crucial for seamless integration.

5. **Evaluation is Essential**: Multi-dimensional testing (style, humor, tool accuracy) ensures quality.

6. **Iteration is Expected**: First pass establishes baseline; refinement based on evaluation is normal.

7. **Weekend is Achievable**: With focused execution and AI coding assistance, complete pipeline is realistic in 72 hours.

---

## Conclusion

This consolidated guide synthesizes best practices from multiple training approaches to create TwainBot. The emphasis on data quality, balanced composition, and efficient training makes this achievable as a weekend project while demonstrating professional-grade AI engineering skills.

**Success depends on**:
- Rigorous data curation and deduplication
- Balanced dataset composition across categories
- Proper LoRA configuration and hyperparameters
- Multi-dimensional evaluation before deployment
- Iterative refinement based on results

**The result**: A specialized model that proves small, focused datasets create distinctive AI capabilities that generic training cannot replicate.
