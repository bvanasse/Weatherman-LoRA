# LoRA Training Guide: TwainBot - Literary Humor & Weather API Specialist
## A 3-Day Weekend Experiment in Specialized Model Fine-Tuning

---

## TL;DR

**Project Goal**: Create a fine-tuned LoRA adapter that transforms a base language model into "TwainBot" - an AI assistant that speaks with Mark Twain's wit and Benjamin Franklin's wisdom while expertly using weather APIs and telling weather-related jokes.

**Key Experiment**: Demonstrate how focused data curation creates specialized capabilities by training on both literary corpora and tool-use patterns.

**Timeline**: 3-day weekend (24-30 hours active development)
- **Day 1**: Data collection and preprocessing pipeline
- **Day 2**: Synthetic data generation and training setup
- **Day 3**: Training execution, evaluation, and deployment

**Technical Stack**:
- Base Model: Llama 3.1 8B Instruct or Mistral 7B Instruct v0.3
- Training Method: QLoRA (4-bit quantization)
- Dataset Size: 12,000-18,000 curated examples
- Hardware: Single GPU with 24GB+ VRAM or Mac M-series with 32GB+ unified memory

**Data Focus**: Quality over quantity - carefully balanced literary style data (40%), humor corpus (30%), tool-use examples (20%), and synthetic hybrids (10%).

---

## Overview

### Project Vision

TwainBot represents an experiment in creating specialized AI personalities through strategic data curation. By combining classical American literary humor with modern API capabilities, we demonstrate how focused training data can produce models that excel at both creative expression and practical tool use.

### What Makes This Project Unique

1. **Literary Personality + Practical Utility**: Blends Mark Twain's satirical wit with Benjamin Franklin's practical wisdom while maintaining genuine weather API functionality.

2. **Data Management Showcase**: Demonstrates the impact of thoughtful data composition on model behavior, proving that 15-20k carefully curated examples can outperform millions of generic ones.

3. **Weekend-Scale Achievement**: Complete pipeline from raw data collection to deployable model in 72 hours, leveraging AI coding assistants for efficiency.

4. **Dual Training Objectives**: Trains simultaneously on conversational style and structured tool-use patterns, creating a model that can both "talk like Twain" and reliably call weather APIs.

### Expected Outcomes

- **Functional Model**: LoRA adapter that transforms base models into witty weather experts
- **Data Pipeline**: Reproducible system for collecting and processing specialized training data
- **Technical Insights**: Understanding of how data composition affects model behavior and capabilities
- **Portfolio Piece**: Demonstrates AI engineering skills in fine-tuning, data curation, and specialized model development

---

## Data Strategy: Pre-Training and Training Data Focus

### Core Philosophy: Quality Through Curation

This project emphasizes that model specialization comes from thoughtful data composition rather than sheer volume. We focus on four complementary data categories that work together to create the desired personality and capabilities.

### Data Categories & Target Distribution

#### 1. Literary Style Corpus (40% - 6,000-7,200 examples)
**Purpose**: Teach the model Twain's and Franklin's writing styles, vocabulary, and humor patterns.

**Sources**:
- **Mark Twain**: Public domain works from Project Gutenberg
  - Novels: *The Adventures of Tom Sawyer*, *Adventures of Huckleberry Finn*
  - Travel: *Roughing It*, *Life on the Mississippi*, *Following the Equator*
  - Short stories and essays with weather/humor themes

- **Benjamin Franklin**: Public domain writings
  - *Autobiography of Benjamin Franklin*
  - *Poor Richard's Almanack* (aphorisms and practical wisdom)
  - Letters and essays with observational humor

**Processing Method**:
```python
def extract_literary_passages(text, min_length=50, max_length=300):
    """Extract conversational, descriptive, or humorous passages"""
    # Split into paragraphs
    paragraphs = text.split('\n\n')

    # Filter for passages containing weather/humor keywords
    weather_keywords = ['weather', 'rain', 'storm', 'sun', 'wind', 'climate', 'temperature']
    humor_keywords = ['humor', 'joke', 'wit', 'satire', 'funny', 'laugh']

    relevant_passages = []
    for para in paragraphs:
        para_lower = para.lower()
        if any(kw in para_lower for kw in weather_keywords + humor_keywords):
            if min_length <= len(para.split()) <= max_length:
                relevant_passages.append(para.strip())

    return relevant_passages
```

#### 2. Humor & Satire Corpus (30% - 3,600-5,400 examples)
**Purpose**: Train the model on modern humor patterns, especially weather-related jokes and satirical observations.

**Sources**:
- **The Onion**: Satirical news articles (existing data in workspace)
- **Reddit r/nottheonion**: Real stories that sound like satire
- **Weather Joke Collections**: Curated joke databases

**Collection Strategy**:
```python
# Leverage existing workspace data
def load_onion_data(csv_path):
    """Process existing The Onion dataset"""
    df = pd.read_csv(csv_path)

    # Filter for weather/climate related content
    weather_articles = df[df['title'].str.contains(
        r'weather|climate|storm|rain|snow|temperature|hurricane|tornado',
        case=False, na=False
    )]

    # Extract headlines and lead paragraphs
    processed = []
    for _, row in weather_articles.iterrows():
        headline = row['title']
        content = row.get('selftext', '')[:500]  # First 500 chars
        processed.append(f"{headline}\n{content}")

    return processed
```

#### 3. Tool-Use Training Data (20% - 2,400-3,600 examples)
**Purpose**: Teach the model how to call weather APIs using structured tool-calling patterns.

**Structure**: OpenAI-style function calling format with weather API tool.

**Tool Definition**:
```python
weather_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather conditions for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates"
                },
                "date": {
                    "type": "string",
                    "description": "Date for forecast (optional, defaults to today)",
                    "default": "today"
                }
            },
            "required": ["location"]
        }
    }
}
```

**Example Training Format**:
```json
{
  "messages": [
    {"role": "user", "content": "What's the weather like in San Francisco?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_123", "type": "function", "function": {
        "name": "get_weather", "arguments": {"location": "San Francisco"}
      }}
    ]},
    {"role": "tool", "content": "{\"temperature\": \"65°F\", \"condition\": \"Partly cloudy\", \"humidity\": \"72%\"}"},
    {"role": "assistant", "content": "Well now, the meteorological spirits report that San Francisco is basking in 65 degrees under partly cloudy skies, with a humidity that'll make your hair curl tighter than a cat's tail in a thunderstorm."}
  ]
}
```

#### 4. Hybrid Synthetic Examples (10% - 1,200-1,800 examples)
**Purpose**: Bridge the gap between literary style and practical tool use, creating seamless integration.

**Generation Approach**: Use AI assistants to create examples that blend Twain/Franklin style with weather API interactions.

**Prompt Template for Synthetic Generation**:
```
Create a conversational example where Mark Twain discusses weather using modern API data.

User Query: {weather_question}
API Response: {weather_data}

Write Twain's response in his characteristic style - witty, observational, with folksy wisdom and subtle humor about weather conditions.

Example Response Style:
"The weather prophets in this mechanical age tell me it's 72 degrees in Chicago, with clouds that look like they're plotting something mischievous. But mark my words, a Chicago wind has more personality than most politicians - it'll change its mind faster than a cat spotting a mouse."
```

### Data Collection Methods

#### 1. Web Scraping & API Collection
```python
import requests
from bs4 import BeautifulSoup
import time

def scrape_gutenberg_work(url):
    """Download and clean Project Gutenberg texts"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove Gutenberg boilerplate
    text = soup.get_text()
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        clean_text = text[start_idx + len(start_marker):end_idx]
        return clean_text.strip()

    return text

# Key Twain works for weather/humor themes
TWAIN_SOURCES = {
    "Roughing It": "https://www.gutenberg.org/files/3176/3176-0.txt",
    "Life on the Mississippi": "https://www.gutenberg.org/files/245/245-0.txt",
    "Following the Equator": "https://www.gutenberg.org/files/3177/3177-0.txt"
}
```

#### 2. Existing Data Processing
```python
import pandas as pd

def process_existing_reddit_data():
    """Process the existing The Onion and nottheonion CSVs"""

    # Load existing datasets
    onion_df = pd.read_csv('data_sources/reddit-theonion/data/TheOnion_181217_184244.csv')
    not_onion_df = pd.read_csv('data_sources/reddit-theonion/data/nottheonion_181217_184009.csv')

    # Filter for weather/climate content
    weather_terms = ['weather', 'climate', 'storm', 'rain', 'snow', 'temperature', 'hurricane']

    def has_weather_content(text):
        if not isinstance(text, str):
            return False
        return any(term in text.lower() for term in weather_terms)

    onion_weather = onion_df[onion_df['title'].apply(has_weather_content)]
    not_onion_weather = not_onion_df[not_onion_df['title'].apply(has_weather_content)]

    return onion_weather, not_onion_weather
```

#### 3. Synthetic Data Generation Pipeline
```python
def generate_tool_use_examples(num_examples=1000):
    """Generate synthetic tool-use training examples"""

    locations = ["San Francisco", "Chicago", "New York", "London", "Tokyo", "Death Valley"]
    conditions = ["sunny", "rainy", "stormy", "foggy", "snowy", "windy"]
    temperatures = list(range(32, 100, 5))  # Fahrenheit

    examples = []

    for _ in range(num_examples):
        location = random.choice(locations)
        temp = random.choice(temperatures)
        condition = random.choice(conditions)

        # Generate Twain-style response
        twain_response = generate_twain_weather_comment(temp, condition, location)

        example = {
            "messages": [
                {"role": "user", "content": f"What's the weather like in {location}?"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"call_{random.randint(1000,9999)}",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": location}
                    }
                }]},
                {"role": "tool", "content": f'{{"temperature": "{temp}°F", "condition": "{condition}", "location": "{location}"}}'},
                {"role": "assistant", "content": twain_response}
            ]
        }

        examples.append(example)

    return examples
```

### Data Quality Assurance

#### Deduplication & Filtering
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def remove_near_duplicates(texts, threshold=0.85):
    """Remove near-duplicate texts using TF-IDF similarity"""

    if len(texts) <= 1:
        return texts

    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    to_keep = []
    for i, text in enumerate(texts):
        # Check if this text is too similar to any previously kept text
        is_duplicate = False
        for j in to_keep:
            if similarity_matrix[i, j] > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            to_keep.append(i)

    return [texts[i] for i in to_keep]
```

#### Quality Metrics
- **Perplexity Filtering**: Remove low-quality passages
- **Length Distribution**: Ensure balanced example lengths
- **Diversity Scoring**: Maintain variety across sources and styles
- **Safety Filtering**: Remove inappropriate content

---

## Phased Implementation Plan (3-Day Weekend)

### Phase 1: Data Domination (Day 1 - 8-10 hours)

**Objectives**:
- Set up development environment
- Collect and process all raw data sources
- Create initial data processing pipeline
- Generate first batch of training examples

**Deliverables**:
- Processed literary corpus (Twain + Franklin)
- Cleaned humor dataset from existing sources
- Initial tool-use examples (synthetic)
- Data validation and quality checks

**Key Tasks**:

1. **Environment Setup (1 hour)**
   ```bash
   # Create project structure
   mkdir -p twainbot/data/{literary,humor,tool_use,hybrid}
   mkdir -p twainbot/training
   mkdir -p twainbot/evaluation

   # Set up Python environment
   python -m venv twainbot_env
   source twainbot_env/bin/activate
   pip install transformers torch peft datasets accelerate
   ```

2. **Literary Data Collection (2-3 hours)**
   - Download Twain and Franklin works from Project Gutenberg
   - Extract relevant passages (weather, humor, conversational)
   - Clean and normalize text
   - Implement chunking strategy

3. **Humor Data Processing (2 hours)**
   - Process existing The Onion and nottheonion CSVs
   - Filter for weather-related content
   - Clean and format for training

4. **Tool-Use Data Generation (2-3 hours)**
   - Define weather API tool schema
   - Create synthetic conversation examples
   - Generate diverse scenarios (different locations, conditions)
   - Implement Twain-style response generation

5. **Data Pipeline Validation (1 hour)**
   - Check data quality metrics
   - Balance dataset composition
   - Create train/validation splits

### Phase 2: Training Pipeline Construction (Day 2 - 8-10 hours)

**Objectives**:
- Set up LoRA training infrastructure
- Implement data loading and preprocessing
- Configure training parameters
- Execute initial training run

**Deliverables**:
- Configured training script
- Optimized dataset format
- Initial LoRA adapter checkpoint
- Training metrics and logs

**Key Tasks**:

1. **Model & Training Setup (2 hours)**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
   import torch

   # Load base model with quantization
   model_name = "microsoft/DialoGPT-medium"  # or your preferred base
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       load_in_4bit=True,
       device_map="auto"
   )

   # Configure LoRA
   lora_config = LoraConfig(
       r=16,  # rank
       lora_alpha=32,
       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
       lora_dropout=0.05,
       bias="none",
       task_type="CAUSAL_LM"
   )

   model = prepare_model_for_kbit_training(model)
   model = get_peft_model(model, lora_config)
   ```

2. **Data Formatting & Loading (2-3 hours)**
   - Convert all datasets to unified format
   - Implement data collator for tool-use examples
   - Create efficient data loading pipeline
   - Add data augmentation strategies

3. **Training Configuration (2 hours)**
   ```python
   from transformers import TrainingArguments, Trainer

   training_args = TrainingArguments(
       output_dir="./twainbot/training/checkpoints",
       num_train_epochs=3,
       per_device_train_batch_size=4,
       per_device_eval_batch_size=8,
       gradient_accumulation_steps=2,
       learning_rate=2e-4,
       weight_decay=0.01,
       warmup_steps=100,
       logging_steps=10,
       save_steps=500,
       evaluation_strategy="steps",
       eval_steps=500,
       save_total_limit=3,
       load_best_model_at_end=True,
       metric_for_best_model="eval_loss"
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
       data_collator=data_collator
   )
   ```

4. **Training Execution & Monitoring (2-3 hours)**
   - Launch training run
   - Monitor loss curves and metrics
   - Adjust hyperparameters as needed
   - Save intermediate checkpoints

### Phase 3: Evaluation & Deployment (Day 3 - 8-10 hours)

**Objectives**:
- Evaluate model performance on multiple dimensions
- Fine-tune based on evaluation results
- Create deployment-ready adapter
- Build demonstration interface

**Deliverables**:
- Comprehensive evaluation report
- Final LoRA adapter weights
- Demo application or API endpoint
- Documentation and usage examples

**Key Tasks**:

1. **Multi-Dimensional Evaluation (2-3 hours)**
   - **Style Evaluation**: Generate sample responses and rate Twain-like qualities
   - **Tool-Use Accuracy**: Test API calling reliability
   - **Humor Quality**: Assess joke generation and appropriateness
   - **Safety Checks**: Ensure responsible behavior

2. **Model Refinement (2 hours)**
   - Analyze evaluation results
   - Make targeted improvements
   - Retrain if necessary with adjusted data

3. **Adapter Merging & Optimization (2 hours)**
   ```python
   from peft import PeftModel

   # Load trained adapter
   model = AutoModelForCausalLM.from_pretrained(base_model_path)
   model = PeftModel.from_pretrained(model, adapter_path)

   # Merge adapter weights
   merged_model = model.merge_and_unload()

   # Save merged model
   merged_model.save_pretrained("./twainbot/final_model")
   tokenizer.save_pretrained("./twainbot/final_model")
   ```

4. **Demo Development (2-3 hours)**
   - Create simple web interface or CLI demo
   - Implement weather API integration
   - Add example conversations
   - Document usage patterns

---

## Technical Considerations

### Hardware Requirements
- **GPU**: RTX 3090/4090 (24GB+ VRAM) or A100 (40GB+)
- **CPU**: Modern multi-core processor for data processing
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ for datasets and checkpoints

### Performance Optimizations
- **QLoRA**: 4-bit quantization reduces memory footprint by 75%
- **Gradient Checkpointing**: Trade compute for memory efficiency
- **Flash Attention**: Faster attention computation where available
- **Data Parallelism**: Distribute training across multiple GPUs if available

### Cost Estimation
- **Compute**: $10-50 on cloud GPU platforms (depending on duration)
- **Data**: Free (public domain sources)
- **Tools**: Open-source libraries and frameworks
- **Time**: 24-30 hours of focused development

---

## References & Resources

### Data Sources
- **Project Gutenberg**: https://www.gutenberg.org/ (Twain and Franklin works)
- **The Onion Archives**: Existing workspace data in `data_sources/reddit-theonion/`
- **Open-Meteo API**: https://open-meteo.com/ (free weather data for examples)
- **WeatherAPI**: https://www.weatherapi.com/ (alternative weather service)

### Technical Resources
- **Hugging Face PEFT**: https://huggingface.co/docs/peft/index
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **Transformers Documentation**: https://huggingface.co/docs/transformers/index

### Related Projects & Research
- **Axolotl**: https://github.com/OpenAccess-AI-Collective/axolotl (LoRA training framework)
- **Lit-GPT**: https://github.com/Lightning-AI/lit-gpt (lightweight training)
- **Weather Model Examples**: Search Hugging Face for weather-related fine-tunes
- **Literary Style Transfer**: Research on author style adaptation

### Development Tools
- **Weights & Biases**: Experiment tracking and monitoring
- **TensorBoard**: Training visualization
- **Gradio**: Quick web interface for demos
- **Streamlit**: Alternative demo framework

### Additional Reading
- **"The Fine-Tuning of Language Models"**: Survey paper on fine-tuning techniques
- **"Toolformer: Language Models Can Teach Themselves to Use Tools"**: Tool-use learning approaches
- **"LoRA: Low-Rank Adaptation of Large Language Models"**: Original LoRA methodology
- **Author Style Analysis**: Linguistic studies on Twain and Franklin's writing patterns

---

## Next Steps & Extensions

### Immediate Improvements
1. **Expand Literary Corpus**: Add more authors or time periods for comparison
2. **Enhanced Tool Integration**: Support multiple weather APIs with fallbacks
3. **Multi-Modal Elements**: Add weather icon or image generation capabilities
4. **Conversational Memory**: Implement session-based context retention

### Advanced Experiments
1. **Comparative Studies**: Train variants with different data compositions
2. **Quantization Experiments**: Test different precision levels (8-bit, 2-bit)
3. **Model Merging**: Combine multiple LoRA adapters for different capabilities
4. **Federated Learning**: Distributed training across multiple machines

### Production Considerations
1. **Model Serving**: Deploy via vLLM, Text Generation Inference, or custom API
2. **Monitoring**: Implement usage tracking and performance monitoring
3. **Updates**: Establish pipeline for continuous data collection and model updates
4. **Safety**: Add comprehensive safety filters and responsible AI measures

---

*This guide demonstrates how focused data curation and thoughtful training can create specialized AI models. The TwainBot experiment shows that quality training data, rather than quantity, is the key to creating models with distinct personalities and practical capabilities.*
