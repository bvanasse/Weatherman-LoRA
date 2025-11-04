# Tech Stack

## Machine Learning & Training

### Core Frameworks
- **PyTorch 2.1+** — Deep learning framework with CUDA 12.1+ support for H100 GPU optimization
- **Transformers 4.36+** — Hugging Face library for loading base models (Llama 3.1 8B Instruct, Mistral 7B Instruct)
- **PEFT 0.7+** — Parameter-Efficient Fine-Tuning library for LoRA/QLoRA adapter training
- **TRL (Transformer Reinforcement Learning)** — Supervised fine-tuning with SFTTrainer for chat-format data
- **Accelerate 0.25+** — Distributed training utilities and device management
- **bitsandbytes 0.41+** — 4-bit quantization for QLoRA (NF4 with double quantization)

### Base Models
- **meta-llama/Meta-Llama-3.1-8B-Instruct** — Primary base model (8B parameters, Apache 2.0 license)
- **mistralai/Mistral-7B-Instruct-v0.3** — Alternative base model (7B parameters, Apache 2.0 license)
- **Sequence Length:** 4096 tokens preferred, 2048 acceptable if memory-constrained

### LoRA Configuration
- **Quantization:** 4-bit NF4 (QLoRA) with double quantization for memory efficiency
- **LoRA Rank (r):** 16-32 (starting point: 16)
- **LoRA Alpha:** 32-64 (typically 2x rank)
- **LoRA Dropout:** 0.05-0.1
- **Target Modules:** `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **Learning Rate:** 2e-4 with cosine decay, 3% warmup
- **Batch Configuration:** Micro-batch 1-4, gradient accumulation to effective batch 64-256
- **Optimizations:** Gradient checkpointing, Flash Attention 2.0, sequence packing for short turns

## Data Processing & Collection

### Web Scraping & Parsing
- **trafilatura** — HTML content extraction with readability algorithms
- **BeautifulSoup4 4.12+** — HTML parsing for structured data extraction
- **requests 2.31+** — HTTP client for web scraping and API calls
- **Playwright** (optional) — Browser automation for JavaScript-heavy sites if needed

### Text Processing
- **datasets 2.15+** — Hugging Face datasets library for JSONL loading and preprocessing
- **NLTK 3.8+** — Natural language processing utilities for text chunking
- **spaCy 3.7+** (optional) — Advanced NLP for sentence segmentation if needed
- **fasttext or langdetect** — Language detection for English filtering

### Deduplication & Quality
- **datasketch** — MinHash/LSH for near-duplicate detection (Jaccard threshold 0.8)
- **pandas 2.1+** — Data manipulation for CSV processing and statistics
- **jsonlines 4.0+** — Efficient JSONL file reading/writing

## Data Sources

### Primary Corpora
- **Project Gutenberg** — Public domain literary texts (Twain, Franklin)
  - API: gutendex.com (unofficial) or direct file downloads
  - Format: Plain text with metadata extraction
- **Reddit APIs** — Existing CSVs in `data_sources/reddit-theonion/`
  - PRAW (Python Reddit API Wrapper) if additional scraping needed
  - Pushshift API for historical data (if accessible)

### Weather APIs
- **Open-Meteo** — Free weather API, no key required, for tool-use examples
- **OpenWeatherMap** (alternative) — Weather API with free tier
- Mock API responses for offline synthetic data generation

## Model Serving & Deployment

### Inference Engines
- **vLLM** — High-throughput LLM serving with OpenAI-compatible API endpoints
- **text-generation-inference** (alternative) — Hugging Face inference server
- **FastAPI** — Lightweight API framework for custom serving layer if needed

### Serving Configuration
- **Tool Schema Format:** OpenAI-style function calling with `tool_calls` messages
- **System Prompts:** Persona steering ("You are a witty assistant who speaks like Mark Twain")
- **Adapter Loading:** Separate LoRA weights or merged model export

## Monitoring & Evaluation

### Training Monitoring
- **Weights & Biases (wandb) 0.16+** — Experiment tracking, loss curves, hyperparameter logging
- **TensorBoard** (alternative) — Local training visualization

### Evaluation Tools
- **Custom evaluation harness** — Python scripts for:
  - Style classifier (Twain vs. neutral) accuracy/F1
  - Tool-call JSON schema validation
  - Groundedness scoring (answer uses tool output)
  - LLM-as-judge quality assessment (wit, Twain-ness scores 1-5)
  - Safety/toxicity filters (openai-moderation or together-ai APIs)

## Development Environment

### Compute Infrastructure
- **Primary:** Single NVIDIA H100 GPU (80GB VRAM)
- **Fallback:** A100 (40GB/80GB) or consumer GPU (RTX 4090/4080 with reduced batch sizes)
- **CPU:** Multi-core for parallel data preprocessing
- **RAM:** 32GB+ system RAM for data loading
- **Storage:** 500GB+ SSD for datasets, checkpoints, and model weights

### Operating System
- **Linux (Ubuntu 22.04 LTS preferred)** — Primary development OS for CUDA compatibility
- **Python 3.10+** — Runtime environment with conda/venv for dependency isolation

### Version Control & Collaboration
- **Git** — Source control for scripts, configs, and documentation
- **GitHub/GitLab** — Repository hosting for code and documentation
- **Hugging Face Hub** — Model weight sharing and versioning (optional)

## Data Format Standards

### Training Data Format
- **JSONL (JSON Lines)** — One JSON object per line for efficient streaming
- **Chat Format:** Messages-based with `role` fields (`system`, `user`, `assistant`, `tool`)
- **Tool Format:** OpenAI-style `tool_calls` with `function` name and `arguments` JSON
- **Metadata Tags:** `persona`, `tone`, `domain`, `source`, `device` for stratification

### Example Schema
```json
{
  "messages": [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User query"},
    {"role": "assistant", "content": "Response or tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "JSON result"}
  ],
  "tags": {"persona": "twain", "tone": "humorous", "domain": "weather", "source": "synthetic"}
}
```

## Licensing & Compliance

### Model Licenses
- **Llama 3.1:** Meta Llama 3.1 Community License (commercial use allowed)
- **Mistral 7B:** Apache 2.0 License (fully permissive)

### Data Licenses
- **Project Gutenberg:** Public domain (US) with attribution
- **Reddit Data:** Follow Reddit API Terms of Service and PRAW guidelines
- **Synthetic Data:** Self-generated, no external license constraints
