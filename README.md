# Weatherman-LoRA

A specialized language model training project that enables rapid fine-tuning of open-source LLMs to create weather-focused conversational AI with literary personality (Mark Twain & Benjamin Franklin) and tool-calling capabilities using efficient LoRA/QLoRA techniques.

## Project Overview

This project trains a LoRA adapter for base LLMs (Mistral 7B) that combines:
- **Literary Style**: Mark Twain's wit and Benjamin Franklin's wisdom
- **Weather Knowledge**: Domain-specific weather information and humor
- **Tool Calling**: OpenAI-style function calling for weather APIs

**Training Goal**: Complete fine-tuning in under 48 hours on a single H100 GPU (or 3-4 days on Mac M4).

## Quick Start

### Platform Detection

Detect your platform and get configuration recommendation:

```bash
python scripts/check_gpu.py
```

This will automatically detect whether you're on:
- **H100 GPU** (CUDA) - Recommended config: `configs/training_config_h100.yaml`
- **Mac M4** (MPS) - Recommended config: `configs/training_config_m4.yaml`

### H100 GPU Setup (Remote Training)

**Recommended for**: Production training, faster iteration (3-4 hours)

```bash
# Create conda environment
conda env create -f environment-remote.yml
conda activate weatherman-lora

# Install Flash Attention 2 (optional but recommended)
pip install flash-attn --no-build-isolation

# Validate environment
python scripts/validate_environment.py --env=h100

# Validate configuration
python scripts/validate_training_config.py --config configs/training_config_h100.yaml
```

See **[docs/SETUP_H100.md](docs/SETUP_H100.md)** for detailed H100 setup instructions.

### Mac M4 Setup (Local Training)

**Recommended for**: Local development, config testing (8-12 hours)

```bash
# Run automated setup
./setup_m4.sh

# Activate environment
source .venv-m4/bin/activate

# Validate environment
python scripts/validate_environment.py --env=m4

# Validate configuration
python scripts/validate_training_config.py --config configs/training_config_m4.yaml
```

See **[docs/SETUP_M4.md](docs/SETUP_M4.md)** for detailed M4 setup instructions.

### Local Environment (Data Processing Only)

For data processing without training:

```bash
# Run setup script
./setup_local.sh

# Verify environment
python3 scripts/validate_environment.py --env=local
```

## QLoRA Training Configuration

This project supports dual-platform QLoRA training optimized for:
- **H100 GPU** (primary): 3-4 hour training with Flash Attention 2
- **Mac M4** (alternative): 8-12 hour training with MPS backend

### Training Configurations

#### H100 Configuration (`configs/training_config_h100.yaml`)

**Optimized for**: Maximum throughput on 80GB VRAM

```yaml
# Key settings
model:
  model_name_or_path: mistralai/Mistral-7B-Instruct-v0.2
  max_seq_length: 4096  # Full context with Flash Attention 2

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch: 16
  num_train_epochs: 3
  learning_rate: 2.0e-4
```

**Expected performance**:
- Training time: 3-4 hours (15K examples, 3 epochs)
- Memory usage: 60-70GB VRAM
- Throughput: 400-500 examples/minute

#### M4 Configuration (`configs/training_config_m4.yaml`)

**Optimized for**: Memory efficiency on 32GB unified memory

```yaml
# Key differences from H100
model:
  max_seq_length: 2048  # Reduced for memory

training:
  per_device_train_batch_size: 1  # Memory-constrained
  gradient_accumulation_steps: 16  # Maintain effective batch of 16
```

**Expected performance**:
- Training time: 8-12 hours (15K examples, 3 epochs)
- Memory usage: 24-28GB unified memory
- Throughput: 150-200 examples/minute

### Loading Training Configurations

```python
from scripts.config_loader import load_training_config

# Load H100 config
h100_config = load_training_config(config_path="configs/training_config_h100.yaml")

# Load M4 config
m4_config = load_training_config(config_path="configs/training_config_m4.yaml")

# Load with overrides
custom_config = load_training_config(
    config_path="configs/training_config_h100.yaml",
    overrides={
        'training': {
            'num_train_epochs': 5,  # Train for 5 epochs instead of 3
            'learning_rate': 1e-4,   # Lower learning rate
        }
    }
)
```

### Platform-Specific Validation

Validate your environment before training:

```bash
# For H100
python scripts/check_gpu.py
python scripts/validate_environment.py --env=h100
python scripts/validate_training_config.py --config configs/training_config_h100.yaml

# For Mac M4
python scripts/check_gpu.py
python scripts/validate_environment.py --env=m4
python scripts/validate_training_config.py --config configs/training_config_m4.yaml
```

### Training Quick Start

See **[docs/TRAINING_QUICKSTART.md](docs/TRAINING_QUICKSTART.md)** for step-by-step training instructions including:
- Platform detection
- Configuration customization
- Monitoring training
- Troubleshooting OOM errors
- Loading trained adapters

## Data Collection

### Literary Corpus Collection

Collect high-quality literary passages from Mark Twain and Benjamin Franklin:

```bash
# Full pipeline (download + extract + serialize)
python scripts/collect_literary_corpus.py

# Skip download if books are cached
python scripts/collect_literary_corpus.py --skip-download

# Filter to specific author
python scripts/collect_literary_corpus.py --books twain
```

**Output**: `data/processed/gutenberg_passages.json` (~500-800 passages, ~1-2MB)

**What's collected:**
- 6 books from Project Gutenberg (4 Twain, 2 Franklin)
- Passages containing weather and/or humor keywords
- 200-500 words per passage with surrounding context
- Rich metadata: author, book, genre, keywords, publication year

**For details**, see [docs/LITERARY_CORPUS.md](docs/LITERARY_CORPUS.md)

### Synthetic Tool-Use Data Generation

Generate 1,000-3,000 OpenAI-style function calling examples using Claude Haiku 4.5 API:

```bash
# Set API key (or will prompt interactively)
export ANTHROPIC_API_KEY="sk-ant-..."

# Generate 1000 tool-use conversation examples
python scripts/generate_synthetic_tool_data.py --count 1000

# Test mode without API calls
python scripts/generate_synthetic_tool_data.py --count 10 --mock
```

**Output**: `data/synthetic/tool_use_examples.jsonl` (~1000-3000 conversations, ~5-15MB)

**What's generated:**
- Weather tool calling examples (get_current_weather, get_forecast, geocode_location)
- Persona distribution: 60% neutral, 25% Twain, 15% Franklin
- Scenario mix: 60-70% success cases, 15-20% error handling, 15-20% multi-turn
- Validation: automatic schema, semantic, and groundedness checks

**Cost & Time** (on M4 Mac):
- 1,000 examples: $0.50-$1.00, ~30-45 minutes
- 3,000 examples: $1.50-$3.00, ~90-135 minutes

**API Key Setup:**
- Get key from [console.anthropic.com](https://console.anthropic.com/)
- Set `ANTHROPIC_API_KEY` environment variable, or script will prompt interactively
- No pre-configuration needed - just run the script

**For details**, see [docs/SYNTHETIC_DATA_GENERATION.md](docs/SYNTHETIC_DATA_GENERATION.md)

## Project Structure

```
Weatherman-LoRA/
├── data/                      # Training and processing data
│   ├── raw/                   # Original downloads (Gutenberg texts, Reddit CSVs)
│   ├── processed/             # Cleaned, deduplicated JSONL datasets
│   └── synthetic/             # Generated tool-use examples
├── data_sources/              # Existing source data
│   └── reddit-theonion/       # Reddit humor dataset CSVs
├── models/                    # Base model storage (~15GB per model)
│   └── .gitkeep               # (Mistral 7B)
├── adapters/                  # LoRA weights and checkpoints
│   ├── weatherman-lora-h100/  # H100-trained adapters
│   └── weatherman-lora-m4/    # M4-trained adapters
├── scripts/                   # Python utilities and processing
│   ├── check_storage.py       # Storage verification
│   ├── check_gpu.py           # Dual-platform GPU/MPS detection
│   ├── paths.py               # Path constants
│   ├── config_loader.py       # Configuration loader with overrides
│   ├── verify_model.py        # Model verification
│   ├── validate_environment.py # Multi-platform environment validation
│   ├── validate_training_config.py # Training config validation
│   ├── download_gutenberg.py  # Gutenberg book downloader
│   ├── keyword_matcher.py     # Keyword matching module
│   ├── extract_passages.py    # Passage extraction engine
│   └── collect_literary_corpus.py # Main collection pipeline
├── configs/                   # Configuration files
│   ├── training_config.yaml   # Original LoRA training parameters
│   ├── training_config_h100.yaml # H100-optimized configuration
│   ├── training_config_m4.yaml   # M4-optimized configuration
│   ├── paths_config.json      # Data path configurations
│   └── gutenberg_books.json   # Book metadata configuration
├── docs/                      # Documentation
│   ├── SETUP_GUIDE.md         # Comprehensive setup instructions
│   ├── SETUP_H100.md          # H100-specific setup guide
│   ├── SETUP_M4.md            # Mac M4-specific setup guide
│   ├── TRAINING_QUICKSTART.md # Fast-track training guide
│   ├── DATA_SYNC.md           # Data transfer workflows
│   ├── MODEL_DOWNLOAD.md      # Model download instructions
│   └── LITERARY_CORPUS.md     # Literary corpus documentation
├── tests/                     # Test suite
│   ├── test_training_configs.py   # Config loading tests
│   ├── test_platform_detection.py # Platform detection tests
│   └── test_environment_setup.py  # Environment setup tests
├── references/                # Implementation guides and research
│   └── IMPLEMENTATION_GUIDE.md # Consolidated training guide
├── agent-os/                  # Agent OS specs and planning
│   ├── product/               # Product roadmap and mission
│   └── specs/                 # Feature specifications
├── requirements-local.txt     # Local environment dependencies
├── requirements-m4.txt        # Mac M4 training dependencies
├── environment-remote.yml     # H100 conda environment spec
├── setup_local.sh             # Local setup automation
├── setup_m4.sh                # Mac M4 setup automation
└── setup_remote.sh            # Remote setup automation
```

## Multi-Environment Workflow

This project uses a **dual-environment architecture** optimized for different hardware:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL (Mac M4)                                │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Data Collection & Processing                           │    │
│  │  • Download Project Gutenberg texts                     │    │
│  │  • Process Reddit humor data                            │    │
│  │  • Clean, deduplicate, format JSONL                     │    │
│  │  • Generate synthetic tool-use examples (Claude API)    │    │
│  │    → 1000-3000 conversations, ~$1-3, 30-90 min         │    │
│  └────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            │ Optional: Train on M4 (8-12 hours)  │
│                            │ Sync processed data to H100         │
│                            │ (rsync/scp ~500MB-1GB)             │
│                            ▼                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              REMOTE (H100 - Lambda/RunPod)                       │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  GPU Training                                           │    │
│  │  • Download base model (~15GB)                          │    │
│  │  • Load processed training data                         │    │
│  │  • Train LoRA adapter (3-4 hours)                       │    │
│  │  • Save adapter checkpoints                             │    │
│  └────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            │ Sync trained adapters               │
│                            │ (rsync/scp ~100-500MB)             │
│                            ▼                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL (Evaluation)                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Model Testing & Evaluation                             │    │
│  │  • Run evaluation metrics                               │    │
│  │  • Generate sample conversations                        │    │
│  │  • Document results                                     │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

- **Cost Efficiency**: Only pay for GPU time during training (3-4 hours on H100), not during data processing
- **API Costs on Local**: Generate synthetic data locally using Claude API ($1-3 total) before expensive GPU time
- **Flexibility**: Train locally on M4 for testing, or use H100 for production
- **Data Processing Speed**: Mac M4 is sufficient for CPU-bound data tasks and API orchestration
- **Bandwidth Optimization**: Sync only processed data (~1GB) instead of raw data and models (~20GB)
- **Platform Choice**: Easily switch between H100, A100, 3090, or Mac M4

## Storage Requirements

**Minimum**: 30GB available disk space
**Recommended**: 50GB for comfortable operation

### Storage Breakdown
- Raw data (Gutenberg, Reddit): ~150-300MB
- Processed JSONL training data: ~500MB-1GB
- Base model (Mistral 7B): ~15GB
- Checkpoints and adapters: ~500MB-2GB
- Working buffer: ~10-15GB

Run `python3 scripts/check_storage.py` to verify available space.

## Roadmap

See [agent-os/product/roadmap.md](agent-os/product/roadmap.md) for the complete development roadmap.

**Current Status**: QLoRA Training Configuration (Item 7) ✅

**Next Steps**:
1. ✅ Environment Setup & Data Infrastructure
2. ✅ Literary Corpus Collection (Gutenberg downloads)
3. [ ] Reddit Humor Dataset Processing
4. [ ] Data Normalization & Deduplication Pipeline
5. [ ] Instructionalization & Tagging
6. [ ] Synthetic Tool-Use Data Generation
7. ✅ **QLoRA Training Configuration** (Dual-platform H100/M4)
8. [ ] Style-Only LoRA Training
9. [ ] Style Model Evaluation & Validation
10. [ ] Combined Style+Tool-Use Training
11. [ ] Tool-Use Evaluation Harness
12. [ ] Model Serving & Deployment

## Technology Stack

### Local Environment
- **Python 3.10** with venv
- **Data Processing**: pandas, BeautifulSoup4, trafilatura, datasets, datasketch, jsonlines, NLTK, requests

### H100 Training Environment
- **Python 3.10** with conda
- **ML Frameworks**: PyTorch 2.1+, Transformers 4.36+, PEFT 0.7+, TRL 0.7.4
- **Training**: QLoRA with 4-bit NF4 quantization (bitsandbytes 0.41+)
- **Optimization**: Flash Attention 2, gradient checkpointing, bfloat16 mixed precision
- **GPU**: CUDA 12.1+ (H100 80GB, A100 40GB+, or RTX 3090 24GB)

### Mac M4 Training Environment
- **Python 3.10** with venv
- **ML Frameworks**: PyTorch 2.1+ with MPS backend, Transformers 4.36+, PEFT 0.7+, TRL 0.7.4
- **Training**: QLoRA with 4-bit NF4 quantization, gradient checkpointing, bfloat16
- **Optimization**: MPS acceleration (no Flash Attention), reduced batch size
- **Hardware**: Apple Silicon M1/M2/M3/M4 with 32GB unified memory

### Base Model
- **Mistral 7B Instruct v0.2** (`mistralai/Mistral-7B-Instruct-v0.2`)

## Documentation

### Setup Guides
- **[H100 Setup Guide](docs/SETUP_H100.md)**: Complete H100 GPU setup instructions
- **[M4 Setup Guide](docs/SETUP_M4.md)**: Complete Mac M4 setup instructions
- **[Setup Guide](docs/SETUP_GUIDE.md)**: General installation and configuration

### Training Guides
- **[Training Quickstart](docs/TRAINING_QUICKSTART.md)**: Fast-track training guide with platform detection
- **[Implementation Guide](references/IMPLEMENTATION_GUIDE.md)**: Consolidated training methodology

### Data Guides
- **[Literary Corpus](docs/LITERARY_CORPUS.md)**: Literary passage collection methodology
- **[Synthetic Data Generation](docs/SYNTHETIC_DATA_GENERATION.md)**: Claude Haiku API-based tool-use data generation
- **[Data Sync](docs/DATA_SYNC.md)**: Transferring data between environments
- **[Model Download](docs/MODEL_DOWNLOAD.md)**: HuggingFace authentication and downloads

## Contributing

This is a personal learning project demonstrating efficient LoRA fine-tuning with limited compute resources. The methodology is designed to be reproducible and educational.

## License

Base model used:
- **Mistral 7B**: Apache 2.0 License

Training data sources:
- **Project Gutenberg**: Public domain
- **Reddit Data**: Following Reddit API Terms of Service
