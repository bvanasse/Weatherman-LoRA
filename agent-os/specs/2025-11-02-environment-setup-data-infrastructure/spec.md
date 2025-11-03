# Specification: Environment Setup & Data Infrastructure

## Goal
Create a dual-environment Python setup supporting local data processing on Mac M4 and remote GPU training on H100/3090 machines, with standardized project structure, dependency management, and data sync workflows for efficient LoRA fine-tuning.

## User Stories
- As an ML engineer, I want separate local and remote Python environments so that I can process data on my Mac and train models on GPU machines without conflicts
- As a developer, I want a standardized project directory structure so that all data, models, and scripts are organized consistently across environments

## Specific Requirements

**Local Environment (Mac M4 - Data Processing)**
- Create Python 3.10 virtual environment using venv (no CUDA dependencies)
- Install data processing libraries: pandas, BeautifulSoup4, trafilatura, datasets, datasketch, jsonlines, NLTK, language detection tools
- Pin exact versions in `requirements-local.txt` for reproducibility
- No GPU or Metal acceleration requirements (CPU-only processing)
- Lightweight setup optimized for data collection, cleaning, deduplication, and JSONL formatting
- Support ARM64 architecture compatibility for Mac M4

**Remote Environment (H100/3090 - GPU Training)**
- Create Python 3.10 conda environment for CUDA 12.1+ compatibility
- Install GPU-accelerated ML libraries: PyTorch 2.1+, Transformers 4.36+, PEFT 0.7+, TRL, Accelerate, bitsandbytes
- Pin exact versions in `environment-remote.yml` (conda) for reproducibility across Lambda/RunPod/home 3090
- Configure QLoRA with 4-bit quantization support (NF4, double quantization, bfloat16)
- Verify CUDA 12.1+ installation and GPU availability (do NOT install CUDA drivers)
- Validate GPU memory availability: minimum 24GB for RTX 3090, 80GB for H100

**Project Directory Structure**
- Create `data/raw/` for original downloads (Gutenberg texts, Reddit CSVs)
- Create `data/processed/` for cleaned/deduplicated JSONL datasets
- Create `data/synthetic/` for generated tool-use examples
- Create `models/` for base model storage (Llama 3.1 8B, Mistral 7B)
- Create `adapters/` for LoRA weights and training checkpoints
- Create `scripts/` for Python training and processing scripts
- Create `configs/` for YAML/JSON configuration files
- Ensure directory structure is identical on local and remote machines

**Storage Verification**
- Check available disk space on both local and remote environments
- Warn if less than 30GB available, error if less than 20GB
- Expected storage breakdown: raw data (~150-300MB), processed JSONL (~500MB-1GB), base model (~15GB), checkpoints (~500MB-2GB), buffer (~10-15GB)
- Display storage requirements summary when running setup
- No need for 500GB (optimized for efficient dataset sizes per implementation guide)

**Dependency Version Pinning**
- Pin exact versions for critical ML libraries to ensure reproducibility
- Align versions with `references/IMPLEMENTATION_GUIDE.md` specifications
- Use `torch==2.1.0` (not `>=2.1.0`) style pinning for core libraries
- Separate local (data processing) and remote (GPU training) dependencies
- Include version comments in requirements files explaining critical pins

**CUDA and GPU Validation (Remote Only)**
- Verify CUDA version 12.1+ is installed (display version, do not install)
- Check GPU availability using `torch.cuda.is_available()`
- Display GPU memory: 24GB for RTX 3090, 80GB for H100
- Validate PyTorch CUDA compatibility with installed CUDA version
- Create diagnostic script that outputs GPU info for troubleshooting

**Data Sync Workflow Documentation**
- Document rsync command for syncing processed data from local to remote
- Provide scp alternative for single-file transfers
- Include example commands with placeholders for remote host/path
- Workflow: Process locally on Mac M4 → Sync to remote → Train on GPU → Sync adapters back
- Optional cloud storage sync approach (S3, Google Drive) for larger transfers

**Base Model Download Strategy**
- Provide Hugging Face Hub authentication instructions (token setup)
- Document model download commands for Llama 3.1 8B Instruct and Mistral 7B Instruct
- Download separately on each GPU machine (Lambda/RunPod/home 3090) to avoid 15GB transfers
- Configure HuggingFace cache directory in `models/` folder
- Include model verification script to confirm successful downloads

**Configuration Management**
- Create template YAML config for training hyperparameters (LoRA rank, alpha, dropout, learning rate)
- Create template JSON config for data paths and processing settings
- Provide path management constants (DATA_RAW, DATA_PROCESSED, MODELS_DIR, etc.)
- Support environment-specific overrides (local vs remote paths)
- Include example configs with comments explaining each parameter

**Setup Scripts and Automation**
- Create `setup_local.sh` script for Mac M4 environment initialization
- Create `setup_remote.sh` script for GPU machine environment initialization
- Include verification steps in each script (check Python version, disk space, dependencies)
- Display summary of installed packages and environment status
- Make scripts idempotent (safe to run multiple times)

## Visual Design
No visual assets provided.

## Existing Code to Leverage

**`references/IMPLEMENTATION_GUIDE.md`**
- Contains exact library versions to pin: PyTorch 2.1+, Transformers 4.36+, PEFT 0.7+, TRL, bitsandbytes 0.41+
- Provides realistic storage estimates (~20-30GB total vs. original 500GB assumption)
- Documents LoRA hyperparameters: rank 16-32, alpha 32-64, dropout 0.05-0.1, target modules list
- Specifies quantization config: 4-bit NF4, double quantization, bfloat16 compute dtype
- Lists data source URLs: Project Gutenberg API (gutendex.com), Open-Meteo weather API

**`data_sources/reddit-theonion/` directory**
- Already contains CSV files: `nottheonion_181217_184009.csv`, `TheOnion_181217_184244.csv`
- Demonstrates existing data storage pattern to follow
- Shows folder structure convention: source type → data subfolder
- Referenced in roadmap Item 3 for Reddit humor dataset processing

## Out of Scope
- Downloading Project Gutenberg texts or processing Reddit data (Roadmap Items 2 & 3)
- Implementing data cleaning, deduplication, or normalization pipelines (Roadmap Item 4)
- Creating instructionalization or tagging logic (Roadmap Item 5)
- Generating synthetic tool-use examples (Roadmap Item 6)
- Running any training experiments or model fine-tuning (Roadmap Items 8, 10)
- Building evaluation harnesses or scoring systems (Roadmap Items 9, 11)
- Setting up model serving infrastructure like vLLM or TGI (Roadmap Item 12)
- Installing CUDA drivers or GPU system software (assumes pre-installed on remote machines)
- Creating actual training scripts (only environment setup, not training logic)
- Downloading or managing base model weights during setup (only provide instructions, don't auto-download)
