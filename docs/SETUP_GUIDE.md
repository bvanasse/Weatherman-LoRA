# Weatherman-LoRA Setup Guide

Complete setup instructions for local data processing and remote GPU training environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Environment Setup (Mac M4)](#local-environment-setup-mac-m4)
- [Remote Environment Setup (H100/3090)](#remote-environment-setup-h1003090)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Local Machine (Mac M4):**
- macOS (ARM64 architecture)
- Python 3.10
- 30-50GB free disk space
- Internet connection

**Remote GPU Machine:**
- Linux (Ubuntu 22.04 LTS recommended)
- NVIDIA GPU: H100 (80GB), A100 (40GB+), or RTX 3090 (24GB)
- CUDA 12.1+ drivers installed
- Python 3.10
- Conda or Miniconda
- 30-50GB free disk space
- Internet connection

### Software Prerequisites

#### Mac M4 (Local)

```bash
# Check Python 3.10
python3.10 --version
# or
python3 --version  # Should be 3.10.x

# If not installed
brew install python@3.10
```

#### GPU Machine (Remote)

```bash
# Check Python 3.10
python3.10 --version

# Check CUDA
nvcc --version  # Should be 12.1 or higher
nvidia-smi      # Should show your GPU

# Check conda
conda --version

# If conda not installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

---

## Local Environment Setup (Mac M4)

The local environment is used for data collection, processing, cleaning, and deduplication.

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-username/weatherman-lora.git
cd weatherman-lora

# Or if already cloned, navigate to it
cd ~/Apps/Weatherman-LoRA
```

### Step 2: Verify Storage

```bash
# Check available disk space
python3 scripts/check_storage.py
```

Expected output:
```
✅ Storage Check Passed!
   Available: 513.86 GB
   Recommended: 30 GB minimum
```

If you see a warning or error, free up disk space before continuing.

### Step 3: Run Local Setup Script

```bash
# Make script executable (if not already)
chmod +x setup_local.sh

# Run setup
./setup_local.sh
```

The script will:
1. Verify Python 3.10 is installed
2. Create virtual environment in `.venv-local/`
3. Install data processing dependencies
4. Run storage verification
5. Display setup summary

Expected output:
```
============================================================
Local Environment Setup Complete!
============================================================

Virtual environment: .venv-local/
Python version: Python 3.10.x

To activate the environment:
  source .venv-local/bin/activate
```

### Step 4: Activate Environment

```bash
# Activate virtual environment
source .venv-local/bin/activate

# Verify installation
python -c "import pandas, beautifulsoup4, trafilatura; print('✅ All imports successful')"
```

### Step 5: Test Path Configuration

```bash
# Test path constants
python scripts/paths.py
```

Expected output:
```
Weatherman-LoRA Path Configuration
============================================================
Base Directory:       /Users/you/Apps/Weatherman-LoRA

Data Directories:
  DATA_RAW:           /Users/you/Apps/Weatherman-LoRA/data/raw
  DATA_PROCESSED:     /Users/you/Apps/Weatherman-LoRA/data/processed
  ...
```

### Step 6: Validate Environment

```bash
# Run validation script
python scripts/validate_environment.py --env=local
```

---

## Remote Environment Setup (H100/3090)

The remote environment is used for GPU-accelerated LoRA training.

### Step 1: Access Remote Machine

```bash
# SSH into your GPU machine
# Lambda Labs
ssh ubuntu@xxx.xxx.xxx.xxx

# RunPod (custom port)
ssh -p 22022 root@runpod.io

# Home 3090
ssh user@192.168.1.100
```

### Step 2: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-username/weatherman-lora.git
cd weatherman-lora
```

### Step 3: Verify GPU and CUDA

```bash
# Check GPU
nvidia-smi

# Check CUDA version
nvcc --version
```

Expected CUDA version: 12.1 or higher

### Step 4: Verify Storage

```bash
# Check available disk space
python3 scripts/check_storage.py
```

### Step 5: Run Remote Setup Script

```bash
# Make script executable
chmod +x setup_remote.sh

# Run setup (takes 10-15 minutes)
./setup_remote.sh
```

The script will:
1. Verify conda installation
2. Check CUDA version
3. Create conda environment from `environment-remote.yml`
4. Install GPU-accelerated ML libraries
5. Run GPU diagnostics
6. Run storage verification
7. Display setup summary

Expected output:
```
============================================================
Remote GPU Environment Setup Complete!
============================================================

Environment: weatherman-lora
Python: Python 3.10.x
PyTorch: 2.1.0
CUDA available: True

To activate the environment:
  conda activate weatherman-lora
```

### Step 6: Activate Environment

```bash
# Activate conda environment
conda activate weatherman-lora

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: True
```

### Step 7: Run GPU Diagnostics

```bash
# Detailed GPU check
python scripts/check_gpu.py
```

Expected output:
```
✅ GPU ENVIRONMENT READY FOR TRAINING

   GPUs available: 1
   Total VRAM: 80.00 GB
   CUDA version: 12.1
```

### Step 8: Validate Environment

```bash
# Run validation script
python scripts/validate_environment.py --env=remote
```

---

## Configuration

### Training Configuration

Edit `configs/training_config.yaml` to customize training parameters:

```yaml
# Key parameters to adjust
lora:
  r: 16              # LoRA rank (16-32)
  lora_alpha: 32     # Scaling factor (typically 2x rank)

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4  # Reduce if OOM
  learning_rate: 2.0e-4

model:
  model_name_or_path: meta-llama/Meta-Llama-3.1-8B-Instruct
  # or: mistralai/Mistral-7B-Instruct-v0.3
```

### Paths Configuration

Edit `configs/paths_config.json` if using non-standard paths:

```json
{
  "data": {
    "processed": "data/processed",
    "raw": "data/raw"
  },
  "models": {
    "dir": "models"
  }
}
```

---

## Verification

### Local Environment Verification Checklist

- [ ] Python 3.10 installed
- [ ] Virtual environment created (`.venv-local/`)
- [ ] All dependencies installed (pandas, beautifulsoup4, etc.)
- [ ] Storage check passes (30GB+ available)
- [ ] Path constants work (`python scripts/paths.py`)
- [ ] Imports successful (pandas, transformers, etc.)
- [ ] Validation script passes

### Remote Environment Verification Checklist

- [ ] CUDA 12.1+ installed
- [ ] GPU detected (`nvidia-smi` shows GPU)
- [ ] Conda environment created (`weatherman-lora`)
- [ ] PyTorch installed with CUDA support
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] GPU diagnostics pass
- [ ] Storage check passes (30GB+ available)
- [ ] Validation script passes

---

## Troubleshooting

### Local Environment Issues

#### Python 3.10 Not Found

```bash
# Install via Homebrew
brew install python@3.10

# Verify
python3.10 --version
```

#### Dependency Installation Fails

```bash
# Upgrade pip
pip install --upgrade pip

# Retry installation
pip install -r requirements-local.txt
```

#### Storage Check Fails

```bash
# Free up disk space
# Delete old downloads, empty trash, etc.

# Re-run check
python scripts/check_storage.py
```

### Remote Environment Issues

#### CUDA Not Found

```bash
# Check if CUDA is installed
ls /usr/local/cuda*/

# Add to PATH if installed but not found
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Add to ~/.bashrc for persistence
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
```

#### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# If not working, may need to install/update drivers
# Lambda/RunPod usually have this pre-configured
```

#### PyTorch CUDA Not Available

```bash
# Reinstall PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

#### Conda Environment Creation Fails

```bash
# Clean conda cache
conda clean --all

# Remove partial environment
conda env remove -n weatherman-lora

# Retry
./setup_remote.sh
```

#### Out of Memory During Training

```yaml
# Reduce batch size in configs/training_config.yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size

# Or reduce sequence length
model:
  max_seq_length: 1024  # Reduce from 2048
```

---

## Next Steps

### After Local Setup

1. **Download Data Sources** (Roadmap Item 2)
   - Project Gutenberg texts
   - Process existing Reddit data

2. **Process Data** (Roadmap Items 3-5)
   - Clean and deduplicate
   - Format as JSONL
   - Create train/validation splits

3. **Sync to Remote** (see [DATA_SYNC.md](DATA_SYNC.md))

### After Remote Setup

1. **Download Base Model** (see [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md))
   - Authenticate with HuggingFace
   - Download Llama 3.1 8B or Mistral 7B

2. **Sync Processed Data** (see [DATA_SYNC.md](DATA_SYNC.md))
   - Transfer from local to remote

3. **Configure Training** (edit `configs/training_config.yaml`)

4. **Run Training** (Roadmap Items 8, 10)

5. **Sync Adapters Back** (see [DATA_SYNC.md](DATA_SYNC.md))

---

## Additional Resources

- [Data Sync Guide](DATA_SYNC.md) - Transfer files between local and remote
- [Model Download Guide](MODEL_DOWNLOAD.md) - Download base models
- [Implementation Guide](../references/IMPLEMENTATION_GUIDE.md) - Full training methodology
- [Product Roadmap](../agent-os/product/roadmap.md) - Development phases

---

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Run validation script: `python scripts/validate_environment.py --env=<local|remote>`
3. Review error messages carefully
4. Check [Implementation Guide](../references/IMPLEMENTATION_GUIDE.md) for detailed context

---

**Last Updated**: 2025-11-02
