# H100 GPU Setup Guide

Complete setup instructions for training Weatherman-LoRA on H100 GPU with CUDA 12.1+.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA H100 (80GB VRAM) or similar high-end GPU (A100, RTX 3090 with 24GB+)
- **CPU**: Multi-core CPU (8+ cores recommended)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free disk space

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **CUDA**: 12.1 or newer
- **Python**: 3.10
- **NVIDIA Drivers**: Latest version compatible with CUDA 12.1+

## Setup Steps

### 1. Verify NVIDIA Drivers and CUDA

Check that NVIDIA drivers are installed:

```bash
nvidia-smi
```

You should see your H100 GPU listed with driver version and CUDA version.

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.1   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA H100 80G...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0    67W / 300W |      0MiB / 81920MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### 2. Install Conda (if not already installed)

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Activate
source ~/.bashrc
```

### 3. Clone the Repository

```bash
git clone <repository-url> Weatherman-LoRA
cd Weatherman-LoRA
```

### 4. Create Conda Environment

```bash
# Create environment from config
conda env create -f environment-remote.yml

# Activate environment
conda activate weatherman-lora
```

This will install:
- PyTorch 2.1.0 with CUDA 12.1 support
- Transformers 4.36.0 (Mistral 7B Instruct compatible)
- PEFT 0.7.0 (LoRA adapters)
- TRL 0.7.4 (SFTTrainer)
- bitsandbytes 0.41.0 (4-bit quantization)
- Other dependencies

### 5. Install Flash Attention 2 (Optional but Recommended)

Flash Attention 2 provides 2-4x speedup for long sequences (4096 tokens):

```bash
# Must be in activated conda environment
conda activate weatherman-lora

# Install Flash Attention 2
pip install flash-attn --no-build-isolation
```

Verify installation:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

Expected output: `2.3.0` or newer

**Note**: If Flash Attention 2 fails to install, training will still work but will be slower. The H100 config is optimized for Flash Attention but falls back gracefully.

### 6. Validate Environment

```bash
# Check Python version
python --version  # Should be 3.10.x

# Validate H100-specific environment
python scripts/validate_environment.py --env=h100
```

Expected output:
```
H100 ENVIRONMENT VALIDATION
✅ Python Version              PASS
✅ Remote Dependencies         PASS
✅ H100-Specific Dependencies  PASS
✅ Mistral Compatibility       PASS
✅ Training Dependencies       PASS
✅ GPU Availability            PASS
✅ Storage                     PASS
✅ Directory Structure         PASS
✅ Configuration Files         PASS
✅ Scripts                     PASS

Overall: 10/10 checks passed
🎉 H100 environment is ready for training!
```

### 7. Run GPU Diagnostics

```bash
python scripts/check_gpu.py
```

Expected output:
```
Weatherman-LoRA Platform Diagnostics
PyTorch Version:     2.1.0+cu121

CUDA Platform Detection
CUDA Available:      True
CUDA Version:        12.1

GPU Details:
GPU 0:
  Name:              NVIDIA H100 80GB HBM3
  Total Memory:      80.00 GB
  Memory Assessment:
    ✅ EXCELLENT: 80.00 GB (H100-class)
       Ideal for fast training with large batch sizes

Recommended Training Configuration:
✅ H100-CLASS GPU DETECTED
   Recommended config: configs/training_config_h100.yaml
   Expected training time: 3-4 hours for 3 epochs
   Sequence length: 4096 tokens
   Batch size: 4-8
   Flash Attention 2: Enabled
```

### 8. Validate Training Configuration

```bash
python scripts/validate_training_config.py --config configs/training_config_h100.yaml
```

All checks should pass.

## Training Configuration

The H100-optimized configuration is in `configs/training_config_h100.yaml`:

### Key Settings:
- **Base Model**: Mistral 7B Instruct v0.3 (`mistralai/Mistral-7B-Instruct-v0.3`)
- **Sequence Length**: 4096 tokens (with Flash Attention 2)
- **Batch Size**: 4 per device
- **Gradient Accumulation**: 4 steps (effective batch size: 16)
- **LoRA Parameters**: r=16, alpha=32, dropout=0.05
- **Target Modules**: All 7 (q/k/v/o/gate/up/down projections)
- **Quantization**: 4-bit NF4 with double quantization
- **Mixed Precision**: bfloat16 (H100 native acceleration)
- **Gradient Checkpointing**: Enabled (saves 30-40% memory)

### Expected Performance:
- **Training Time**: 3-4 hours for 3 epochs (15K examples)
- **Memory Usage**: ~60-70GB VRAM peak
- **Throughput**: ~400-500 examples per minute

## Data Preparation

Ensure your training data is in the correct location:

```bash
# Training data (JSONL format with chat messages)
ls -lh data/processed/train.jsonl

# Validation data (optional)
ls -lh data/processed/val.jsonl
```

If data files don't exist, prepare them using the data pipeline (see main README.md).

## Starting Training

```bash
# Activate environment
conda activate weatherman-lora

# Start training (example command, update with your actual training script)
python scripts/train.py --config configs/training_config_h100.yaml
```

Monitor training with Weights & Biases:
```bash
# Set up wandb (first time only)
wandb login

# View logs
wandb login --relogin  # If needed
```

## Troubleshooting

### OOM (Out of Memory) Errors

If you encounter OOM errors even with H100 80GB:

1. **Reduce batch size** in `configs/training_config_h100.yaml`:
   ```yaml
   training:
     per_device_train_batch_size: 2  # Changed from 4
     gradient_accumulation_steps: 8  # Changed from 4 to maintain effective batch
   ```

2. **Reduce sequence length** (if still OOM):
   ```yaml
   model:
     max_seq_length: 2048  # Changed from 4096
   ```

3. **Verify gradient checkpointing is enabled**:
   ```yaml
   training:
     gradient_checkpointing: true
   ```

4. **Check for other processes using GPU memory**:
   ```bash
   nvidia-smi
   # Kill any unnecessary processes
   ```

### CUDA Out of Date

If CUDA version is older than 12.1:

```bash
# Update CUDA toolkit
conda install cuda-toolkit=12.1 -c nvidia

# Reinstall PyTorch
conda install pytorch=2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Flash Attention 2 Installation Fails

If Flash Attention 2 won't install:

```bash
# Check CUDA is available in Python
python -c "import torch; print(torch.cuda.is_available())"

# Try installing with verbose output
pip install flash-attn --no-build-isolation -v
```

**Fallback**: Training will work without Flash Attention 2, but will be slower (5-6 hours instead of 3-4 hours).

## Performance Monitoring

### During Training

Monitor GPU utilization:
```bash
watch -n 1 nvidia-smi
```

Expected values:
- **GPU Utilization**: 90-100%
- **Memory Used**: 60-70GB / 80GB
- **Temperature**: 60-80°C
- **Power Usage**: 250-300W

### After Training

Check training logs:
```bash
# View checkpoints
ls -lh adapters/weatherman-lora-h100/

# Check adapter size
du -sh adapters/weatherman-lora-h100/
```

Expected adapter size: ~200-400MB (LoRA adapters are much smaller than full model)

## Next Steps

After successful training:

1. **Evaluate adapters** (Roadmap Item 9)
2. **Test style consistency** (Roadmap Item 11)
3. **Deploy for inference** (Roadmap Item 12)

## Additional Resources

- [Mistral 7B Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691)
