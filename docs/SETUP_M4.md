# Mac M4 Setup Guide

Complete setup instructions for training Weatherman-LoRA on Mac M4 with MPS backend and 32GB unified memory.

## Prerequisites

### Hardware Requirements
- **Mac**: M1, M2, M3, or M4 chip (optimized for M4)
- **Unified Memory**: 32GB recommended (16GB minimum but may require aggressive optimization)
- **Storage**: 50GB+ free disk space

### Software Requirements
- **OS**: macOS 12.3+ (Monterey or newer)
- **Python**: 3.10 or newer
- **Xcode Command Line Tools**: Latest version

## Setup Steps

### 1. Verify Apple Silicon

Check that you're running on Apple Silicon:

```bash
uname -m
```

Expected output: `arm64`

Check system memory:

```bash
sysctl hw.memsize
```

Expected output: `hw.memsize: 34359738368` (32GB) or higher

### 2. Install Xcode Command Line Tools

```bash
xcode-select --install
```

Follow the prompts to complete installation.

### 3. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 4. Install Python 3.10

```bash
# Install Python 3.10
brew install python@3.10

# Verify installation
python3.10 --version
```

Expected output: `Python 3.10.x`

### 5. Clone the Repository

```bash
git clone <repository-url> Weatherman-LoRA
cd Weatherman-LoRA
```

### 6. Run Setup Script

The automated setup script will:
- Verify Python 3.10+
- Check Apple Silicon and memory
- Create virtual environment
- Install MPS-compatible dependencies
- Validate MPS backend

```bash
# Make script executable (if needed)
chmod +x setup_m4.sh

# Run setup
./setup_m4.sh
```

Expected output:
```
Mac M4 Training Environment Setup Complete!
Virtual environment: .venv-m4/
Python version: Python 3.10.x
Unified memory: 32GB
MPS backend: Available ✓
```

### 7. Activate Virtual Environment

```bash
source .venv-m4/bin/activate
```

To deactivate later:
```bash
deactivate
```

### 8. Validate Environment

```bash
# Validate M4-specific environment
python scripts/validate_environment.py --env=m4
```

Expected output:
```
M4 ENVIRONMENT VALIDATION
✅ Python Version              PASS
✅ Local Dependencies          PASS
✅ M4-Specific Dependencies    PASS
✅ Mistral Compatibility       PASS
✅ Training Dependencies       PASS
✅ Storage                     PASS
✅ Directory Structure         PASS
✅ Configuration Files         PASS
✅ Scripts                     PASS

Overall: 9/9 checks passed
🎉 Mac M4 environment is ready for training!
```

### 9. Run Platform Diagnostics

```bash
python scripts/check_gpu.py
```

Expected output:
```
Weatherman-LoRA Platform Diagnostics
PyTorch Version:     2.1.0

MPS (Metal) Platform Detection
MPS Available:       True
Unified Memory:      32.00 GB

Memory Assessment:
  ✅ SUFFICIENT: 32.00 GB
     Suitable for M4 training with reduced batch size
     Use batch_size=1-2, seq_length=2048

Recommended Training Configuration:
✅ MAC M4 (MPS) PLATFORM DETECTED
   Recommended config: configs/training_config_m4.yaml
   Expected training time: 8-12 hours for 3 epochs
   Sequence length: 2048 tokens (reduced for memory)
   Batch size: 1-2
   Flash Attention 2: Not available (CUDA-only)
```

### 10. Validate Training Configuration

```bash
python scripts/validate_training_config.py --config configs/training_config_m4.yaml
```

All checks should pass.

## Training Configuration

The M4-optimized configuration is in `configs/training_config_m4.yaml`:

### Key Settings:
- **Base Model**: Mistral 7B Instruct v0.3 (`mistralai/Mistral-7B-Instruct-v0.3`)
- **Sequence Length**: 2048 tokens (reduced for 32GB memory)
- **Batch Size**: 1 per device (conservative for memory)
- **Gradient Accumulation**: 16 steps (effective batch size: 16)
- **LoRA Parameters**: r=16, alpha=32, dropout=0.05 (same as H100)
- **Target Modules**: All 7 (q/k/v/o/gate/up/down projections)
- **Quantization**: 4-bit NF4 with double quantization
- **Mixed Precision**: bfloat16 (MPS supports bfloat16)
- **Gradient Checkpointing**: Enabled (essential for 32GB)

### Key Differences from H100:
- **No Flash Attention 2**: CUDA-only, not available on MPS
- **Smaller sequence length**: 2048 vs 4096 (saves ~50% activation memory)
- **Smaller batch size**: 1 vs 4 (memory-constrained)
- **Higher gradient accumulation**: 16 vs 4 (maintains effective batch)
- **Fewer dataloader workers**: 2 vs 4 (reduces memory pressure)

### Expected Performance:
- **Training Time**: 8-12 hours for 3 epochs (15K examples)
- **Memory Usage**: ~24-28GB unified memory peak
- **Throughput**: ~150-200 examples per minute (2-3x slower than H100)

## Memory Optimization

### Before Training

**Close all unnecessary applications** to free unified memory:

```bash
# Check current memory usage
top -l 1 | grep PhysMem

# Close:
# - Web browsers (Chrome, Safari, Firefox)
# - Slack, Discord, communication apps
# - IDEs (VSCode, PyCharm) if not needed
# - Docker Desktop
# - Other memory-intensive apps
```

**Monitor Activity Monitor**:
- Open Activity Monitor
- Select "Memory" tab
- Watch "Memory Pressure" indicator
- Keep it in "Green" zone during training

### During Training

**Do not**:
- Open web browsers
- Run other Python processes
- Start Docker containers
- Use memory-intensive apps

**Do**:
- Leave terminal open with training running
- Use `wandb` for remote monitoring
- Check Activity Monitor occasionally
- Train overnight or during off-hours

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
source .venv-m4/bin/activate

# Start training (example command, update with your actual training script)
python scripts/train.py --config configs/training_config_m4.yaml
```

**Recommended**: Train overnight (8-12 hours):

```bash
# Disable auto-sleep
sudo pmset -a disablesleep 1

# Start training
python scripts/train.py --config configs/training_config_m4.yaml

# After training completes, re-enable auto-sleep
sudo pmset -a disablesleep 0
```

Monitor training with Weights & Biases:
```bash
# Set up wandb (first time only)
wandb login

# Access from any device
# Go to wandb.ai and view your run
```

## Troubleshooting

### OOM (Out of Memory) Errors

If you encounter OOM errors:

1. **Close all unnecessary applications**
   - Use Activity Monitor to identify memory hogs
   - Force quit if needed

2. **Reduce batch size to 1** (if not already):
   ```yaml
   training:
     per_device_train_batch_size: 1
     gradient_accumulation_steps: 16
   ```

3. **Reduce sequence length to 1024**:
   ```yaml
   model:
     max_seq_length: 1024  # Halves activation memory
   ```

4. **Reduce LoRA rank** (last resort):
   ```yaml
   lora:
     r: 8  # Changed from 16, smaller adapters
     lora_alpha: 16  # Keep 2x rank
   ```

5. **Verify gradient checkpointing is enabled**:
   ```yaml
   training:
     gradient_checkpointing: true
   ```

6. **Check system memory**:
   ```bash
   # View memory usage
   vm_stat

   # Check memory pressure
   sysctl vm.memory_pressure_percentage
   ```

### MPS Backend Not Available

If MPS is not detected:

```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"
# Should be 2.1.0 or newer

# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print True

# Update PyTorch if needed
pip install --upgrade torch torchvision torchaudio
```

### Training Too Slow

M4 training is inherently 2-3x slower than H100. If training seems abnormally slow:

1. **Verify MPS is being used**:
   ```python
   import torch
   print(torch.backends.mps.is_available())  # Should be True
   device = torch.device("mps")
   x = torch.randn(100, 100, device=device)  # Should not error
   ```

2. **Check Activity Monitor**:
   - "GPU History" should show activity
   - "Memory Pressure" should be in green/yellow (not red)

3. **Reduce dataloader workers** if CPU-bound:
   ```yaml
   training:
     dataloader_num_workers: 1  # Reduce from 2
   ```

### System Freezing

If macOS becomes unresponsive during training:

1. **Force quit the training process**:
   - Press `Cmd + Option + Esc`
   - Select Python process
   - Click "Force Quit"

2. **Reduce memory usage**:
   - Lower batch size to 1
   - Lower sequence length to 1024
   - Close all other apps before retrying

3. **Use `nice` to lower process priority**:
   ```bash
   nice -n 10 python scripts/train.py --config configs/training_config_m4.yaml
   ```

## Performance Monitoring

### Memory Pressure

Monitor memory pressure during training:

```bash
# View memory pressure percentage
sysctl vm.memory_pressure_percentage

# View detailed memory stats
vm_stat 1
```

Target: Keep memory pressure below 50%

### Activity Monitor

Watch these indicators:
- **Memory Pressure**: Green or Yellow (not Red)
- **GPU History**: Should show consistent activity
- **CPU Usage**: 200-400% (2-4 cores active)
- **Energy Impact**: High (expected during training)

### Training Logs

Check progress in terminal or wandb:
- **Examples/second**: 2-3 examples/sec (expected for batch size 1)
- **Loss**: Should decrease over time
- **Eval loss**: Monitor for overfitting

## Expected Training Timeline

For 15K examples, 3 epochs, batch size 1, gradient accumulation 16:

- **Total steps**: ~2,813 steps
- **Time per step**: ~10-15 seconds
- **Total time**: 8-12 hours

Progress checkpoints:
- **After 1 hour**: ~7-10% complete
- **After 4 hours**: ~30-40% complete
- **After 8 hours**: ~60-80% complete
- **After 10-12 hours**: Complete

## Next Steps

After successful training:

1. **Find trained adapters**:
   ```bash
   ls -lh adapters/weatherman-lora-m4/
   ```

2. **Evaluate adapters** (Roadmap Item 9)
3. **Test style consistency** (Roadmap Item 11)
4. **Optionally transfer to H100 for inference** (faster)

## Tips for Better M4 Training Experience

1. **Train overnight**: Start training before bed, check in morning
2. **Use wandb**: Monitor remotely without keeping Mac awake
3. **Disable screen saver**: Prevent interruptions
4. **Disable notifications**: Reduce background activity
5. **Keep Mac plugged in**: Training uses significant power
6. **Good ventilation**: M4 may run warm (60-80°C is normal)
7. **Use Time Machine**: Back up before long training runs

## Additional Resources

- [PyTorch MPS Backend Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)
- [Mistral 7B Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [PEFT Documentation](https://huggingface.co/docs/peft)
