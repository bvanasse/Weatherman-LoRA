# Training Quickstart Guide

Fast-track guide to training Weatherman-LoRA on H100 or Mac M4.

## Quick Platform Detection

**Step 1**: Detect your platform and get configuration recommendation

```bash
python scripts/check_gpu.py
```

This will tell you:
- Which platform you're on (H100/CUDA or Mac M4/MPS)
- Which config file to use
- Expected training time
- Memory requirements

## Platform-Specific Setup

### H100 GPU Training

**Recommended for**: Production training, faster iteration (3-4 hours)

```bash
# 1. Create conda environment
conda env create -f environment-remote.yml
conda activate weatherman-lora

# 2. Install Flash Attention 2 (optional but recommended)
pip install flash-attn --no-build-isolation

# 3. Validate environment
python scripts/validate_environment.py --env=h100

# 4. Validate configuration
python scripts/validate_training_config.py --config configs/training_config_h100.yaml
```

See [SETUP_H100.md](./SETUP_H100.md) for detailed instructions.

### Mac M4 Training

**Recommended for**: Local development, testing configs (8-12 hours)

```bash
# 1. Run automated setup
./setup_m4.sh

# 2. Activate environment
source .venv-m4/bin/activate

# 3. Validate environment
python scripts/validate_environment.py --env=m4

# 4. Validate configuration
python scripts/validate_training_config.py --config configs/training_config_m4.yaml
```

See [SETUP_M4.md](./SETUP_M4.md) for detailed instructions.

## Training Configurations Explained

### H100 Configuration (`configs/training_config_h100.yaml`)

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

### M4 Configuration (`configs/training_config_m4.yaml`)

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

### Why Same LoRA Parameters?

Both configs use identical LoRA parameters (r=16, alpha=32, dropout=0.05) to ensure **adapter compatibility**:
- Adapters trained on M4 work on H100
- Adapters trained on H100 work on M4
- Reproducible results across platforms
- Only training speed differs, not final quality

## Data Preparation

Ensure your training data is ready:

```bash
# Check training data exists
ls -lh data/processed/train.jsonl
ls -lh data/processed/val.jsonl

# Verify data format (should be chat format with messages)
head -n 1 data/processed/train.jsonl | python -m json.tool
```

Expected format:
```json
{
  "messages": [
    {"role": "system", "content": "You are a witty weather assistant..."},
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "Well, as Mark Twain might say..."}
  ]
}
```

If data doesn't exist, run the data pipeline first (see main README).

## Starting Training

### H100 Training

```bash
# Activate environment
conda activate weatherman-lora

# Optional: Set up wandb for monitoring
wandb login

# Start training (replace with your actual training command)
python scripts/train.py --config configs/training_config_h100.yaml

# Monitor GPU
watch -n 1 nvidia-smi
```

### M4 Training

```bash
# Activate environment
source .venv-m4/bin/activate

# Close unnecessary apps to free memory
# Check Activity Monitor: Memory Pressure should be Green

# Optional: Set up wandb for remote monitoring
wandb login

# Disable auto-sleep for overnight training
sudo pmset -a disablesleep 1

# Start training (replace with your actual training command)
python scripts/train.py --config configs/training_config_m4.yaml

# After training, re-enable auto-sleep
sudo pmset -a disablesleep 0
```

## Config Customization

You can override config values without modifying the base files:

### Option 1: Command-line Overrides (Recommended)

```python
# In your training script
from scripts.config_loader import load_training_config

# Load with overrides
config = load_training_config(
    config_path="configs/training_config_h100.yaml",
    overrides={
        'training': {
            'num_train_epochs': 5,  # Train for 5 epochs instead of 3
            'learning_rate': 1e-4,   # Lower learning rate
        }
    }
)
```

### Option 2: Create Custom Config

```bash
# Copy base config
cp configs/training_config_h100.yaml configs/my_custom_config.yaml

# Edit your custom config
nano configs/my_custom_config.yaml

# Use your custom config
python scripts/train.py --config configs/my_custom_config.yaml
```

### Common Customizations

**Reduce memory usage (if OOM)**:
```yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase from 4
```

**Faster iteration (fewer epochs)**:
```yaml
training:
  num_train_epochs: 1  # Quick test run
```

**Adjust LoRA capacity**:
```yaml
lora:
  r: 32  # Increase rank for more capacity (uses more memory)
  lora_alpha: 64  # Keep 2x rank
```

**Change learning rate**:
```yaml
training:
  learning_rate: 1.0e-4  # Lower for more stable training
  learning_rate: 3.0e-4  # Higher for faster convergence
```

## Monitoring Training

### Weights & Biases (Recommended)

Best for remote monitoring:

```bash
# Set up once
wandb login

# Training will automatically log to wandb
# View at: https://wandb.ai/your-username/weatherman-lora
```

Monitor:
- Training loss (should decrease)
- Evaluation loss (should decrease, watch for overfitting)
- Learning rate (should decay with cosine schedule)
- GPU utilization (should be ~90-100%)

### Terminal Logs

Watch for:
```
Epoch 1/3: 100%|████████| 2813/2813 [1:15:23<00:00, 1.61s/it]
Train Loss: 1.234
Eval Loss: 1.456
```

### GPU Monitoring (H100 only)

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Expected:
# - GPU Utilization: 90-100%
# - Memory Used: 60-70GB / 80GB
# - Temperature: 60-80°C
```

### Memory Monitoring (M4 only)

```bash
# Check memory pressure
sysctl vm.memory_pressure_percentage

# Use Activity Monitor
# - Memory tab
# - Memory Pressure: Green or Yellow (not Red)
```

## Checkpoints and Outputs

Training saves checkpoints periodically:

```bash
# View checkpoints
ls -lh adapters/weatherman-lora-h100/
# or
ls -lh adapters/weatherman-lora-m4/
```

Expected files:
```
adapter_config.json    # LoRA configuration
adapter_model.bin      # Trained adapter weights (~200-400MB)
checkpoint-500/        # Intermediate checkpoint
checkpoint-1000/       # Intermediate checkpoint
checkpoint-2500/       # Final checkpoint (best model)
```

## Troubleshooting

### Out of Memory (OOM)

**H100**:
1. Reduce batch size: `per_device_train_batch_size: 2`
2. Increase gradient accumulation: `gradient_accumulation_steps: 8`
3. Reduce sequence length: `max_seq_length: 2048`

**M4**:
1. Close all unnecessary apps
2. Reduce sequence length: `max_seq_length: 1024`
3. Check Activity Monitor for memory pressure
4. Reduce LoRA rank: `r: 8`

### Training Too Slow

**H100**:
- Verify Flash Attention 2 is installed: `python -c "import flash_attn"`
- Check GPU utilization: `nvidia-smi` (should be 90-100%)
- Verify CUDA 12.1+: `nvidia-smi` header

**M4**:
- M4 is inherently 2-3x slower (expected)
- Verify MPS is being used: Check Activity Monitor GPU history
- Close background apps

### Validation Loss Not Decreasing

1. Check learning rate isn't too low: Try 2e-4
2. Increase number of epochs: Try 5 instead of 3
3. Check data quality: Verify `train.jsonl` format
4. Monitor for overfitting: Compare train vs eval loss

### Training Crashes

**H100**:
```bash
# Check CUDA errors
dmesg | grep -i cuda

# Verify GPU health
nvidia-smi
```

**M4**:
```bash
# Check crash logs
log show --predicate 'process == "Python"' --last 1h
```

## Next Steps After Training

1. **Find your trained adapters**:
   ```bash
   ls -lh adapters/weatherman-lora-*/adapter_model.bin
   ```

2. **Load and test adapters**:
   ```python
   from peft import PeftModel
   from transformers import AutoModelForCausalLM

   base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
   model = PeftModel.from_pretrained(base_model, "adapters/weatherman-lora-h100")
   ```

3. **Evaluate style consistency** (Roadmap Item 11)
4. **Deploy for inference** (Roadmap Item 12)

## Quick Reference

### Platform Detection
```bash
python scripts/check_gpu.py
```

### Environment Validation
```bash
# H100
python scripts/validate_environment.py --env=h100

# M4
python scripts/validate_environment.py --env=m4
```

### Config Validation
```bash
# H100
python scripts/validate_training_config.py --config configs/training_config_h100.yaml

# M4
python scripts/validate_training_config.py --config configs/training_config_m4.yaml
```

### Load Config in Python
```python
from scripts.config_loader import load_training_config

config = load_training_config(config_path="configs/training_config_h100.yaml")
```

## Additional Resources

- [H100 Setup Guide](./SETUP_H100.md) - Detailed H100 setup
- [M4 Setup Guide](./SETUP_M4.md) - Detailed M4 setup
- [Mistral 7B Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
