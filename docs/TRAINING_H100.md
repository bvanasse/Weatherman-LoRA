# Weatherman-LoRA H100 Training Guide

Complete guide for training the Weatherman LoRA adapter on NVIDIA H100 GPUs via RunPod.

## Overview

- **Platform:** RunPod H100 (80GB VRAM)
- **Training Time:** 3-4 hours
- **Estimated Cost:** $2-3 USD
- **Dataset:** 14,399 training examples (combined tool-use + humor)
- **Base Model:** Mistral 7B Instruct v0.3

---

## Prerequisites

### 1. RunPod Account

Create account at [runpod.io](https://www.runpod.io/):
- Sign up with email or GitHub
- Add payment method (credit card)
- Minimum balance: $5 recommended

### 2. System Requirements

**RunPod Instance:**
- GPU: NVIDIA H100 (80GB VRAM)
- CUDA: 12.1+ (pre-installed on RunPod)
- Storage: 50GB+ free space
- RAM: 32GB+ recommended
- Conda: Optional (auto-installed to /workspace/miniconda3 if missing)

**Local Machine (for setup):**
- SSH client
- Git
- Internet connection for data sync

### 3. Repository Setup

```bash
# Clone repository to local machine first
git clone https://github.com/yourusername/Weatherman-LoRA.git
cd Weatherman-LoRA

# Verify training data exists
ls -lh data/synthetic/final_train.jsonl
ls -lh data/synthetic/final_validation.jsonl

# Expected output:
# -rw-r--r--  39M final_train.jsonl
# -rw-r--r--  4.3M final_validation.jsonl
```

---

## Step 1: Provision RunPod H100 Instance

### Launch Instance

1. **Navigate to RunPod Console**
   - Go to https://www.runpod.io/console/pods
   - Click "Deploy" or "Rent"

2. **Select GPU**
   - Filter: GPU = H100
   - Select: "NVIDIA H100 PCIe 80GB"
   - Region: Choose closest to you
   - Expected rate: $2.49-$2.99/hour

3. **Configure Instance**
   ```
   Template: RunPod PyTorch 2.1
   Container Disk: 50GB minimum
   Volume: Not required (optional for persistence)
   Expose HTTP/SSH: Enable SSH (port 22)
   ```

4. **Launch**
   - Click "Deploy On-Demand"
   - Wait 1-2 minutes for provisioning
   - Note SSH connection details

### Connect to Instance

```bash
# SSH connection (provided by RunPod)
ssh root@<pod-id>.pods.runpod.io -p <port>

# Or use RunPod web terminal
# Click "Connect" → "Start Web Terminal"
```

---

## Step 2: Setup Training Environment

### Upload Repository

**Option A: Direct Git Clone (Recommended)**

```bash
# On RunPod instance
cd /workspace
git clone https://github.com/yourusername/Weatherman-LoRA.git
cd Weatherman-LoRA
```

**Option B: SCP from Local Machine**

```bash
# From local machine
scp -P <port> -r Weatherman-LoRA/ root@<pod-id>.pods.runpod.io:/workspace/
```

### Run Setup Script

```bash
# On RunPod instance
cd /workspace/Weatherman-LoRA

# Make script executable (if needed)
chmod +x setup_runpod_h100.sh

# Run setup
./setup_runpod_h100.sh
```

**Expected Output:**

```
============================================================
Weatherman-LoRA H100 RunPod Environment Setup
============================================================

[SETUP-H100] Starting RunPod H100 environment setup

Step 1: Verifying RunPod H100 environment...
⚠ WARNING: Conda not found. Installing Miniconda to /workspace/miniconda3...
Downloading Miniconda installer...
Installing Miniconda to /workspace/miniconda3...
✓ Miniconda installed successfully to /workspace/miniconda3
✓ Found: conda 24.1.2
✓ Found: CUDA 12.1
✓ Found: H100 GPU with 80GB VRAM

Step 2: Setting up conda environment...
✓ Conda environment created: weatherman-lora
✓ Activated weatherman-lora environment

Step 3: Installing Flash Attention 2...
✓ Flash Attention 2 installed

Step 4: Running validation checks...
✓ Environment validation passed
✓ GPU check passed
✓ Storage check passed
✓ Training config validation passed

Step 5: Setting up data paths...
✓ Data paths verified

Step 6: Configuring Weights & Biases...
✓ Weights & Biases authenticated

============================================================
✓ [SETUP-H100-COMPLETE] Ready for training
============================================================
```

**Setup Time:** 10-15 minutes

### Verify Installation

```bash
# Activate conda environment
conda activate weatherman-lora

# Check Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import flash_attn; print('Flash Attention 2: Installed')"

# Expected output:
# PyTorch: 2.1.0+cu121
# CUDA: True
# Flash Attention 2: Installed

# Verify GPU
nvidia-smi

# Expected: H100 with 80GB VRAM shown
```

---

## Step 3: Configure Training

### Review Training Config

```bash
# View configuration
cat configs/training_config_h100.yaml
```

**Key Parameters:**

```yaml
# H100 Optimized Settings
per_device_train_batch_size: 4      # H100 can handle larger batches
gradient_accumulation_steps: 2      # Effective batch size = 8
max_seq_length: 4096                # Full context for Mistral
learning_rate: 2e-4                 # Standard LoRA LR
num_train_epochs: 3                 # 3 epochs over 14K examples
save_steps: 500                     # Checkpoint every 500 steps
logging_steps: 10                   # Log every 10 steps

# LoRA Configuration
lora_r: 64                          # LoRA rank
lora_alpha: 128                     # Scaling factor
lora_dropout: 0.05                  # Dropout for regularization

# Flash Attention 2
use_flash_attention_2: true         # Faster training

# Weights & Biases
report_to: "wandb"                  # Remote monitoring
run_name: "weatherman-h100-$(date +%Y%m%d)"
```

### Optional: Customize Config

```bash
# Edit if needed
nano configs/training_config_h100.yaml

# Or create custom config
cp configs/training_config_h100.yaml configs/my_config.yaml
nano configs/my_config.yaml
```

---

## Step 4: Launch Training

### Start Training Script

```bash
# From /workspace/Weatherman-LoRA
./train_h100_runpod.sh

# Or with custom config
./train_h100_runpod.sh --config configs/my_config.yaml
```

**Expected Output:**

```
============================================================
Weatherman-LoRA H100 Training Execution
============================================================

[TRAINING-H100] Pre-flight checks starting

Step 1: Running pre-flight validation...
✓ Found training config: configs/training_config_h100.yaml
✓ Environment validation passed
✓ GPU check passed
✓ Storage check passed
✓ Training config validation passed
✓ Training data verified

Step 2: Setting up logging...
✓ Log file created: logs/training_20251103_184530.log

Step 3: Checking for existing checkpoints...
No existing checkpoints found. Starting fresh training.

Step 4: Launching training in tmux session...
[TRAINING-H100] Starting training in tmux session: weatherman-training
✓ Training launched successfully

Step 5: Monitoring initial training steps...
✓ Training started successfully (first 100 steps)

============================================================
✓ [TRAINING-H100-STARTED] Training in progress
============================================================

[TRAINING-H100] Estimated duration: 3-4 hours

Monitoring Commands:
  Reconnect to session:  tmux attach -t weatherman-training
  Detach from session:   Ctrl+B, then D
  View log file:         tail -f logs/training_20251103_184530.log
  Check GPU usage:       nvidia-smi

Remote Monitoring:
  Weights & Biases:      https://wandb.ai
[TRAINING-H100] Monitor remotely via W&B dashboard
```

### Detach from SSH

Training runs in `tmux`, so you can safely disconnect:

```bash
# Detach from tmux session
# Press: Ctrl+B, then D

# Exit SSH
exit
```

**Training continues running even after you disconnect!**

---

## Step 5: Monitor Training

### Option A: Weights & Biases (Recommended)

1. **Open W&B Dashboard**
   - Go to https://wandb.ai
   - Sign in with your account
   - Find project: `weatherman-lora`

2. **Monitor Metrics**
   - Loss curves (train/validation)
   - Learning rate schedule
   - GPU utilization
   - Training speed (steps/second)
   - ETA to completion

3. **Alerts**
   - Set up email/Slack alerts for:
     - Training completion
     - Loss spikes
     - Errors

### Option B: SSH + tmux

```bash
# Reconnect to RunPod
ssh root@<pod-id>.pods.runpod.io -p <port>

# Attach to training session
tmux attach -t weatherman-training

# Detach again: Ctrl+B, then D
```

### Option C: Log File

```bash
# SSH to RunPod
ssh root@<pod-id>.pods.runpod.io -p <port>

# Tail log file
tail -f /workspace/Weatherman-LoRA/logs/training_*.log

# Check specific metrics
grep "loss" logs/training_*.log | tail -20
grep "epoch" logs/training_*.log
```

### Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Expected during training:
# GPU Utilization: 95-100%
# Memory Used: ~60-70GB / 80GB
# Power: ~350-400W
# Temperature: 60-80°C
```

---

## Step 6: Training Completion

### Verify Completion

After 3-4 hours, check training status:

```bash
# SSH to RunPod
ssh root@<pod-id>.pods.runpod.io -p <port>

# Check tmux session status
tmux ls

# If session exited, training is complete
# Attach to check final output
tmux attach -t weatherman-training

# Or check log file
tail -50 /workspace/Weatherman-LoRA/logs/training_*.log
```

**Expected Final Output:**

```
Training Epoch 3/3: 100%|██████████| 4800/4800 [1:23:45<00:00, 57.23 steps/s]
Validation Loss: 0.3421
Saving final model checkpoint...
[TRAINING-H100-COMPLETE] Training finished
Model saved to: adapters/weatherman-lora-h100/
```

### Verify Model Files

```bash
cd /workspace/Weatherman-LoRA/adapters/weatherman-lora-h100

# List files
ls -lh

# Expected output:
# adapter_config.json (1KB)
# adapter_model.bin (500MB-2GB)
# tokenizer files...
# training_args.bin
```

### Download Model to Local Machine

```bash
# From local machine
scp -P <port> -r root@<pod-id>.pods.runpod.io:/workspace/Weatherman-LoRA/adapters/weatherman-lora-h100 ./adapters/

# Or use rsync for resume capability
rsync -avz --progress -e "ssh -p <port>" \
  root@<pod-id>.pods.runpod.io:/workspace/Weatherman-LoRA/adapters/weatherman-lora-h100/ \
  ./adapters/weatherman-lora-h100/
```

**Download Size:** 500MB-2GB (5-15 minutes depending on connection)

---

## Step 7: Cleanup RunPod Instance

### Stop Instance

```bash
# From RunPod Console
# Click "Stop" on your pod

# Or via CLI
runpod pod stop <pod-id>
```

**Important:** Stop the pod immediately after download to avoid charges!

### Estimated Cost Summary

```
Training Time: 3.5 hours
Hourly Rate: $2.79/hour
Total Training: $9.77

Setup/Download: 0.5 hours
Total: $11.16

(Rates vary by region and availability)
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**

```bash
# Symptoms
RuntimeError: CUDA out of memory

# Solutions
# 1. Reduce batch size in config
per_device_train_batch_size: 2  # Down from 4

# 2. Reduce sequence length
max_seq_length: 2048  # Down from 4096

# 3. Increase gradient accumulation
gradient_accumulation_steps: 4  # Up from 2
```

**2. CUDA Version Mismatch**

```bash
# Symptoms
RuntimeError: CUDA version mismatch

# Solution
# Re-run setup script
./setup_runpod_h100.sh
```

**3. Flash Attention Installation Fails**

```bash
# Symptoms
ERROR: Failed to build flash-attn

# Solution
# Install manually without build isolation
pip install flash-attn --no-build-isolation --no-cache-dir

# Or disable in config
use_flash_attention_2: false
```

**4. Training Crashes After Resume**

```bash
# Symptoms
ERROR: Crash loop detected

# Solution
# Remove checkpoint and restart
rm -rf adapters/weatherman-lora-h100/checkpoint-*
./train_h100_runpod.sh
```

**5. Slow Training Speed**

```bash
# Expected: 50-60 steps/second
# If seeing <20 steps/second:

# Check GPU utilization
nvidia-smi

# Ensure Flash Attention is enabled
python -c "import flash_attn; print('OK')"

# Check CPU bottleneck
htop

# Increase dataloader workers in config
dataloader_num_workers: 4
```

### Validation Failures

```bash
# Re-run validation manually
python scripts/validate_environment.py --env=h100
python scripts/check_gpu.py
python scripts/validate_training_config.py --config configs/training_config_h100.yaml
```

---

## Best Practices

### Before Training

1. **Verify dataset integrity**
   ```bash
   wc -l data/synthetic/final_train.jsonl
   # Should show 14399
   ```

2. **Test on small subset first**
   ```yaml
   # In config
   max_steps: 100  # Just 100 steps for testing
   ```

3. **Set up W&B monitoring**
   ```bash
   wandb login
   ```

### During Training

1. **Monitor first 30 minutes closely**
   - Loss should decrease steadily
   - GPU utilization should be 95-100%
   - No memory errors

2. **Check checkpoints periodically**
   ```bash
   ls -lh adapters/weatherman-lora-h100/checkpoint-*/
   ```

3. **Don't stop mid-training**
   - Use tmux persistence
   - Let it run to completion

### After Training

1. **Verify model quality**
   ```bash
   # Quick test inference
   python scripts/test_inference.py --adapter adapters/weatherman-lora-h100
   ```

2. **Back up to multiple locations**
   ```bash
   # Local + Cloud storage
   rsync -avz adapters/weatherman-lora-h100/ ~/backup/
   aws s3 sync adapters/weatherman-lora-h100/ s3://my-bucket/weatherman/
   ```

3. **Document training run**
   - Save W&B run link
   - Note final loss values
   - Record any issues encountered

---

## Next Steps

After successful training:

1. **Deploy Model** → See [DEPLOYMENT.md](./DEPLOYMENT.md)
2. **Test Inference** → See Section 4 of DEPLOYMENT.md
3. **Fine-tune Further** → Adjust hyperparameters and retrain if needed

---

## Additional Resources

- [RunPod Documentation](https://docs.runpod.io/)
- [H100 GPU Specs](https://www.nvidia.com/en-us/data-center/h100/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [LoRA Training Guide](https://huggingface.co/docs/peft/task_guides/lora)
- [Weights & Biases Docs](https://docs.wandb.ai/)

For issues, see project repository or open a GitHub issue.
