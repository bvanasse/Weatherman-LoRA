# Weatherman-LoRA M4 Local Training Guide

Complete guide for training the Weatherman LoRA adapter on Mac M4 (Apple Silicon) locally.

## Overview

- **Platform:** Mac M4 (Apple Silicon)
- **Backend:** MPS (Metal Performance Shaders)
- **Training Time:** 12-18 hours
- **Estimated Cost:** $0 (local training)
- **Dataset:** 14,399 training examples (combined tool-use + humor)
- **Base Model:** Mistral 7B Instruct v0.2

---

## Prerequisites

### 1. Hardware Requirements

**Mac M4 Specifications:**
- Chip: Apple M4 (any variant: base, Pro, Max)
- Unified Memory: 16GB minimum, **32GB+ strongly recommended**
- Storage: 50GB+ free space
- macOS: 12.3+ (for MPS support)

**Memory Considerations:**
- 16GB: Possible but will experience memory pressure. Use batch_size=1, seq_length=1024
- 24GB: Workable with batch_size=1, seq_length=2048
- 32GB+: Recommended for stable training with batch_size=1, seq_length=2048

### 2. Software Requirements

**System Software:**
- macOS 12.3 or later (for MPS backend)
- Python 3.10+ (recommended: 3.10 or 3.11)
- Xcode Command Line Tools

**Python Environment:**
- PyTorch 2.1+ with MPS support
- Transformers, PEFT, TRL libraries
- All dependencies in `requirements-local.txt`

### 3. Repository Setup

```bash
# Clone repository
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

## Step 1: Environment Setup

### Run Setup Script

```bash
# Make script executable (if needed)
chmod +x setup_m4_local.sh

# Run setup
./setup_m4_local.sh
```

**Expected Output:**

```
============================================================
Weatherman-LoRA Mac M4 Local Training Environment Setup
============================================================

[SETUP-M4] Starting Mac M4 local setup

Step 1: Verifying Mac M4 hardware...
✓ Found: Apple M4 Pro
✓ Found: Python 3.11.5

Step 2: Checking unified memory...
Unified Memory: 32GB
SUFFICIENT: 32GB
✓ Unified Memory: 32GB

Step 3: Creating virtual environment...
✓ Virtual environment created
✓ Activated .venv-local
✓ pip upgraded to 23.3.1

Step 4: Installing packages from requirements-local.txt...
Installing dependencies (this may take 5-10 minutes)...
✓ Packages installed
PyTorch version: 2.1.0

Step 5: Validating MPS backend...
✓ MPS backend available
Testing MPS computation...
✓ MPS computation test passed

Step 6: Running M4-specific validation...
✓ Environment validation passed
✓ Training config validation passed
✓ Storage check passed
Setting up data paths...
✓ Created symlink: data/processed/train.jsonl
✓ Created symlink: data/processed/val.jsonl

============================================================
✓ [SETUP-M4-COMPLETE] Ready for local training
============================================================

⚠ WARNING: M4 training takes 12-18 hours (vs 3-4 hours on H100)
⚠ Close unnecessary applications to free unified memory

Environment Details:
  Virtual Environment: .venv-local
  Python: 3.11.5
  PyTorch: 2.1.0
  Chip: Apple M4 Pro
  Unified Memory: 32GB
  MPS Backend: Available ✓

Training Configuration:
  Config file: configs/training_config_m4.yaml
  Estimated time: 12-18 hours
  Batch size: 1 (vs 4-8 on H100)
  Sequence length: 2048 tokens (vs 4096 on H100)
  Checkpoints: every 250 steps (vs 500 on H100)

Memory Optimization Tips:
  - Close all browsers, IDEs, and unnecessary apps
  - Monitor Activity Monitor for memory pressure
  - Train overnight or during off-hours
  - Use wandb for remote monitoring

Next Steps:
  1. Activate environment: source .venv-local/bin/activate
  2. Start training: ./train_m4_local.sh

[SETUP-M4] Setup completed successfully
```

**Setup Time:** 10-15 minutes

### Verify Installation

```bash
# Activate virtual environment
source .venv-local/bin/activate

# Check Python packages
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"

# Expected output:
# PyTorch: 2.1.0
# MPS Available: True

# Check system info
system_profiler SPHardwareDataType | grep "Chip\|Memory"

# Expected: Shows Apple M4 variant and total memory
```

---

## Step 2: Pre-Training Preparation

### Close Unnecessary Applications

**Critical for stable training:**

```bash
# Close these applications before training:
# - Web browsers (Safari, Chrome, Firefox)
# - IDEs (Xcode, VS Code, IntelliJ)
# - Docker Desktop
# - Slack, Discord, messaging apps
# - Media players
# - Any other memory-intensive apps

# Keep only:
# - Terminal (for training)
# - Activity Monitor (for monitoring)
# - Finder (minimal usage)
```

### Monitor System Resources

```bash
# Open Activity Monitor
open -a "Activity Monitor"

# In Activity Monitor:
# 1. Click "Memory" tab
# 2. Check "Memory Pressure" graph (should be green)
# 3. Monitor "App Memory" and "Cached Files"

# Goal: At least 20GB free memory before training
```

### Review Training Config

```bash
# View configuration
cat configs/training_config_m4.yaml
```

**Key Parameters:**

```yaml
# M4 Optimized Settings
per_device_train_batch_size: 1      # M4 memory constraint
gradient_accumulation_steps: 8      # Effective batch size = 8
max_seq_length: 2048                # Half of H100 (4096)
learning_rate: 2e-4                 # Standard LoRA LR
num_train_epochs: 3                 # 3 epochs over 14K examples
save_steps: 250                     # Checkpoint every 250 steps
logging_steps: 10                   # Log every 10 steps

# LoRA Configuration
lora_r: 64                          # LoRA rank
lora_alpha: 128                     # Scaling factor
lora_dropout: 0.05                  # Dropout for regularization

# MPS Backend
use_mps: true                       # Enable MPS backend
bf16: false                         # MPS doesn't support bfloat16
fp16: false                         # Use full precision on MPS

# Weights & Biases
report_to: "wandb"                  # Remote monitoring
run_name: "weatherman-m4-$(date +%Y%m%d)"
```

### Optional: Customize Config

```bash
# Edit if needed (e.g., reduce sequence length for 16GB M4)
nano configs/training_config_m4.yaml

# For 16GB M4, use these settings:
# max_seq_length: 1024
# gradient_accumulation_steps: 16
# save_steps: 100
```

---

## Step 3: Launch Training

### Start Training Script

```bash
# Ensure virtual environment is activated
source .venv-local/bin/activate

# Launch training
./train_m4_local.sh

# Or with custom config
./train_m4_local.sh --config configs/my_config.yaml
```

**Expected Output:**

```
============================================================
Weatherman-LoRA M4 Local Training Execution
============================================================

[TRAINING-M4] Pre-flight checks starting

Step 1: Running pre-flight validation...
✓ Found training config: configs/training_config_m4.yaml
✓ Environment validation passed
✓ Training config validation passed
✓ Storage check passed
✓ Training data verified

Step 2: Verifying M4-specific configuration...
Training configuration from configs/training_config_m4.yaml:
  Batch size per device: 1
  Gradient accumulation: 8
  Checkpoint frequency: 250 steps
  Max sequence length: 2048 tokens
✓ M4 configuration verified
Recommended: batch_size=1, grad_accum=8, seq_length=2048, checkpoint_steps=250

Step 3: Setting up logging...
[TRAINING-M4] Log file: logs/training_m4_20251103_140530.log
✓ Log file created: logs/training_m4_20251103_140530.log

============================================================
[TRAINING-M4] Starting training on MPS backend
============================================================

Training will run in foreground (not in tmux)
Press Ctrl+C to stop training (progress will be saved)

⚠ If training slows, check Activity Monitor for thermal throttling
To monitor thermal state manually:
  sudo powermetrics --samplers smc -n 1

Launching training...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Training output begins here...]
```

**Training Process:**

Training will run in the foreground (not in tmux like H100). You'll see:

```
Epoch 1/3: [██░░░░░░░░░░░░░░░░░░] 10% | Step 480/4800 | Loss: 0.542 | 2.3 steps/s
Epoch 1/3: [████░░░░░░░░░░░░░░░░] 20% | Step 960/4800 | Loss: 0.489 | 2.4 steps/s
Saving checkpoint at step 1000...
Epoch 1/3: [██████░░░░░░░░░░░░░░] 30% | Step 1440/4800 | Loss: 0.431 | 2.3 steps/s
...
```

**Training Time Estimate:**

After first 100 steps, the script will estimate total time:
- Typical: 12-18 hours for 14K examples (3 epochs)
- Depends on: M4 variant, memory, thermal throttling

### Monitoring During Training

**Option 1: Activity Monitor (Recommended)**

Keep Activity Monitor open in separate window:

```bash
# Open Activity Monitor
open -a "Activity Monitor"

# Monitor:
# - Memory Pressure (should stay green/yellow)
# - CPU usage (should be 400-800% for M4)
# - GPU usage (look for "WindowServer" high GPU %)
# - Temperature (no direct indicator, but watch for slowdowns)
```

**Expected Resource Usage:**
- **CPU**: 400-800% (4-8 cores active)
- **Memory Pressure**: Green to yellow (not red)
- **App Memory**: 25-30GB used (out of 32GB)
- **GPU**: Metal tasks active in WindowServer

**Option 2: Weights & Biases (Remote)**

Setup W&B for remote monitoring (optional):

```bash
# Before training, login to W&B
wandb login

# Enter your API key from https://wandb.ai/authorize
```

Then monitor training from any device:
- Go to https://wandb.ai
- Find project: `weatherman-lora`
- View real-time metrics, loss curves, ETA

**Option 3: Log File (Separate Terminal)**

Open a new terminal window:

```bash
# Navigate to project
cd Weatherman-LoRA

# Tail log file
tail -f logs/training_m4_*.log

# Check specific metrics
grep "loss" logs/training_m4_*.log | tail -20
grep "epoch" logs/training_m4_*.log
```

### Thermal Management

**M4 may throttle if temperature gets too high:**

**Signs of Throttling:**
- Training speed drops from 2.5 steps/s to <1.5 steps/s
- Fan noise increases significantly
- System feels hot to touch

**Mitigation:**
```bash
# Check thermal state
sudo powermetrics --samplers smc -n 1 | grep -i temp

# Improve cooling:
# - Place Mac on hard, flat surface (not bed/couch)
# - Ensure vents are clear
# - Use laptop stand for better airflow
# - Lower room temperature if possible

# If throttling persists, reduce batch size or sequence length
```

---

## Step 4: Training Completion

### Verify Completion

After 12-18 hours, training should complete. You'll see:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

============================================================
✓ [TRAINING-M4-COMPLETE] Training finished
============================================================

Training Summary:
  Duration: 14h 32m
  Config: configs/training_config_m4.yaml
  Log file: logs/training_m4_20251103_140530.log

Model Location:
  adapters/weatherman-lora-m4

Next Steps:
  1. Validate the trained model
  2. Test with sample prompts
  3. See docs/DEPLOYMENT.md for deployment instructions

Model size: 1.8GB
```

### Verify Model Files

```bash
cd adapters/weatherman-lora-m4

# List files
ls -lh

# Expected output:
# adapter_config.json (1KB)
# adapter_model.bin (500MB-2GB)
# tokenizer files...
# training_args.bin
```

**Expected Files:**
```
adapters/weatherman-lora-m4/
├── adapter_config.json         # LoRA configuration
├── adapter_model.bin            # LoRA weights (~1-2GB)
├── adapter_model.safetensors    # Alternative format
├── tokenizer.json               # Tokenizer configuration
├── tokenizer_config.json        # Tokenizer settings
├── special_tokens_map.json      # Special tokens
├── training_args.bin            # Training configuration
└── checkpoint-*/                # Intermediate checkpoints
```

### Quick Validation Test

```bash
# Activate environment
source .venv-local/bin/activate

# Test inference (if script exists)
python scripts/test_inference.py --adapter adapters/weatherman-lora-m4

# Expected: Model loads successfully and generates responses
```

---

## Step 5: Post-Training Cleanup

### Free Up Space (Optional)

```bash
# Remove intermediate checkpoints (keep final model)
rm -rf adapters/weatherman-lora-m4/checkpoint-*

# This saves ~2-4GB per checkpoint
```

### Backup Model

```bash
# Create backup directory
mkdir -p ~/Documents/weatherman-backups

# Copy model
cp -r adapters/weatherman-lora-m4 ~/Documents/weatherman-backups/

# Or compress for storage
tar -czf ~/Documents/weatherman-lora-m4.tar.gz adapters/weatherman-lora-m4
```

### Cloud Backup (Recommended)

```bash
# Upload to cloud storage (if available)

# iCloud Drive
cp -r adapters/weatherman-lora-m4 ~/Library/Mobile\ Documents/com~apple~CloudDocs/

# Or use cloud CLI tools
# aws s3 sync adapters/weatherman-lora-m4/ s3://my-bucket/weatherman-m4/
# rclone copy adapters/weatherman-lora-m4/ gdrive:weatherman/
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**

```bash
# Symptoms
RuntimeError: MPS backend out of memory

# Solutions
# 1. Close all other applications
# 2. Reduce batch size (already at 1)
# 3. Reduce sequence length in config
max_seq_length: 1024  # Down from 2048

# 4. Increase gradient accumulation
gradient_accumulation_steps: 16  # Up from 8

# 5. Restart Mac to clear memory
sudo reboot
```

**2. MPS Not Available**

```bash
# Symptoms
RuntimeError: MPS backend not available

# Check macOS version
sw_vers

# Expected: macOS 12.3 or later

# Solution
# Upgrade macOS to 12.3+
# Or use CPU backend (very slow):
# Edit config: use_mps: false
```

**3. Slow Training Speed**

```bash
# Expected: 2.0-2.5 steps/second
# If seeing <1.5 steps/second:

# Check thermal throttling
sudo powermetrics --samplers smc -n 1 | grep -i temp

# Check memory pressure
open -a "Activity Monitor"
# Look at Memory tab → Memory Pressure graph

# Close background apps
# Improve cooling (see Thermal Management section)

# If persistent, reduce workload:
max_seq_length: 1024
```

**4. Training Crashes/Hangs**

```bash
# Symptoms
Training stops progressing, no error message

# Solutions
# 1. Check log file for errors
tail -100 logs/training_m4_*.log

# 2. Check system logs
log show --predicate 'eventMessage contains "python"' --last 5m

# 3. Restart training (will resume from last checkpoint)
./train_m4_local.sh

# 4. If checkpoint is corrupted, remove and restart
rm -rf adapters/weatherman-lora-m4/checkpoint-*
./train_m4_local.sh
```

**5. Validation Failures Before Training**

```bash
# Re-run setup script
./setup_m4_local.sh

# Or manually run validations
source .venv-local/bin/activate

python scripts/validate_environment.py --env=m4
python scripts/validate_training_config.py --config configs/training_config_m4.yaml
python scripts/check_storage.py
```

**6. ImportError or Package Issues**

```bash
# Reinstall dependencies
source .venv-local/bin/activate
pip install -r requirements-local.txt --force-reinstall

# Or recreate virtual environment
rm -rf .venv-local
./setup_m4_local.sh
```

### Performance Comparison

| Metric | M4 (32GB) | M4 (16GB) | H100 |
|--------|-----------|-----------|------|
| Training Time | 12-15 hours | 18-24 hours | 3-4 hours |
| Steps/Second | 2.0-2.5 | 1.0-1.5 | 50-60 |
| Batch Size | 1 | 1 | 4 |
| Seq Length | 2048 | 1024 | 4096 |
| Checkpoints | Every 250 steps | Every 100 steps | Every 500 steps |
| Cost | $0 | $0 | $9-12 |

---

## Best Practices

### Before Training

1. **Verify dataset integrity**
   ```bash
   wc -l data/synthetic/final_train.jsonl
   # Should show 14399
   ```

2. **Free up memory**
   ```bash
   # Close all unnecessary apps
   # Restart Mac if needed
   sudo reboot
   ```

3. **Test on small subset first**
   ```yaml
   # In config
   max_steps: 100  # Just 100 steps for testing
   ```

4. **Set up W&B monitoring**
   ```bash
   wandb login
   ```

5. **Plan training time**
   ```bash
   # Start training overnight or during off-hours
   # M4 training takes 12-18 hours
   ```

### During Training

1. **Don't use Mac for heavy tasks**
   - Avoid video editing, gaming, compiling
   - Light web browsing is okay
   - Email and messaging are fine

2. **Monitor first hour closely**
   - Loss should decrease steadily
   - Memory pressure should stay green/yellow
   - No thermal throttling

3. **Check progress periodically**
   ```bash
   # Every few hours, check:
   tail -20 logs/training_m4_*.log
   ```

4. **Don't interrupt training**
   - Let it run to completion
   - Checkpoints save every 250 steps
   - Use Ctrl+C only if necessary

5. **Keep Mac plugged in**
   - Don't rely on battery
   - Prevent sleep mode
   - Consider caffeine tool: `brew install caffeine`

### After Training

1. **Verify model quality**
   ```bash
   # Quick test inference
   python scripts/test_inference.py --adapter adapters/weatherman-lora-m4
   ```

2. **Back up immediately**
   ```bash
   # Local backup
   cp -r adapters/weatherman-lora-m4 ~/Documents/backup/

   # Cloud backup
   tar -czf weatherman-m4.tar.gz adapters/weatherman-lora-m4
   # Upload to cloud storage
   ```

3. **Document training run**
   - Note final loss values
   - Record any issues encountered
   - Save W&B run link

4. **Test deployment**
   - Follow docs/DEPLOYMENT.md
   - Test with sample prompts
   - Validate tool-use functionality

---

## Memory Configuration Guide

### For 16GB M4

```yaml
# configs/training_config_m4_16gb.yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
max_seq_length: 1024
save_steps: 100
logging_steps: 5

# Training time: 18-24 hours
```

### For 24GB M4

```yaml
# configs/training_config_m4_24gb.yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 12
max_seq_length: 1536
save_steps: 200
logging_steps: 10

# Training time: 14-18 hours
```

### For 32GB+ M4 (Recommended)

```yaml
# configs/training_config_m4.yaml (default)
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
max_seq_length: 2048
save_steps: 250
logging_steps: 10

# Training time: 12-15 hours
```

---

## Next Steps

After successful training:

1. **Deploy Model** → See [DEPLOYMENT.md](./DEPLOYMENT.md)
2. **Test Inference** → Section 4 of DEPLOYMENT.md
3. **Compare with H100** → Train on H100 for faster iterations

---

## Additional Resources

- [Apple Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [LoRA Training Guide](https://huggingface.co/docs/peft/task_guides/lora)
- [Weights & Biases Docs](https://docs.wandb.ai/)

For issues, see project repository or open a GitHub issue.
