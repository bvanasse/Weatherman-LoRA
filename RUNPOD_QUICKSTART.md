# RunPod H100 Quick Start Guide

Quick setup guide for training Weatherman-LoRA on RunPod H100 instances with pre-installed Axolotl.

## Prerequisites

- RunPod H100 instance with Axolotl pre-installed
- GitHub account with access to the repository

## Setup (5 minutes)

### 1. Clone Repository

Using HTTPS (recommended):

```bash
cd /workspace
git clone https://github.com/bvanasse/Weatherman-LoRA.git
cd Weatherman-LoRA
```

If prompted for credentials:
- **Username**: your-github-username
- **Password**: [GitHub Personal Access Token](https://github.com/settings/tokens)

### 2. Verify Axolotl Installation

```bash
# Check Axolotl is available
python -c "import axolotl; print(f'Axolotl {axolotl.__version__}')"

# Check accelerate CLI
which accelerate
# Should show: /usr/local/bin/accelerate or similar

# Verify GPU
nvidia-smi
# Should show H100 with 80GB
```

### 3. Verify Training Data

```bash
# Check data files exist
ls -lh data/synthetic/final_train_diverse.jsonl
ls -lh data/synthetic/final_validation_diverse.jsonl

# Count examples
wc -l data/synthetic/final_*_diverse.jsonl
# Should show: 14399 train, 1601 validation
```

## Training

### Start Training

```bash
# Make script executable (if not already)
chmod +x train_axolotl_simple.sh

# Launch training
./train_axolotl_simple.sh
```

**What happens:**
1. Validates config and data files
2. Checks for existing checkpoints (auto-resumes if found)
3. Creates timestamped log file
4. Launches training in tmux session
5. Waits 30 seconds to verify startup
6. Shows monitoring commands

**Expected output:**
```
============================================================
Weatherman-LoRA Axolotl Training (H100)
============================================================

✓ Found config: axolotl_config_h100.yaml
✓ Training data: 14399 examples
✓ Validation data: 1601 examples

Log file: logs/axolotl_20250109_143022.log

============================================================
Launching Training in Persistent Session
============================================================

Base Model: mistralai/Mistral-7B-Instruct-v0.3
Training Examples: 14399
Validation Examples: 1601
Output: adapters/weatherman-lora-axolotl-h100
Estimated Duration: 3-4 hours

✓ Training launched in tmux session: weatherman-training
Waiting for training to initialize (30 seconds)...
✓ Training initialized successfully

============================================================
✓ Training Session Active
============================================================

Session Type: tmux
Session Name: weatherman-training

Monitor Training:
  Attach to session:   tmux attach -t weatherman-training
  Detach from session: Ctrl+B, then D
  View log:            tail -f logs/axolotl_20250109_143022.log
  Check GPU:           nvidia-smi
```

### Monitor Training

```bash
# View log in real-time
tail -f logs/axolotl_*.log

# Attach to tmux session (see live output)
tmux attach -t weatherman-training
# Press Ctrl+B, then D to detach

# Check GPU usage
watch -n 1 nvidia-smi

# Check training progress in log
grep -E "Epoch|Step|Loss" logs/axolotl_*.log | tail -20
```

### Training Progress

Expected timeline for 14,399 examples, 3 epochs:

- **Steps per epoch**: ~900 (14,399 / (4 batch × 4 accumulation))
- **Total steps**: ~2,700 (3 epochs)
- **Time per step**: ~4-5 seconds on H100
- **Estimated total**: 3-4 hours
- **Checkpoints**: Every 500 steps (6 checkpoints total)

You can safely disconnect from SSH. Training continues in tmux session.

## After Training Completes

### 1. Verify Training Completed

```bash
# Check if training finished
tail -100 logs/axolotl_*.log

# Look for completion messages like:
# "Training completed"
# "Saving final checkpoint"

# Check final checkpoint exists
ls -lh adapters/weatherman-lora-axolotl-h100/
```

### 2. Download Model to Local Machine

From your local machine:

```bash
# Using rsync (recommended - shows progress)
rsync -avz --progress -e "ssh -p <runpod-port>" \
  root@<runpod-ip>:/workspace/Weatherman-LoRA/adapters/weatherman-lora-axolotl-h100/ \
  ~/Downloads/weatherman-lora-axolotl-h100/

# Or using scp
scp -r -P <runpod-port> \
  root@<runpod-ip>:/workspace/Weatherman-LoRA/adapters/weatherman-lora-axolotl-h100 \
  ~/Downloads/
```

Expected download size: ~2-4GB (LoRA adapter weights)

### 3. Deploy on M4 Mac

See [M4_DEPLOYMENT.md](docs/M4_DEPLOYMENT.md) for complete deployment guide.

Quick summary:
1. Install Ollama on M4
2. Merge LoRA adapter with base model
3. Convert to GGUF format
4. Import to Ollama
5. Test inference

## Troubleshooting

### Training Stops Immediately

```bash
# Check for errors in log
tail -50 logs/axolotl_*.log | grep -i error

# Common issues:
# - Data files missing: Check data/synthetic/ exists
# - Config invalid: Run ./check_gpu.sh to validate config
# - GPU memory: Check with nvidia-smi

# Validate config syntax
./check_gpu.sh
```

**Note**: The config has been updated to remove fields that caused Pydantic validation errors:
- Removed `wandb_run_id` (replaced with `wandb_entity`)
- Commented out `resume_from_checkpoint`, `chat_template`, and `special_tokens`
- These will auto-detect from the model's tokenizer

### Out of Memory

If you see "CUDA out of memory":

```bash
# Edit axolotl_config_h100.yaml
# Reduce micro_batch_size from 4 to 2:
micro_batch_size: 2

# Or reduce sequence length:
sequence_len: 2048  # Down from 4096
```

### Training Not Resuming from Checkpoint

```bash
# Check if checkpoints exist
ls -la adapters/weatherman-lora-axolotl-h100/checkpoint-*

# Manually specify checkpoint in config
# Edit axolotl_config_h100.yaml:
resume_from_checkpoint: adapters/weatherman-lora-axolotl-h100/checkpoint-2000
```

### Tmux Session Not Found

```bash
# List all tmux sessions
tmux ls

# If no session, training may have completed or crashed
# Check logs:
tail -100 logs/axolotl_*.log

# Restart training (will auto-resume from latest checkpoint):
./train_axolotl_simple.sh
```

## Useful Commands

```bash
# Stop training
tmux kill-session -t weatherman-training

# View all tmux sessions
tmux ls

# Attach to session
tmux attach -t weatherman-training

# Monitor GPU continuously
watch -n 1 nvidia-smi

# Check disk space
df -h

# Check process running
ps aux | grep axolotl

# View checkpoint history
ls -lth adapters/weatherman-lora-axolotl-h100/checkpoint-*
```

## Cost Optimization

RunPod H100 costs ~$2-3/hour. Training takes 3-4 hours = ~$8-12 total.

To minimize cost:
1. Verify data and config before starting
2. Monitor first 30 minutes to ensure no errors
3. Can safely disconnect - training continues
4. Download model immediately after completion
5. Stop/terminate pod when done

## Next Steps

After successful training:
1. Download adapter weights
2. Follow [M4_DEPLOYMENT.md](docs/M4_DEPLOYMENT.md) to deploy locally
3. Test model with sample weather queries
4. Integrate into your application

## Support

For issues:
- Check logs: `tail -100 logs/axolotl_*.log`
- Verify data: `head -2 data/synthetic/final_train_diverse.jsonl`
- Check GPU: `nvidia-smi`
- GitHub issues: https://github.com/bvanasse/Weatherman-LoRA/issues
