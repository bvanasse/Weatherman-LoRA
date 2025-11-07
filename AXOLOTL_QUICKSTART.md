# Axolotl Training Quickstart

This guide explains how to train the Weatherman-LoRA model using [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), a battle-tested LoRA training framework.

## Why Axolotl?

- ✅ **Production-ready**: Used by thousands of practitioners
- ✅ **Better error handling**: Automatic checkpoint resumption
- ✅ **Zero code changes needed**: Your data is already compatible!
- ✅ **Active community**: Better support and documentation
- ✅ **Flexible configs**: Easy hyperparameter experimentation

## Your Data Format (Already Compatible!)

Your training data is in **OpenAI Chat Completions format with tool calls** - which Axolotl supports natively!

**Files:**
- `data/synthetic/final_train_diverse.jsonl` (14,399 examples)
- `data/synthetic/final_validation_diverse.jsonl` (1,601 examples)

**Data Quality:**
- ✨ **Diverse responses** - Regenerated to remove repetitive templates
- Original data had 59% template repetition
- New diverse data has <3% templates for better generalization
- See [DATA_REGENERATION_GUIDE.md](DATA_REGENERATION_GUIDE.md) for full details

**Format:**
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Before You Train

### Verify Data Quality

Before starting training, verify that you have the diverse dataset:

```bash
# Check if diverse dataset exists
ls -lh data/synthetic/final_train_diverse.jsonl

# If it doesn't exist, you need to run regeneration first
# See DATA_REGENERATION_GUIDE.md for instructions

# Verify diversity metrics
python3 scripts/analyze_data_diversity.py data/synthetic/final_train_diverse.jsonl

# Should show:
# - Total templated: <3% (good!)
# - Diversity score: 90+/100 (excellent!)
```

**If you see high template percentage (>10%)**: You need to regenerate the data. Run:
```bash
./regenerate_training_data.sh
```

This will take ~3-4 hours and cost ~$70 in Claude API usage, but it's crucial for training a high-quality model.

## Quick Start

### Option 1: H100 GPU Training (RunPod)

```bash
# On RunPod H100 instance
cd /workspace/Weatherman-LoRA

# Activate environment
conda activate weatherman-lora

# Run training (automatic Axolotl installation)
./train_with_axolotl_h100.sh

# Training will take ~3-4 hours
```

### Option 2: M4 Mac Training (Local)

```bash
# On Mac M4
cd ~/Apps/Weatherman-LoRA

# Activate environment (if using conda)
conda activate weatherman-lora
# OR activate venv
source .venv-local/bin/activate

# Run training (automatic Axolotl installation)
./train_with_axolotl_m4.sh

# Training will take ~8-12 hours (recommend overnight)
```

## Configuration Files

Two configs are provided, optimized for each platform:

### `axolotl_config_h100.yaml`
- **Platform**: H100 GPU (80GB VRAM)
- **Sequence Length**: 4096 tokens
- **Batch Size**: 4 (gradient accumulation: 4)
- **Flash Attention**: Enabled
- **Precision**: bfloat16
- **Duration**: 3-4 hours

### `axolotl_config_m4.yaml`
- **Platform**: Mac M4 MPS (32GB unified memory)
- **Sequence Length**: 2048 tokens
- **Batch Size**: 1 (gradient accumulation: 16)
- **Flash Attention**: Disabled (MPS doesn't support it)
- **Precision**: float16
- **Duration**: 8-12 hours

## Monitoring Training

### Weights & Biases (Recommended)

Training automatically logs to W&B if configured:

```bash
# Set up W&B
wandb login

# W&B will automatically track:
# - Loss curves
# - Learning rate schedule
# - GPU/MPS utilization
# - Gradient norms
# - Validation metrics
```

### Local Logs

Logs are saved to `logs/axolotl_training_*.log`:

```bash
# Watch training progress
tail -f logs/axolotl_training_h100_*.log

# Check for errors
grep -i error logs/axolotl_training_h100_*.log
```

### Terminal Output

The training script shows:
- Current epoch/step
- Training loss
- Validation loss (every 500 steps for H100, 1000 for M4)
- Estimated time remaining
- GPU/MPS utilization

## Checkpoint Management

Checkpoints are automatically saved to:
- **H100**: `adapters/weatherman-lora-axolotl-h100/`
- **M4**: `adapters/weatherman-lora-axolotl-m4/`

### Automatic Resumption

If training crashes or is interrupted:

```bash
# Just re-run the training script
./train_with_axolotl_h100.sh

# Axolotl automatically detects and resumes from latest checkpoint
```

### Manual Checkpoint Control

Edit the config to resume from specific checkpoint:

```yaml
# In axolotl_config_h100.yaml
resume_from_checkpoint: ./adapters/weatherman-lora-axolotl-h100/checkpoint-1000
```

## After Training

### 1. Merge Adapter (Optional)

Merge LoRA adapter with base model for faster inference:

```bash
# H100
python -m axolotl.cli.merge \
  axolotl_config_h100.yaml \
  --lora-model-dir=./adapters/weatherman-lora-axolotl-h100

# M4
python3 -m axolotl.cli.merge \
  axolotl_config_m4.yaml \
  --lora-model-dir=./adapters/weatherman-lora-axolotl-m4
```

### 2. Test Inference

Quick test using Axolotl's inference CLI:

```bash
# H100
python -m axolotl.cli.inference \
  axolotl_config_h100.yaml \
  --lora-model-dir=./adapters/weatherman-lora-axolotl-h100 \
  --prompt="What's the weather in Boston?"

# M4
python3 -m axolotl.cli.inference \
  axolotl_config_m4.yaml \
  --lora-model-dir=./adapters/weatherman-lora-axolotl-m4 \
  --prompt="What's the weather in Boston?"
```

### 3. Deploy

Deploy the trained adapter using Ollama or AnythingLLM:

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed instructions.

## Advanced: Hyperparameter Tuning

Edit the config files to experiment:

### Learning Rate

```yaml
# Default: 0.0002
learning_rate: 0.0001  # Lower for more stable training
learning_rate: 0.0003  # Higher for faster convergence
```

### LoRA Rank

```yaml
# Default: r=16, alpha=32
lora_r: 32        # Higher rank = more capacity (more VRAM)
lora_alpha: 64    # Keep alpha = 2 * r
```

### Sequence Length

```yaml
# H100 default: 4096
sequence_len: 8192  # Longer sequences (requires more VRAM)

# M4 default: 2048
sequence_len: 1024  # Shorter to save memory
```

### Batch Size

```yaml
# H100 default: micro_batch_size=4, gradient_accumulation=4
micro_batch_size: 8              # Larger batches (more VRAM)
gradient_accumulation_steps: 2   # Reduce accumulation

# M4 default: micro_batch_size=1, gradient_accumulation=16
micro_batch_size: 2              # Try larger if you have 64GB
gradient_accumulation_steps: 8   # Reduce accumulation
```

## Troubleshooting

### Out of Memory (H100)

```yaml
# Reduce batch size
micro_batch_size: 2
gradient_accumulation_steps: 8

# Or reduce sequence length
sequence_len: 2048
```

### Out of Memory (M4)

```yaml
# Ensure smallest batch size
micro_batch_size: 1
gradient_accumulation_steps: 16

# Reduce sequence length
sequence_len: 1024

# Close all other applications
```

### Slow Training (M4)

This is expected! MPS is 2-3x slower than CUDA. Recommendations:

- Train overnight (8-12 hours)
- Monitor Activity Monitor for memory pressure
- Close browser tabs and heavy apps
- Consider using H100 for faster iterations

### Training Diverges (Loss increases)

```yaml
# Lower learning rate
learning_rate: 0.0001

# Add more warmup
warmup_steps: 200

# Reduce batch size
micro_batch_size: 2
```

### Tool Calls Not Working

Verify your data format:

```bash
# Check first example
head -1 data/synthetic/final_train.jsonl | python -m json.tool | grep tool_calls

# Should show tool_calls in assistant messages
```

## Comparison: Custom Scripts vs Axolotl

| Feature | Custom Scripts | Axolotl |
|---------|---------------|---------|
| Setup Complexity | Higher | Lower |
| Error Handling | Manual | Automatic |
| Checkpoint Resume | Custom logic | Built-in |
| Multi-GPU Support | Limited | Full DeepSpeed |
| Community Support | Project-specific | Large community |
| Documentation | This repo | Extensive docs |
| Configuration | Python code | YAML configs |
| Experimentation | Code changes | Config tweaks |

## Additional Resources

- [Axolotl Documentation](https://github.com/axolotl-ai-cloud/axolotl/tree/main/docs)
- [Axolotl Discord](https://discord.gg/HhrNrHJPRb)
- [Example Configs](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## Questions?

- Check logs: `logs/axolotl_training_*.log`
- Review config: `axolotl_config_h100.yaml` or `axolotl_config_m4.yaml`
- Axolotl issues: https://github.com/axolotl-ai-cloud/axolotl/issues
- Project issues: https://github.com/your-username/Weatherman-LoRA/issues
