# Axolotl Training Configuration Fixes

This document explains the fixes made to the Axolotl training configuration and script for H100 GPU training.

## Issues Fixed

### 1. Configuration File (`axolotl_config_h100.yaml`)

**Problems:**
- Missing `train_on_inputs` and `group_by_length` fields
- Removed unnecessary `local_rank` field (handled by accelerate)
- Improved dataset configuration comments

**Fixes:**
- ✅ Added `train_on_inputs: false` (prevents training on input tokens)
- ✅ Added `group_by_length: false` (better for tool calling sequences)
- ✅ Removed `local_rank:` empty field (not needed)
- ✅ Kept `test_datasets` for validation (matches M4 config)
- ✅ Improved comments and documentation

### 2. Training Script (`train_with_axolotl_h100.sh`)

**Problems:**
- Complex dependency installation that could fail
- Missing PEFT version compatibility checks
- No proper error handling for package conflicts
- Used `accelerate launch` directly instead of `axolotl train` CLI

**Fixes:**
- ✅ Simplified installation process with better error handling
- ✅ Added PEFT version compatibility check and upgrade
- ✅ Added bitsandbytes verification for QLoRA
- ✅ Uses `axolotl train` CLI when available (simpler)
- ✅ Falls back to `accelerate launch` if CLI not available
- ✅ Better GPU detection and verification
- ✅ Improved logging and error messages
- ✅ Better checkpoint detection and resumption

## Key Changes

### Configuration Format

The config now follows Axolotl's expected format more closely:

```yaml
# Dataset configuration
datasets:
  - path: data/synthetic/final_train_diverse.jsonl
    ds_type: json
    type: chat_template
    field_messages: messages
    message_field_role: role
    message_field_content: content

# Validation dataset
val_set_size: 0
test_datasets:
  - path: data/synthetic/final_validation_diverse.jsonl
    ds_type: json
    type: chat_template
    field_messages: messages
    message_field_role: role
    message_field_content: content
```

### Training Script Improvements

1. **Better Dependency Management:**
   - Checks for PEFT compatibility issues
   - Upgrades PEFT, accelerate, and bitsandbytes automatically
   - Handles flash-attn installation failures gracefully

2. **Simplified Training Command:**
   - Uses `axolotl train` CLI when available (recommended)
   - Falls back to `accelerate launch` if needed
   - Better error reporting

3. **Enhanced Verification:**
   - Validates data files exist
   - Counts examples in datasets
   - Checks GPU availability via nvidia-smi and PyTorch
   - Verifies checkpoint directory

## Usage

### Quick Start

```bash
# On RunPod H100 instance
cd /workspace/Weatherman-LoRA

# Activate environment
conda activate weatherman-lora

# Run training (automatic setup)
./train_with_axolotl_h100.sh
```

### What the Script Does

1. **Environment Check:**
   - Verifies conda environment is activated
   - Checks config file exists
   - Validates training and validation data files

2. **Dependency Installation:**
   - Installs PyTorch with CUDA 12.1 support
   - Installs Flash Attention (if possible)
   - Installs/upgrades Axolotl with DeepSpeed support
   - Ensures PEFT, accelerate, and bitsandbytes are compatible

3. **GPU Verification:**
   - Checks GPU via nvidia-smi
   - Verifies CUDA availability in PyTorch
   - Reports GPU name and memory

4. **Training Launch:**
   - Detects existing checkpoints for resumption
   - Launches training with proper logging
   - Provides next steps on completion

## Troubleshooting

### Common Issues

#### 1. PEFT Import Error
```
ImportError: cannot import name 'QuantLinear' from 'peft.tuners.lora'
```
**Solution:** The script now automatically upgrades PEFT. If this still occurs:
```bash
pip install --upgrade peft
```

#### 2. Flash Attention Build Fails
**Solution:** Training will continue without Flash Attention (slower but works). To retry:
```bash
pip install flash-attn==2.8.2 --no-build-isolation
```

#### 3. Out of Memory
**Solution:** Reduce batch size or sequence length in config:
```yaml
micro_batch_size: 2  # Reduce from 4
sequence_len: 2048    # Reduce from 4096
```

#### 4. Dataset Format Error
**Solution:** Verify your data format:
```bash
head -1 data/synthetic/final_train_diverse.jsonl | python -m json.tool
```
Should show `messages` array with `role` and `content` fields.

## Expected Behavior

### Successful Training

You should see:
- ✅ Config file found
- ✅ Training data verified (14,399 examples)
- ✅ Validation data verified (1,601 examples)
- ✅ Axolotl installed/verified
- ✅ GPU detected and available
- ✅ Training starts with progress logs

### Training Progress

- **Steps per epoch:** ~900 (14,399 / (4 * 4))
- **Total steps:** ~2,700 (3 epochs)
- **Checkpoints:** Every 500 steps
- **Evaluation:** Every 500 steps
- **Duration:** 3-4 hours on H100

### Output Location

- **Adapters:** `adapters/weatherman-lora-axolotl-h100/`
- **Logs:** `logs/axolotl_training_h100_YYYYMMDD_HHMMSS.log`
- **Checkpoints:** `adapters/weatherman-lora-axolotl-h100/checkpoint-*/`

## Next Steps After Training

1. **Merge Adapter (Optional):**
   ```bash
   axolotl merge-lora axolotl_config_h100.yaml \
     --lora-model-dir=./adapters/weatherman-lora-axolotl-h100
   ```

2. **Test Inference:**
   ```bash
   axolotl inference axolotl_config_h100.yaml \
     --lora-model-dir=./adapters/weatherman-lora-axolotl-h100 \
     --prompt="What's the weather in Boston?"
   ```

3. **Deploy:**
   - See `docs/DEPLOYMENT.md` for Ollama/AnythingLLM deployment

## References

- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples)
- [Axolotl Documentation](https://docs.axolotl.ai)
- [Mistral Example](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/mistral)

