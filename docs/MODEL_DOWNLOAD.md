# Model Download Guide

This guide explains how to download base models (Llama 3.1 8B Instruct or Mistral 7B Instruct) from HuggingFace Hub for the Weatherman-LoRA project.

## Overview

Base models (~15GB each) should be downloaded **separately on each GPU machine** to avoid transferring large files. The models are cached in the `models/` directory for reuse.

## Prerequisites

- HuggingFace account (free)
- Access token with read permissions
- 20GB free disk space per model
- Internet connection on GPU machine

## Step 1: Create HuggingFace Account

1. Go to https://huggingface.co/join
2. Sign up with email or GitHub
3. Verify your email address

## Step 2: Request Model Access (Llama 3.1 Only)

Llama 3.1 requires accepting Meta's license agreement.

1. Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Click "Agree and access repository"
3. Fill out the form (takes ~5 minutes to approve)

**Note**: Mistral 7B does not require access request.

## Step 3: Create Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `weatherman-lora-training`
4. Type: **Read** (not Write)
5. Click "Generate token"
6. **Copy the token** (you won't see it again)

## Step 4: Authenticate on GPU Machine

### Method A: Using huggingface-cli (Recommended)

```bash
# SSH into GPU machine
ssh user@remote

# Activate conda environment
conda activate weatherman-lora

# Login to HuggingFace
huggingface-cli login

# Paste your access token when prompted
# Token will be saved to ~/.cache/huggingface/token
```

### Method B: Using Python

```python
# In Python script or notebook
from huggingface_hub import login

login(token="hf_your_token_here")
```

### Method C: Environment Variable

```bash
# Add to ~/.bashrc or ~/.zshrc
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"

# Or set for current session only
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
```

## Step 5: Configure Cache Directory

By default, models download to `~/.cache/huggingface/`. To use the project's `models/` directory:

```bash
# Set environment variable
export HF_HOME="/path/to/weatherman-lora/models/.cache"

# Or add to ~/.bashrc
echo 'export HF_HOME="/path/to/weatherman-lora/models/.cache"' >> ~/.bashrc
source ~/.bashrc
```

## Step 6: Download Models

### Option A: Download via Python (Recommended)

Create a script or run in Python:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set cache directory
import os
os.environ['HF_HOME'] = '/path/to/weatherman-lora/models/.cache'

# Download Llama 3.1 8B Instruct
print("Downloading Llama 3.1 8B Instruct...")
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=None  # Don't load to GPU yet
)
print(f"Model downloaded and cached in {os.environ['HF_HOME']}")

# Alternative: Download Mistral 7B Instruct
# print("Downloading Mistral 7B Instruct...")
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map=None
# )
```

### Option B: Download via huggingface-cli

```bash
# Download Llama 3.1 8B Instruct
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --local-dir models/Meta-Llama-3.1-8B-Instruct \
    --local-dir-use-symlinks False

# Or download Mistral 7B Instruct
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
    --local-dir models/Mistral-7B-Instruct-v0.3 \
    --local-dir-use-symlinks False
```

### Option C: Download Specific Files Only

```bash
# Download only necessary files (saves space)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --include "*.json" "*.safetensors" "tokenizer.model" \
    --local-dir models/Meta-Llama-3.1-8B-Instruct
```

## Step 7: Verify Download

Use the provided verification script:

```bash
# Verify Llama 3.1
python scripts/verify_model.py meta-llama/Meta-Llama-3.1-8B-Instruct

# Or verify Mistral
python scripts/verify_model.py mistralai/Mistral-7B-Instruct-v0.3
```

Expected output:
```
✅ Model verified: meta-llama/Meta-Llama-3.1-8B-Instruct
   Location: /path/to/models/.cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct
   Size: ~15.2 GB
   Architecture: LlamaForCausalLM
   Vocabulary size: 128,256 tokens
```

## Model Selection Guide

### Llama 3.1 8B Instruct (Recommended)

**Pros:**
- Excellent instruction following
- Strong reasoning capabilities
- Better at maintaining personality
- Apache 2.0 license (commercial use allowed)

**Cons:**
- Requires access request (~5 min approval)
- Slightly larger vocabulary (128K vs 32K)

**Best for:** Production use, commercial applications

### Mistral 7B Instruct v0.3

**Pros:**
- No access request needed
- Faster inference (7B vs 8B parameters)
- Smaller vocabulary (more memory efficient)
- Apache 2.0 license

**Cons:**
- Less capable at complex instructions
- May struggle with personality consistency

**Best for:** Quick prototyping, personal projects

## Download Times

| Connection Speed | Model Size | Estimated Time |
|------------------|------------|----------------|
| 1 Gbps (Lambda Labs) | ~15 GB | 2-3 minutes |
| 100 Mbps | ~15 GB | 20-30 minutes |
| 10 Mbps | ~15 GB | 3-4 hours |

## Storage Requirements

```
models/
├── .cache/                          # HuggingFace cache
│   └── hub/
│       ├── models--meta-llama--Meta-Llama-3.1-8B-Instruct/
│       │   ├── snapshots/
│       │   │   └── [hash]/          # ~15.2 GB
│       │   │       ├── config.json
│       │   │       ├── model-*.safetensors  (8 files)
│       │   │       ├── tokenizer.json
│       │   │       └── ...
│       │   └── refs/
│       └── models--mistralai--Mistral-7B-Instruct-v0.3/
│           └── snapshots/
│               └── [hash]/          # ~14.8 GB
```

## Troubleshooting

### Authentication Failed

```
Error: 401 Client Error: Unauthorized
```

**Solutions:**
1. Verify token has read permissions
2. Re-run `huggingface-cli login`
3. Check token hasn't expired
4. Ensure you accepted Llama license (if using Llama)

### Access Denied (403)

```
Error: 403 Client Error: Forbidden
```

**For Llama 3.1:**
1. Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Click "Agree and access repository"
3. Wait 5-10 minutes for approval
4. Try download again

### Out of Disk Space

```bash
# Check available space
df -h /path/to/models

# If low, clean up old cache
rm -rf ~/.cache/huggingface/hub/models--*

# Or use a different drive
export HF_HOME="/mnt/large_drive/huggingface_cache"
```

### Slow Download

```bash
# Resume interrupted download (huggingface-cli handles this automatically)
# Just re-run the same download command

# Or download in parts
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --include "model-00001-of-00008.safetensors" \
    --local-dir models/Meta-Llama-3.1-8B-Instruct

# Then download remaining files
```

### Model Not Found During Training

```python
# Error: Can't load model from 'meta-llama/Meta-Llama-3.1-8B-Instruct'

# Fix: Ensure HF_HOME is set before importing transformers
import os
os.environ['HF_HOME'] = '/path/to/weatherman-lora/models/.cache'

# Then import transformers
from transformers import AutoModelForCausalLM
```

## Best Practices

1. **Download Once Per Machine**: Don't transfer models between machines
2. **Use Cache Directory**: Configure `HF_HOME` to project's `models/` folder
3. **Verify After Download**: Run `scripts/verify_model.py` to confirm
4. **Keep Token Secret**: Never commit tokens to git
5. **Monitor Disk Space**: Ensure 20GB free before downloading
6. **Use Symlinks**: Default HuggingFace behavior (saves space)
7. **Clean Up**: Remove old model versions when not needed

## Switching Models

To switch between Llama and Mistral during training:

```yaml
# In configs/training_config.yaml
model:
  # Use Llama 3.1
  model_name_or_path: meta-llama/Meta-Llama-3.1-8B-Instruct

  # Or use Mistral
  # model_name_or_path: mistralai/Mistral-7B-Instruct-v0.3
```

## Model Download Checklist

- [ ] HuggingFace account created
- [ ] Access token created (read permission)
- [ ] Llama 3.1 access approved (if using Llama)
- [ ] Authenticated on GPU machine
- [ ] `HF_HOME` environment variable set
- [ ] 20GB free disk space confirmed
- [ ] Model downloaded successfully
- [ ] Model verified with `verify_model.py`
- [ ] Ready to start training

## See Also

- [Setup Guide](SETUP_GUIDE.md) - Environment configuration
- [Data Sync Guide](DATA_SYNC.md) - Transferring training data
- [Implementation Guide](../references/IMPLEMENTATION_GUIDE.md) - Full training workflow
- [Training Config](../configs/training_config.yaml) - Model configuration
