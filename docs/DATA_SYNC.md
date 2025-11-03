# Data Synchronization Guide

This guide explains how to transfer data between your local machine (Mac M4) and remote GPU machines (Lambda Labs, RunPod, or home 3090) for the Weatherman-LoRA project.

## Overview

The Weatherman-LoRA workflow separates data processing and GPU training:

1. **Local (Mac M4)**: Process and prepare training data
2. **Sync to Remote**: Transfer processed data to GPU machine
3. **Remote (H100/3090)**: Train LoRA adapter
4. **Sync from Remote**: Retrieve trained adapters

## Data Transfer Volumes

| Data Type | Direction | Size | Frequency |
|-----------|-----------|------|-----------|
| Processed JSONL | Local → Remote | ~500MB-1GB | Once per dataset |
| Training Checkpoints | Remote → Local | ~100-500MB | After training |
| Final Adapters | Remote → Local | ~100-500MB | After training |

**Why not sync base models?** Base models (~15GB) are downloaded directly on GPU machines to avoid large transfers.

## Method 1: rsync (Recommended)

### Prerequisites
- SSH access to remote machine
- rsync installed (standard on Linux/macOS)

### Sync Processed Data to Remote

```bash
# Basic syntax
rsync -avz --progress data/processed/ user@remote:/path/to/weatherman-lora/data/processed/

# Example: Lambda Labs
rsync -avz --progress data/processed/ ubuntu@123.45.67.89:~/weatherman-lora/data/processed/

# Example: RunPod
rsync -avz --progress -e "ssh -p 22022" data/processed/ root@runpod.io:~/weatherman-lora/data/processed/

# Example: Home 3090
rsync -avz --progress data/processed/ user@192.168.1.100:~/weatherman-lora/data/processed/
```

**Flags explained:**
- `-a`: Archive mode (preserves permissions, timestamps)
- `-v`: Verbose output
- `-z`: Compress during transfer
- `--progress`: Show transfer progress

### Sync Adapters from Remote

```bash
# After training completes, retrieve adapters
rsync -avz --progress user@remote:/path/to/weatherman-lora/adapters/ adapters/

# Example: Lambda Labs
rsync -avz --progress ubuntu@123.45.67.89:~/weatherman-lora/adapters/weatherman-lora/ adapters/weatherman-lora/
```

### Dry Run (Test Without Transferring)

```bash
# Add --dry-run flag to see what would be transferred
rsync -avz --dry-run --progress data/processed/ user@remote:/path/to/weatherman-lora/data/processed/
```

### Exclude Files

```bash
# Exclude certain files or patterns
rsync -avz --progress --exclude='*.tmp' --exclude='.DS_Store' data/processed/ user@remote:~/weatherman-lora/data/processed/
```

## Method 2: scp (Single Files)

### Prerequisites
- SSH access to remote machine
- scp installed (standard on Linux/macOS)

### Transfer Single File to Remote

```bash
# Basic syntax
scp /local/path/file.jsonl user@remote:/remote/path/

# Example: Transfer training data
scp data/processed/train.jsonl ubuntu@123.45.67.89:~/weatherman-lora/data/processed/

# Example: Transfer with custom SSH port
scp -P 22022 data/processed/train.jsonl root@runpod.io:~/weatherman-lora/data/processed/
```

### Transfer Single File from Remote

```bash
# Retrieve trained adapter
scp user@remote:/path/to/adapter.bin adapters/

# Example: Lambda Labs
scp ubuntu@123.45.67.89:~/weatherman-lora/adapters/weatherman-lora/adapter_model.bin adapters/
```

### Transfer Directory with scp

```bash
# Use -r flag for directories
scp -r data/processed/ user@remote:~/weatherman-lora/data/

# Retrieve adapter directory
scp -r user@remote:~/weatherman-lora/adapters/weatherman-lora/ adapters/
```

## Method 3: Cloud Storage (Optional)

For larger datasets or when direct SSH is unavailable, use cloud storage as an intermediary.

### Using Google Drive (rclone)

```bash
# Install rclone
brew install rclone  # macOS
# or download from https://rclone.org/

# Configure Google Drive
rclone config

# Upload from local
rclone copy data/processed/ gdrive:weatherman-lora/data/processed/ --progress

# Download on remote
rclone copy gdrive:weatherman-lora/data/processed/ data/processed/ --progress
```

### Using AWS S3

```bash
# Install AWS CLI
brew install awscli  # macOS

# Configure credentials
aws configure

# Upload from local
aws s3 sync data/processed/ s3://your-bucket/weatherman-lora/data/processed/

# Download on remote
aws s3 sync s3://your-bucket/weatherman-lora/data/processed/ data/processed/
```

## Helper Scripts

The project includes helper scripts for common sync operations.

### sync_to_remote.sh

```bash
# Edit script to set your remote host and path
vim scripts/sync_to_remote.sh

# Run sync
./scripts/sync_to_remote.sh
```

### sync_from_remote.sh

```bash
# Edit script to set your remote host and path
vim scripts/sync_from_remote.sh

# Run sync
./scripts/sync_from_remote.sh
```

## SSH Configuration Tips

### Save SSH Connection Details

Create or edit `~/.ssh/config`:

```
# Lambda Labs
Host lambda-h100
    HostName 123.45.67.89
    User ubuntu
    IdentityFile ~/.ssh/lambda_key

# RunPod
Host runpod-3090
    HostName runpod.io
    Port 22022
    User root
    IdentityFile ~/.ssh/runpod_key

# Home 3090
Host home-3090
    HostName 192.168.1.100
    User your_username
    IdentityFile ~/.ssh/id_rsa
```

Then use short aliases:

```bash
# Sync using saved config
rsync -avz --progress data/processed/ lambda-h100:~/weatherman-lora/data/processed/

# SSH into machine
ssh lambda-h100
```

## Workflow Example

### Full Training Workflow

```bash
# 1. LOCAL: Process data
cd ~/weatherman-lora
source .venv-local/bin/activate
python scripts/process_data.py  # Your data processing script

# 2. LOCAL: Verify processed data
ls -lh data/processed/
# Should see train.jsonl and val.jsonl (~500MB-1GB total)

# 3. LOCAL: Sync to remote
rsync -avz --progress data/processed/ lambda-h100:~/weatherman-lora/data/processed/

# 4. REMOTE: Verify data received
ssh lambda-h100
cd ~/weatherman-lora
ls -lh data/processed/

# 5. REMOTE: Run training
conda activate weatherman-lora
python scripts/train.py  # Training takes 3-6 hours on H100

# 6. LOCAL: Sync adapters back
rsync -avz --progress lambda-h100:~/weatherman-lora/adapters/weatherman-lora/ adapters/weatherman-lora/

# 7. LOCAL: Verify adapters
ls -lh adapters/weatherman-lora/
# Should see adapter_model.bin, adapter_config.json (~100-500MB)
```

## Troubleshooting

### Permission Denied

```bash
# Fix SSH key permissions
chmod 600 ~/.ssh/your_key

# Fix directory permissions on remote
ssh remote "chmod -R 755 ~/weatherman-lora/data/"
```

### Connection Timeout

```bash
# Test connection first
ssh user@remote echo "Connection successful"

# Check firewall rules on remote machine
# Ensure port 22 (or custom SSH port) is open
```

### Slow Transfer Speeds

```bash
# Use compression (already included with -z)
# Remove compression if CPU-bound
rsync -av --progress data/processed/ user@remote:~/weatherman-lora/data/processed/

# Limit bandwidth (useful on shared connections)
rsync -avz --progress --bwlimit=5000 data/processed/ user@remote:~/weatherman-lora/data/processed/
# (5000 = 5 MB/s)
```

### Interrupted Transfer

```bash
# rsync resumes automatically
# Just re-run the same command
rsync -avz --progress data/processed/ user@remote:~/weatherman-lora/data/processed/

# For scp, use rsync instead (scp doesn't resume)
```

## Best Practices

1. **Verify Before Syncing**: Check data integrity locally before transfer
2. **Use Dry Run**: Test with `--dry-run` flag first
3. **Compress Data**: Use `.tar.gz` for very large datasets
4. **Check Disk Space**: Verify remote has sufficient space
5. **Keep Backups**: Maintain local copies of processed data
6. **Document Sync**: Note when data was synced for reproducibility
7. **Clean Up**: Remove old checkpoints after successful training

## Data Transfer Checklist

- [ ] Processed data ready in `data/processed/`
- [ ] Verified data format (JSONL with chat messages)
- [ ] Checked file sizes (~500MB-1GB is normal)
- [ ] SSH access to remote machine confirmed
- [ ] Remote project directory exists
- [ ] Remote has 30-50GB free space
- [ ] Initiated rsync/scp transfer
- [ ] Verified data received on remote
- [ ] Ready to start training

## See Also

- [Setup Guide](SETUP_GUIDE.md) - Environment configuration
- [Model Download Guide](MODEL_DOWNLOAD.md) - Base model download
- [Implementation Guide](../references/IMPLEMENTATION_GUIDE.md) - Full training workflow
