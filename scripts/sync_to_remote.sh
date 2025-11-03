#!/bin/bash
#
# Sync Processed Data to Remote GPU Machine
#
# This script syncs processed training data from local to remote.
# Edit the variables below to match your remote machine setup.
#
# Usage: ./scripts/sync_to_remote.sh

set -e

echo "============================================================"
echo "Sync Processed Data to Remote"
echo "============================================================"
echo ""

# ============================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================

# TODO: Set your remote connection details
REMOTE_USER="TODO_USERNAME"
REMOTE_HOST="TODO_HOSTNAME_OR_IP"
REMOTE_PORT="22"  # Default SSH port (change if using custom port)
REMOTE_PATH="TODO_REMOTE_PATH/weatherman-lora"

# Local paths (usually don't need to change these)
LOCAL_DATA_PROCESSED="data/processed"

# ============================================================
# VALIDATION
# ============================================================

if [ "$REMOTE_USER" = "TODO_USERNAME" ] || [ "$REMOTE_HOST" = "TODO_HOSTNAME_OR_IP" ] || [ "$REMOTE_PATH" = "TODO_REMOTE_PATH/weatherman-lora" ]; then
    echo "❌ ERROR: Please edit this script and set your remote connection details"
    echo ""
    echo "Open the script and set:"
    echo "  REMOTE_USER=\"your_username\""
    echo "  REMOTE_HOST=\"your.remote.host.com or IP\""
    echo "  REMOTE_PATH=\"/path/to/weatherman-lora\""
    echo ""
    exit 1
fi

# Check if local data exists
if [ ! -d "$LOCAL_DATA_PROCESSED" ]; then
    echo "❌ ERROR: Local processed data directory not found: $LOCAL_DATA_PROCESSED"
    echo ""
    echo "Please process your data first before syncing."
    exit 1
fi

# Check if directory has files
if [ -z "$(ls -A $LOCAL_DATA_PROCESSED)" ]; then
    echo "⚠️  WARNING: Processed data directory is empty: $LOCAL_DATA_PROCESSED"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# ============================================================
# SYNC
# ============================================================

echo "Syncing processed data to remote..."
echo ""
echo "  Local:  $LOCAL_DATA_PROCESSED/"
echo "  Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/data/processed/"
echo ""

# Test connection first
echo "Testing SSH connection..."
if ssh -p "$REMOTE_PORT" -o ConnectTimeout=5 "$REMOTE_USER@$REMOTE_HOST" "echo 'Connection successful'" 2>/dev/null; then
    echo "✅ Connection successful"
    echo ""
else
    echo "❌ ERROR: Could not connect to remote machine"
    echo ""
    echo "Please check:"
    echo "  1. Remote host is reachable"
    echo "  2. SSH credentials are correct"
    echo "  3. Port $REMOTE_PORT is open"
    echo ""
    exit 1
fi

# Ensure remote directory exists
echo "Ensuring remote directory exists..."
ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_PATH/data/processed"
echo ""

# Show what will be synced (dry run)
echo "Preview of files to sync (dry run):"
echo "---"
rsync -avz --dry-run --progress \
    -e "ssh -p $REMOTE_PORT" \
    "$LOCAL_DATA_PROCESSED/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/data/processed/"
echo "---"
echo ""

# Confirm before syncing
read -p "Proceed with sync? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Sync cancelled."
    exit 0
fi

# Perform actual sync
echo ""
echo "Syncing..."
rsync -avz --progress \
    -e "ssh -p $REMOTE_PORT" \
    "$LOCAL_DATA_PROCESSED/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/data/processed/"

# ============================================================
# VERIFICATION
# ============================================================

echo ""
echo "Verifying sync..."
REMOTE_FILES=$(ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "ls -1 $REMOTE_PATH/data/processed/ | wc -l")
echo "Files on remote: $REMOTE_FILES"
echo ""

# ============================================================
# SUCCESS
# ============================================================

echo "============================================================"
echo "✅ Sync Complete!"
echo "============================================================"
echo ""
echo "Processed data synced to:"
echo "  $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/data/processed/"
echo ""
echo "Next steps on remote machine:"
echo "  1. SSH into remote: ssh $REMOTE_USER@$REMOTE_HOST"
echo "  2. Activate environment: conda activate weatherman-lora"
echo "  3. Verify data: ls -lh $REMOTE_PATH/data/processed/"
echo "  4. Download base model (see docs/MODEL_DOWNLOAD.md)"
echo "  5. Start training!"
echo ""
