#!/bin/bash
#
# Sync Trained Adapters from Remote GPU Machine
#
# This script syncs trained LoRA adapters from remote back to local.
# Edit the variables below to match your remote machine setup.
#
# Usage: ./scripts/sync_from_remote.sh [adapter_name]
#   If adapter_name is not provided, syncs all adapters

set -e

echo "============================================================"
echo "Sync Trained Adapters from Remote"
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
LOCAL_ADAPTERS_DIR="adapters"

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

# Check if local adapters directory exists
if [ ! -d "$LOCAL_ADAPTERS_DIR" ]; then
    echo "Creating local adapters directory..."
    mkdir -p "$LOCAL_ADAPTERS_DIR"
fi

# ============================================================
# DETERMINE WHAT TO SYNC
# ============================================================

ADAPTER_NAME="${1:-}"

if [ -z "$ADAPTER_NAME" ]; then
    # Sync all adapters
    REMOTE_SOURCE="$REMOTE_PATH/adapters/"
    LOCAL_DEST="$LOCAL_ADAPTERS_DIR/"
    SYNC_DESC="all adapters"
else
    # Sync specific adapter
    REMOTE_SOURCE="$REMOTE_PATH/adapters/$ADAPTER_NAME/"
    LOCAL_DEST="$LOCAL_ADAPTERS_DIR/$ADAPTER_NAME/"
    SYNC_DESC="adapter: $ADAPTER_NAME"
fi

# ============================================================
# CONNECTION TEST
# ============================================================

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

# ============================================================
# CHECK REMOTE FILES
# ============================================================

echo "Checking remote adapters directory..."
if ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "[ -d $REMOTE_PATH/adapters ]"; then
    echo "✅ Remote adapters directory exists"
else
    echo "❌ ERROR: Remote adapters directory not found"
    echo ""
    echo "Expected location: $REMOTE_PATH/adapters/"
    echo ""
    echo "Have you completed training yet?"
    exit 1
fi

# List available adapters
echo ""
echo "Available adapters on remote:"
ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "ls -lh $REMOTE_PATH/adapters/" || {
    echo "No adapters found on remote"
    exit 1
}
echo ""

# ============================================================
# SYNC
# ============================================================

echo "Syncing $SYNC_DESC from remote..."
echo ""
echo "  Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_SOURCE"
echo "  Local:  $LOCAL_DEST"
echo ""

# Show what will be synced (dry run)
echo "Preview of files to sync (dry run):"
echo "---"
rsync -avz --dry-run --progress \
    -e "ssh -p $REMOTE_PORT" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SOURCE" \
    "$LOCAL_DEST"
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
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SOURCE" \
    "$LOCAL_DEST"

# ============================================================
# VERIFICATION
# ============================================================

echo ""
echo "Verifying sync..."
if [ -z "$ADAPTER_NAME" ]; then
    LOCAL_DIRS=$(ls -1 "$LOCAL_ADAPTERS_DIR" | wc -l)
    echo "Adapter directories in local: $LOCAL_DIRS"
else
    if [ -d "$LOCAL_DEST" ]; then
        LOCAL_FILES=$(ls -1 "$LOCAL_DEST" | wc -l)
        echo "Files in local adapter: $LOCAL_FILES"
    else
        echo "⚠️  Warning: Local adapter directory not found after sync"
    fi
fi
echo ""

# ============================================================
# SUCCESS
# ============================================================

echo "============================================================"
echo "✅ Sync Complete!"
echo "============================================================"
echo ""
echo "Adapters synced to: $LOCAL_DEST"
echo ""
echo "Adapter contents:"
ls -lh "$LOCAL_DEST" 2>/dev/null || echo "  (empty)"
echo ""
echo "Next steps:"
echo "  1. Verify adapter files are present"
echo "  2. Run evaluation metrics"
echo "  3. Test model with sample queries"
echo "  4. Merge adapter if deploying (optional)"
echo ""
echo "Key files to check:"
echo "  - adapter_config.json  (LoRA configuration)"
echo "  - adapter_model.bin    (LoRA weights)"
echo "  - training_args.bin    (training configuration)"
echo ""
