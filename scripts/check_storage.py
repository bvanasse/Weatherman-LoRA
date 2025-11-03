#!/usr/bin/env python3
"""
Storage Verification Script for Weatherman-LoRA Project

Checks available disk space and ensures sufficient storage for:
- Raw data: ~150-300MB
- Processed JSONL: ~500MB-1GB
- Base models: ~15GB
- Checkpoints/adapters: ~500MB-2GB
- Working buffer: ~10-15GB
Total recommended: 30-50GB minimum
"""

import shutil
import sys
from pathlib import Path


def get_disk_usage(path="."):
    """Get disk usage statistics for the given path."""
    try:
        usage = shutil.disk_usage(path)
        return usage
    except Exception as e:
        print(f"Error checking disk usage: {e}")
        sys.exit(1)


def bytes_to_gb(bytes_value):
    """Convert bytes to gigabytes."""
    return bytes_value / (1024 ** 3)


def format_size(bytes_value):
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def main():
    """Check storage availability and display requirements."""
    print("=" * 60)
    print("Weatherman-LoRA Storage Verification")
    print("=" * 60)
    print()

    # Get current directory
    project_root = Path(__file__).parent.parent

    # Get disk usage
    usage = get_disk_usage(project_root)
    total_gb = bytes_to_gb(usage.total)
    used_gb = bytes_to_gb(usage.used)
    free_gb = bytes_to_gb(usage.free)

    # Display current storage status
    print(f"Current Storage Status:")
    print(f"  Total Space:     {format_size(usage.total)} ({total_gb:.2f} GB)")
    print(f"  Used Space:      {format_size(usage.used)} ({used_gb:.2f} GB)")
    print(f"  Available Space: {format_size(usage.free)} ({free_gb:.2f} GB)")
    print()

    # Display expected storage breakdown
    print("Expected Storage Requirements:")
    print("  Raw Data:                ~150-300 MB")
    print("  Processed JSONL:         ~500 MB - 1 GB")
    print("  Base Model (Llama/Mistral): ~15 GB")
    print("  Checkpoints/Adapters:    ~500 MB - 2 GB")
    print("  Working Buffer:          ~10-15 GB")
    print("  " + "-" * 50)
    print("  Total Recommended:       30-50 GB minimum")
    print()

    # Check if sufficient space is available
    min_required_gb = 20
    recommended_gb = 30

    if free_gb < min_required_gb:
        print(f"❌ ERROR: Insufficient storage!")
        print(f"   Available: {free_gb:.2f} GB")
        print(f"   Required: {min_required_gb} GB minimum")
        print()
        print("   Please free up disk space before proceeding.")
        sys.exit(1)
    elif free_gb < recommended_gb:
        print(f"⚠️  WARNING: Low storage available!")
        print(f"   Available: {free_gb:.2f} GB")
        print(f"   Recommended: {recommended_gb} GB minimum")
        print()
        print("   You may encounter issues during training.")
        print("   Consider freeing up additional space.")
        sys.exit(0)
    else:
        print(f"✅ Storage Check Passed!")
        print(f"   Available: {free_gb:.2f} GB")
        print(f"   Recommended: {recommended_gb} GB minimum")
        print()
        print("   Sufficient storage available for project.")
        sys.exit(0)


if __name__ == "__main__":
    main()
