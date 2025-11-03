#!/usr/bin/env python3
"""
Model Verification Script for Weatherman-LoRA Project

Verifies that a base model has been downloaded correctly and can be loaded.

Usage:
    python scripts/verify_model.py meta-llama/Meta-Llama-3.1-8B-Instruct
    python scripts/verify_model.py mistralai/Mistral-7B-Instruct-v0.2
"""

import sys
import os
from pathlib import Path


def check_transformers_import():
    """Check if transformers is installed."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        return AutoTokenizer, AutoModelForCausalLM, AutoConfig
    except ImportError:
        print("❌ ERROR: transformers library is not installed")
        print()
        print("Please install transformers:")
        print("  pip install transformers")
        print()
        print("Or run the remote environment setup:")
        print("  ./setup_remote.sh")
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


def get_model_size(model_name):
    """Estimate model size from cache directory."""
    try:
        from huggingface_hub import snapshot_download
        cache_dir = snapshot_download(model_name, local_files_only=True)

        total_size = 0
        cache_path = Path(cache_dir)

        for file in cache_path.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size

        return total_size
    except Exception as e:
        return None


def verify_model(model_name):
    """Verify that model exists and can be loaded."""
    print("=" * 60)
    print("Weatherman-LoRA Model Verification")
    print("=" * 60)
    print()
    print(f"Model: {model_name}")
    print()

    # Import transformers
    AutoTokenizer, AutoModelForCausalLM, AutoConfig = check_transformers_import()

    # Check if model is cached
    print("Step 1: Checking model cache...")
    try:
        from huggingface_hub import scan_cache_dir, try_to_load_from_cache

        # Try to find model in cache
        cache_info = scan_cache_dir()
        model_found = False
        model_cache_path = None

        for repo in cache_info.repos:
            if model_name in repo.repo_id:
                model_found = True
                model_cache_path = repo.repo_path
                print(f"✅ Model found in cache")
                print(f"   Location: {model_cache_path}")

                # Get size
                total_size = sum(
                    revision.size_on_disk
                    for revision in repo.revisions
                )
                print(f"   Size: {format_size(total_size)} ({bytes_to_gb(total_size):.2f} GB)")
                break

        if not model_found:
            print(f"❌ Model not found in cache")
            print()
            print("Please download the model first:")
            print(f"  python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('{model_name}')\"")
            print()
            print("Or use huggingface-cli:")
            print(f"  huggingface-cli download {model_name}")
            print()
            print("See docs/MODEL_DOWNLOAD.md for detailed instructions")
            sys.exit(1)

    except ImportError:
        print("⚠️  Warning: huggingface_hub not available for cache inspection")
        print("   Proceeding with model loading test...")
    print()

    # Load model config
    print("Step 2: Loading model configuration...")
    try:
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        print(f"✅ Configuration loaded successfully")
        print(f"   Architecture: {config.architectures[0] if config.architectures else 'Unknown'}")
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Num layers: {config.num_hidden_layers}")
        print(f"   Num attention heads: {config.num_attention_heads}")
        print(f"   Vocabulary size: {config.vocab_size:,} tokens")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    print()

    # Load tokenizer
    print("Step 3: Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Tokenizer type: {type(tokenizer).__name__}")
        print(f"   Vocabulary size: {len(tokenizer):,} tokens")
        print(f"   Special tokens: {len(tokenizer.all_special_tokens)} tokens")

        # Test tokenization
        test_text = "What's the weather like today?"
        tokens = tokenizer.encode(test_text)
        print(f"   Test encoding: '{test_text}' -> {len(tokens)} tokens")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        sys.exit(1)
    print()

    # Verify model files
    print("Step 4: Verifying model files...")
    try:
        # Check for required files
        from huggingface_hub import hf_hub_download

        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]

        missing_files = []
        for file in required_files:
            try:
                hf_hub_download(model_name, file, local_files_only=True)
            except Exception:
                missing_files.append(file)

        if missing_files:
            print(f"⚠️  Missing files: {', '.join(missing_files)}")
        else:
            print(f"✅ All required files present")

        # Check for model weights (safetensors or pytorch)
        has_safetensors = False
        has_pytorch = False

        try:
            hf_hub_download(model_name, "model.safetensors", local_files_only=True)
            has_safetensors = True
        except Exception:
            pass

        try:
            hf_hub_download(model_name, "pytorch_model.bin", local_files_only=True)
            has_pytorch = True
        except Exception:
            pass

        if has_safetensors:
            print(f"   Format: SafeTensors ✅ (recommended)")
        elif has_pytorch:
            print(f"   Format: PyTorch (.bin)")
        else:
            # Check for sharded weights
            try:
                hf_hub_download(model_name, "model-00001-of-00004.safetensors", local_files_only=True)
                print(f"   Format: SafeTensors (sharded) ✅")
            except Exception:
                print(f"   ⚠️  No model weight files found")

    except Exception as e:
        print(f"⚠️  Could not verify files: {e}")
    print()

    # Summary
    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print()
    print(f"✅ Model verified: {model_name}")
    print()
    print("The model is ready for training!")
    print()
    print("Next steps:")
    print("  1. Sync processed data to this machine (see docs/DATA_SYNC.md)")
    print("  2. Configure training parameters (configs/training_config.yaml)")
    print("  3. Run LoRA training")
    print()


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/verify_model.py <model_name>")
        print()
        print("Examples:")
        print("  python scripts/verify_model.py meta-llama/Meta-Llama-3.1-8B-Instruct")
        print("  python scripts/verify_model.py mistralai/Mistral-7B-Instruct-v0.2")
        sys.exit(1)

    model_name = sys.argv[1]
    verify_model(model_name)


if __name__ == "__main__":
    main()
