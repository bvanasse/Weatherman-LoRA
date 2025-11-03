#!/usr/bin/env python3
"""
Training Configuration Validation Script

Validates training configuration files for completeness and correctness.
Checks LoRA parameters, memory requirements, and data paths.

Usage:
    python scripts/validate_training_config.py --config configs/training_config_h100.yaml
    python scripts/validate_training_config.py --config configs/training_config_m4.yaml
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.config_loader import load_training_config, get_config_value


def print_header(title):
    """Print section header."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print()


def print_check(passed, message):
    """Print check result."""
    if passed:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
    return passed


def validate_required_fields(config):
    """Validate that all required fields are present."""
    print_header("Required Fields Check")

    required_fields = [
        'model.model_name_or_path',
        'lora.r',
        'lora.lora_alpha',
        'lora.target_modules',
        'training.learning_rate',
        'training.num_train_epochs',
        'training.per_device_train_batch_size',
        'training.gradient_accumulation_steps',
    ]

    all_passed = True

    for field_path in required_fields:
        value = get_config_value(config, field_path)
        if value is not None:
            print_check(True, f"{field_path}: {value}")
        else:
            print_check(False, f"{field_path}: MISSING")
            all_passed = False

    return all_passed


def validate_lora_params(config):
    """Validate LoRA parameters are in acceptable ranges."""
    print_header("LoRA Parameters Check")

    all_passed = True

    # Check LoRA rank
    r = get_config_value(config, 'lora.r')
    if r is not None:
        if 8 <= r <= 64:
            print_check(True, f"LoRA rank (r={r}) in acceptable range [8, 64]")
        else:
            print_check(False, f"LoRA rank (r={r}) outside recommended range [8, 64]")
            all_passed = False
    else:
        print_check(False, "LoRA rank not specified")
        all_passed = False

    # Check LoRA alpha
    alpha = get_config_value(config, 'lora.lora_alpha')
    if alpha is not None and r is not None:
        if alpha == 2 * r:
            print_check(True, f"LoRA alpha ({alpha}) = 2 * rank ({r}) [recommended]")
        else:
            print(f"⚠️  LoRA alpha ({alpha}) != 2 * rank ({r}) [not critical but recommended]")
    else:
        print_check(False, "LoRA alpha not specified")
        all_passed = False

    # Check LoRA dropout
    dropout = get_config_value(config, 'lora.lora_dropout')
    if dropout is not None:
        if 0.0 <= dropout <= 0.2:
            print_check(True, f"LoRA dropout ({dropout}) in acceptable range [0.0, 0.2]")
        else:
            print_check(False, f"LoRA dropout ({dropout}) outside recommended range [0.0, 0.2]")
            all_passed = False
    else:
        print_check(False, "LoRA dropout not specified")
        all_passed = False

    # Check target modules
    target_modules = get_config_value(config, 'lora.target_modules')
    if target_modules is not None:
        if len(target_modules) >= 4:
            print_check(True, f"Target modules ({len(target_modules)}): {', '.join(target_modules)}")
        else:
            print_check(False, f"Too few target modules ({len(target_modules)}): need at least 4")
            all_passed = False
    else:
        print_check(False, "Target modules not specified")
        all_passed = False

    return all_passed


def validate_memory_requirements(config):
    """Validate memory requirements based on platform."""
    print_header("Memory Requirements Check")

    all_passed = True

    seq_length = get_config_value(config, 'model.max_seq_length')
    batch_size = get_config_value(config, 'training.per_device_train_batch_size')
    model_name = get_config_value(config, 'model.model_name_or_path', '')

    # Estimate memory requirements (rough approximation)
    # Base model (4-bit): ~5-6GB
    # Activations scale with seq_length^2 and batch_size
    # For Mistral 7B in 4-bit with gradient checkpointing

    if seq_length and batch_size:
        # Very rough estimate
        base_memory = 6  # GB for 4-bit model
        activation_memory_per_token = 0.015  # GB per token per batch (approximate)
        activation_memory = seq_length * batch_size * activation_memory_per_token
        optimizer_memory = 4  # GB for paged AdamW
        total_memory = base_memory + activation_memory + optimizer_memory

        print(f"Estimated memory usage (approximate):")
        print(f"  Sequence length: {seq_length}")
        print(f"  Batch size: {batch_size}")
        print(f"  Base model (4-bit): ~{base_memory:.1f} GB")
        print(f"  Activations: ~{activation_memory:.1f} GB")
        print(f"  Optimizer: ~{optimizer_memory:.1f} GB")
        print(f"  Total estimate: ~{total_memory:.1f} GB")
        print()

        # Platform-specific warnings
        if seq_length == 4096:
            # H100 platform
            if total_memory > 70:
                print_check(False, f"Estimated {total_memory:.1f}GB may exceed H100 80GB VRAM")
                print("  Consider reducing batch size")
                all_passed = False
            else:
                print_check(True, f"Estimated {total_memory:.1f}GB should fit in H100 80GB VRAM")
        elif seq_length == 2048:
            # M4 platform
            if total_memory > 28:
                print_check(False, f"Estimated {total_memory:.1f}GB may exceed M4 32GB unified memory")
                print("  Consider reducing batch size to 1")
                all_passed = False
            else:
                print_check(True, f"Estimated {total_memory:.1f}GB should fit in M4 32GB unified memory")
        else:
            print(f"⚠️  Non-standard sequence length ({seq_length})")

    return all_passed


def validate_data_paths(config):
    """Validate that data file paths exist."""
    print_header("Data Paths Check")

    all_passed = True

    train_file = get_config_value(config, 'dataset.train_file')
    val_file = get_config_value(config, 'dataset.val_file')

    if train_file:
        train_path = Path(train_file)
        if train_path.exists():
            print_check(True, f"Training file exists: {train_file}")
        else:
            print_check(False, f"Training file NOT FOUND: {train_file}")
            print(f"  Create this file or update config path")
            all_passed = False
    else:
        print_check(False, "Training file path not specified")
        all_passed = False

    if val_file:
        val_path = Path(val_file)
        if val_path.exists():
            print_check(True, f"Validation file exists: {val_file}")
        else:
            print(f"⚠️  Validation file NOT FOUND: {val_file}")
            print(f"  Will use train/val split instead (not critical)")
    else:
        print(f"⚠️  Validation file path not specified")
        print(f"  Will use train/val split instead (not critical)")

    return all_passed


def validate_wandb_config(config):
    """Validate Weights & Biases configuration."""
    print_header("Weights & Biases Check")

    all_passed = True

    report_to = get_config_value(config, 'training.report_to')
    run_name = get_config_value(config, 'training.run_name')

    if report_to:
        if 'wandb' in report_to.lower():
            print_check(True, f"Reporting to: {report_to}")

            if run_name:
                print_check(True, f"Run name: {run_name}")
            else:
                print(f"⚠️  No run name specified (will use default)")
        else:
            print(f"ℹ️  Not using wandb (reporting to: {report_to})")
    else:
        print(f"⚠️  No reporting configured")

    return all_passed


def validate_model_compatibility(config):
    """Validate model compatibility."""
    print_header("Model Compatibility Check")

    all_passed = True

    model_name = get_config_value(config, 'model.model_name_or_path')

    if model_name:
        if 'Mistral-7B-Instruct' in model_name:
            print_check(True, f"Using Mistral 7B Instruct: {model_name}")
        else:
            print(f"⚠️  Using non-standard model: {model_name}")
            print(f"  This config is optimized for Mistral 7B Instruct")
    else:
        print_check(False, "Model name not specified")
        all_passed = False

    # Check quantization config
    load_in_4bit = get_config_value(config, 'quantization.load_in_4bit')
    if load_in_4bit:
        print_check(True, "4-bit quantization enabled (QLoRA)")
    else:
        print_check(False, "4-bit quantization not enabled")
        all_passed = False

    return all_passed


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description='Validate training configuration file'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config file (e.g., configs/training_config_h100.yaml)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Training Configuration Validation")
    print("=" * 60)
    print()
    print(f"Config file: {args.config}")

    # Load configuration
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            print()
            print(f"❌ ERROR: Config file not found: {args.config}")
            sys.exit(1)

        config = load_training_config(config_path=config_path)
    except Exception as e:
        print()
        print(f"❌ ERROR loading config: {e}")
        sys.exit(1)

    # Run validation checks
    checks = [
        ("Required Fields", validate_required_fields(config)),
        ("LoRA Parameters", validate_lora_params(config)),
        ("Memory Requirements", validate_memory_requirements(config)),
        ("Data Paths", validate_data_paths(config)),
        ("Weights & Biases", validate_wandb_config(config)),
        ("Model Compatibility", validate_model_compatibility(config)),
    ]

    # Summary
    print_header("Validation Summary")

    passed_count = sum(1 for _, passed in checks if passed)
    total_count = len(checks)

    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:30s} {status}")

    print()
    print(f"Overall: {passed_count}/{total_count} checks passed")
    print()

    if passed_count == total_count:
        print("✅ Configuration is valid and ready for training!")
        print()
        print("Next steps:")
        print("  1. Check platform: python scripts/check_gpu.py")
        print("  2. Validate environment: python scripts/validate_environment.py --env=h100  (or --env=m4)")
        print("  3. Start training with this config")
        return 0
    else:
        print("❌ Configuration has issues. Please fix the errors above.")
        print()
        print("For help, see:")
        print("  - configs/training_config_h100.yaml (H100 example)")
        print("  - configs/training_config_m4.yaml (M4 example)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
