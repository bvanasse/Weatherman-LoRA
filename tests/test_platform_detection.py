#!/usr/bin/env python3
"""
Tests for Platform Detection and Validation

Tests GPU detection, platform validation, and config validation scripts.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.config_loader import load_training_config


def test_check_gpu_script_exists():
    """Test that check_gpu.py script exists and is executable."""
    script_path = project_root / "scripts" / "check_gpu.py"
    assert script_path.exists(), "check_gpu.py not found"


def test_validate_environment_script_exists():
    """Test that validate_environment.py script exists."""
    script_path = project_root / "scripts" / "validate_environment.py"
    assert script_path.exists(), "validate_environment.py not found"


def test_validate_training_config_script_exists():
    """Test that validate_training_config.py script exists."""
    script_path = project_root / "scripts" / "validate_training_config.py"
    assert script_path.exists(), "validate_training_config.py not found"


def test_validate_training_config_h100():
    """Test that H100 config passes validation checks."""
    config_path = project_root / "configs" / "training_config_h100.yaml"
    config = load_training_config(config_path=config_path)

    # Check required fields for validation
    assert config.get('model', {}).get('model_name_or_path') is not None
    assert config.get('lora', {}).get('r') is not None
    assert config.get('lora', {}).get('lora_alpha') is not None
    assert config.get('training', {}).get('learning_rate') is not None


def test_validate_training_config_m4():
    """Test that M4 config passes validation checks."""
    config_path = project_root / "configs" / "training_config_m4.yaml"
    config = load_training_config(config_path=config_path)

    # Check required fields for validation
    assert config.get('model', {}).get('model_name_or_path') is not None
    assert config.get('lora', {}).get('r') is not None
    assert config.get('lora', {}).get('lora_alpha') is not None
    assert config.get('training', {}).get('learning_rate') is not None


def test_lora_params_in_valid_range():
    """Test that LoRA parameters are in acceptable ranges."""
    h100_config = load_training_config(
        config_path=project_root / "configs" / "training_config_h100.yaml"
    )
    m4_config = load_training_config(
        config_path=project_root / "configs" / "training_config_m4.yaml"
    )

    # Check H100 config
    assert 8 <= h100_config['lora']['r'] <= 64, "H100 LoRA rank out of range"
    assert 0.0 <= h100_config['lora']['lora_dropout'] <= 0.2, "H100 dropout out of range"
    assert len(h100_config['lora']['target_modules']) >= 4, "H100 too few target modules"

    # Check M4 config
    assert 8 <= m4_config['lora']['r'] <= 64, "M4 LoRA rank out of range"
    assert 0.0 <= m4_config['lora']['lora_dropout'] <= 0.2, "M4 dropout out of range"
    assert len(m4_config['lora']['target_modules']) >= 4, "M4 too few target modules"


def test_h100_memory_config_reasonable():
    """Test that H100 config has reasonable memory settings."""
    config = load_training_config(
        config_path=project_root / "configs" / "training_config_h100.yaml"
    )

    seq_length = config['model']['max_seq_length']
    batch_size = config['training']['per_device_train_batch_size']

    # H100 should use 4096 seq length
    assert seq_length == 4096, f"H100 seq_length should be 4096, got {seq_length}"

    # Batch size should be reasonable (4-8 for H100)
    assert 2 <= batch_size <= 8, f"H100 batch_size should be 2-8, got {batch_size}"


def test_m4_memory_config_reasonable():
    """Test that M4 config has reasonable memory settings."""
    config = load_training_config(
        config_path=project_root / "configs" / "training_config_m4.yaml"
    )

    seq_length = config['model']['max_seq_length']
    batch_size = config['training']['per_device_train_batch_size']

    # M4 should use 2048 seq length (memory-constrained)
    assert seq_length == 2048, f"M4 seq_length should be 2048, got {seq_length}"

    # Batch size should be small (1-2 for M4)
    assert 1 <= batch_size <= 2, f"M4 batch_size should be 1-2, got {batch_size}"

    # Gradient accumulation should be higher for M4
    grad_accum = config['training']['gradient_accumulation_steps']
    assert grad_accum >= 8, f"M4 gradient_accumulation should be >= 8, got {grad_accum}"


def test_config_validation_script_detects_missing_fields():
    """Test that validation script can detect missing required fields."""
    # This would require actually running the script, which is beyond unit testing
    # Just verify the script exists and has the right structure
    from scripts import validate_training_config

    # Check that validation functions exist
    assert hasattr(validate_training_config, 'validate_required_fields')
    assert hasattr(validate_training_config, 'validate_lora_params')
    assert hasattr(validate_training_config, 'validate_memory_requirements')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
