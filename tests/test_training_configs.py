#!/usr/bin/env python3
"""
Tests for Training Configuration Files

Validates that H100 and M4 training configs load correctly
and contain all required fields.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.config_loader import load_training_config, validate_required_fields


def test_h100_config_loads():
    """Test that H100 config file loads without errors."""
    config_path = project_root / "configs" / "training_config_h100.yaml"
    assert config_path.exists(), f"H100 config not found at {config_path}"

    config = load_training_config(config_path=config_path)
    assert config is not None
    assert isinstance(config, dict)


def test_m4_config_loads():
    """Test that M4 config file loads without errors."""
    config_path = project_root / "configs" / "training_config_m4.yaml"
    assert config_path.exists(), f"M4 config not found at {config_path}"

    config = load_training_config(config_path=config_path)
    assert config is not None
    assert isinstance(config, dict)


def test_h100_config_has_required_fields():
    """Test that H100 config contains all required fields."""
    config_path = project_root / "configs" / "training_config_h100.yaml"
    config = load_training_config(config_path=config_path)

    required_fields = [
        'lora.r',
        'lora.lora_alpha',
        'lora.target_modules',
        'training.learning_rate',
        'training.num_train_epochs',
        'model.model_name_or_path',
    ]

    # Should not raise ValueError
    validate_required_fields(config, required_fields, "training_config_h100.yaml")


def test_m4_config_has_required_fields():
    """Test that M4 config contains all required fields."""
    config_path = project_root / "configs" / "training_config_m4.yaml"
    config = load_training_config(config_path=config_path)

    required_fields = [
        'lora.r',
        'lora.lora_alpha',
        'lora.target_modules',
        'training.learning_rate',
        'training.num_train_epochs',
        'model.model_name_or_path',
    ]

    # Should not raise ValueError
    validate_required_fields(config, required_fields, "training_config_m4.yaml")


def test_h100_config_optimized_for_platform():
    """Test that H100 config has platform-specific optimizations."""
    config_path = project_root / "configs" / "training_config_h100.yaml"
    config = load_training_config(config_path=config_path)

    # H100 should use 4096 sequence length
    assert config['model']['max_seq_length'] == 4096

    # H100 should use larger batch size (4-8)
    assert config['training']['per_device_train_batch_size'] >= 4

    # Should use Mistral 7B Instruct
    assert 'Mistral-7B-Instruct' in config['model']['model_name_or_path']

    # Should use bfloat16
    assert config['training']['bf16'] is True
    assert config['training']['fp16'] is False

    # Should enable gradient checkpointing
    assert config['training']['gradient_checkpointing'] is True


def test_m4_config_optimized_for_platform():
    """Test that M4 config has platform-specific optimizations."""
    config_path = project_root / "configs" / "training_config_m4.yaml"
    config = load_training_config(config_path=config_path)

    # M4 should use 2048 sequence length (memory-constrained)
    assert config['model']['max_seq_length'] == 2048

    # M4 should use smaller batch size (1-2)
    assert config['training']['per_device_train_batch_size'] <= 2

    # M4 should use higher gradient accumulation
    assert config['training']['gradient_accumulation_steps'] >= 8

    # Should use Mistral 7B Instruct
    assert 'Mistral-7B-Instruct' in config['model']['model_name_or_path']

    # Should use bfloat16
    assert config['training']['bf16'] is True
    assert config['training']['fp16'] is False

    # Should enable gradient checkpointing
    assert config['training']['gradient_checkpointing'] is True


def test_configs_have_consistent_lora_params():
    """Test that H100 and M4 configs use identical LoRA parameters."""
    h100_config = load_training_config(
        config_path=project_root / "configs" / "training_config_h100.yaml"
    )
    m4_config = load_training_config(
        config_path=project_root / "configs" / "training_config_m4.yaml"
    )

    # LoRA parameters should be identical for reproducibility
    assert h100_config['lora']['r'] == m4_config['lora']['r']
    assert h100_config['lora']['lora_alpha'] == m4_config['lora']['lora_alpha']
    assert h100_config['lora']['lora_dropout'] == m4_config['lora']['lora_dropout']
    assert h100_config['lora']['target_modules'] == m4_config['lora']['target_modules']


def test_configs_use_same_base_model():
    """Test that both configs use the same base model."""
    h100_config = load_training_config(
        config_path=project_root / "configs" / "training_config_h100.yaml"
    )
    m4_config = load_training_config(
        config_path=project_root / "configs" / "training_config_m4.yaml"
    )

    # Both should use Mistral 7B Instruct v0.3
    assert h100_config['model']['model_name_or_path'] == m4_config['model']['model_name_or_path']
    assert h100_config['model']['model_name_or_path'] == 'mistralai/Mistral-7B-Instruct-v0.3'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
