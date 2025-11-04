#!/usr/bin/env python3
"""
Integration Tests for QLoRA Training Configuration

Tests end-to-end workflows for dual-platform training setup.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.config_loader import load_training_config, get_config_value


def test_h100_config_end_to_end():
    """Test H100 config loads and can be used for training setup."""
    # Load H100 config
    config_path = project_root / "configs" / "training_config_h100.yaml"
    config = load_training_config(config_path=config_path)

    # Verify key training parameters are accessible
    assert get_config_value(config, 'model.model_name_or_path') == 'mistralai/Mistral-7B-Instruct-v0.3'
    assert get_config_value(config, 'model.max_seq_length') == 4096
    assert get_config_value(config, 'lora.r') == 16
    assert get_config_value(config, 'lora.lora_alpha') == 32
    assert get_config_value(config, 'training.per_device_train_batch_size') >= 4
    assert get_config_value(config, 'training.gradient_checkpointing') is True
    assert get_config_value(config, 'quantization.load_in_4bit') is True


def test_m4_config_end_to_end():
    """Test M4 config loads and can be used for training setup."""
    # Load M4 config
    config_path = project_root / "configs" / "training_config_m4.yaml"
    config = load_training_config(config_path=config_path)

    # Verify key training parameters are accessible
    assert get_config_value(config, 'model.model_name_or_path') == 'mistralai/Mistral-7B-Instruct-v0.3'
    assert get_config_value(config, 'model.max_seq_length') == 2048
    assert get_config_value(config, 'lora.r') == 16
    assert get_config_value(config, 'lora.lora_alpha') == 32
    assert get_config_value(config, 'training.per_device_train_batch_size') <= 2
    assert get_config_value(config, 'training.gradient_accumulation_steps') >= 8
    assert get_config_value(config, 'training.gradient_checkpointing') is True
    assert get_config_value(config, 'quantization.load_in_4bit') is True


def test_config_override_functionality():
    """Test that config override functionality works."""
    h100_config_path = project_root / "configs" / "training_config_h100.yaml"

    # Load with overrides
    overrides = {
        'training': {
            'num_train_epochs': 5,
            'learning_rate': 1e-4,
        },
        'lora': {
            'r': 32,
        }
    }

    config = load_training_config(config_path=h100_config_path, overrides=overrides)

    # Verify overrides were applied
    assert get_config_value(config, 'training.num_train_epochs') == 5
    assert get_config_value(config, 'training.learning_rate') == 1e-4
    assert get_config_value(config, 'lora.r') == 32

    # Verify other values remain unchanged
    assert get_config_value(config, 'model.model_name_or_path') == 'mistralai/Mistral-7B-Instruct-v0.3'
    assert get_config_value(config, 'lora.lora_alpha') == 32  # Should still be from original config


def test_config_loader_validates_mistral_fields():
    """Test that config loader validates required Mistral 7B fields."""
    h100_config_path = project_root / "configs" / "training_config_h100.yaml"
    config = load_training_config(config_path=h100_config_path)

    # Required fields should be present
    assert 'model' in config
    assert 'model_name_or_path' in config['model']
    assert 'Mistral' in config['model']['model_name_or_path']

    # LoRA configuration should be complete
    assert 'lora' in config
    assert 'r' in config['lora']
    assert 'lora_alpha' in config['lora']
    assert 'target_modules' in config['lora']
    assert len(config['lora']['target_modules']) == 7  # All projection modules


def test_platform_detection_recommends_correct_config():
    """Test that platform detection logic recommends correct config."""
    # This is a simplified test - actual platform detection happens in check_gpu.py
    # We just verify the config files exist and have the right characteristics

    h100_config = load_training_config(
        config_path=project_root / "configs" / "training_config_h100.yaml"
    )
    m4_config = load_training_config(
        config_path=project_root / "configs" / "training_config_m4.yaml"
    )

    # H100 should have higher seq length (Flash Attention)
    assert get_config_value(h100_config, 'model.max_seq_length') == 4096

    # M4 should have lower seq length (memory-constrained)
    assert get_config_value(m4_config, 'model.max_seq_length') == 2048

    # H100 should have larger batch size
    h100_batch = get_config_value(h100_config, 'training.per_device_train_batch_size')
    m4_batch = get_config_value(m4_config, 'training.per_device_train_batch_size')
    assert h100_batch > m4_batch

    # M4 should have higher gradient accumulation
    h100_grad_accum = get_config_value(h100_config, 'training.gradient_accumulation_steps')
    m4_grad_accum = get_config_value(m4_config, 'training.gradient_accumulation_steps')
    assert m4_grad_accum > h100_grad_accum


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
