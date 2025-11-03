#!/usr/bin/env python3
"""
Configuration Loader for Weatherman-LoRA Project

Loads and merges YAML training configs and JSON path configs.
Supports environment-specific overrides for local vs remote environments.

Usage:
    from scripts.config_loader import load_training_config, load_paths_config

    # Load configurations
    training_config = load_training_config()
    paths_config = load_paths_config()

    # Access nested values
    lora_rank = training_config['lora']['r']
    data_dir = paths_config['data']['processed']

    # Load with overrides
    custom_config = load_training_config(overrides={'training': {'num_train_epochs': 5}})
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Default config file paths
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training_config.yaml"
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths_config.json"


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config if config is not None else {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Load JSON configuration file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, 'r') as f:
        try:
            config = json.load(f)
            # Filter out comment fields
            return {k: v for k, v in config.items() if not k.startswith('_')}
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error parsing JSON file {file_path}: {e}")


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with overrides taking precedence.

    Args:
        base: Base configuration dictionary
        overrides: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    result = base.copy()

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


def validate_required_fields(config: Dict[str, Any], required_fields: list, config_name: str) -> None:
    """
    Validate that required fields are present in configuration.

    Args:
        config: Configuration dictionary
        required_fields: List of required field paths (e.g., ['lora.r', 'training.learning_rate'])
        config_name: Name of config for error messages

    Raises:
        ValueError: If required fields are missing
    """
    missing_fields = []

    for field_path in required_fields:
        keys = field_path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                missing_fields.append(field_path)
                break

    if missing_fields:
        raise ValueError(
            f"Missing required fields in {config_name}: {', '.join(missing_fields)}"
        )


def load_training_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to config file (default: configs/training_config.yaml)
        overrides: Dictionary of values to override in config

    Returns:
        Training configuration dictionary

    Example:
        # Load default config
        config = load_training_config()

        # Load with custom path
        config = load_training_config(Path("custom_config.yaml"))

        # Load with overrides
        config = load_training_config(overrides={
            'lora': {'r': 32, 'lora_alpha': 64},
            'training': {'num_train_epochs': 5}
        })
    """
    config_path = config_path or DEFAULT_TRAINING_CONFIG
    config = load_yaml(config_path)

    # Apply overrides if provided
    if overrides:
        config = deep_merge(config, overrides)

    # Validate required fields
    required_fields = [
        'lora.r',
        'lora.lora_alpha',
        'lora.target_modules',
        'training.learning_rate',
        'training.num_train_epochs',
        'model.model_name_or_path',
    ]
    validate_required_fields(config, required_fields, "training_config.yaml")

    return config


def load_paths_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load paths configuration from JSON file.

    Args:
        config_path: Path to config file (default: configs/paths_config.json)
        overrides: Dictionary of values to override in config

    Returns:
        Paths configuration dictionary

    Example:
        # Load default config
        config = load_paths_config()

        # Load with overrides for remote environment
        config = load_paths_config(overrides={
            'data': {'processed': '/remote/data/processed'}
        })
    """
    config_path = config_path or DEFAULT_PATHS_CONFIG
    config = load_json(config_path)

    # Apply overrides if provided
    if overrides:
        config = deep_merge(config, overrides)

    # Validate required fields
    required_fields = [
        'data.raw',
        'data.processed',
        'models.dir',
        'adapters.dir',
    ]
    validate_required_fields(config, required_fields, "paths_config.json")

    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'lora.r')
        default: Default value if key doesn't exist

    Returns:
        Configuration value or default

    Example:
        config = load_training_config()
        lora_rank = get_config_value(config, 'lora.r', default=16)
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """
    Pretty print configuration dictionary.

    Args:
        config: Configuration dictionary
        indent: Current indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


if __name__ == "__main__":
    # Demo: Load and display configurations
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    training_config = load_training_config()
    print_config(training_config)
    print()

    print("=" * 60)
    print("Paths Configuration")
    print("=" * 60)
    paths_config = load_paths_config()
    print_config(paths_config)
    print()

    # Demo: Access specific values
    print("=" * 60)
    print("Accessing Specific Values")
    print("=" * 60)
    print(f"LoRA rank: {get_config_value(training_config, 'lora.r')}")
    print(f"Learning rate: {get_config_value(training_config, 'training.learning_rate')}")
    print(f"Processed data path: {get_config_value(paths_config, 'data.processed')}")
    print(f"Models directory: {get_config_value(paths_config, 'models.dir')}")
