"""
Path Constants for Weatherman-LoRA Project

Provides centralized path management for data, models, and outputs.
Supports environment-specific overrides for local vs remote environments.

Usage:
    from scripts.paths import DATA_RAW, DATA_PROCESSED, MODELS_DIR

    # Read raw data
    with open(DATA_RAW / "example.txt") as f:
        content = f.read()

    # Override for remote environment
    from scripts.paths import override_paths
    override_paths(base_dir="/remote/path/to/weatherman-lora")
"""

import os
from pathlib import Path
from typing import Optional


# Base project directory (parent of scripts/)
_BASE_DIR = Path(__file__).parent.parent.resolve()

# Environment variable for base directory override
if "WEATHERMAN_BASE_DIR" in os.environ:
    _BASE_DIR = Path(os.environ["WEATHERMAN_BASE_DIR"]).resolve()


# Data directories
DATA_DIR = _BASE_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_SYNTHETIC = DATA_DIR / "synthetic"

# Source data (existing datasets)
DATA_SOURCES = _BASE_DIR / "data_sources"
REDDIT_DATA = DATA_SOURCES / "reddit-theonion" / "data"

# Model and adapter directories
MODELS_DIR = _BASE_DIR / "models"
ADAPTERS_DIR = _BASE_DIR / "adapters"

# Scripts and configuration
SCRIPTS_DIR = _BASE_DIR / "scripts"
CONFIGS_DIR = _BASE_DIR / "configs"

# Documentation
DOCS_DIR = _BASE_DIR / "docs"
REFERENCES_DIR = _BASE_DIR / "references"

# Agent OS directories
AGENT_OS_DIR = _BASE_DIR / "agent-os"
PRODUCT_DIR = AGENT_OS_DIR / "product"
SPECS_DIR = AGENT_OS_DIR / "specs"


def override_paths(base_dir: str) -> None:
    """
    Override all path constants with a new base directory.

    Useful when running on remote machines with different directory structures.

    Args:
        base_dir: New base directory path (string or Path)

    Example:
        # On remote GPU machine
        override_paths("/home/user/weatherman-lora")
    """
    global _BASE_DIR, DATA_DIR, DATA_RAW, DATA_PROCESSED, DATA_SYNTHETIC
    global DATA_SOURCES, REDDIT_DATA, MODELS_DIR, ADAPTERS_DIR
    global SCRIPTS_DIR, CONFIGS_DIR, DOCS_DIR, REFERENCES_DIR
    global AGENT_OS_DIR, PRODUCT_DIR, SPECS_DIR

    _BASE_DIR = Path(base_dir).resolve()

    # Update all derived paths
    DATA_DIR = _BASE_DIR / "data"
    DATA_RAW = DATA_DIR / "raw"
    DATA_PROCESSED = DATA_DIR / "processed"
    DATA_SYNTHETIC = DATA_DIR / "synthetic"

    DATA_SOURCES = _BASE_DIR / "data_sources"
    REDDIT_DATA = DATA_SOURCES / "reddit-theonion" / "data"

    MODELS_DIR = _BASE_DIR / "models"
    ADAPTERS_DIR = _BASE_DIR / "adapters"

    SCRIPTS_DIR = _BASE_DIR / "scripts"
    CONFIGS_DIR = _BASE_DIR / "configs"

    DOCS_DIR = _BASE_DIR / "docs"
    REFERENCES_DIR = _BASE_DIR / "references"

    AGENT_OS_DIR = _BASE_DIR / "agent-os"
    PRODUCT_DIR = AGENT_OS_DIR / "product"
    SPECS_DIR = AGENT_OS_DIR / "specs"


def get_base_dir() -> Path:
    """
    Get the current base directory.

    Returns:
        Path object representing the project base directory
    """
    return _BASE_DIR


def ensure_dirs_exist() -> None:
    """
    Create all required directories if they don't exist.

    Safe to call multiple times (idempotent).
    """
    dirs_to_create = [
        DATA_RAW,
        DATA_PROCESSED,
        DATA_SYNTHETIC,
        MODELS_DIR,
        ADAPTERS_DIR,
        SCRIPTS_DIR,
        CONFIGS_DIR,
        DOCS_DIR,
    ]

    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)


def print_paths() -> None:
    """Print all configured paths for debugging."""
    print("Weatherman-LoRA Path Configuration")
    print("=" * 60)
    print(f"Base Directory:       {_BASE_DIR}")
    print()
    print("Data Directories:")
    print(f"  DATA_RAW:           {DATA_RAW}")
    print(f"  DATA_PROCESSED:     {DATA_PROCESSED}")
    print(f"  DATA_SYNTHETIC:     {DATA_SYNTHETIC}")
    print(f"  REDDIT_DATA:        {REDDIT_DATA}")
    print()
    print("Model Directories:")
    print(f"  MODELS_DIR:         {MODELS_DIR}")
    print(f"  ADAPTERS_DIR:       {ADAPTERS_DIR}")
    print()
    print("Project Directories:")
    print(f"  SCRIPTS_DIR:        {SCRIPTS_DIR}")
    print(f"  CONFIGS_DIR:        {CONFIGS_DIR}")
    print(f"  DOCS_DIR:           {DOCS_DIR}")
    print(f"  REFERENCES_DIR:     {REFERENCES_DIR}")
    print()


if __name__ == "__main__":
    # Print paths when run directly
    print_paths()

    # Check if directories exist
    print("Directory Status:")
    print(f"  DATA_RAW exists:       {DATA_RAW.exists()}")
    print(f"  DATA_PROCESSED exists: {DATA_PROCESSED.exists()}")
    print(f"  MODELS_DIR exists:     {MODELS_DIR.exists()}")
    print(f"  ADAPTERS_DIR exists:   {ADAPTERS_DIR.exists()}")
