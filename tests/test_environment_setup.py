#!/usr/bin/env python3
"""
Tests for Environment Setup Files

Tests environment configurations and setup scripts for H100 and M4 platforms.
"""

import pytest
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent


def test_environment_remote_yml_exists():
    """Test that environment-remote.yml exists."""
    env_file = project_root / "environment-remote.yml"
    assert env_file.exists(), "environment-remote.yml not found"


def test_requirements_m4_txt_exists():
    """Test that requirements-m4.txt exists."""
    req_file = project_root / "requirements-m4.txt"
    assert req_file.exists(), "requirements-m4.txt not found"


def test_setup_m4_sh_exists():
    """Test that setup_m4.sh exists and is executable."""
    setup_file = project_root / "setup_m4.sh"
    assert setup_file.exists(), "setup_m4.sh not found"
    # Check if executable (on Unix systems)
    import os
    assert os.access(setup_file, os.X_OK), "setup_m4.sh is not executable"


def test_environment_remote_has_h100_dependencies():
    """Test that environment-remote.yml has H100-specific dependencies."""
    env_file = project_root / "environment-remote.yml"

    with open(env_file, 'r') as f:
        env_config = yaml.safe_load(f)

    # Check conda channels
    assert 'pytorch' in env_config['channels']
    assert 'nvidia' in env_config['channels']

    # Check Python version
    python_dep = [d for d in env_config['dependencies'] if isinstance(d, str) and 'python' in d]
    assert len(python_dep) > 0, "Python version not specified"

    # Check PyTorch
    pytorch_dep = [d for d in env_config['dependencies'] if isinstance(d, str) and 'pytorch' in d]
    assert len(pytorch_dep) > 0, "PyTorch not specified"

    # Check CUDA toolkit
    cuda_dep = [d for d in env_config['dependencies'] if isinstance(d, str) and 'cuda' in d]
    assert len(cuda_dep) > 0, "CUDA toolkit not specified"


def test_environment_remote_has_training_libraries():
    """Test that environment-remote.yml has required training libraries."""
    env_file = project_root / "environment-remote.yml"

    with open(env_file, 'r') as f:
        env_config = yaml.safe_load(f)

    # Find pip section
    pip_deps = None
    for dep in env_config['dependencies']:
        if isinstance(dep, dict) and 'pip' in dep:
            pip_deps = dep['pip']
            break

    assert pip_deps is not None, "pip dependencies section not found"

    # Check for required libraries (as strings or in strings)
    pip_str = '\n'.join(pip_deps)
    assert 'transformers' in pip_str, "transformers not in pip dependencies"
    assert 'peft' in pip_str, "peft not in pip dependencies"
    assert 'trl' in pip_str, "trl not in pip dependencies"
    assert 'bitsandbytes' in pip_str, "bitsandbytes not in pip dependencies"
    assert 'accelerate' in pip_str, "accelerate not in pip dependencies"


def test_requirements_m4_has_pytorch():
    """Test that requirements-m4.txt has PyTorch with MPS support."""
    req_file = project_root / "requirements-m4.txt"

    with open(req_file, 'r') as f:
        requirements = f.read()

    assert 'torch' in requirements, "torch not in requirements-m4.txt"
    # Check for version 2.1+ (supports MPS)
    assert '2.1' in requirements or 'torch>=' in requirements, "PyTorch version not specified or too old"


def test_requirements_m4_has_training_libraries():
    """Test that requirements-m4.txt has training libraries."""
    req_file = project_root / "requirements-m4.txt"

    with open(req_file, 'r') as f:
        requirements = f.read()

    required_libs = ['transformers', 'peft', 'trl', 'bitsandbytes', 'accelerate', 'datasets']

    for lib in required_libs:
        assert lib in requirements, f"{lib} not in requirements-m4.txt"


def test_version_pinning_in_requirements():
    """Test that requirements-m4.txt has version pinning."""
    req_file = project_root / "requirements-m4.txt"

    with open(req_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    # Filter actual package lines
    package_lines = [line for line in requirements if '=' in line or '>=' in line]

    # Most packages should have version constraints
    assert len(package_lines) > 10, "Too few packages with version constraints"

    # Check critical packages have exact versions
    critical_packages = ['transformers==', 'peft==', 'trl==', 'bitsandbytes==']
    requirements_str = '\n'.join(requirements)

    for pkg in critical_packages:
        assert pkg in requirements_str, f"{pkg} not pinned to exact version"


def test_documentation_files_exist():
    """Test that setup documentation files exist."""
    docs_dir = project_root / "docs"

    assert (docs_dir / "SETUP_H100.md").exists(), "SETUP_H100.md not found"
    assert (docs_dir / "SETUP_M4.md").exists(), "SETUP_M4.md not found"


def test_setup_h100_doc_has_required_sections():
    """Test that SETUP_H100.md has required sections."""
    doc_file = project_root / "docs" / "SETUP_H100.md"

    with open(doc_file, 'r') as f:
        content = f.read()

    required_sections = [
        'Prerequisites',
        'Setup Steps',
        'Training Configuration',
        'Troubleshooting',
    ]

    for section in required_sections:
        assert section in content, f"Section '{section}' not found in SETUP_H100.md"


def test_setup_m4_doc_has_required_sections():
    """Test that SETUP_M4.md has required sections."""
    doc_file = project_root / "docs" / "SETUP_M4.md"

    with open(doc_file, 'r') as f:
        content = f.read()

    required_sections = [
        'Prerequisites',
        'Setup Steps',
        'Training Configuration',
        'Memory Optimization',
        'Troubleshooting',
    ]

    for section in required_sections:
        assert section in content, f"Section '{section}' not found in SETUP_M4.md"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
