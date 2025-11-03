#!/usr/bin/env python3
"""
Environment Validation Script for Weatherman-LoRA Project

Validates that local or remote environment is correctly set up.
Supports dual-platform validation: H100/CUDA and Mac M4/MPS.

Usage:
    python scripts/validate_environment.py --env=local
    python scripts/validate_environment.py --env=remote
    python scripts/validate_environment.py --env=h100
    python scripts/validate_environment.py --env=m4
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse


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


def check_python_version():
    """Check Python version."""
    print_header("Python Version Check")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    print(f"Python version: {version_str}")

    # Accept Python 3.10 or 3.11 (relaxed from strict 3.10)
    passed = version.major == 3 and version.minor >= 10
    print_check(passed, f"Python 3.10+ required (found {version.major}.{version.minor})")

    return passed


def check_imports_local():
    """Check required imports for local environment."""
    print_header("Local Dependencies Check")

    required = {
        'pandas': 'Data manipulation',
        'bs4': 'BeautifulSoup4 (HTML parsing)',
        'trafilatura': 'Content extraction',
        'datasets': 'HuggingFace datasets',
        'datasketch': 'MinHash deduplication',
        'jsonlines': 'JSONL file handling',
        'nltk': 'Natural language processing',
        'langdetect': 'Language detection',
        'requests': 'HTTP client',
    }

    all_passed = True

    for module, description in required.items():
        try:
            __import__(module)
            print_check(True, f"{description:40s} ({module})")
        except ImportError:
            print_check(False, f"{description:40s} ({module}) - NOT INSTALLED")
            all_passed = False

    return all_passed


def check_imports_remote():
    """Check required imports for remote environment (H100/CUDA)."""
    print_header("Remote Dependencies Check (H100/CUDA)")

    required = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'peft': 'Parameter-Efficient Fine-Tuning',
        'trl': 'Transformer Reinforcement Learning',
        'accelerate': 'Training acceleration',
        'bitsandbytes': '4-bit quantization',
        'datasets': 'HuggingFace datasets',
    }

    all_passed = True

    for module, description in required.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print_check(True, f"{description:40s} ({module} v{version})")
        except ImportError:
            print_check(False, f"{description:40s} ({module}) - NOT INSTALLED")
            all_passed = False

    return all_passed


def check_imports_h100():
    """Check H100-specific dependencies (CUDA 12.1+, Flash Attention)."""
    print_header("H100-Specific Dependencies")

    all_passed = True

    # Check PyTorch CUDA version
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"CUDA version: {cuda_version}")

            if cuda_version:
                cuda_major, cuda_minor = map(int, cuda_version.split('.')[:2])
                if cuda_major >= 12 and cuda_minor >= 1:
                    print_check(True, "CUDA 12.1+ available")
                else:
                    print_check(False, f"CUDA {cuda_version} < 12.1 required")
                    all_passed = False
            else:
                print_check(False, "CUDA version not detected")
                all_passed = False
        else:
            print_check(False, "CUDA not available")
            all_passed = False
    except Exception as e:
        print_check(False, f"Could not check CUDA: {e}")
        all_passed = False

    # Check Flash Attention 2 (optional but recommended for H100)
    try:
        import flash_attn
        version = getattr(flash_attn, '__version__', 'unknown')
        print_check(True, f"Flash Attention 2 installed (v{version})")
    except ImportError:
        print_check(False, "Flash Attention 2 not installed (recommended for H100)")
        print("  Note: Training will work without it but will be slower")
        print("  Install with: pip install flash-attn --no-build-isolation")
        # Not a critical failure, so don't set all_passed = False

    return all_passed


def check_imports_m4():
    """Check M4-specific dependencies (MPS backend, MPS-compatible PyTorch)."""
    print_header("M4-Specific Dependencies (MPS)")

    all_passed = True

    # Check PyTorch MPS backend
    try:
        import torch
        if hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                print_check(True, "MPS backend available")

                # Test MPS computation
                try:
                    test_tensor = torch.randn(100, 100, device='mps')
                    result = torch.matmul(test_tensor, test_tensor)
                    print_check(True, "MPS computation test passed")
                except Exception as e:
                    print_check(False, f"MPS computation failed: {e}")
                    all_passed = False
            else:
                print_check(False, "MPS backend not available")
                print("  Ensure you're on Apple Silicon (M1/M2/M3/M4)")
                all_passed = False
        else:
            print_check(False, "MPS backend not supported by PyTorch version")
            print("  Update PyTorch: pip install --upgrade torch")
            all_passed = False
    except Exception as e:
        print_check(False, f"Could not check MPS: {e}")
        all_passed = False

    # Check PyTorch version for MPS support (2.1+)
    try:
        import torch
        version = torch.__version__
        major, minor = map(int, version.split('.')[:2])
        if major >= 2 and minor >= 1:
            print_check(True, f"PyTorch {version} supports MPS (2.1+ required)")
        else:
            print_check(False, f"PyTorch {version} < 2.1 (MPS requires 2.1+)")
            all_passed = False
    except Exception as e:
        print_check(False, f"Could not check PyTorch version: {e}")
        all_passed = False

    return all_passed


def check_mistral_compatibility():
    """Check Mistral 7B model compatibility (Transformers 4.36+)."""
    print_header("Mistral 7B Compatibility")

    all_passed = True

    try:
        import transformers
        version = transformers.__version__
        major, minor = map(int, version.split('.')[:2])

        if major >= 4 and minor >= 36:
            print_check(True, f"Transformers {version} supports Mistral 7B (4.36+ required)")
        else:
            print_check(False, f"Transformers {version} < 4.36 (Mistral requires 4.36+)")
            all_passed = False
    except Exception as e:
        print_check(False, f"Could not check Transformers version: {e}")
        all_passed = False

    return all_passed


def check_training_dependencies():
    """Check training dependencies (PEFT 0.7+, TRL, bitsandbytes 0.41+)."""
    print_header("Training Dependencies")

    all_passed = True

    # Check PEFT version
    try:
        import peft
        version = peft.__version__
        major, minor = map(int, version.split('.')[:2])

        if major >= 0 and minor >= 7:
            print_check(True, f"PEFT {version} (0.7+ required)")
        else:
            print_check(False, f"PEFT {version} < 0.7")
            all_passed = False
    except Exception as e:
        print_check(False, f"PEFT not available: {e}")
        all_passed = False

    # Check TRL
    try:
        import trl
        version = getattr(trl, '__version__', 'unknown')
        print_check(True, f"TRL {version} installed")
    except Exception as e:
        print_check(False, f"TRL not available: {e}")
        all_passed = False

    # Check bitsandbytes version
    try:
        import bitsandbytes
        version = bitsandbytes.__version__
        major, minor = map(int, version.split('.')[:2])

        if major >= 0 and minor >= 41:
            print_check(True, f"bitsandbytes {version} (0.41+ required)")
        else:
            print_check(False, f"bitsandbytes {version} < 0.41")
            all_passed = False
    except Exception as e:
        print_check(False, f"bitsandbytes not available: {e}")
        all_passed = False

    return all_passed


def check_gpu():
    """Check GPU availability (remote only)."""
    print_header("GPU Check")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print_check(cuda_available, "CUDA available")

        if cuda_available:
            gpu_count = torch.cuda.device_count()
            print(f"   Number of GPUs: {gpu_count}")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                properties = torch.cuda.get_device_properties(i)
                memory_gb = properties.total_memory / (1024 ** 3)

                print(f"   GPU {i}: {gpu_name}")
                print(f"   Memory: {memory_gb:.2f} GB")

                if memory_gb >= 24:
                    print_check(True, f"GPU {i} has sufficient memory (24GB+ required)")
                else:
                    print_check(False, f"GPU {i} has insufficient memory ({memory_gb:.2f}GB < 24GB)")

            cuda_version = torch.version.cuda
            print(f"   CUDA version: {cuda_version}")

            return True
        else:
            print_check(False, "No GPU detected")
            return False

    except ImportError:
        print_check(False, "PyTorch not installed")
        return False


def check_storage():
    """Check available disk space."""
    print_header("Storage Check")

    try:
        import shutil
        usage = shutil.disk_usage(".")

        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)

        print(f"Total space: {total_gb:.2f} GB")
        print(f"Free space:  {free_gb:.2f} GB")

        if free_gb >= 30:
            print_check(True, f"Sufficient storage ({free_gb:.2f} GB >= 30 GB recommended)")
            return True
        elif free_gb >= 20:
            print_check(True, f"Minimum storage met ({free_gb:.2f} GB >= 20 GB minimum)")
            print("⚠️  Warning: Consider freeing more space for comfortable operation")
            return True
        else:
            print_check(False, f"Insufficient storage ({free_gb:.2f} GB < 20 GB minimum)")
            return False

    except Exception as e:
        print_check(False, f"Could not check storage: {e}")
        return False


def check_directories():
    """Check required directories exist."""
    print_header("Directory Structure Check")

    required_dirs = [
        'data/raw',
        'data/processed',
        'data/synthetic',
        'models',
        'adapters',
        'scripts',
        'configs',
    ]

    all_passed = True

    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        print_check(exists, f"Directory exists: {dir_path}")
        if not exists:
            all_passed = False

    return all_passed


def check_config_files():
    """Check configuration files exist."""
    print_header("Configuration Files Check")

    required_files = [
        'requirements-local.txt',
        'environment-remote.yml',
        'configs/training_config.yaml',
        'configs/training_config_h100.yaml',
        'configs/training_config_m4.yaml',
        'configs/paths_config.json',
    ]

    all_passed = True

    for file_path in required_files:
        exists = Path(file_path).exists()
        print_check(exists, f"File exists: {file_path}")
        if not exists:
            all_passed = False

    return all_passed


def check_scripts():
    """Check required scripts exist and are executable."""
    print_header("Scripts Check")

    required_scripts = [
        'scripts/check_storage.py',
        'scripts/check_gpu.py',
        'scripts/paths.py',
        'scripts/config_loader.py',
        'scripts/verify_model.py',
        'scripts/validate_training_config.py',
        'setup_local.sh',
        'setup_remote.sh',
    ]

    all_passed = True

    for script_path in required_scripts:
        path = Path(script_path)
        exists = path.exists()

        if exists:
            is_executable = os.access(path, os.X_OK)
            if is_executable or script_path.endswith('.py'):
                print_check(True, f"Script exists: {script_path}")
            else:
                print_check(False, f"Script exists but not executable: {script_path}")
                all_passed = False
        else:
            print_check(False, f"Script missing: {script_path}")
            all_passed = False

    return all_passed


def validate_local():
    """Validate local environment."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "LOCAL ENVIRONMENT VALIDATION" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")

    checks = [
        ("Python Version", check_python_version()),
        ("Local Dependencies", check_imports_local()),
        ("Storage", check_storage()),
        ("Directory Structure", check_directories()),
        ("Configuration Files", check_config_files()),
        ("Scripts", check_scripts()),
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
        print("🎉 Local environment is correctly configured!")
        print()
        print("Next steps:")
        print("  1. Download Project Gutenberg texts (Roadmap Item 2)")
        print("  2. Process Reddit humor data (Roadmap Item 3)")
        print("  3. Run data cleaning pipeline (Roadmap Item 4)")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print()
        print("For help, see:")
        print("  - docs/SETUP_GUIDE.md")
        print("  - ./setup_local.sh")
        return 1


def validate_remote():
    """Validate remote environment (generic H100/3090)."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "REMOTE ENVIRONMENT VALIDATION" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝")

    checks = [
        ("Python Version", check_python_version()),
        ("Remote Dependencies", check_imports_remote()),
        ("GPU Availability", check_gpu()),
        ("Storage", check_storage()),
        ("Directory Structure", check_directories()),
        ("Configuration Files", check_config_files()),
        ("Scripts", check_scripts()),
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
        print("🎉 Remote GPU environment is correctly configured!")
        print()
        print("Next steps:")
        print("  1. Sync processed data from local (docs/DATA_SYNC.md)")
        print("  2. Download base model (docs/MODEL_DOWNLOAD.md)")
        print("  3. Configure training (configs/training_config.yaml)")
        print("  4. Run LoRA training (Roadmap Items 8, 10)")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print()
        print("For help, see:")
        print("  - docs/SETUP_GUIDE.md")
        print("  - ./setup_remote.sh")
        return 1


def validate_h100():
    """Validate H100-specific environment."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "H100 ENVIRONMENT VALIDATION" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝")

    checks = [
        ("Python Version", check_python_version()),
        ("Remote Dependencies", check_imports_remote()),
        ("H100-Specific Dependencies", check_imports_h100()),
        ("Mistral Compatibility", check_mistral_compatibility()),
        ("Training Dependencies", check_training_dependencies()),
        ("GPU Availability", check_gpu()),
        ("Storage", check_storage()),
        ("Directory Structure", check_directories()),
        ("Configuration Files", check_config_files()),
        ("Scripts", check_scripts()),
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
        print("🎉 H100 environment is ready for training!")
        print()
        print("Recommended configuration: configs/training_config_h100.yaml")
        print()
        print("Next steps:")
        print("  1. Check GPU: python scripts/check_gpu.py")
        print("  2. Validate config: python scripts/validate_training_config.py --config configs/training_config_h100.yaml")
        print("  3. Start training with H100 config")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print()
        print("For help, see:")
        print("  - docs/SETUP_H100.md")
        print("  - ./setup_remote.sh")
        return 1


def validate_m4():
    """Validate M4-specific environment."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 13 + "M4 ENVIRONMENT VALIDATION" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")

    checks = [
        ("Python Version", check_python_version()),
        ("Local Dependencies", check_imports_local()),
        ("M4-Specific Dependencies", check_imports_m4()),
        ("Mistral Compatibility", check_mistral_compatibility()),
        ("Training Dependencies", check_training_dependencies()),
        ("Storage", check_storage()),
        ("Directory Structure", check_directories()),
        ("Configuration Files", check_config_files()),
        ("Scripts", check_scripts()),
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
        print("🎉 Mac M4 environment is ready for training!")
        print()
        print("Recommended configuration: configs/training_config_m4.yaml")
        print()
        print("Next steps:")
        print("  1. Check MPS: python scripts/check_gpu.py")
        print("  2. Validate config: python scripts/validate_training_config.py --config configs/training_config_m4.yaml")
        print("  3. Start training with M4 config (expect 8-12 hours)")
        print()
        print("Important: Close unnecessary applications to free unified memory")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print()
        print("For help, see:")
        print("  - docs/SETUP_M4.md")
        print("  - ./setup_m4.sh")
        return 1


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Validate Weatherman-LoRA environment setup'
    )
    parser.add_argument(
        '--env',
        choices=['local', 'remote', 'h100', 'm4'],
        required=True,
        help='Environment to validate (local, remote, h100, or m4)'
    )

    args = parser.parse_args()

    if args.env == 'local':
        exit_code = validate_local()
    elif args.env == 'remote':
        exit_code = validate_remote()
    elif args.env == 'h100':
        exit_code = validate_h100()
    elif args.env == 'm4':
        exit_code = validate_m4()
    else:
        print(f"Unknown environment: {args.env}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
