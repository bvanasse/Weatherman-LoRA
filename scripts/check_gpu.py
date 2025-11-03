#!/usr/bin/env python3
"""
GPU Diagnostics Script for Weatherman-LoRA Project

Verifies GPU availability and platform compatibility for training.
Supports both CUDA (H100/3090) and MPS (Mac M4) platforms.

Requirements:
- CUDA 12.1+ for H100/3090 (remote)
- MPS backend for Mac M4 (local)
- PyTorch 2.1+ with appropriate backend support

Usage:
    python scripts/check_gpu.py
"""

import sys


def check_pytorch_import():
    """Check if PyTorch is installed."""
    try:
        import torch
        return torch
    except ImportError:
        print("❌ ERROR: PyTorch is not installed")
        print()
        print("Please install PyTorch:")
        print("  For CUDA (H100/3090):")
        print("    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        print("  For Mac M4 (MPS):")
        print("    pip install torch torchvision torchaudio")
        print()
        print("Or run the setup script:")
        print("  ./setup_remote.sh  (for H100/3090)")
        print("  ./setup_m4.sh      (for Mac M4)")
        sys.exit(1)


def bytes_to_gb(bytes_value):
    """Convert bytes to gigabytes."""
    return bytes_value / (1024 ** 3)


def check_mps_backend(torch):
    """
    Check if MPS (Metal Performance Shaders) backend is available.

    Args:
        torch: PyTorch module

    Returns:
        bool: True if MPS is available and working
    """
    if not hasattr(torch.backends, 'mps'):
        return False

    if not torch.backends.mps.is_available():
        return False

    # Test MPS with a small computation
    try:
        test_tensor = torch.randn(100, 100, device='mps')
        result = torch.matmul(test_tensor, test_tensor)
        return True
    except Exception:
        return False


def check_cuda_platform(torch):
    """
    Check CUDA platform (H100 or other GPU).

    Args:
        torch: PyTorch module

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print("=" * 60)
    print("CUDA Platform Detection")
    print("=" * 60)
    print()

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:      {cuda_available}")

    if not cuda_available:
        print()
        print("❌ ERROR: CUDA is not available")
        print()
        print("Possible issues:")
        print("  1. CUDA drivers not installed")
        print("  2. PyTorch installed without CUDA support")
        print("  3. GPU not detected by system")
        print()
        print("To fix:")
        print("  1. Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
        print("  2. Install CUDA toolkit 12.1+")
        print("  3. Reinstall PyTorch with CUDA: ./setup_remote.sh")
        return 1

    # Get CUDA version
    cuda_version = torch.version.cuda
    print(f"CUDA Version:        {cuda_version}")
    print()

    # Check CUDA version compatibility
    if cuda_version:
        cuda_major, cuda_minor = map(int, cuda_version.split('.')[:2])
        if cuda_major < 12 or (cuda_major == 12 and cuda_minor < 1):
            print(f"⚠️  WARNING: CUDA {cuda_version} is older than required 12.1")
            print("    Training may encounter compatibility issues")
            print()

    # Get GPU count
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs:      {gpu_count}")
    print()

    if gpu_count == 0:
        print("❌ ERROR: No GPUs detected")
        print()
        print("Please ensure:")
        print("  1. GPU is properly installed")
        print("  2. NVIDIA drivers are loaded (run 'nvidia-smi')")
        return 1

    # Display information for each GPU
    print("GPU Details:")
    print("-" * 60)

    h100_detected = False
    total_memory = 0

    for i in range(gpu_count):
        print(f"\nGPU {i}:")

        # Get GPU name
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  Name:              {gpu_name}")

        # Check if H100
        if 'H100' in gpu_name:
            h100_detected = True

        # Get GPU memory
        gpu_properties = torch.cuda.get_device_properties(i)
        total_memory_gb = bytes_to_gb(gpu_properties.total_memory)
        total_memory += total_memory_gb
        print(f"  Total Memory:      {total_memory_gb:.2f} GB")

        # Get current memory usage
        if torch.cuda.is_available():
            torch.cuda.set_device(i)
            allocated = bytes_to_gb(torch.cuda.memory_allocated(i))
            reserved = bytes_to_gb(torch.cuda.memory_reserved(i))
            print(f"  Allocated Memory:  {allocated:.2f} GB")
            print(f"  Reserved Memory:   {reserved:.2f} GB")
            print(f"  Free Memory:       {total_memory_gb - reserved:.2f} GB")

        # Check compute capability
        compute_capability = f"{gpu_properties.major}.{gpu_properties.minor}"
        print(f"  Compute Capability: {compute_capability}")

        # Memory requirements check
        print()
        print(f"  Memory Assessment:")
        if total_memory_gb < 20:
            print(f"    ❌ INSUFFICIENT: {total_memory_gb:.2f} GB < 24 GB required")
            print(f"       Training will likely fail due to OOM errors")
        elif total_memory_gb < 24:
            print(f"    ⚠️  LOW: {total_memory_gb:.2f} GB (24 GB minimum for 3090)")
            print(f"       May need to reduce batch size or sequence length")
        elif total_memory_gb < 70:
            print(f"    ✅ ADEQUATE: {total_memory_gb:.2f} GB (24GB+ for QLoRA training)")
            print(f"       Suitable for training with standard batch sizes")
        else:
            print(f"    ✅ EXCELLENT: {total_memory_gb:.2f} GB (H100-class)")
            print(f"       Ideal for fast training with large batch sizes")

    print()
    print("-" * 60)

    # Test GPU computation
    print()
    print("Testing GPU computation...")
    try:
        # Create a small tensor on GPU
        test_tensor = torch.randn(1000, 1000, device='cuda')
        result = torch.matmul(test_tensor, test_tensor)
        print("✅ GPU computation test PASSED")
    except Exception as e:
        print(f"❌ GPU computation test FAILED: {e}")
        return 1

    print()

    # Recommend configuration
    print("=" * 60)
    print("Recommended Training Configuration")
    print("=" * 60)
    print()

    if h100_detected or total_memory >= 70:
        print("✅ H100-CLASS GPU DETECTED")
        print()
        print(f"   Recommended config: configs/training_config_h100.yaml")
        print(f"   Expected training time: 3-4 hours for 3 epochs")
        print(f"   Sequence length: 4096 tokens")
        print(f"   Batch size: 4-8")
        print(f"   Flash Attention 2: Enabled")
        print()
        print("   To use this config:")
        print("     python scripts/train.py --config configs/training_config_h100.yaml")
    else:
        print("✅ GPU DETECTED (RTX 3090 or similar)")
        print()
        print(f"   Available VRAM: {total_memory:.2f} GB")
        print(f"   Recommended config: configs/training_config_h100.yaml")
        print(f"   Expected training time: 8-12 hours for 3 epochs")
        print()
        print("   Note: You may need to adjust batch size if OOM occurs")
        print("   See config file for troubleshooting guidance")

    return 0


def check_mps_platform(torch):
    """
    Check MPS platform (Mac M4).

    Args:
        torch: PyTorch module

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print("=" * 60)
    print("MPS (Metal) Platform Detection")
    print("=" * 60)
    print()

    # Check MPS availability
    mps_available = check_mps_backend(torch)
    print(f"MPS Available:       {mps_available}")

    if not mps_available:
        print()
        print("❌ ERROR: MPS (Metal Performance Shaders) is not available")
        print()
        print("Possible issues:")
        print("  1. Not running on Apple Silicon (M1/M2/M3/M4)")
        print("  2. PyTorch version too old (need 2.0+)")
        print("  3. macOS version too old (need 12.3+)")
        print()
        print("To fix:")
        print("  1. Verify you're on Apple Silicon: system_profiler SPHardwareDataType")
        print("  2. Update PyTorch: pip install --upgrade torch")
        print("  3. Run setup script: ./setup_m4.sh")
        return 1

    print()

    # Get system memory (unified memory on M4)
    try:
        import subprocess
        result = subprocess.run(
            ['sysctl', 'hw.memsize'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            memory_bytes = int(result.stdout.split(':')[1].strip())
            memory_gb = bytes_to_gb(memory_bytes)
            print(f"Unified Memory:      {memory_gb:.2f} GB")
            print()

            # Memory assessment for M4
            print("Memory Assessment:")
            if memory_gb < 16:
                print(f"  ❌ INSUFFICIENT: {memory_gb:.2f} GB < 32 GB recommended")
                print(f"     Training will likely fail due to OOM errors")
                print(f"     Consider cloud GPU (H100) instead")
            elif memory_gb < 32:
                print(f"  ⚠️  LIMITED: {memory_gb:.2f} GB (32 GB recommended)")
                print(f"     Training possible but may need aggressive memory optimization")
                print(f"     Use batch_size=1, seq_length=1024, gradient_accumulation=16")
            else:
                print(f"  ✅ SUFFICIENT: {memory_gb:.2f} GB")
                print(f"     Suitable for M4 training with reduced batch size")
                print(f"     Use batch_size=1-2, seq_length=2048")
            print()
    except Exception as e:
        print(f"⚠️  Could not detect system memory: {e}")
        print()

    # Test MPS computation
    print("Testing MPS computation...")
    try:
        test_tensor = torch.randn(1000, 1000, device='mps')
        result = torch.matmul(test_tensor, test_tensor)
        print("✅ MPS computation test PASSED")
    except Exception as e:
        print(f"❌ MPS computation test FAILED: {e}")
        return 1

    print()

    # Recommend configuration
    print("=" * 60)
    print("Recommended Training Configuration")
    print("=" * 60)
    print()
    print("✅ MAC M4 (MPS) PLATFORM DETECTED")
    print()
    print(f"   Recommended config: configs/training_config_m4.yaml")
    print(f"   Expected training time: 8-12 hours for 3 epochs")
    print(f"   Sequence length: 2048 tokens (reduced for memory)")
    print(f"   Batch size: 1-2")
    print(f"   Flash Attention 2: Not available (CUDA-only)")
    print()
    print("   To use this config:")
    print("     python scripts/train.py --config configs/training_config_m4.yaml")
    print()
    print("   Important notes:")
    print("     - Training will be 2-3x slower than H100")
    print("     - Close unnecessary applications to free memory")
    print("     - Monitor Activity Monitor for memory pressure")
    print("     - Consider training overnight due to longer duration")

    return 0


def main():
    """Run GPU/platform diagnostics and display results."""
    print("=" * 60)
    print("Weatherman-LoRA Platform Diagnostics")
    print("=" * 60)
    print()

    # Import PyTorch
    torch = check_pytorch_import()

    # Check PyTorch version
    print(f"PyTorch Version:     {torch.__version__}")
    print()

    # Detect platform and run appropriate checks
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    if cuda_available:
        # CUDA platform (H100, 3090, etc.)
        exit_code = check_cuda_platform(torch)
    elif mps_available:
        # MPS platform (Mac M4)
        exit_code = check_mps_platform(torch)
    else:
        # No accelerator detected
        print("❌ ERROR: No GPU accelerator detected")
        print()
        print("Neither CUDA nor MPS backend is available.")
        print()
        print("For CUDA (H100/3090):")
        print("  - Install NVIDIA drivers and CUDA toolkit")
        print("  - Reinstall PyTorch with CUDA support: ./setup_remote.sh")
        print()
        print("For Mac M4 (MPS):")
        print("  - Verify you're on Apple Silicon (M1/M2/M3/M4)")
        print("  - Update PyTorch: pip install --upgrade torch")
        print("  - Run setup script: ./setup_m4.sh")
        exit_code = 1

    print()
    print("=" * 60)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
