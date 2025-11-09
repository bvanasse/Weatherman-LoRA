#!/bin/bash
#
# Quick GPU and Config Diagnostic
#

echo "=== GPU Check ==="
echo ""

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi output:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    echo ""
else
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

# Check CUDA environment
echo "CUDA Environment Variables:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
echo ""

# Check PyTorch CUDA
echo "PyTorch CUDA Check:"
python3 -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda}'); print(f'  Device count: {torch.cuda.device_count()}'); print(f'  Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1

echo ""
echo "=== Config Validation ==="
echo ""

# Try to load and validate config
python3 << 'EOF'
import yaml
import sys

try:
    with open('axolotl_config_h100.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("✓ YAML parses successfully")
    print("")

    # Check critical fields
    critical_fields = {
        'base_model': str,
        'datasets': list,
        'adapter': str,
        'output_dir': str,
    }

    for field, expected_type in critical_fields.items():
        if field in config:
            value = config[field]
            if isinstance(value, expected_type):
                print(f"✓ {field}: {value if expected_type == str else type(value).__name__}")
            else:
                print(f"✗ {field}: Wrong type! Expected {expected_type.__name__}, got {type(value).__name__}")
                print(f"   Value: {value}")
        else:
            print(f"✗ {field}: MISSING")

    print("")

    # Check for common problematic fields
    if 'chat_template' in config:
        print(f"chat_template: {config['chat_template']}")

    if 'special_tokens' in config:
        print(f"special_tokens: {config['special_tokens']}")

    if 'wandb_project' in config:
        print(f"wandb_project: {config['wandb_project']}")

except Exception as e:
    print(f"✗ Error loading config: {e}")
    sys.exit(1)
EOF

echo ""
echo "=== Axolotl Version ==="
python3 -c "import axolotl; print(f'Axolotl version: {axolotl.__version__}')" 2>&1
