#!/usr/bin/env python3
"""
Merge LoRA Adapter with Base Model

This script merges a trained LoRA adapter with its base model to create
a single merged model file. Useful for deployment scenarios where you want
a standalone model rather than base + adapter.

Usage:
    python scripts/merge_lora_adapter.py \
        --adapter adapters/weatherman-lora-axolotl-h100 \
        --output models/weatherman-merged

Requirements:
    pip install transformers peft accelerate torch
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_adapter_config(adapter_path: str) -> dict:
    """Load adapter configuration to determine base model."""
    config_path = Path(adapter_path) / "adapter_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_path}\n"
            "Ensure you're pointing to a valid LoRA adapter directory."
        )

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def merge_lora(
    adapter_path: str,
    output_path: str,
    base_model_override: str = None,
    dtype: str = "float16",
    device_map: str = "auto"
):
    """
    Merge LoRA adapter with base model.

    Args:
        adapter_path: Path to LoRA adapter directory
        output_path: Path to save merged model
        base_model_override: Optional override for base model name
        dtype: Model dtype (float16, bfloat16, float32)
        device_map: Device placement strategy
    """
    print("=" * 60)
    print("LoRA Adapter Merge Tool")
    print("=" * 60)
    print()

    # Load adapter configuration
    print(f"📂 Loading adapter config from: {adapter_path}")
    adapter_config = load_adapter_config(adapter_path)

    # Determine base model
    base_model_name = base_model_override or adapter_config.get("base_model_name_or_path")

    if not base_model_name:
        raise ValueError(
            "Could not determine base model. Either:\n"
            "1. Ensure adapter_config.json has 'base_model_name_or_path', or\n"
            "2. Provide --base-model argument"
        )

    print(f"🔧 Base model: {base_model_name}")
    print(f"🔧 LoRA adapter: {adapter_path}")
    print(f"💾 Output path: {output_path}")
    print(f"🔢 Data type: {dtype}")
    print()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    # Load base model
    print("⏳ Loading base model...")
    print("   This may take several minutes and will download ~14GB if not cached")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    print("✅ Base model loaded")
    print()

    # Load LoRA adapter
    print(f"⏳ Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch_dtype,
    )
    print("✅ LoRA adapter loaded")
    print()

    # Print adapter info
    print("📊 Adapter Information:")
    print(f"   LoRA Rank: {adapter_config.get('r', 'unknown')}")
    print(f"   LoRA Alpha: {adapter_config.get('lora_alpha', 'unknown')}")
    print(f"   Target Modules: {', '.join(adapter_config.get('target_modules', []))}")
    print()

    # Merge adapter weights into base model
    print("⏳ Merging LoRA weights into base model...")
    print("   This creates a single model with adapter weights merged")
    model = model.merge_and_unload()
    print("✅ Merge complete")
    print()

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save merged model
    print(f"💾 Saving merged model to: {output_path}")
    model.save_pretrained(
        output_path,
        safe_serialization=True,  # Save as .safetensors (recommended)
    )
    print("✅ Model saved")
    print()

    # Save tokenizer
    print("💾 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)
    print("✅ Tokenizer saved")
    print()

    # Print file info
    print("📁 Output Files:")
    for file_path in sorted(output_dir.glob("*")):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"   {file_path.name:40} {size_mb:8.1f} MB")

    total_size = sum(f.stat().st_size for f in output_dir.glob("*")) / (1024 * 1024)
    print(f"   {'Total:':40} {total_size:8.1f} MB")
    print()

    print("=" * 60)
    print("✅ Merge Complete!")
    print("=" * 60)
    print()
    print("Next Steps:")
    print(f"  1. Test merged model: python scripts/test_model.py --model {output_path}")
    print(f"  2. Convert to GGUF: See docs/M4_DEPLOYMENT.md Step 4")
    print(f"  3. Create Ollama model: See docs/M4_DEPLOYMENT.md Step 5")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge adapter (auto-detect base model from config)
  python scripts/merge_lora_adapter.py \\
    --adapter adapters/weatherman-lora-axolotl-h100 \\
    --output models/weatherman-merged

  # Merge with specific base model
  python scripts/merge_lora_adapter.py \\
    --adapter adapters/weatherman-lora-axolotl-h100 \\
    --base-model mistralai/Mistral-7B-Instruct-v0.3 \\
    --output models/weatherman-merged

  # Merge with bfloat16 precision
  python scripts/merge_lora_adapter.py \\
    --adapter adapters/weatherman-lora-axolotl-h100 \\
    --output models/weatherman-merged \\
    --dtype bfloat16
        """
    )

    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged model"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name or path (auto-detected from adapter config if not provided)"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model data type (default: float16)"
    )

    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device placement strategy (default: auto)"
    )

    args = parser.parse_args()

    # Validate adapter path exists
    if not os.path.exists(args.adapter):
        print(f"❌ Error: Adapter path not found: {args.adapter}")
        print()
        print("Expected adapter directory structure:")
        print("  adapters/weatherman-lora-axolotl-h100/")
        print("  ├── adapter_config.json")
        print("  ├── adapter_model.safetensors")
        print("  └── ...")
        return 1

    try:
        merge_lora(
            adapter_path=args.adapter,
            output_path=args.output,
            base_model_override=args.base_model,
            dtype=args.dtype,
            device_map=args.device_map
        )
        return 0

    except Exception as e:
        print(f"❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
