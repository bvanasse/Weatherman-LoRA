#!/usr/bin/env python3
"""
Merge LoRA adapter with base model for Ollama/GGUF conversion
This creates a standalone merged model that can be converted to GGUF format
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "./adapters/weatherman-lora-axolotl-h100"
OUTPUT_PATH = "./merged_model"

def merge_lora():
    print("=" * 60)
    print("Merging LoRA Adapter with Base Model")
    print("=" * 60)

    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    print("\n2. Loading base model (this will take a few minutes)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("\n3. Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("\n4. Merging adapter weights into base model...")
    merged_model = model.merge_and_unload()

    print(f"\n5. Saving merged model to {OUTPUT_PATH}...")
    merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("✓ Merge complete!")
    print(f"✓ Merged model saved to: {OUTPUT_PATH}")
    print("\nNext steps:")
    print("1. Convert to GGUF using llama.cpp")
    print("2. Import into Ollama using the Modelfile")
    print("=" * 60)

if __name__ == "__main__":
    merge_lora()
