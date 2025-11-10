#!/usr/bin/env python3
"""
Local inference script for Weatherman-LoRA trained model
Mac-compatible version (no bitsandbytes/quantization)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# Configuration
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "./adapters/weatherman-lora-axolotl-h100"

def load_model():
    """Load the base model with LoRA adapter (Mac-compatible)"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    print("Loading base model (this may take a few minutes and ~30GB RAM)...")
    print("Note: This uses full precision on Mac - expect slower performance than GPU")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Model loaded successfully!")
    print(f"Using device: {next(model.parameters()).device}")
    return model, tokenizer

def create_weather_query(location, query_type="current"):
    """Create a sample weather query message"""
    if query_type == "current":
        content = f"What's the weather like in {location}?"
    elif query_type == "forecast":
        content = f"What's the weather forecast for {location} this week?"
    elif query_type == "comparison":
        content = f"Compare the weather in {location} and compare it to New York"
    else:
        content = query_type  # Use custom query

    messages = [
        {"role": "user", "content": content}
    ]
    return messages

def generate_response(model, tokenizer, messages, max_new_tokens=512):
    """Generate a response using the trained model"""
    # Format messages using the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate
    print("\nGenerating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    return response

def main():
    print("=" * 60)
    print("Weatherman-LoRA Local Inference Test (Mac Version)")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model()

    # Test queries
    test_cases = [
        ("San Francisco", "current"),
        ("Tokyo", "forecast"),
        ("What's the weather like in London and should I bring an umbrella?", "custom"),
    ]

    for i, (location, query_type) in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"Test Case {i}")
        print(f"{'=' * 60}")

        if query_type == "custom":
            messages = [{"role": "user", "content": location}]
            print(f"Query: {location}")
        else:
            messages = create_weather_query(location, query_type)
            print(f"Location: {location}")
            print(f"Query Type: {query_type}")
            print(f"Query: {messages[0]['content']}")

        response = generate_response(model, tokenizer, messages)

        print(f"\nResponse:\n{response}")

        # Try to parse tool calls if present
        if "<tool_call>" in response or "get_weather" in response:
            print("\n✓ Model appears to be using tool calling!")

    print(f"\n{'=' * 60}")
    print("Testing complete!")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
