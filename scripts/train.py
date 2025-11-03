#!/usr/bin/env python3
"""
Weatherman-LoRA Training Script
QLoRA fine-tuning for Mistral 7B with HuggingFace TRL
"""

import argparse
import os
import sys
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_quantization_config(config: Dict[str, Any]) -> BitsAndBytesConfig:
    """Setup 4-bit quantization configuration for QLoRA."""
    quant_config = config['quantization']

    # Map string dtype to torch dtype
    compute_dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }
    compute_dtype = compute_dtype_map.get(
        quant_config['bnb_4bit_compute_dtype'],
        torch.bfloat16
    )

    return BitsAndBytesConfig(
        load_in_4bit=quant_config['load_in_4bit'],
        bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
        bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=compute_dtype,
    )


def setup_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """Setup LoRA configuration."""
    lora_config = config['lora']

    return LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config['bias'],
        task_type=lora_config['task_type'],
    )


def load_model_and_tokenizer(config: Dict[str, Any], bnb_config: BitsAndBytesConfig):
    """Load base model and tokenizer with quantization."""
    model_config = config['model']
    model_name = model_config['model_name_or_path']

    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side=model_config.get('padding_side', 'right'),
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def load_datasets(config: Dict[str, Any]):
    """Load training and validation datasets."""
    dataset_config = config['dataset']

    train_file = dataset_config['train_file']
    val_file = dataset_config.get('val_file')

    print(f"Loading training data from: {train_file}")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")

    # Load training data
    train_dataset = load_dataset('json', data_files=train_file, split='train')

    # Load validation data if provided
    eval_dataset = None
    if val_file and os.path.exists(val_file):
        print(f"Loading validation data from: {val_file}")
        eval_dataset = load_dataset('json', data_files=val_file, split='train')

    return train_dataset, eval_dataset


def format_chat_template(example, tokenizer):
    """Format conversation using chat template."""
    # Apply chat template to messages
    text = tokenizer.apply_chat_template(
        example['messages'],
        tokenize=False,
        add_generation_prompt=False
    )
    return {'text': text}


def setup_training_arguments(config: Dict[str, Any]) -> TrainingArguments:
    """Setup training arguments from config."""
    training_config = config['training']

    return TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size',
                                                        training_config['per_device_train_batch_size']),
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        warmup_ratio=training_config['warmup_ratio'],
        weight_decay=training_config['weight_decay'],
        max_grad_norm=training_config['max_grad_norm'],
        optim=training_config['optim'],
        bf16=training_config.get('bf16', False),
        fp16=training_config.get('fp16', False),
        logging_steps=training_config['logging_steps'],
        logging_first_step=training_config.get('logging_first_step', True),
        evaluation_strategy=training_config.get('evaluation_strategy', 'no'),
        eval_steps=training_config.get('eval_steps', 500),
        save_steps=training_config['save_steps'],
        save_strategy=training_config['save_strategy'],
        save_total_limit=training_config.get('save_total_limit', 3),
        load_best_model_at_end=training_config.get('load_best_model_at_end', False),
        metric_for_best_model=training_config.get('metric_for_best_model', 'loss'),
        greater_is_better=training_config.get('greater_is_better', False),
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
        dataloader_pin_memory=training_config.get('dataloader_pin_memory', True),
        disable_tqdm=training_config.get('disable_tqdm', False),
        report_to=training_config.get('report_to', 'wandb'),
        run_name=training_config.get('run_name', 'weatherman-lora'),
    )


def main():
    parser = argparse.ArgumentParser(description='Train Weatherman LoRA adapter')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training configuration YAML file'
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Setup configurations
    bnb_config = setup_quantization_config(config)
    lora_config = setup_lora_config(config)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, bnb_config)

    # Apply LoRA
    print("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    train_dataset, eval_dataset = load_datasets(config)

    # Format datasets with chat template
    print("Formatting datasets with chat template...")
    train_dataset = train_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: format_chat_template(x, tokenizer),
            remove_columns=eval_dataset.column_names
        )

    # Setup training arguments
    training_args = setup_training_arguments(config)

    # Create trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=config['model']['max_seq_length'],
        dataset_text_field='text',
        tokenizer=tokenizer,
        packing=False,  # Don't pack sequences for chat format
    )

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    trainer.train()

    # Save final model
    print("\n" + "="*60)
    print("Training complete! Saving final model...")
    print("="*60 + "\n")

    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    print(f"Model saved to: {training_args.output_dir}")
    print("\nTraining finished successfully!")


if __name__ == "__main__":
    main()
