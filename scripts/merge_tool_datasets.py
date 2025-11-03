#!/usr/bin/env python3
"""
Merge Tool-Use Datasets

Combines multiple tool-use JSONL files into a single final training dataset with
stratified train/validation split.

Usage:
    python scripts/merge_tool_datasets.py \\
        --inputs data/synthetic/tool_use_examples.jsonl data/synthetic/tool_use_examples_humor.jsonl \\
        --output-train data/synthetic/final_train.jsonl \\
        --output-val data/synthetic/final_validation.jsonl \\
        --split-ratio 0.9
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
from datetime import datetime


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    conversations = []
    with open(filepath, 'r') as f:
        for line in f:
            conversations.append(json.loads(line.strip()))
    return conversations


def write_jsonl(conversations: List[Dict[str, Any]], filepath: Path):
    """Write JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')
    print(f"  Wrote {len(conversations)} conversations to {filepath}")


def stratified_split(
    conversations: List[Dict[str, Any]],
    split_ratio: float = 0.9,
    random_seed: int = 42
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Perform stratified train/validation split by persona.

    Args:
        conversations: List of conversation dictionaries
        split_ratio: Ratio for train split (default 0.9 = 90/10)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_conversations, val_conversations)
    """
    random.seed(random_seed)

    # Group by persona
    by_persona = defaultdict(list)
    for conv in conversations:
        persona = conv.get('tags', {}).get('persona', 'unknown')
        by_persona[persona].append(conv)

    train_convs = []
    val_convs = []

    # Split each persona group
    for persona, convs in by_persona.items():
        random.shuffle(convs)
        split_idx = int(len(convs) * split_ratio)
        train_convs.extend(convs[:split_idx])
        val_convs.extend(convs[split_idx:])

    # Shuffle final splits
    random.shuffle(train_convs)
    random.shuffle(val_convs)

    return train_convs, val_convs


def print_statistics(conversations: List[Dict[str, Any]], label: str):
    """Print dataset statistics."""
    persona_counts = defaultdict(int)
    scenario_counts = defaultdict(int)

    for conv in conversations:
        tags = conv.get('tags', {})
        persona = tags.get('persona', 'unknown')
        scenario = tags.get('scenario', 'unknown')

        persona_counts[persona] += 1
        scenario_counts[scenario] += 1

    total = len(conversations)

    print(f"\n{label} Statistics:")
    print(f"  Total: {total}")
    print(f"\n  Persona Distribution:")
    for persona, count in sorted(persona_counts.items()):
        pct = (count / total) * 100
        print(f"    {persona}: {count} ({pct:.1f}%)")

    humor_count = persona_counts.get('twain', 0) + persona_counts.get('franklin', 0)
    humor_pct = (humor_count / total) * 100
    print(f"\n  Humor Ratio (Twain + Franklin): {humor_pct:.1f}%")

    print(f"\n  Scenario Distribution:")
    for scenario, count in sorted(scenario_counts.items()):
        pct = (count / total) * 100
        print(f"    {scenario}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple tool-use JSONL files into final training dataset'
    )
    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='Input JSONL files to merge'
    )
    parser.add_argument(
        '--output-train',
        type=str,
        default='data/synthetic/final_train.jsonl',
        help='Output path for training set'
    )
    parser.add_argument(
        '--output-val',
        type=str,
        default='data/synthetic/final_validation.jsonl',
        help='Output path for validation set'
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.9,
        help='Train/validation split ratio (default: 0.9 = 90/10)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Tool-Use Dataset Merger")
    print("=" * 60)

    # Load all input files
    all_conversations = []
    for input_file in args.inputs:
        filepath = Path(input_file)
        if not filepath.exists():
            print(f"Warning: {input_file} not found, skipping")
            continue

        conversations = load_jsonl(filepath)
        print(f"\nLoaded {len(conversations)} conversations from {filepath.name}")
        all_conversations.extend(conversations)

    print(f"\n{'=' * 60}")
    print(f"Total conversations loaded: {len(all_conversations)}")

    # Print combined statistics
    print_statistics(all_conversations, "Combined Dataset")

    # Perform stratified split
    print(f"\n{'=' * 60}")
    print(f"Performing stratified split (ratio: {args.split_ratio})")
    train_convs, val_convs = stratified_split(
        all_conversations,
        split_ratio=args.split_ratio,
        random_seed=args.random_seed
    )

    # Print split statistics
    print_statistics(train_convs, "Training Set")
    print_statistics(val_convs, "Validation Set")

    # Write output files
    print(f"\n{'=' * 60}")
    print("Writing output files...")
    write_jsonl(train_convs, Path(args.output_train))
    write_jsonl(val_convs, Path(args.output_val))

    # Write metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'input_files': args.inputs,
        'total_conversations': len(all_conversations),
        'train_count': len(train_convs),
        'val_count': len(val_convs),
        'split_ratio': args.split_ratio,
        'random_seed': args.random_seed
    }

    metadata_path = Path(args.output_train).parent / 'merge_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote metadata to {metadata_path}")

    print(f"\n{'=' * 60}")
    print("Merge complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
