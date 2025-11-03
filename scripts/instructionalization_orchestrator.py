#!/usr/bin/env python3
"""
Instructionalization & Tagging Pipeline Orchestrator

Main orchestration script that coordinates the complete pipeline:
Loading → Tag Assignment → Chat Conversion → Stratified Split → Output → Statistics

Usage:
    python instructionalization_orchestrator.py
    python instructionalization_orchestrator.py --input data/processed/training_data_clean.jsonl
    python instructionalization_orchestrator.py --dry-run --split-ratio 0.8
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add scripts directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from paths import DATA_PROCESSED, ensure_dirs_exist
from data_loader import load_jsonl_file
from persona_tagger import determine_persona_from_item
from tone_tagger import determine_tone_from_item
from domain_tagger import determine_domains_from_item
from metadata_filter import filter_metadata, merge_tags
from chat_converter import convert_to_chat_format
from chat_format_validator import validate_chat_entry
from stratified_splitter import stratified_split
from output_writer import write_jsonl_output, validate_jsonl_output
from instructionalization_stats import calculate_instructionalization_stats
from instructionalization_reporter import write_instructionalization_reports


def validate_environment(input_path: Path, output_train: Path, output_val: Path) -> bool:
    """
    Validate environment before running pipeline.

    Args:
        input_path: Input file path
        output_train: Train output path
        output_val: Validation output path

    Returns:
        True if environment is valid, False otherwise
    """
    print("\n" + "=" * 60)
    print("Environment Validation")
    print("=" * 60)

    all_valid = True

    # Check input file
    print(f"\nChecking input file:")
    if input_path.exists():
        size_mb = input_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {input_path.name} ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ {input_path.name} (NOT FOUND)")
        all_valid = False

    # Check output directory
    output_dir = output_train.parent
    if output_dir.exists():
        print(f"\n✓ Output directory exists: {output_dir}")
    else:
        print(f"\n⚠ Output directory does not exist, will create: {output_dir}")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created output directory")
        except Exception as e:
            print(f"  ✗ Failed to create output directory: {e}")
            all_valid = False

    # Check write permissions
    try:
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        print(f"  ✓ Output directory is writable")
    except Exception as e:
        print(f"  ✗ Output directory not writable: {e}")
        all_valid = False

    return all_valid


def apply_tags(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply persona, tone, and domain tags to item.

    Args:
        item: Data item from normalization pipeline

    Returns:
        Item with tags field added
    """
    # Determine tags
    persona = determine_persona_from_item(item)
    tone = determine_tone_from_item(item)
    domains = determine_domains_from_item(item)

    # Filter metadata
    metadata = filter_metadata(item)

    # Merge into tags
    tags = merge_tags(persona, tone, domains, metadata)

    return {**item, 'tags': tags}


def convert_item_to_chat(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert tagged item to chat format.

    Args:
        item: Item with tags field

    Returns:
        Dictionary with messages and tags
    """
    persona = item['tags']['persona']

    # Convert to chat format (auto-select single/multi-turn)
    chat_entry = convert_to_chat_format(item, persona)

    # Merge tags
    chat_entry['tags'] = item['tags']

    return chat_entry


def run_pipeline(
    input_path: Path,
    output_train: Path,
    output_val: Path,
    split_ratio: float = 0.9,
    seed: int = 42,
    dry_run: bool = False
) -> bool:
    """
    Run the complete instructionalization pipeline.

    Args:
        input_path: Path to normalized training data
        output_train: Path to train output file
        output_val: Path to validation output file
        split_ratio: Train/validation split ratio (default: 0.9)
        seed: Random seed for reproducibility (default: 42)
        dry_run: If True, don't write output files

    Returns:
        True if successful, False otherwise

    Pipeline stages:
        1. Validate environment
        2. Load normalized data
        3. Apply tag assignment (persona, tone, domain)
        4. Convert to chat format
        5. Validate chat entries
        6. Stratified train/validation split
        7. Write output files (atomic)
        8. Generate statistics reports
        9. Validate output
    """
    print("\n" + "=" * 60)
    print("Instructionalization & Tagging Pipeline")
    print("=" * 60)

    # Stage 1: Validate environment
    if not validate_environment(input_path, output_train, output_val):
        print("\n✗ Environment validation failed")
        return False

    print("\n✓ Environment validation passed")

    # Stage 2: Load normalized data
    print("\n" + "=" * 60)
    print("Stage 1: Loading Normalized Data")
    print("=" * 60)
    print(f"Loading from: {input_path}")

    items = load_jsonl_file(input_path)

    if not items:
        print("\n✗ No data loaded")
        return False

    print(f"  ✓ Loaded {len(items)} items")

    # Stage 3: Apply tag assignment
    print("\n" + "=" * 60)
    print("Stage 2: Tag Assignment")
    print("=" * 60)
    print(f"Applying persona/tone/domain tags to {len(items)} items...")

    tagged_items = []
    for item in items:
        tagged_item = apply_tags(item)
        tagged_items.append(tagged_item)

    print(f"  ✓ Tagged {len(tagged_items)} items")

    # Quick tag distribution summary
    persona_counts = {}
    tone_counts = {}
    for item in tagged_items:
        persona = item['tags']['persona']
        tone = item['tags']['tone']
        persona_counts[persona] = persona_counts.get(persona, 0) + 1
        tone_counts[tone] = tone_counts.get(tone, 0) + 1

    print(f"  Persona distribution: {persona_counts}")
    print(f"  Tone distribution: {tone_counts}")

    # Stage 4: Convert to chat format
    print("\n" + "=" * 60)
    print("Stage 3: Chat Format Conversion")
    print("=" * 60)
    print(f"Converting {len(tagged_items)} items to chat format...")

    chat_items = []
    for item in tagged_items:
        chat_item = convert_item_to_chat(item)
        chat_items.append(chat_item)

    print(f"  ✓ Converted {len(chat_items)} items")

    # Count single vs multi-turn
    single_turn = sum(1 for item in chat_items if len(item['messages']) == 3)
    multi_turn = len(chat_items) - single_turn
    print(f"  Single-turn: {single_turn}, Multi-turn: {multi_turn}")

    # Stage 5: Validate chat entries
    print("\n" + "=" * 60)
    print("Stage 4: Chat Format Validation")
    print("=" * 60)
    print(f"Validating {len(chat_items)} chat entries...")

    invalid_count = 0
    for i, item in enumerate(chat_items):
        is_valid, error = validate_chat_entry(item)
        if not is_valid:
            print(f"  ⚠ Item {i}: {error}")
            invalid_count += 1

    if invalid_count > 0:
        print(f"  ⚠ {invalid_count} items failed validation")
        return False
    else:
        print(f"  ✓ All {len(chat_items)} items valid")

    # Stage 6: Stratified split
    print("\n" + "=" * 60)
    print("Stage 5: Stratified Train/Validation Split")
    print("=" * 60)
    print(f"Splitting {len(chat_items)} items (ratio: {split_ratio}, seed: {seed})...")

    train_items, val_items = stratified_split(chat_items, ratio=split_ratio, seed=seed)

    print(f"  ✓ Train: {len(train_items)} items ({len(train_items)/len(chat_items)*100:.1f}%)")
    print(f"  ✓ Validation: {len(val_items)} items ({len(val_items)/len(chat_items)*100:.1f}%)")

    # Stage 7: Write output files
    if not dry_run:
        print("\n" + "=" * 60)
        print("Stage 6: Writing Output Files")
        print("=" * 60)

        write_jsonl_output(train_items, output_train)
        write_jsonl_output(val_items, output_val)

        # Validate output
        print("\nValidating output files...")
        train_valid = validate_jsonl_output(output_train, expected_count=len(train_items))
        val_valid = validate_jsonl_output(output_val, expected_count=len(val_items))

        if not (train_valid and val_valid):
            print("  ⚠ Output validation failed")
            return False

    # Stage 8: Generate statistics reports
    print("\n" + "=" * 60)
    print("Stage 7: Generating Statistics Reports")
    print("=" * 60)

    stats = calculate_instructionalization_stats(train_items, val_items)

    if not dry_run:
        write_instructionalization_reports(stats, DATA_PROCESSED)

    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"  Input items: {len(items):,}")
    print(f"  Tagged items: {len(tagged_items):,}")
    print(f"  Chat items: {len(chat_items):,}")
    print(f"  Train items: {len(train_items):,}")
    print(f"  Validation items: {len(val_items):,}")
    print(f"  Split ratio: {stats['split']['split_ratio']:.4f}")
    print(f"  Stratification quality: {stats['stratification_quality']['average_similarity']:.4f}")

    if dry_run:
        print("\n  DRY RUN: No files written")

    return True


def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Process training data through instructionalization pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default input
  python instructionalization_orchestrator.py

  # Process specific file
  python instructionalization_orchestrator.py --input data/processed/training_data_clean.jsonl

  # Custom output paths
  python instructionalization_orchestrator.py --output-train output/train.jsonl --output-val output/val.jsonl

  # Custom split ratio
  python instructionalization_orchestrator.py --split-ratio 0.8

  # Dry run (no output)
  python instructionalization_orchestrator.py --dry-run
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        default=DATA_PROCESSED / 'training_data_clean.jsonl',
        help='Input file path (default: data/processed/training_data_clean.jsonl)'
    )

    parser.add_argument(
        '--output-train',
        type=Path,
        default=DATA_PROCESSED / 'train.jsonl',
        help='Train output file path (default: data/processed/train.jsonl)'
    )

    parser.add_argument(
        '--output-val',
        type=Path,
        default=DATA_PROCESSED / 'validation.jsonl',
        help='Validation output file path (default: data/processed/validation.jsonl)'
    )

    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.9,
        help='Train/validation split ratio (default: 0.9)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run pipeline without writing output files'
    )

    args = parser.parse_args()

    # Ensure directories exist
    ensure_dirs_exist()

    # Run pipeline
    success = run_pipeline(
        input_path=args.input,
        output_train=args.output_train,
        output_val=args.output_val,
        split_ratio=args.split_ratio,
        seed=args.seed,
        dry_run=args.dry_run
    )

    if success:
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        if not args.dry_run:
            print(f"\nTrain output: {args.output_train}")
            print(f"Validation output: {args.output_val}")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("Pipeline Failed")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
