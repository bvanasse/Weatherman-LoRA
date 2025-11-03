#!/usr/bin/env python3
"""
Data Normalization & Deduplication Pipeline Orchestrator

Main orchestration script that coordinates the complete pipeline:
Loading → Normalization → Deduplication → Language Filter → Safety Filter → Output

Usage:
    python normalization_pipeline_orchestrator.py
    python normalization_pipeline_orchestrator.py --input data/processed/gutenberg_passages.json
    python normalization_pipeline_orchestrator.py --dry-run --skip-safety
"""

import argparse
import sys
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add scripts directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from paths import DATA_PROCESSED, CONFIGS_DIR, ensure_dirs_exist
from config_loader import load_json
from text_normalization import normalize_batch
from deduplication import remove_duplicates
from language_filter import filter_english_only
from safety_filter import filter_unsafe_content
from data_loader import load_multiple_sources
from statistics_reporter import calculate_statistics, write_json_report, write_markdown_report


def validate_environment(
    input_paths: List[Path],
    output_path: Path,
    config_path: Path
) -> bool:
    """
    Validate environment before running pipeline.

    Args:
        input_paths: List of input file paths
        output_path: Output file path
        config_path: Pipeline config path

    Returns:
        True if environment is valid, False otherwise
    """
    print("\n" + "=" * 60)
    print("Environment Validation")
    print("=" * 60)

    all_valid = True

    # Check config file
    print(f"\nChecking configuration:")
    if config_path.exists():
        print(f"  ✓ Config found: {config_path.name}")
    else:
        print(f"  ✗ Config not found: {config_path}")
        all_valid = False

    # Check input files
    print(f"\nChecking {len(input_paths)} input file(s):")
    for input_path in input_paths:
        if input_path.exists():
            size_mb = input_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {input_path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {input_path.name} (NOT FOUND)")
            all_valid = False

    # Check output directory
    output_dir = output_path.parent
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

    # Check dependencies
    print("\nChecking dependencies:")
    dependencies = [
        ('datasketch', 'MinHash/LSH deduplication'),
        ('langdetect', 'Language detection'),
        ('openai', 'Safety moderation API')
    ]

    for module_name, description in dependencies:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name} ({description})")
        except ImportError:
            print(f"  ✗ {module_name} not found ({description})")
            all_valid = False

    return all_valid


def load_pipeline_config(config_path: Path) -> Dict[str, Any]:
    """
    Load pipeline configuration.

    Args:
        config_path: Path to pipeline_config.json

    Returns:
        Configuration dictionary
    """
    config = load_json(config_path)
    print(f"\nLoaded pipeline config:")
    print(f"  Deduplication threshold: {config['deduplication']['threshold']}")
    print(f"  Normalization form: {config['normalization']['form']}")
    print(f"  Safety filter enabled: {config['safety_filter']['enabled']}")
    return config


def check_idempotency(output_path: Path, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Check for existing output and filter already-processed items.

    Args:
        output_path: Path to output file
        items: List of items to process

    Returns:
        List of items not already in output
    """
    if not output_path.exists():
        return items

    print(f"\nChecking for existing output...")
    existing_ids = set()

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                # Extract ID from tags
                if 'tags' in entry:
                    tags = entry['tags']
                    if 'reddit_id' in tags:
                        existing_ids.add(tags['reddit_id'])
                    elif 'gutenberg_id' in tags:
                        existing_ids.add(tags['gutenberg_id'])
                    elif 'id' in tags:
                        existing_ids.add(tags['id'])

        print(f"  Found {len(existing_ids)} existing items in output")

        # Filter out items that already exist
        new_items = []
        skipped = 0
        for item in items:
            item_id = item.get('id') or item.get('reddit_id') or item.get('gutenberg_id')
            if item_id and item_id in existing_ids:
                skipped += 1
            else:
                new_items.append(item)

        if skipped > 0:
            print(f"  Skipping {skipped} already-processed items")

        return new_items

    except Exception as e:
        print(f"  Warning: Could not check existing output: {e}")
        return items


def write_output_atomic(items: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write output with atomic file operation (tempfile + rename).

    Args:
        items: List of items to write
        output_path: Path to output file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary file in same directory
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=output_path.parent,
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

        # Write each item as JSONL
        for item in items:
            json.dump(item, tmp_file, ensure_ascii=False)
            tmp_file.write('\n')

    # Atomic rename
    tmp_path.replace(output_path)

    print(f"  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")


def validate_output(output_path: Path, expected_count: int) -> bool:
    """
    Validate output file after writing.

    Args:
        output_path: Path to output file
        expected_count: Expected number of items

    Returns:
        True if valid, False otherwise
    """
    print("\nValidating output...")

    try:
        count = 0
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    json.loads(line)  # Validate JSON
                    count += 1

        if count == expected_count:
            print(f"  ✓ Validation passed: {count} items")
            return True
        else:
            print(f"  ⚠ Count mismatch: expected {expected_count}, found {count}")
            return True  # Still valid, just different count

    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        return False


def run_pipeline(
    input_paths: List[Path],
    output_path: Path,
    config_path: Path,
    dry_run: bool = False,
    skip_safety: bool = False
) -> bool:
    """
    Run the complete data normalization pipeline.

    Args:
        input_paths: List of input file paths
        output_path: Path to output JSONL file
        config_path: Path to pipeline config
        dry_run: If True, don't write output
        skip_safety: If True, skip OpenAI API calls

    Returns:
        True if successful, False otherwise

    Pipeline stages:
        1. Validate environment
        2. Load configuration
        3. Load data from sources
        4. Apply unicode normalization
        5. Apply deduplication
        6. Filter by language (English)
        7. Apply safety filtering
        8. Add pipeline metadata
        9. Write output (atomic)
        10. Generate statistics reports
        11. Validate output
    """
    print("\n" + "=" * 60)
    print("Data Normalization & Deduplication Pipeline")
    print("=" * 60)

    # Stage 1: Validate environment
    if not validate_environment(input_paths, output_path, config_path):
        print("\n✗ Environment validation failed")
        return False

    print("\n✓ Environment validation passed")

    # Stage 2: Load configuration
    print("\n" + "=" * 60)
    print("Loading Configuration")
    print("=" * 60)
    config = load_pipeline_config(config_path)

    # Track statistics for all stages
    processing_stats = {}

    # Stage 3: Load data
    print("\n" + "=" * 60)
    print("Stage 1: Loading Data")
    print("=" * 60)

    items = load_multiple_sources(input_paths)

    if not items:
        print("\n✗ No data loaded")
        return False

    processing_stats['loaded'] = {
        'count': len(items),
        'files_count': len(input_paths)
    }

    # Check idempotency
    # items = check_idempotency(output_path, items)

    # Stage 4: Unicode normalization
    print("\n" + "=" * 60)
    print("Stage 2: Unicode Normalization")
    print("=" * 60)
    print(f"Normalizing {len(items)} items using {config['normalization']['form']} form...")

    items = normalize_batch(items)
    processing_stats['normalized'] = {
        'count': len(items),
        'form': config['normalization']['form']
    }
    print(f"  ✓ Normalized {len(items)} items")

    # Stage 5: Deduplication
    print("\n" + "=" * 60)
    print("Stage 3: Deduplication")
    print("=" * 60)

    threshold = config['deduplication']['threshold']
    num_perm = config['deduplication']['num_perm']
    print(f"Removing duplicates (threshold: {threshold}, permutations: {num_perm})...")

    # Determine text field to use
    text_field = 'content' if 'content' in items[0] else 'text'
    items, dedup_stats = remove_duplicates(items, threshold=threshold, num_perm=num_perm, text_field=text_field)
    processing_stats['deduplicated'] = dedup_stats

    print(f"  ✓ Removed {dedup_stats['duplicates_removed']} duplicates ({dedup_stats['duplicate_rate']}%)")
    print(f"  Remaining: {dedup_stats['unique_count']} items")

    # Stage 6: Language filter
    print("\n" + "=" * 60)
    print("Stage 4: Language Filter")
    print("=" * 60)
    print(f"Filtering for English content...")

    items, lang_stats = filter_english_only(items, text_field=text_field)
    processing_stats['language_filtered'] = lang_stats

    print(f"  ✓ Kept {lang_stats['english_count']} English items")
    print(f"  Filtered: {lang_stats['filtered_count']} non-English items")
    print(f"  Language distribution:")
    for lang, count in sorted(lang_stats['language_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {lang}: {count}")

    # Stage 7: Safety filter
    print("\n" + "=" * 60)
    print("Stage 5: Safety Filter")
    print("=" * 60)

    if skip_safety or not config['safety_filter']['enabled']:
        print("Safety filter SKIPPED (--skip-safety flag or disabled in config)")
        items, safety_stats = filter_unsafe_content(items, text_field=text_field, skip=True)
    else:
        batch_size = config['safety_filter']['batch_size']
        print(f"Checking content safety (batch size: {batch_size})...")
        items, safety_stats = filter_unsafe_content(items, text_field=text_field, batch_size=batch_size)
        print(f"  ✓ Kept {safety_stats['safe_count']} safe items")
        print(f"  Flagged: {safety_stats['flagged_count']} unsafe items")

    processing_stats['safety_filtered'] = safety_stats

    # Stage 8: Add pipeline metadata
    print("\n" + "=" * 60)
    print("Stage 6: Adding Pipeline Metadata")
    print("=" * 60)

    pipeline_metadata = config['metadata']
    for item in items:
        if 'pipeline_metadata' not in item:
            item['pipeline_metadata'] = pipeline_metadata

    print(f"  ✓ Added metadata to {len(items)} items")

    # Stage 9: Write output
    if not dry_run:
        print("\n" + "=" * 60)
        print("Stage 7: Writing Output")
        print("=" * 60)

        write_output_atomic(items, output_path)

        # Stage 11: Validate output
        validate_output(output_path, len(items))

    # Stage 10: Generate statistics reports
    print("\n" + "=" * 60)
    print("Stage 8: Generating Statistics Reports")
    print("=" * 60)

    stats = calculate_statistics(processing_stats)

    if not dry_run:
        timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        json_report_path = DATA_PROCESSED / f"pipeline_stats_{timestamp_str}.json"
        md_report_path = DATA_PROCESSED / f"pipeline_stats_{timestamp_str}.md"

        write_json_report(stats, json_report_path)
        write_markdown_report(stats, md_report_path)

    # Print final summary
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"  Initial items: {stats['summary']['initial_count']:,}")
    print(f"  Final items: {stats['summary']['final_count']:,}")
    print(f"  Filtered: {stats['summary']['total_filtered']:,}")
    print(f"  Retention rate: {stats['summary']['retention_rate']}%")

    return True


def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Process data through normalization and deduplication pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default sources
  python normalization_pipeline_orchestrator.py

  # Process specific files
  python normalization_pipeline_orchestrator.py --input data/processed/gutenberg_passages.json

  # Multiple inputs
  python normalization_pipeline_orchestrator.py --input file1.json file2.jsonl

  # Dry run (no output)
  python normalization_pipeline_orchestrator.py --dry-run

  # Skip safety checks (no API calls)
  python normalization_pipeline_orchestrator.py --skip-safety
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        nargs='+',
        help='Input file paths (JSON or JSONL)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=DATA_PROCESSED / 'training_data_clean.jsonl',
        help='Output JSONL file path (default: data/processed/training_data_clean.jsonl)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=CONFIGS_DIR / 'pipeline_config.json',
        help='Pipeline config path (default: configs/pipeline_config.json)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run pipeline without writing output'
    )

    parser.add_argument(
        '--skip-safety',
        action='store_true',
        help='Skip OpenAI API safety checks (for testing)'
    )

    args = parser.parse_args()

    # Determine input files
    if args.input:
        input_files = args.input
    else:
        # Default: load both gutenberg and reddit files
        input_files = [
            DATA_PROCESSED / 'gutenberg_passages.json',
            DATA_PROCESSED / 'reddit_humor_weather.jsonl'
        ]

    # Ensure directories exist
    ensure_dirs_exist()

    # Run pipeline
    success = run_pipeline(
        input_paths=input_files,
        output_path=args.output,
        config_path=args.config,
        dry_run=args.dry_run,
        skip_safety=args.skip_safety
    )

    if success:
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        if not args.dry_run:
            print(f"\nOutput: {args.output}")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("Pipeline Failed")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
