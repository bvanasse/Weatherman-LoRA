#!/usr/bin/env python3
"""
Reddit Pipeline Orchestrator for Humor Dataset Processing

Main orchestration script that coordinates the complete pipeline:
CSV loading → keyword filtering → quality filtering → JSONL conversion

Usage:
    python reddit_pipeline_orchestrator.py
    python reddit_pipeline_orchestrator.py --output data/processed/custom.jsonl --max-examples 3000
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import pandas as pd

# Add scripts directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from paths import REDDIT_DATA, DATA_PROCESSED, ensure_dirs_exist
from reddit_csv_processor import process_all_csvs, calculate_pipeline_statistics
from reddit_jsonl_converter import convert_to_jsonl, validate_jsonl_output, print_sample_entries


def validate_environment(csv_paths: List[Path], output_path: Path = None) -> bool:
    """
    Validate environment before running pipeline.

    Args:
        csv_paths: List of CSV file paths to validate
        output_path: Output file path (optional, for directory check)

    Returns:
        True if environment is valid, False otherwise

    Checks:
        - All CSV files exist
        - Output directory exists or can be created
        - Required dependencies available (pandas, jsonlines)
    """
    print("\n" + "=" * 60)
    print("Environment Validation")
    print("=" * 60)

    all_valid = True

    # Check CSV files
    print(f"\nChecking {len(csv_paths)} CSV files:")
    for csv_path in csv_paths:
        if csv_path.exists():
            size_mb = csv_path.stat().st_size / (1024 * 1024)
            print(f"  \u2713 {csv_path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  \u2717 {csv_path.name} (NOT FOUND)")
            all_valid = False

    # Check output directory
    if output_path:
        output_dir = output_path.parent
        if output_dir.exists():
            print(f"\n\u2713 Output directory exists: {output_dir}")
        else:
            print(f"\n\u26a0 Output directory does not exist, will create: {output_dir}")
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"  \u2713 Created output directory")
            except Exception as e:
                print(f"  \u2717 Failed to create output directory: {e}")
                all_valid = False

    # Check dependencies
    print("\nChecking dependencies:")
    try:
        import pandas
        print(f"  \u2713 pandas {pandas.__version__}")
    except ImportError:
        print(f"  \u2717 pandas not found")
        all_valid = False

    try:
        import jsonlines
        print(f"  \u2713 jsonlines available")
    except ImportError:
        print(f"  \u26a0 jsonlines not found (optional, using json module)")

    return all_valid


def calculate_keyword_distribution(df: pd.DataFrame, top_n: int = 10) -> Dict[str, int]:
    """
    Calculate distribution of matched weather keywords.

    Args:
        df: DataFrame with 'matched_keywords' column
        top_n: Number of top keywords to return

    Returns:
        Dictionary of top N keywords with counts
    """
    all_keywords = []
    for keywords_list in df.get('matched_keywords', []):
        if isinstance(keywords_list, list):
            all_keywords.extend(keywords_list)

    keyword_counts = Counter(all_keywords)
    return dict(keyword_counts.most_common(top_n))


def calculate_subreddit_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate subreddit-specific statistics.

    Args:
        df: DataFrame with subreddit and title data

    Returns:
        Dictionary with subreddit statistics
    """
    stats = {}

    # Examples per subreddit
    subreddit_counts = df['subreddit'].value_counts().to_dict()
    stats['examples_per_subreddit'] = subreddit_counts

    # Average title length
    if 'cleaned_title' in df.columns:
        avg_length = df['cleaned_title'].str.len().mean()
        stats['avg_title_length'] = avg_length

    return stats


def print_pipeline_statistics(
    df: pd.DataFrame,
    pipeline_stats: Dict = None
) -> None:
    """
    Print comprehensive pipeline statistics.

    Args:
        df: Final processed DataFrame
        pipeline_stats: Optional pipeline processing statistics
    """
    print("\n" + "=" * 60)
    print("Pipeline Statistics")
    print("=" * 60)

    if pipeline_stats:
        print(f"\nProcessing stages:")
        print(f"  Total rows loaded:     {pipeline_stats.get('total_rows_loaded', 0)}")
        print(f"  After keyword filter:  {pipeline_stats.get('total_after_keywords', 0)}")
        print(f"  After quality filter:  {pipeline_stats.get('total_after_quality', 0)}")
        print(f"  Final output count:    {pipeline_stats.get('final_count', 0)}")

    # Subreddit distribution
    subreddit_stats = calculate_subreddit_statistics(df)
    print(f"\nExamples per subreddit:")
    for subreddit, count in subreddit_stats.get('examples_per_subreddit', {}).items():
        print(f"  {subreddit}: {count}")

    # Keyword distribution
    keyword_dist = calculate_keyword_distribution(df, top_n=10)
    print(f"\nTop 10 matched weather keywords:")
    for keyword, count in sorted(keyword_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {keyword}: {count}")

    # Title length statistics
    if 'avg_title_length' in subreddit_stats:
        print(f"\nAverage title length: {subreddit_stats['avg_title_length']:.1f} characters")

    # Metadata coverage
    if len(df) > 0:
        required_fields = ['id', 'subreddit', 'created_utc', 'url']
        complete_metadata = df[required_fields].notna().all(axis=1).sum()
        coverage_pct = (complete_metadata / len(df)) * 100
        print(f"Metadata coverage: {coverage_pct:.1f}%")


def validate_output_count(
    count: int,
    target_range: Tuple[int, int] = (2000, 4000)
) -> Tuple[bool, str]:
    """
    Validate that output count is within target range.

    Args:
        count: Actual output count
        target_range: Tuple of (min, max) target range

    Returns:
        Tuple of (is_valid, message)
    """
    min_target, max_target = target_range

    if count < min_target:
        return False, f"Output count {count} is below target range ({min_target}-{max_target})"
    elif count > max_target:
        return False, f"Output count {count} is above target range ({min_target}-{max_target})"
    else:
        return True, f"Output count {count} is within target range ({min_target}-{max_target})"


def run_pipeline(
    csv_paths: List[Path],
    output_path: Path,
    max_samples: int = 4000,
    target_range: Tuple[int, int] = (2000, 4000),
    dry_run: bool = False
) -> pd.DataFrame:
    """
    Run the complete Reddit humor dataset processing pipeline.

    Args:
        csv_paths: List of paths to CSV files
        output_path: Path to output JSONL file
        max_samples: Maximum samples to include
        target_range: Target range for output count
        dry_run: If True, don't write output file

    Returns:
        Final processed DataFrame

    Pipeline stages:
        1. Validate environment
        2. Load and process CSVs
        3. Filter by keywords and quality
        4. Balance across subreddits
        5. Convert to JSONL
        6. Validate output
        7. Print statistics
    """
    print("\n" + "=" * 60)
    print("Reddit Humor Dataset Processing Pipeline")
    print("=" * 60)

    # Stage 1: Validate environment
    if not validate_environment(csv_paths, output_path):
        print("\n\u2717 Environment validation failed")
        return None

    print("\n\u2713 Environment validation passed")

    # Stage 2-4: Process CSVs (load, filter, balance)
    print("\n" + "=" * 60)
    print("Processing CSV Files")
    print("=" * 60)

    processed_df = process_all_csvs(
        csv_paths=csv_paths,
        max_samples=max_samples,
        target_range=target_range
    )

    if processed_df is None or len(processed_df) == 0:
        print("\n\u2717 No data after processing")
        return None

    print(f"\n\u2713 Processing complete: {len(processed_df)} examples ready")

    # Stage 5: Convert to JSONL
    if not dry_run:
        print("\n" + "=" * 60)
        print("Converting to JSONL")
        print("=" * 60)

        convert_to_jsonl(processed_df, output_path)

        # Stage 6: Validate output
        print("\nValidating output...")
        if validate_jsonl_output(output_path):
            print("\u2713 Output validation passed")

            # Show samples
            print_sample_entries(output_path, num_samples=2)
        else:
            print("\u2717 Output validation failed")

    # Stage 7: Print statistics
    print_pipeline_statistics(processed_df)

    # Validate count
    is_valid, msg = validate_output_count(len(processed_df), target_range)
    print(f"\n{'✓' if is_valid else '⚠'} {msg}")

    return processed_df


def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Process Reddit humor dataset for Weatherman-LoRA training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all CSVs with default settings
  python reddit_pipeline_orchestrator.py

  # Custom output path and max examples
  python reddit_pipeline_orchestrator.py --output data/custom.jsonl --max-examples 3000

  # Dry run (don't write output)
  python reddit_pipeline_orchestrator.py --dry-run
        """
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=DATA_PROCESSED / 'reddit_humor_weather.jsonl',
        help='Output JSONL file path (default: data/processed/reddit_humor_weather.jsonl)'
    )

    parser.add_argument(
        '--max-examples',
        type=int,
        default=4000,
        help='Maximum number of examples to include (default: 4000)'
    )

    parser.add_argument(
        '--min-target',
        type=int,
        default=2000,
        help='Minimum target examples (default: 2000)'
    )

    parser.add_argument(
        '--max-target',
        type=int,
        default=4000,
        help='Maximum target examples (default: 4000)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run pipeline without writing output file'
    )

    parser.add_argument(
        '--csv-dir',
        type=Path,
        default=REDDIT_DATA,
        help='Directory containing CSV files (default: data_sources/reddit-theonion/data)'
    )

    args = parser.parse_args()

    # Find CSV files
    csv_files = list(args.csv_dir.glob('*.csv'))

    if not csv_files:
        print(f"ERROR: No CSV files found in {args.csv_dir}")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} CSV files to process")

    # Ensure output directory exists
    ensure_dirs_exist()

    # Run pipeline
    result = run_pipeline(
        csv_paths=csv_files,
        output_path=args.output,
        max_samples=args.max_examples,
        target_range=(args.min_target, args.max_target),
        dry_run=args.dry_run
    )

    if result is not None:
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        if not args.dry_run:
            print(f"\nOutput: {args.output}")
            print(f"Examples: {len(result)}")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("Pipeline Failed")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
