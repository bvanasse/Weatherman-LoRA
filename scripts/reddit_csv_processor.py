#!/usr/bin/env python3
"""
Reddit CSV Processing Module for Humor Dataset

Handles loading, filtering, and processing of Reddit CSV files from
r/TheOnion and r/nottheonion subreddits.

Usage:
    from reddit_csv_processor import process_all_csvs
    from paths import REDDIT_DATA

    csv_files = list(REDDIT_DATA.glob("*.csv"))
    processed_df = process_all_csvs(csv_files, max_samples=4000)
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

from reddit_text_processing import (
    matches_weather_keywords,
    get_unique_weather_keywords,
    clean_reddit_text,
    is_valid_cleaned_text
)


def load_reddit_csv(csv_path: Path, encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
    """
    Load Reddit CSV file with encoding error handling.

    Args:
        csv_path: Path to CSV file
        encoding: Encoding to try (default: utf-8)

    Returns:
        DataFrame with CSV data, or None if loading fails

    Notes:
        - Tries UTF-8 first, falls back to latin-1 if needed
        - Handles malformed rows by skipping them
        - Preserves all columns from original CSV
    """
    try:
        df = pd.read_csv(csv_path, encoding=encoding, on_bad_lines='skip')
        return df
    except UnicodeDecodeError:
        # Try fallback encoding
        try:
            df = pd.read_csv(csv_path, encoding='latin-1', on_bad_lines='skip')
            return df
        except Exception as e:
            print(f"ERROR: Failed to load {csv_path}: {e}")
            return None
    except Exception as e:
        print(f"ERROR: Failed to load {csv_path}: {e}")
        return None


def filter_by_weather_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only rows with weather-related titles.

    Args:
        df: DataFrame with 'title' column

    Returns:
        Filtered DataFrame with additional 'matched_keywords' column

    Notes:
        - Applies expanded weather keyword matching
        - Tracks which keywords were matched for each row
        - Case-insensitive whole-word matching
    """
    # Create list to store results
    filtered_rows = []

    for idx, row in df.iterrows():
        title = row.get('title', '')
        if not title or not isinstance(title, str):
            continue

        # Check if title matches weather keywords
        if matches_weather_keywords(title):
            # Get matched keywords
            keywords = get_unique_weather_keywords(title)

            # Add matched_keywords to row
            row_dict = row.to_dict()
            row_dict['matched_keywords'] = keywords
            filtered_rows.append(row_dict)

    # Create new DataFrame from filtered rows
    if filtered_rows:
        return pd.DataFrame(filtered_rows)
    else:
        return pd.DataFrame()


def apply_quality_filters(df: pd.DataFrame, min_length: int = 10) -> pd.DataFrame:
    """
    Apply quality filters and text cleaning.

    Args:
        df: DataFrame to filter
        min_length: Minimum text length after cleaning (default: 10)

    Returns:
        Filtered DataFrame with 'cleaned_title' column added

    Quality filters:
        - Clean Reddit artifacts from titles
        - Remove entries with cleaned text shorter than min_length
        - Remove entries with missing critical metadata (id, subreddit)
        - Validate cleaned text is non-empty and contains alphanumeric content
    """
    filtered_rows = []

    for idx, row in df.iterrows():
        # Check for missing critical metadata
        if pd.isna(row.get('id')) or pd.isna(row.get('subreddit')):
            continue

        title = row.get('title', '')
        if not title or not isinstance(title, str):
            continue

        # Clean the text
        cleaned = clean_reddit_text(title)

        # Validate cleaned text
        if not is_valid_cleaned_text(cleaned, min_length=min_length):
            continue

        # Add cleaned_title to row
        row_dict = row.to_dict()
        row_dict['cleaned_title'] = cleaned
        filtered_rows.append(row_dict)

    # Create new DataFrame from filtered rows
    if filtered_rows:
        return pd.DataFrame(filtered_rows)
    else:
        return pd.DataFrame()


def balance_subreddit_samples(
    df: pd.DataFrame,
    max_samples: int = 4000,
    target_ratio: float = 0.5
) -> pd.DataFrame:
    """
    Balance samples across subreddits and prioritize quality.

    Args:
        df: DataFrame with 'subreddit' and 'num_comments' columns
        max_samples: Maximum total samples to return
        target_ratio: Target ratio for balancing (default: 0.5 for 50/50)

    Returns:
        Balanced DataFrame with up to max_samples rows

    Strategy:
        - Sort by num_comments (descending) as quality proxy
        - Try to balance between TheOnion and nottheonion subreddits
        - If one subreddit has fewer posts, take all and fill remainder from other
    """
    if len(df) == 0:
        return df

    # Sort by num_comments descending (higher engagement = higher quality)
    df_sorted = df.sort_values('num_comments', ascending=False)

    # Get subreddit counts
    subreddit_counts = df_sorted['subreddit'].value_counts()

    if len(subreddit_counts) == 1:
        # Only one subreddit, just take top max_samples
        return df_sorted.head(max_samples)

    # Calculate target counts per subreddit
    target_per_subreddit = max_samples // 2

    # Separate by subreddit
    subreddit_dfs = {}
    for subreddit in subreddit_counts.index:
        subreddit_dfs[subreddit] = df_sorted[df_sorted['subreddit'] == subreddit]

    # Sample from each subreddit
    sampled_dfs = []
    for subreddit, sub_df in subreddit_dfs.items():
        # Take up to target_per_subreddit from this subreddit
        sample_count = min(len(sub_df), target_per_subreddit)
        sampled_dfs.append(sub_df.head(sample_count))

    # Combine samples
    combined = pd.concat(sampled_dfs, ignore_index=True)

    # If we're under max_samples, we can take more from the larger subreddit
    if len(combined) < max_samples:
        remaining = max_samples - len(combined)

        # Find which subreddit has more available
        for subreddit, sub_df in subreddit_dfs.items():
            already_taken = len(sampled_dfs[list(subreddit_dfs.keys()).index(subreddit)])
            available = len(sub_df) - already_taken

            if available > 0:
                additional = min(available, remaining)
                additional_df = sub_df.iloc[already_taken:already_taken + additional]
                combined = pd.concat([combined, additional_df], ignore_index=True)
                remaining -= additional

                if remaining <= 0:
                    break

    # Final limit to max_samples
    return combined.head(max_samples)


def process_all_csvs(
    csv_paths: List[Path],
    max_samples: int = 4000,
    target_range: tuple = (3500, 4000)
) -> pd.DataFrame:
    """
    Process all Reddit CSV files through the complete pipeline.

    Args:
        csv_paths: List of paths to CSV files
        max_samples: Maximum samples to return
        target_range: Target range for final sample count

    Returns:
        Combined and filtered DataFrame with all processed data

    Pipeline stages:
        1. Load all CSV files
        2. Filter by weather keywords
        3. Apply quality filters and cleaning
        4. Balance across subreddits
        5. Sample to target range

    Statistics tracked:
        - Total rows per CSV
        - Rows after keyword filtering
        - Rows after quality filtering
        - Final row count after balancing
    """
    all_dataframes = []
    stats = {
        'csv_files': {},
        'total_rows_loaded': 0,
        'total_after_keywords': 0,
        'total_after_quality': 0,
        'final_count': 0
    }

    print(f"\nProcessing {len(csv_paths)} CSV files...")

    for csv_path in csv_paths:
        print(f"\n  Loading: {csv_path.name}")

        # Load CSV
        df = load_reddit_csv(csv_path)
        if df is None or len(df) == 0:
            print(f"    WARNING: No data loaded from {csv_path.name}")
            continue

        csv_stats = {
            'total_rows': len(df),
            'after_keywords': 0,
            'after_quality': 0
        }

        stats['total_rows_loaded'] += len(df)
        print(f"    Loaded {len(df)} rows")

        # Filter by weather keywords
        df_keywords = filter_by_weather_keywords(df)
        csv_stats['after_keywords'] = len(df_keywords)
        stats['total_after_keywords'] += len(df_keywords)
        print(f"    After keyword filtering: {len(df_keywords)} rows")

        # Apply quality filters
        df_quality = apply_quality_filters(df_keywords)
        csv_stats['after_quality'] = len(df_quality)
        stats['total_after_quality'] += len(df_quality)
        print(f"    After quality filtering: {len(df_quality)} rows")

        if len(df_quality) > 0:
            all_dataframes.append(df_quality)

        stats['csv_files'][csv_path.name] = csv_stats

    # Combine all dataframes
    if not all_dataframes:
        print("\nWARNING: No data after filtering!")
        return pd.DataFrame()

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n  Combined total: {len(combined_df)} rows")

    # Balance and sample
    final_df = balance_subreddit_samples(combined_df, max_samples=max_samples)
    stats['final_count'] = len(final_df)

    print(f"  After balancing: {len(final_df)} rows")

    # Check if within target range
    min_target, max_target = target_range
    if min_target <= len(final_df) <= max_target:
        print(f"  \u2713 Final count within target range: {min_target}-{max_target}")
    else:
        print(f"  \u26a0 WARNING: Final count {len(final_df)} outside target range {min_target}-{max_target}")

    return final_df


def calculate_pipeline_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate statistics for processed dataset.

    Args:
        df: Processed DataFrame

    Returns:
        Dictionary containing:
            - total_examples: Total number of examples
            - examples_per_subreddit: Count by subreddit
            - keyword_distribution: Top 10 most common keywords
            - avg_title_length: Average cleaned title length
            - metadata_coverage: Percentage of rows with complete metadata
    """
    if len(df) == 0:
        return {
            'total_examples': 0,
            'examples_per_subreddit': {},
            'keyword_distribution': {},
            'avg_title_length': 0,
            'metadata_coverage': 0
        }

    # Examples per subreddit
    subreddit_counts = df['subreddit'].value_counts().to_dict()

    # Keyword distribution
    all_keywords = []
    for keywords_list in df['matched_keywords']:
        if isinstance(keywords_list, list):
            all_keywords.extend(keywords_list)

    keyword_counts = Counter(all_keywords)
    top_keywords = dict(keyword_counts.most_common(10))

    # Average title length
    avg_length = df['cleaned_title'].str.len().mean()

    # Metadata coverage
    required_fields = ['id', 'subreddit', 'created_utc', 'url']
    complete_metadata = df[required_fields].notna().all(axis=1).sum()
    metadata_coverage = (complete_metadata / len(df)) * 100

    return {
        'total_examples': len(df),
        'examples_per_subreddit': subreddit_counts,
        'keyword_distribution': top_keywords,
        'avg_title_length': avg_length,
        'metadata_coverage': metadata_coverage
    }


if __name__ == "__main__":
    # Test with sample data
    from paths import REDDIT_DATA

    print("Reddit CSV Processor Test")
    print("=" * 60)

    # Find CSV files
    csv_files = list(REDDIT_DATA.glob("*.csv"))
    print(f"\nFound {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")

    # Process first CSV as test
    if csv_files:
        print(f"\nProcessing {csv_files[0].name} as test...")
        df = load_reddit_csv(csv_files[0])
        if df is not None:
            print(f"Loaded {len(df)} rows")
            print(f"Columns: {', '.join(df.columns)}")

            # Test keyword filtering
            filtered = filter_by_weather_keywords(df)
            print(f"\nAfter weather keyword filtering: {len(filtered)} rows")

            if len(filtered) > 0:
                # Show first match
                print(f"\nFirst match:")
                print(f"  Title: {filtered.iloc[0]['title']}")
                print(f"  Keywords: {filtered.iloc[0]['matched_keywords']}")
