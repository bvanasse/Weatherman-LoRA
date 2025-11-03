#!/usr/bin/env python3
"""
Statistics Reporter Module for Pipeline

Generates comprehensive statistics reports in JSON and Markdown formats
with timestamps for tracking pipeline runs over time.

Usage:
    from scripts.statistics_reporter import calculate_statistics, write_json_report, write_markdown_report

    # Calculate stats
    stats = calculate_statistics(processing_stats)

    # Write reports
    write_json_report(stats, output_path)
    write_markdown_report(stats, output_path)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


def calculate_statistics(processing_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate aggregate statistics from processing stages.

    Args:
        processing_stats: Dictionary with stats from each stage:
            - loaded: Items loaded
            - normalized: Normalization stats
            - deduplicated: Deduplication stats
            - language_filtered: Language filter stats
            - safety_filtered: Safety filter stats

    Returns:
        Dictionary with comprehensive statistics

    Example:
        >>> stats_input = {
        ...     'loaded': {'count': 100},
        ...     'deduplicated': {'unique_count': 90, 'duplicates_removed': 10}
        ... }
        >>> stats = calculate_statistics(stats_input)
        >>> 'pipeline_version' in stats
        True
    """
    timestamp = datetime.now().isoformat()

    # Initialize aggregated stats
    stats = {
        'timestamp': timestamp,
        'pipeline_version': '1.0',
        'stages': {}
    }

    # Extract stage-specific stats
    if 'loaded' in processing_stats:
        stats['stages']['loading'] = processing_stats['loaded']

    if 'normalized' in processing_stats:
        stats['stages']['normalization'] = processing_stats['normalized']

    if 'deduplicated' in processing_stats:
        dedup = processing_stats['deduplicated']
        stats['stages']['deduplication'] = {
            'original_count': dedup.get('original_count', 0),
            'unique_count': dedup.get('unique_count', 0),
            'duplicates_removed': dedup.get('duplicates_removed', 0),
            'duplicate_rate': dedup.get('duplicate_rate', 0.0)
        }

    if 'language_filtered' in processing_stats:
        lang = processing_stats['language_filtered']
        stats['stages']['language_filter'] = {
            'original_count': lang.get('original_count', 0),
            'english_count': lang.get('english_count', 0),
            'filtered_count': lang.get('filtered_count', 0),
            'language_distribution': lang.get('language_distribution', {})
        }

    if 'safety_filtered' in processing_stats:
        safety = processing_stats['safety_filtered']
        stats['stages']['safety_filter'] = {
            'original_count': safety.get('original_count', 0),
            'safe_count': safety.get('safe_count', 0),
            'flagged_count': safety.get('flagged_count', 0),
            'flagged_categories': safety.get('flagged_categories', {}),
            'skipped': safety.get('skipped', False)
        }

    # Calculate overall summary
    final_count = 0
    if 'safety_filtered' in processing_stats:
        final_count = processing_stats['safety_filtered'].get('safe_count', 0)
    elif 'language_filtered' in processing_stats:
        final_count = processing_stats['language_filtered'].get('english_count', 0)
    elif 'deduplicated' in processing_stats:
        final_count = processing_stats['deduplicated'].get('unique_count', 0)
    elif 'loaded' in processing_stats:
        final_count = processing_stats['loaded'].get('count', 0)

    initial_count = processing_stats.get('loaded', {}).get('count', 0)

    stats['summary'] = {
        'initial_count': initial_count,
        'final_count': final_count,
        'total_filtered': initial_count - final_count,
        'retention_rate': round((final_count / initial_count * 100), 2) if initial_count > 0 else 0.0
    }

    # Calculate character length statistics if provided
    if 'length_stats' in processing_stats:
        stats['length_statistics'] = processing_stats['length_stats']

    return stats


def write_json_report(stats: Dict[str, Any], output_path: Path) -> None:
    """
    Write statistics report in JSON format.

    Args:
        stats: Statistics dictionary
        output_path: Path to output JSON file

    Creates:
        JSON file with formatted statistics and timestamp
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Replace {timestamp} placeholder if present
    if '{timestamp}' in str(output_path):
        timestamp_str = stats.get('timestamp', datetime.now().isoformat()).replace(':', '-')
        output_path = Path(str(output_path).replace('{timestamp}', timestamp_str))

    # Write JSON with pretty formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"  JSON report saved: {output_path}")


def write_markdown_report(stats: Dict[str, Any], output_path: Path) -> None:
    """
    Write statistics report in Markdown format.

    Args:
        stats: Statistics dictionary
        output_path: Path to output Markdown file

    Creates:
        Markdown file with formatted tables and sections
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Replace {timestamp} placeholder if present
    if '{timestamp}' in str(output_path):
        timestamp_str = stats.get('timestamp', datetime.now().isoformat()).replace(':', '-')
        output_path = Path(str(output_path).replace('{timestamp}', timestamp_str))

    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# Data Normalization Pipeline Statistics\n\n")
        f.write(f"**Generated:** {stats.get('timestamp', 'N/A')}  \n")
        f.write(f"**Pipeline Version:** {stats.get('pipeline_version', 'N/A')}\n\n")

        f.write("---\n\n")

        # Summary
        f.write("## Summary\n\n")
        summary = stats.get('summary', {})
        f.write(f"- **Initial Count:** {summary.get('initial_count', 0):,}\n")
        f.write(f"- **Final Count:** {summary.get('final_count', 0):,}\n")
        f.write(f"- **Total Filtered:** {summary.get('total_filtered', 0):,}\n")
        f.write(f"- **Retention Rate:** {summary.get('retention_rate', 0.0)}%\n\n")

        # Stages
        f.write("---\n\n")
        f.write("## Pipeline Stages\n\n")

        stages = stats.get('stages', {})

        # Loading
        if 'loading' in stages:
            f.write("### Stage 1: Data Loading\n\n")
            loading = stages['loading']
            f.write(f"- **Files Loaded:** {loading.get('files_count', 0)}\n")
            f.write(f"- **Total Items:** {loading.get('count', 0):,}\n\n")

        # Normalization
        if 'normalization' in stages:
            f.write("### Stage 2: Unicode Normalization\n\n")
            norm = stages['normalization']
            f.write(f"- **Items Processed:** {norm.get('count', 0):,}\n")
            f.write(f"- **Form:** {norm.get('form', 'NFC')}\n\n")

        # Deduplication
        if 'deduplication' in stages:
            f.write("### Stage 3: Deduplication\n\n")
            dedup = stages['deduplication']
            f.write(f"- **Original Count:** {dedup.get('original_count', 0):,}\n")
            f.write(f"- **Unique Count:** {dedup.get('unique_count', 0):,}\n")
            f.write(f"- **Duplicates Removed:** {dedup.get('duplicates_removed', 0):,}\n")
            f.write(f"- **Duplicate Rate:** {dedup.get('duplicate_rate', 0.0)}%\n\n")

        # Language Filter
        if 'language_filter' in stages:
            f.write("### Stage 4: Language Filter\n\n")
            lang = stages['language_filter']
            f.write(f"- **Original Count:** {lang.get('original_count', 0):,}\n")
            f.write(f"- **English Count:** {lang.get('english_count', 0):,}\n")
            f.write(f"- **Filtered Count:** {lang.get('filtered_count', 0):,}\n\n")

            # Language distribution
            lang_dist = lang.get('language_distribution', {})
            if lang_dist:
                f.write("**Language Distribution (before filtering):**\n\n")
                f.write("| Language | Count |\n")
                f.write("|----------|-------|\n")
                for lang_code, count in sorted(lang_dist.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"| {lang_code} | {count:,} |\n")
                f.write("\n")

        # Safety Filter
        if 'safety_filter' in stages:
            f.write("### Stage 5: Safety Filter\n\n")
            safety = stages['safety_filter']
            f.write(f"- **Original Count:** {safety.get('original_count', 0):,}\n")
            f.write(f"- **Safe Count:** {safety.get('safe_count', 0):,}\n")
            f.write(f"- **Flagged Count:** {safety.get('flagged_count', 0):,}\n")
            f.write(f"- **Skipped:** {safety.get('skipped', False)}\n\n")

            # Flagged categories
            flagged_cats = safety.get('flagged_categories', {})
            if flagged_cats:
                f.write("**Flagged Categories:**\n\n")
                f.write("| Category | Count |\n")
                f.write("|----------|-------|\n")
                for category, count in sorted(flagged_cats.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"| {category} | {count:,} |\n")
                f.write("\n")

        # Length Statistics
        if 'length_statistics' in stats:
            f.write("---\n\n")
            f.write("## Character Length Statistics\n\n")
            length_stats = stats['length_statistics']
            f.write(f"- **Minimum Length:** {length_stats.get('min', 0):,} characters\n")
            f.write(f"- **Maximum Length:** {length_stats.get('max', 0):,} characters\n")
            f.write(f"- **Mean Length:** {length_stats.get('mean', 0.0):.1f} characters\n")
            f.write(f"- **Median Length:** {length_stats.get('median', 0):.1f} characters\n\n")

    print(f"  Markdown report saved: {output_path}")


if __name__ == "__main__":
    # Test statistics reporter
    print("Statistics Reporter Test")
    print("=" * 60)

    # Sample processing stats
    processing_stats = {
        'loaded': {
            'count': 1000,
            'files_count': 2
        },
        'normalized': {
            'count': 1000,
            'form': 'NFC'
        },
        'deduplicated': {
            'original_count': 1000,
            'unique_count': 850,
            'duplicates_removed': 150,
            'duplicate_rate': 15.0
        },
        'language_filtered': {
            'original_count': 850,
            'english_count': 800,
            'filtered_count': 50,
            'language_distribution': {
                'en': 800,
                'fr': 30,
                'es': 20
            }
        },
        'safety_filtered': {
            'original_count': 800,
            'safe_count': 780,
            'flagged_count': 20,
            'flagged_categories': {
                'hate': 5,
                'violence': 10,
                'sexual': 5
            },
            'skipped': False
        },
        'length_stats': {
            'min': 10,
            'max': 5000,
            'mean': 250.5,
            'median': 200.0
        }
    }

    # Calculate statistics
    stats = calculate_statistics(processing_stats)

    print("\nCalculated Statistics:")
    print(f"  Timestamp: {stats['timestamp']}")
    print(f"  Initial count: {stats['summary']['initial_count']}")
    print(f"  Final count: {stats['summary']['final_count']}")
    print(f"  Retention rate: {stats['summary']['retention_rate']}%")

    # Test report writing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        json_path = tmpdir / "stats_test.json"
        md_path = tmpdir / "stats_test.md"

        print("\nWriting reports...")
        write_json_report(stats, json_path)
        write_markdown_report(stats, md_path)

        print(f"\nJSON report size: {json_path.stat().st_size} bytes")
        print(f"Markdown report size: {md_path.stat().st_size} bytes")
