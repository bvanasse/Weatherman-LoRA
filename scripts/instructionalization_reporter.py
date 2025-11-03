#!/usr/bin/env python3
"""
Instructionalization Reporter Module

Generates comprehensive statistics reports for instructionalization pipeline
in JSON and Markdown formats with tag distributions and split quality metrics.

Usage:
    from scripts.instructionalization_reporter import write_instructionalization_reports

    write_instructionalization_reports(stats, output_dir)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def write_instructionalization_json(stats: Dict[str, Any], output_path: Path) -> None:
    """
    Write instructionalization statistics in JSON format.

    Args:
        stats: Statistics dictionary from calculate_instructionalization_stats
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
    return output_path


def write_instructionalization_markdown(stats: Dict[str, Any], output_path: Path) -> None:
    """
    Write instructionalization statistics in Markdown format.

    Args:
        stats: Statistics dictionary from calculate_instructionalization_stats
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
        f.write("# Instructionalization Pipeline Statistics\n\n")
        f.write(f"**Generated:** {stats.get('timestamp', 'N/A')}  \n")
        f.write(f"**Pipeline Version:** {stats.get('pipeline_version', 'N/A')}  \n")
        f.write(f"**Stage:** {stats.get('pipeline_stage', 'N/A')}\n\n")

        f.write("---\n\n")

        # Split Summary
        f.write("## Train/Validation Split\n\n")
        split = stats.get('split', {})
        f.write(f"- **Total Items:** {split.get('total_count', 0):,}\n")
        f.write(f"- **Train Count:** {split.get('train_count', 0):,}\n")
        f.write(f"- **Validation Count:** {split.get('val_count', 0):,}\n")
        f.write(f"- **Split Ratio:** {split.get('split_ratio', 0):.4f} (target: {split.get('target_ratio', 0.9)})\n\n")

        # Persona Distribution
        f.write("---\n\n")
        f.write("## Persona Distribution\n\n")

        persona = stats.get('persona_distribution', {})
        train_persona = persona.get('train', {})
        val_persona = persona.get('validation', {})
        train_pct = persona.get('train_percentages', {})
        val_pct = persona.get('val_percentages', {})

        if train_persona or val_persona:
            f.write("| Persona | Train Count | Train % | Val Count | Val % |\n")
            f.write("|---------|-------------|---------|-----------|-------|\n")

            all_personas = set(train_persona.keys()) | set(val_persona.keys())
            for p in sorted(all_personas):
                train_count = train_persona.get(p, 0)
                val_count = val_persona.get(p, 0)
                train_percent = train_pct.get(p, 0)
                val_percent = val_pct.get(p, 0)
                f.write(f"| {p:10s} | {train_count:11,d} | {train_percent:6.2f}% | {val_count:9,d} | {val_percent:5.2f}% |\n")
            f.write("\n")

        # Tone Distribution
        f.write("---\n\n")
        f.write("## Tone Distribution\n\n")

        tone = stats.get('tone_distribution', {})
        train_tone = tone.get('train', {})
        val_tone = tone.get('validation', {})
        train_tone_pct = tone.get('train_percentages', {})
        val_tone_pct = tone.get('val_percentages', {})

        if train_tone or val_tone:
            f.write("| Tone | Train Count | Train % | Val Count | Val % |\n")
            f.write("|------|-------------|---------|-----------|-------|\n")

            all_tones = set(train_tone.keys()) | set(val_tone.keys())
            for t in sorted(all_tones):
                train_count = train_tone.get(t, 0)
                val_count = val_tone.get(t, 0)
                train_percent = train_tone_pct.get(t, 0)
                val_percent = val_tone_pct.get(t, 0)
                f.write(f"| {t:12s} | {train_count:11,d} | {train_percent:6.2f}% | {val_count:9,d} | {val_percent:5.2f}% |\n")
            f.write("\n")

        # Domain Distribution
        f.write("---\n\n")
        f.write("## Domain Distribution\n\n")

        domain = stats.get('domain_distribution', {})
        train_domain = domain.get('train', {})
        val_domain = domain.get('validation', {})
        train_domain_pct = domain.get('train_percentages', {})
        val_domain_pct = domain.get('val_percentages', {})

        if train_domain or val_domain:
            f.write("| Domain | Train Count | Train % | Val Count | Val % |\n")
            f.write("|--------|-------------|---------|-----------|-------|\n")

            all_domains = set(train_domain.keys()) | set(val_domain.keys())
            for d in sorted(all_domains):
                train_count = train_domain.get(d, 0)
                val_count = val_domain.get(d, 0)
                train_percent = train_domain_pct.get(d, 0)
                val_percent = val_domain_pct.get(d, 0)
                f.write(f"| {d:10s} | {train_count:11,d} | {train_percent:6.2f}% | {val_count:9,d} | {val_percent:5.2f}% |\n")
            f.write("\n")

        # Message Format
        f.write("---\n\n")
        f.write("## Message Format Analysis\n\n")

        msg_format = stats.get('message_format', {})
        f.write(f"- **Single-turn (Train):** {msg_format.get('train_single_turn', 0):,}\n")
        f.write(f"- **Multi-turn (Train):** {msg_format.get('train_multi_turn', 0):,}\n")
        f.write(f"- **Single-turn (Val):** {msg_format.get('val_single_turn', 0):,}\n")
        f.write(f"- **Multi-turn (Val):** {msg_format.get('val_multi_turn', 0):,}\n")
        f.write(f"- **Total Single-turn:** {msg_format.get('total_single_turn', 0):,}\n")
        f.write(f"- **Total Multi-turn:** {msg_format.get('total_multi_turn', 0):,}\n\n")

        # Message Length
        f.write("---\n\n")
        f.write("## Message Length Statistics\n\n")

        msg_length = stats.get('message_length', {})
        f.write("| Metric | Train | Validation |\n")
        f.write("|--------|-------|------------|\n")
        f.write(f"| Average | {msg_length.get('train_avg_chars', 0):.2f} chars | {msg_length.get('val_avg_chars', 0):.2f} chars |\n")
        f.write(f"| Minimum | {msg_length.get('train_min_chars', 0):,} chars | {msg_length.get('val_min_chars', 0):,} chars |\n")
        f.write(f"| Maximum | {msg_length.get('train_max_chars', 0):,} chars | {msg_length.get('val_max_chars', 0):,} chars |\n\n")

        # Token Estimates
        token_est = stats.get('token_estimates', {})
        f.write(f"**Estimated Token Counts** (approx 4 chars/token):\n")
        f.write(f"- Train: ~{token_est.get('train_avg_tokens', 0):.2f} tokens/message\n")
        f.write(f"- Validation: ~{token_est.get('val_avg_tokens', 0):.2f} tokens/message\n\n")

        # Stratification Quality
        f.write("---\n\n")
        f.write("## Stratification Quality\n\n")

        strat = stats.get('stratification_quality', {})
        f.write("Distribution similarity between train and validation sets (1.0 = perfect match):\n\n")
        f.write(f"- **Persona Similarity:** {strat.get('persona_similarity', 0):.4f}\n")
        f.write(f"- **Tone Similarity:** {strat.get('tone_similarity', 0):.4f}\n")
        f.write(f"- **Domain Similarity:** {strat.get('domain_similarity', 0):.4f}\n")
        f.write(f"- **Average Similarity:** {strat.get('average_similarity', 0):.4f}\n\n")

        quality_score = strat.get('average_similarity', 0)
        if quality_score >= 0.95:
            f.write("**Quality Assessment:** Excellent stratification\n")
        elif quality_score >= 0.85:
            f.write("**Quality Assessment:** Good stratification\n")
        elif quality_score >= 0.75:
            f.write("**Quality Assessment:** Acceptable stratification\n")
        else:
            f.write("**Quality Assessment:** Poor stratification - review needed\n")

    print(f"  Markdown report saved: {output_path}")
    return output_path


def write_instructionalization_reports(
    stats: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Write both JSON and Markdown reports for instructionalization.

    Args:
        stats: Statistics dictionary
        output_dir: Output directory path

    Creates:
        - instructionalization_stats_{timestamp}.json
        - instructionalization_stats_{timestamp}.md
    """
    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    json_path = output_dir / f"instructionalization_stats_{timestamp_str}.json"
    md_path = output_dir / f"instructionalization_stats_{timestamp_str}.md"

    print("\nGenerating statistics reports...")
    write_instructionalization_json(stats, json_path)
    write_instructionalization_markdown(stats, md_path)


if __name__ == "__main__":
    # Test reporter with sample stats
    print("Instructionalization Reporter Test")
    print("=" * 60)

    # Sample statistics
    sample_stats = {
        'timestamp': datetime.now().isoformat(),
        'pipeline_version': '1.0',
        'pipeline_stage': 'instructionalization',
        'split': {
            'train_count': 452,
            'val_count': 51,
            'total_count': 503,
            'split_ratio': 0.8986,
            'target_ratio': 0.9
        },
        'persona_distribution': {
            'train': {'twain': 200, 'franklin': 150, 'neutral': 102},
            'validation': {'twain': 23, 'franklin': 17, 'neutral': 11},
            'train_percentages': {'twain': 44.25, 'franklin': 33.19, 'neutral': 22.57},
            'val_percentages': {'twain': 45.10, 'franklin': 33.33, 'neutral': 21.57}
        },
        'tone_distribution': {
            'train': {'humorous': 250, 'didactic': 152, 'satirical': 50},
            'validation': {'humorous': 28, 'didactic': 17, 'satirical': 6},
            'train_percentages': {'humorous': 55.31, 'didactic': 33.63, 'satirical': 11.06},
            'val_percentages': {'humorous': 54.90, 'didactic': 33.33, 'satirical': 11.76}
        },
        'domain_distribution': {
            'train': {'weather': 452, 'humor': 300},
            'validation': {'weather': 51, 'humor': 34},
            'train_percentages': {'weather': 100.00, 'humor': 66.37},
            'val_percentages': {'weather': 100.00, 'humor': 66.67}
        },
        'message_format': {
            'train_single_turn': 400,
            'train_multi_turn': 52,
            'val_single_turn': 45,
            'val_multi_turn': 6,
            'total_single_turn': 445,
            'total_multi_turn': 58
        },
        'message_length': {
            'train_avg_chars': 542.5,
            'train_min_chars': 50,
            'train_max_chars': 2500,
            'val_avg_chars': 538.2,
            'val_min_chars': 48,
            'val_max_chars': 2400
        },
        'token_estimates': {
            'train_avg_tokens': 135.6,
            'val_avg_tokens': 134.5
        },
        'stratification_quality': {
            'persona_similarity': 0.9985,
            'tone_similarity': 0.9972,
            'domain_similarity': 0.9995,
            'average_similarity': 0.9984
        }
    }

    # Write reports to temp directory
    import tempfile
    import shutil

    test_dir = Path(tempfile.mkdtemp())

    try:
        write_instructionalization_reports(sample_stats, test_dir)

        print("\nGenerated files:")
        for file in test_dir.glob("*"):
            print(f"  - {file.name} ({file.stat().st_size} bytes)")

    finally:
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory")
