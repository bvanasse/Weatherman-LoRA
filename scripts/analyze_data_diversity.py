#!/usr/bin/env python3
"""
Analyze Training Data Diversity

Checks for repetitive patterns and templates in training data.

Usage:
    python scripts/analyze_data_diversity.py data/synthetic/final_train.jsonl
"""

import json
import sys
from pathlib import Path
from collections import Counter
import re


TEMPLATE_PATTERNS = {
    'twain_forecast_promise': 'weather forecasts and promises are alike',
    'twain_climate_expect': 'Climate is what we expect, weather is what we get',
    'franklin_early_bed': 'Early to bed and early to rise',
    'neutral_info_requested': "Here's the information you requested for",
}


def analyze_diversity(file_path: Path):
    """Analyze diversity of responses in training data."""

    print(f"\nAnalyzing: {file_path}")
    print("=" * 70)

    # Load data
    with open(file_path, 'r') as f:
        examples = [json.loads(line) for line in f]

    print(f"Total examples: {len(examples):,}\n")

    # Extract final responses
    responses = []
    personas = []

    for example in examples:
        messages = example['messages']
        persona = example.get('tags', {}).get('persona', 'unknown')
        personas.append(persona)

        # Find final assistant response
        for msg in reversed(messages):
            if msg['role'] == 'assistant' and 'tool_calls' not in msg:
                responses.append(msg['content'])
                break

    # Analyze templates
    print("Template Pattern Analysis:")
    print("-" * 70)

    template_counts = Counter()
    for response in responses:
        response_lower = response.lower()
        for pattern_name, pattern_text in TEMPLATE_PATTERNS.items():
            if pattern_text.lower() in response_lower:
                template_counts[pattern_name] += 1

    total_templated = sum(template_counts.values())

    for pattern, count in sorted(template_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(responses)) * 100
        print(f"  {pattern:35s}: {count:5,} ({percentage:5.1f}%)")

    print(f"\n  {'TOTAL TEMPLATED':35s}: {total_templated:5,} ({total_templated/len(responses)*100:5.1f}%)")
    print(f"  {'UNIQUE/DIVERSE':35s}: {len(responses)-total_templated:5,} ({(len(responses)-total_templated)/len(responses)*100:5.1f}%)")

    # Analyze phrase frequency
    print("\n\nPhrase Repetition Analysis:")
    print("-" * 70)

    # Extract 4-word phrases
    phrase_counter = Counter()
    for response in responses:
        words = response.lower().split()
        for i in range(len(words) - 3):
            phrase = ' '.join(words[i:i+4])
            phrase_counter[phrase] += 1

    # Find most common phrases
    most_common = [p for p, count in phrase_counter.most_common(20) if count > 5]

    if most_common:
        print("  Most repeated 4-word phrases (appearing >5 times):")
        for phrase in most_common[:10]:
            count = phrase_counter[phrase]
            print(f"    \"{phrase}\" - {count} times")
    else:
        print("  ✓ No significantly repeated phrases found!")

    # Analyze response length distribution
    print("\n\nResponse Length Distribution:")
    print("-" * 70)

    lengths = [len(r.split()) for r in responses]
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)

    print(f"  Average length: {avg_length:.1f} words")
    print(f"  Range: {min_length} - {max_length} words")

    # Length buckets
    buckets = {
        '1-10 words': sum(1 for l in lengths if l <= 10),
        '11-25 words': sum(1 for l in lengths if 10 < l <= 25),
        '26-50 words': sum(1 for l in lengths if 25 < l <= 50),
        '51+ words': sum(1 for l in lengths if l > 50),
    }

    print("\n  Length distribution:")
    for bucket, count in buckets.items():
        percentage = (count / len(lengths)) * 100
        print(f"    {bucket:20s}: {count:5,} ({percentage:5.1f}%)")

    # Persona distribution
    print("\n\nPersona Distribution:")
    print("-" * 70)

    persona_counts = Counter(personas)
    for persona, count in sorted(persona_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(personas)) * 100
        print(f"  {persona:15s}: {count:5,} ({percentage:5.1f}%)")

    # Diversity score
    print("\n\nDiversity Score:")
    print("-" * 70)

    # Calculate unique response ratio
    unique_responses = len(set(responses))
    unique_ratio = unique_responses / len(responses)

    # Calculate template-free ratio
    template_free_ratio = (len(responses) - total_templated) / len(responses)

    # Calculate phrase diversity (1 - most_common_phrase_frequency)
    if phrase_counter:
        most_common_phrase_freq = phrase_counter.most_common(1)[0][1] / len(responses)
        phrase_diversity = 1 - most_common_phrase_freq
    else:
        phrase_diversity = 1.0

    # Overall diversity score (weighted average)
    diversity_score = (
        unique_ratio * 0.3 +
        template_free_ratio * 0.5 +
        phrase_diversity * 0.2
    ) * 100

    print(f"  Unique response ratio: {unique_ratio:.1%}")
    print(f"  Template-free ratio: {template_free_ratio:.1%}")
    print(f"  Phrase diversity: {phrase_diversity:.1%}")
    print(f"\n  Overall Diversity Score: {diversity_score:.1f}/100")

    if diversity_score >= 90:
        print("  ✓ EXCELLENT - Very diverse dataset!")
    elif diversity_score >= 75:
        print("  ✓ GOOD - Reasonably diverse")
    elif diversity_score >= 60:
        print("  ⚠ FAIR - Some improvement needed")
    else:
        print("  ✗ POOR - Significant template repetition")

    print("\n" + "=" * 70 + "\n")

    return {
        'total_examples': len(examples),
        'total_templated': total_templated,
        'template_percentage': total_templated / len(responses) * 100,
        'diversity_score': diversity_score,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_data_diversity.py <data_file.jsonl>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    analyze_diversity(file_path)


if __name__ == '__main__':
    main()
