#!/usr/bin/env python3
"""
Instructionalization Statistics Module

Calculates comprehensive statistics for instructionalization pipeline,
including tag distributions, split quality, and message analysis.

Usage:
    from scripts.instructionalization_stats import calculate_instructionalization_stats

    stats = calculate_instructionalization_stats(train_items, val_items)
"""

from typing import Dict, Any, List
from collections import defaultdict
from datetime import datetime


def calculate_instructionalization_stats(
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for instructionalization pipeline.

    Args:
        train_items: Training set items
        val_items: Validation set items

    Returns:
        Dictionary with statistics including:
        - Counts (train/val/total)
        - Persona distribution (train/val)
        - Tone distribution (train/val)
        - Domain distribution (train/val)
        - Multi-turn vs single-turn counts
        - Average message length
        - Token estimates
        - Stratification validation

    Examples:
        >>> train = [{"messages": [...], "tags": {...}}] * 90
        >>> val = [{"messages": [...], "tags": {...}}] * 10
        >>> stats = calculate_instructionalization_stats(train, val)
        >>> stats['split']['train_count']
        90
    """
    timestamp = datetime.now().isoformat()

    stats = {
        'timestamp': timestamp,
        'pipeline_stage': 'instructionalization',
        'pipeline_version': '1.0'
    }

    # Split counts
    train_count = len(train_items)
    val_count = len(val_items)
    total_count = train_count + val_count

    stats['split'] = {
        'train_count': train_count,
        'val_count': val_count,
        'total_count': total_count,
        'split_ratio': round(train_count / total_count, 4) if total_count > 0 else 0,
        'target_ratio': 0.9
    }

    # Tag distributions
    def count_tags(items, tag_name):
        counts = defaultdict(int)
        for item in items:
            tags = item.get('tags', {})
            value = tags.get(tag_name)
            if value:
                if isinstance(value, list):
                    for v in value:
                        counts[v] += 1
                else:
                    counts[value] += 1
        return dict(counts)

    # Persona distribution
    train_persona = count_tags(train_items, 'persona')
    val_persona = count_tags(val_items, 'persona')

    stats['persona_distribution'] = {
        'train': train_persona,
        'validation': val_persona,
        'train_percentages': {k: round(v/train_count*100, 2) for k, v in train_persona.items()} if train_count > 0 else {},
        'val_percentages': {k: round(v/val_count*100, 2) for k, v in val_persona.items()} if val_count > 0 else {}
    }

    # Tone distribution
    train_tone = count_tags(train_items, 'tone')
    val_tone = count_tags(val_items, 'tone')

    stats['tone_distribution'] = {
        'train': train_tone,
        'validation': val_tone,
        'train_percentages': {k: round(v/train_count*100, 2) for k, v in train_tone.items()} if train_count > 0 else {},
        'val_percentages': {k: round(v/val_count*100, 2) for k, v in val_tone.items()} if val_count > 0 else {}
    }

    # Domain distribution
    train_domain = count_tags(train_items, 'domain')
    val_domain = count_tags(val_items, 'domain')

    stats['domain_distribution'] = {
        'train': train_domain,
        'validation': val_domain,
        'train_percentages': {k: round(v/train_count*100, 2) for k, v in train_domain.items()} if train_count > 0 else {},
        'val_percentages': {k: round(v/val_count*100, 2) for k, v in val_domain.items()} if val_count > 0 else {}
    }

    # Message format analysis
    train_single = sum(1 for item in train_items if len(item.get('messages', [])) == 3)
    train_multi = train_count - train_single

    val_single = sum(1 for item in val_items if len(item.get('messages', [])) == 3)
    val_multi = val_count - val_single

    stats['message_format'] = {
        'train_single_turn': train_single,
        'train_multi_turn': train_multi,
        'val_single_turn': val_single,
        'val_multi_turn': val_multi,
        'total_single_turn': train_single + val_single,
        'total_multi_turn': train_multi + val_multi
    }

    # Message length statistics
    def calculate_message_lengths(items):
        lengths = []
        for item in items:
            for msg in item.get('messages', []):
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    lengths.append(len(content))
        return lengths

    train_lengths = calculate_message_lengths(train_items)
    val_lengths = calculate_message_lengths(val_items)

    stats['message_length'] = {
        'train_avg_chars': round(sum(train_lengths) / len(train_lengths), 2) if train_lengths else 0,
        'train_min_chars': min(train_lengths) if train_lengths else 0,
        'train_max_chars': max(train_lengths) if train_lengths else 0,
        'val_avg_chars': round(sum(val_lengths) / len(val_lengths), 2) if val_lengths else 0,
        'val_min_chars': min(val_lengths) if val_lengths else 0,
        'val_max_chars': max(val_lengths) if val_lengths else 0
    }

    # Token estimates (rough: ~4 chars per token)
    stats['token_estimates'] = {
        'train_avg_tokens': round(stats['message_length']['train_avg_chars'] / 4, 2),
        'val_avg_tokens': round(stats['message_length']['val_avg_chars'] / 4, 2)
    }

    # Stratification validation
    def calculate_distribution_similarity(dist1, dist2):
        """Calculate similarity between two distributions (0-1)."""
        if not dist1 or not dist2:
            return 0.0

        keys = set(dist1.keys()) | set(dist2.keys())
        total1 = sum(dist1.values())
        total2 = sum(dist2.values())

        if total1 == 0 or total2 == 0:
            return 0.0

        diff_sum = 0
        for key in keys:
            pct1 = dist1.get(key, 0) / total1
            pct2 = dist2.get(key, 0) / total2
            diff_sum += abs(pct1 - pct2)

        # Similarity is 1 - average_difference
        similarity = 1 - (diff_sum / 2)
        return round(similarity, 4)

    persona_similarity = calculate_distribution_similarity(train_persona, val_persona)
    tone_similarity = calculate_distribution_similarity(train_tone, val_tone)
    domain_similarity = calculate_distribution_similarity(train_domain, val_domain)

    stats['stratification_quality'] = {
        'persona_similarity': persona_similarity,
        'tone_similarity': tone_similarity,
        'domain_similarity': domain_similarity,
        'average_similarity': round((persona_similarity + tone_similarity + domain_similarity) / 3, 4)
    }

    return stats


if __name__ == "__main__":
    # Test statistics calculation
    print("Instructionalization Statistics Test")
    print("=" * 60)

    # Create test dataset
    train_items = []
    for i in range(90):
        train_items.append({
            'messages': [
                {'role': 'system', 'content': 'System message'},
                {'role': 'user', 'content': 'User query'},
                {'role': 'assistant', 'content': 'A' * (100 + i*10)}
            ],
            'tags': {
                'persona': 'twain' if i < 50 else 'franklin',
                'tone': 'humorous' if i < 60 else 'didactic',
                'domain': ['weather', 'humor'] if i < 70 else ['weather']
            }
        })

    val_items = []
    for i in range(10):
        val_items.append({
            'messages': [
                {'role': 'system', 'content': 'System message'},
                {'role': 'user', 'content': 'User query'},
                {'role': 'assistant', 'content': 'A' * 200}
            ],
            'tags': {
                'persona': 'twain' if i < 5 else 'franklin',
                'tone': 'humorous' if i < 7 else 'didactic',
                'domain': ['weather', 'humor'] if i < 8 else ['weather']
            }
        })

    # Calculate stats
    stats = calculate_instructionalization_stats(train_items, val_items)

    # Print summary
    print(f"\nTimestamp: {stats['timestamp']}")
    print(f"\nSplit:")
    print(f"  Train: {stats['split']['train_count']}")
    print(f"  Val: {stats['split']['val_count']}")
    print(f"  Ratio: {stats['split']['split_ratio']}")

    print(f"\nPersona distribution:")
    print(f"  Train: {stats['persona_distribution']['train']}")
    print(f"  Val: {stats['persona_distribution']['validation']}")

    print(f"\nMessage format:")
    print(f"  Single-turn: {stats['message_format']['total_single_turn']}")
    print(f"  Multi-turn: {stats['message_format']['total_multi_turn']}")

    print(f"\nMessage length:")
    print(f"  Train avg: {stats['message_length']['train_avg_chars']} chars")
    print(f"  Val avg: {stats['message_length']['val_avg_chars']} chars")

    print(f"\nStratification quality:")
    print(f"  Persona similarity: {stats['stratification_quality']['persona_similarity']}")
    print(f"  Tone similarity: {stats['stratification_quality']['tone_similarity']}")
    print(f"  Average: {stats['stratification_quality']['average_similarity']}")
