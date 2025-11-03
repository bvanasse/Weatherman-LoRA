#!/usr/bin/env python3
"""
Stratified Splitter Module for Train/Validation Split

Implements stratified sampling to maintain balanced representation
of persona/tone/domain combinations in train and validation splits.

Usage:
    from scripts.stratified_splitter import stratified_split

    train, val = stratified_split(items, ratio=0.9, seed=42)
"""

import random
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def create_stratification_key(item: Dict[str, Any]) -> str:
    """
    Create stratification key from item tags.

    Args:
        item: Item with tags field

    Returns:
        String key combining persona, tone, and primary domain

    Examples:
        >>> item = {"tags": {"persona": "twain", "tone": "humorous", "domain": ["weather", "humor"]}}
        >>> create_stratification_key(item)
        'twain-humorous-weather'
    """
    tags = item.get('tags', {})

    persona = tags.get('persona', 'unknown')
    tone = tags.get('tone', 'unknown')
    domain_list = tags.get('domain', ['unknown'])

    # Use primary domain (first in list)
    primary_domain = domain_list[0] if domain_list else 'unknown'

    return f"{persona}-{tone}-{primary_domain}"


def stratified_split(
    items: List[Dict[str, Any]],
    ratio: float = 0.9,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Perform stratified train/validation split.

    Args:
        items: List of items to split
        ratio: Train/validation split ratio (default: 0.9 = 90/10)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_items, validation_items)

    Process:
        1. Group items by stratification key (persona-tone-domain)
        2. Shuffle each group with seed
        3. Split each group according to ratio
        4. Combine train and validation groups
        5. Shuffle final lists

    Examples:
        >>> items = [{"tags": {"persona": "twain", "tone": "humorous", "domain": ["weather"]}}] * 100
        >>> train, val = stratified_split(items, ratio=0.9, seed=42)
        >>> len(train)
        90
        >>> len(val)
        10
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Group items by stratification key
    strata = defaultdict(list)
    for item in items:
        key = create_stratification_key(item)
        strata[key].append(item)

    # Split each stratum
    train_items = []
    val_items = []

    for key, stratum_items in strata.items():
        # Shuffle items in this stratum
        shuffled = stratum_items.copy()
        random.shuffle(shuffled)

        # Calculate split point
        split_point = int(len(shuffled) * ratio)

        # Ensure at least one item in validation if stratum has multiple items
        if split_point == len(shuffled) and len(shuffled) > 1:
            split_point = len(shuffled) - 1

        # Split stratum
        train_items.extend(shuffled[:split_point])
        val_items.extend(shuffled[split_point:])

    # Final shuffle of combined lists
    random.shuffle(train_items)
    random.shuffle(val_items)

    return train_items, val_items


def get_split_statistics(
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate statistics about the train/validation split.

    Args:
        train_items: Training set items
        val_items: Validation set items

    Returns:
        Dictionary with split statistics

    Statistics include:
        - Total counts
        - Persona distributions
        - Tone distributions
        - Domain distributions
        - Stratification quality metrics

    Examples:
        >>> train = [{"tags": {"persona": "twain", "tone": "humorous", "domain": ["weather"]}}] * 90
        >>> val = [{"tags": {"persona": "twain", "tone": "humorous", "domain": ["weather"]}}] * 10
        >>> stats = get_split_statistics(train, val)
        >>> stats['train_count']
        90
    """
    stats = {
        'train_count': len(train_items),
        'val_count': len(val_items),
        'total_count': len(train_items) + len(val_items),
        'split_ratio': len(train_items) / (len(train_items) + len(val_items)) if (len(train_items) + len(val_items)) > 0 else 0
    }

    # Calculate persona distributions
    def count_tag_values(items, tag_name):
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

    stats['train_persona_dist'] = count_tag_values(train_items, 'persona')
    stats['val_persona_dist'] = count_tag_values(val_items, 'persona')

    stats['train_tone_dist'] = count_tag_values(train_items, 'tone')
    stats['val_tone_dist'] = count_tag_values(val_items, 'tone')

    stats['train_domain_dist'] = count_tag_values(train_items, 'domain')
    stats['val_domain_dist'] = count_tag_values(val_items, 'domain')

    return stats


if __name__ == "__main__":
    # Test stratified splitter
    print("Stratified Splitter Test")
    print("=" * 60)

    # Create test dataset with multiple strata
    test_items = []

    # Stratum 1: Twain + humorous + weather
    for i in range(50):
        test_items.append({
            'text': f'Twain passage {i}',
            'tags': {
                'persona': 'twain',
                'tone': 'humorous',
                'domain': ['weather', 'humor']
            }
        })

    # Stratum 2: Franklin + didactic + weather
    for i in range(30):
        test_items.append({
            'text': f'Franklin passage {i}',
            'tags': {
                'persona': 'franklin',
                'tone': 'didactic',
                'domain': ['weather']
            }
        })

    # Stratum 3: Neutral + satirical + weather/humor
    for i in range(20):
        test_items.append({
            'text': f'Reddit post {i}',
            'tags': {
                'persona': 'neutral',
                'tone': 'satirical',
                'domain': ['weather', 'humor']
            }
        })

    print(f"\nTotal items: {len(test_items)}")

    # Perform split
    train, val = stratified_split(test_items, ratio=0.9, seed=42)

    print(f"\nSplit results:")
    print(f"  Train: {len(train)} items ({len(train)/len(test_items)*100:.1f}%)")
    print(f"  Val:   {len(val)} items ({len(val)/len(test_items)*100:.1f}%)")

    # Get statistics
    stats = get_split_statistics(train, val)

    print(f"\nPersona distribution:")
    print(f"  Train: {stats['train_persona_dist']}")
    print(f"  Val:   {stats['val_persona_dist']}")

    print(f"\nTone distribution:")
    print(f"  Train: {stats['train_tone_dist']}")
    print(f"  Val:   {stats['val_tone_dist']}")

    print(f"\nDomain distribution:")
    print(f"  Train: {stats['train_domain_dist']}")
    print(f"  Val:   {stats['val_domain_dist']}")
