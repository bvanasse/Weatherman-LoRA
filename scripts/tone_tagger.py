#!/usr/bin/env python3
"""
Tone Tagger Module for Instructionalization Pipeline

Determines tone tags based on source metadata and existing tags.
Maps Reddit sources and literary content to appropriate tones.

Usage:
    from scripts.tone_tagger import determine_tone

    tone = determine_tone("reddit-theonion", {})
    # Returns: "satirical"
"""

from typing import Dict, Any, List


def determine_tone(source: str, tags: Dict[str, Any] = None) -> str:
    """
    Determine tone tag from source and existing tags.

    Args:
        source: Source identifier string
        tags: Existing tags dictionary (may contain tone from normalization)

    Returns:
        Tone tag string: "satirical", "ironic", "humorous", or "didactic"

    Mapping rules:
        - r/TheOnion → "satirical"
        - r/nottheonion → "ironic"
        - Literary passages with humor keywords → "humorous"
        - Franklin passages → "didactic"
        - Preserve existing tone tags from normalization pipeline
        - Ensure exactly one tone tag per entry

    Examples:
        >>> determine_tone("reddit-theonion", {})
        'satirical'
        >>> determine_tone("reddit-nottheonion", {})
        'ironic'
        >>> determine_tone("gutenberg-franklin", {})
        'didactic'
        >>> determine_tone("any-source", {"tone": "existing-tone"})
        'existing-tone'
    """
    if tags is None:
        tags = {}

    # Check for existing tone tag (preserve from normalization)
    if 'tone' in tags and tags['tone']:
        return tags['tone']

    if not source:
        return 'humorous'

    source_lower = source.lower()

    # Reddit source mappings
    if 'theonion' in source_lower and 'not' not in source_lower:
        return 'satirical'

    if 'nottheonion' in source_lower:
        return 'ironic'

    # Franklin passages are didactic
    if 'franklin' in source_lower:
        return 'didactic'

    # Default to humorous for literary passages
    return 'humorous'


def determine_tone_from_item(item: Dict[str, Any]) -> str:
    """
    Determine tone from a full data item.

    Args:
        item: Dictionary containing item metadata

    Returns:
        Tone tag string

    Checks:
        - Existing tone in tags or at item level
        - Subreddit for Reddit sources
        - Author and genre tags for literary content
        - Matched keywords for humor indicators

    Examples:
        >>> item = {"subreddit": "TheOnion", "matched_keywords": ["storm"]}
        >>> determine_tone_from_item(item)
        'satirical'
    """
    # Check existing tags
    existing_tags = item.get('tags', {})
    if 'tone' in existing_tags and existing_tags['tone']:
        return existing_tags['tone']

    # Check top-level tone field
    if 'tone' in item and item['tone']:
        return item['tone']

    # Check subreddit field (Reddit sources)
    if 'subreddit' in item and item['subreddit']:
        subreddit = item['subreddit'].lower()
        if 'theonion' in subreddit and 'not' not in subreddit:
            return 'satirical'
        if 'nottheonion' in subreddit:
            return 'ironic'

    # Check source_file field
    if 'source_file' in item and item['source_file']:
        source = item['source_file'].lower()
        if 'reddit' in source:
            if 'theonion' in source and 'not' not in source:
                return 'satirical'
            if 'nottheonion' in source:
                return 'ironic'
            return 'humorous'

    # Check author for Franklin (didactic)
    if 'author_name' in item and item['author_name']:
        author = item['author_name'].lower()
        if 'franklin' in author:
            return 'didactic'

    # Check genre tags for humor
    if 'genre_tags' in item and item['genre_tags']:
        genres = [g.lower() for g in item['genre_tags']]
        if 'humor' in genres or 'satire' in genres:
            return 'humorous'

    # Check matched keywords for humor indicators
    if 'matched_keywords' in item and item['matched_keywords']:
        keywords = [k.lower() for k in item['matched_keywords']]
        humor_keywords = ['funny', 'joke', 'hilarious', 'amusing', 'witty']
        if any(kw in keywords for kw in humor_keywords):
            return 'humorous'

    # Default to humorous
    return 'humorous'


if __name__ == "__main__":
    # Test tone tagger
    print("Tone Tagger Test")
    print("=" * 60)

    # Test basic source strings
    test_cases = [
        ("reddit-theonion", {}, "satirical"),
        ("reddit-nottheonion", {}, "ironic"),
        ("gutenberg-franklin", {}, "didactic"),
        ("gutenberg-twain", {}, "humorous"),
        ("any-source", {"tone": "existing"}, "existing"),
        ("", {}, "humorous")
    ]

    print("\nTesting source string detection:")
    for source, tags, expected in test_cases:
        result = determine_tone(source, tags)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {source:30s} + {str(tags):20s} → {result:12s} (expected: {expected})")

    # Test full items
    test_items = [
        {
            'subreddit': 'TheOnion',
            'matched_keywords': ['storm', 'funny']
        },
        {
            'subreddit': 'nottheonion',
            'source_file': 'reddit_humor_weather.jsonl'
        },
        {
            'author_name': 'Benjamin Franklin',
            'genre_tags': ['biography', 'non-fiction']
        },
        {
            'author_name': 'Mark Twain',
            'genre_tags': ['humor', 'satire', 'adventure']
        },
        {
            'tags': {'tone': 'satirical'},
            'author_name': 'Unknown'
        }
    ]

    print("\nTesting full item detection:")
    for i, item in enumerate(test_items, 1):
        tone = determine_tone_from_item(item)
        print(f"  Item {i}: {tone}")
        print(f"    {item}")
