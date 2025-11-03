#!/usr/bin/env python3
"""
Persona Tagger Module for Instructionalization Pipeline

Determines persona tags based on source metadata for training data.
Maps literary passages to author personas (Twain/Franklin) and Reddit to neutral.

Usage:
    from scripts.persona_tagger import determine_persona

    persona = determine_persona("gutenberg-twain-tom_sawyer")
    # Returns: "twain"
"""

from typing import Dict, Any


def determine_persona(source: str) -> str:
    """
    Extract persona from source metadata.

    Args:
        source: Source identifier string containing author/source information

    Returns:
        Persona tag string: "twain", "franklin", or "neutral"

    Mapping rules:
        - Gutenberg passages with Twain books → "twain"
        - Gutenberg passages with Franklin books → "franklin"
        - Reddit posts → "neutral"
        - Unknown/missing → "neutral" (fallback)

    Examples:
        >>> determine_persona("gutenberg-twain-tom_sawyer")
        'twain'
        >>> determine_persona("gutenberg-franklin-autobiography")
        'franklin'
        >>> determine_persona("reddit-theonion")
        'neutral'
        >>> determine_persona("")
        'neutral'
    """
    if not source:
        return 'neutral'

    source_lower = source.lower()

    # Check for Twain references
    if 'twain' in source_lower:
        return 'twain'

    # Check for Franklin references
    if 'franklin' in source_lower:
        return 'franklin'

    # Check for Reddit sources
    if 'reddit' in source_lower:
        return 'neutral'

    # Fallback to neutral for unknown sources
    return 'neutral'


def determine_persona_from_item(item: Dict[str, Any]) -> str:
    """
    Determine persona from a full data item.

    Args:
        item: Dictionary containing item metadata with source info

    Returns:
        Persona tag string

    Checks multiple fields:
        - source_file: File-level source tracking
        - source_url: URL-based source identification
        - author_name: Direct author attribution
        - book_title: Title-based inference

    Examples:
        >>> item = {"author_name": "Mark Twain", "source_file": "gutenberg_passages.json"}
        >>> determine_persona_from_item(item)
        'twain'
    """
    # Check author_name field first (most authoritative)
    if 'author_name' in item and item['author_name']:
        author_lower = item['author_name'].lower()
        if 'twain' in author_lower:
            return 'twain'
        if 'franklin' in author_lower:
            return 'franklin'

    # Check source_file field
    if 'source_file' in item and item['source_file']:
        source = item['source_file'].lower()
        if 'twain' in source:
            return 'twain'
        if 'franklin' in source:
            return 'franklin'
        if 'reddit' in source:
            return 'neutral'

    # Check source_url field
    if 'source_url' in item and item['source_url']:
        url = item['source_url'].lower()
        if 'twain' in url:
            return 'twain'
        if 'franklin' in url:
            return 'franklin'
        if 'reddit' in url:
            return 'neutral'

    # Check book_title field
    if 'book_title' in item and item['book_title']:
        title = item['book_title'].lower()
        if 'tom sawyer' in title or 'huckleberry finn' in title or 'yankee' in title:
            return 'twain'
        if 'autobiography' in title and 'franklin' not in title:
            # Could be Franklin's autobiography, but need more context
            pass

    # Fallback to neutral
    return 'neutral'


if __name__ == "__main__":
    # Test persona tagger
    print("Persona Tagger Test")
    print("=" * 60)

    # Test basic source strings
    test_sources = [
        "gutenberg-twain-tom_sawyer",
        "gutenberg-franklin-autobiography",
        "reddit-theonion",
        "reddit-nottheonion",
        "",
        "unknown-source",
        "Mark Twain's Adventures"
    ]

    print("\nTesting source string detection:")
    for source in test_sources:
        persona = determine_persona(source)
        print(f"  {source:40s} → {persona}")

    # Test full items
    test_items = [
        {
            'author_name': 'Mark Twain',
            'source_file': 'gutenberg_passages.json'
        },
        {
            'author_name': 'Benjamin Franklin',
            'book_title': 'Autobiography'
        },
        {
            'source_file': 'reddit_humor_weather.jsonl',
            'subreddit': 'TheOnion'
        },
        {
            'book_title': 'The Adventures of Tom Sawyer',
            'source_url': 'https://www.gutenberg.org/ebooks/74'
        }
    ]

    print("\nTesting full item detection:")
    for i, item in enumerate(test_items, 1):
        persona = determine_persona_from_item(item)
        print(f"  Item {i}: {persona}")
        print(f"    {item}")
