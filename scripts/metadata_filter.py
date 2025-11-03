#!/usr/bin/env python3
"""
Metadata Filter Module for Instructionalization Pipeline

Filters metadata to preserve essential provenance fields while
removing unnecessary data that bloats file size.

Usage:
    from scripts.metadata_filter import filter_metadata, merge_tags

    filtered = filter_metadata(original_item)
    tags = merge_tags(persona, tone, domains, filtered)
"""

from typing import Dict, Any, List


# Essential metadata fields to preserve (provenance)
ESSENTIAL_FIELDS = {
    'source_file',
    'author_name',
    'book_title',
    'book_id',
    'publication_year',
    'genre_tags',
    'source_url',
    'extraction_date',
    'matched_keywords',
    'subreddit',
    'passage_id'
}

# Fields to exclude (bloat/unnecessary for training)
EXCLUDED_FIELDS = {
    'reddit_id',
    'created_utc',
    'url',
    'score',
    'num_comments',
    'word_count',
    'character_names',
    'chapter_section',
    'context_type',
    'relevance_score',
    'pipeline_metadata'
}


def filter_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter metadata to preserve essential fields only.

    Args:
        item: Original data item with all metadata

    Returns:
        Dictionary with essential metadata fields

    Preserves:
        - source_file: Source tracking
        - author_name: Attribution
        - book_title: Title reference
        - book_id: Gutenberg ID
        - publication_year: Temporal context
        - genre_tags: Genre classification
        - source_url: Original URL
        - extraction_date: Timestamp
        - matched_keywords: Keyword matches
        - subreddit: Reddit source
        - passage_id: Unique identifier

    Excludes:
        - reddit_id: Not needed for training
        - created_utc: Not needed for training
        - url: Redundant with source_url
        - score: Not needed for training
        - word_count: Can be recalculated
        - pipeline_metadata: Internal tracking

    Examples:
        >>> item = {"author_name": "Mark Twain", "reddit_id": "abc123", "matched_keywords": ["storm"]}
        >>> filtered = filter_metadata(item)
        >>> "author_name" in filtered
        True
        >>> "reddit_id" in filtered
        False
    """
    filtered = {}

    for key, value in item.items():
        # Include essential fields if they have values
        if key in ESSENTIAL_FIELDS and value is not None:
            # Filter out empty strings and empty lists
            if value == '' or value == [] or value == {}:
                continue
            filtered[key] = value

    return filtered


def merge_tags(
    persona: str,
    tone: str,
    domains: List[str],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge persona/tone/domain tags with retained metadata.

    Args:
        persona: Persona tag ("twain", "franklin", "neutral")
        tone: Tone tag ("satirical", "ironic", "humorous", "didactic")
        domains: List of domain tags (["weather", "humor"])
        metadata: Filtered metadata dictionary

    Returns:
        Dictionary with merged tags (persona, tone, domain, metadata)

    Ensures:
        - JSON-serializable output
        - Compact representation
        - All required tags present

    Examples:
        >>> merged = merge_tags("twain", "humorous", ["weather", "humor"], {"source_file": "gutenberg.json"})
        >>> merged["persona"]
        'twain'
        >>> merged["tone"]
        'humorous'
        >>> merged["domain"]
        ['weather', 'humor']
    """
    tags = {
        'persona': persona,
        'tone': tone,
        'domain': domains
    }

    # Merge with metadata
    tags.update(metadata)

    return tags


def extract_essential_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract essential metadata from item for tags field.

    This is a convenience function that combines filtering and
    common field extraction.

    Args:
        item: Original data item

    Returns:
        Dictionary with essential metadata ready for tags field

    Examples:
        >>> item = {"author_name": "Mark Twain", "text": "Some content", "reddit_id": "123"}
        >>> essential = extract_essential_from_item(item)
        >>> "author_name" in essential
        True
        >>> "text" in essential
        False
    """
    return filter_metadata(item)


if __name__ == "__main__":
    # Test metadata filter
    print("Metadata Filter Test")
    print("=" * 60)

    # Test item with mixed metadata
    test_item = {
        'text': 'Some passage content...',
        'author_name': 'Mark Twain',
        'book_title': 'The Adventures of Tom Sawyer',
        'book_id': 74,
        'publication_year': 1876,
        'genre_tags': ['humor', 'satire', 'adventure'],
        'source_url': 'https://www.gutenberg.org/ebooks/74',
        'source_file': 'gutenberg_passages.json',
        'extraction_date': '2025-11-02T19:48:02.514087Z',
        'matched_keywords': ['storm', 'lightning', 'funny'],
        'passage_id': 'twain_tom_sawyer_0001',
        'reddit_id': 'abc123',
        'created_utc': 1234567890,
        'url': 'https://reddit.com/r/test',
        'score': 42,
        'word_count': 150,
        'character_names': ['Tom', 'Becky'],
        'chapter_section': 'Chapter 5',
        'context_type': 'both',
        'relevance_score': 2,
        'pipeline_metadata': {'version': '1.0'}
    }

    print("\nFiltering metadata:")
    filtered = filter_metadata(test_item)

    print("\nEssential fields (preserved):")
    for key in sorted(filtered.keys()):
        print(f"  ✓ {key:20s} = {filtered[key]}")

    print("\nExcluded fields (removed):")
    for key in test_item.keys():
        if key not in filtered and key != 'text':
            print(f"  ✗ {key}")

    # Test merging tags
    print("\n" + "=" * 60)
    print("Testing tag merging:")
    merged = merge_tags(
        persona='twain',
        tone='humorous',
        domains=['weather', 'humor'],
        metadata=filtered
    )

    print("\nMerged tags:")
    print(f"  persona: {merged['persona']}")
    print(f"  tone: {merged['tone']}")
    print(f"  domain: {merged['domain']}")
    print(f"  metadata fields: {len(merged) - 3}")

    # Check JSON serialization
    import json
    print("\nJSON serialization test:")
    try:
        json_str = json.dumps(merged, ensure_ascii=False)
        print(f"  ✓ Successfully serialized ({len(json_str)} chars)")
    except Exception as e:
        print(f"  ✗ Serialization failed: {e}")
