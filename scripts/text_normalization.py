#!/usr/bin/env python3
"""
Text Normalization Module for Data Pipeline

Implements Unicode normalization using NFC (Canonical Decomposition + Composition)
to preserve semantic distinctions while ensuring consistent representation.

Usage:
    from scripts.text_normalization import normalize_unicode, normalize_item

    # Normalize a single text
    clean_text = normalize_unicode("Smart quotes and em-dashes")

    # Normalize a data item (dict with text fields)
    normalized_item = normalize_item(item)
"""

import unicodedata
from typing import Dict, Any, List


def normalize_unicode(text: str, form: str = 'NFC') -> str:
    """
    Normalize Unicode text using specified normalization form.

    Args:
        text: Text to normalize
        form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
              Default: NFC (Canonical Decomposition + Composition)

    Returns:
        Normalized text

    NFC vs NFKC:
        - NFC: Preserves semantic distinctions (superscripts, fractions, ligatures)
        - NFKC: Converts to compatibility characters (more aggressive)
        - For literary texts, NFC is preferred to preserve author's intent

    Examples:
        >>> normalize_unicode("café")  # é as single character
        'café'
        >>> normalize_unicode("café")  # e + combining accent
        'café'
    """
    if not text or not isinstance(text, str):
        return ""

    # Apply Unicode normalization
    normalized = unicodedata.normalize(form, text)

    return normalized


def normalize_item(item: Dict[str, Any], text_fields: List[str] = None) -> Dict[str, Any]:
    """
    Normalize all text fields in a data item.

    Args:
        item: Data item dictionary
        text_fields: List of field names to normalize
                     If None, normalizes common fields: 'content', 'text', 'cleaned_title'

    Returns:
        Item with normalized text fields

    Note:
        - Preserves all other fields unchanged
        - Creates a copy to avoid mutating original

    Example:
        >>> item = {'content': 'Test text', 'source': 'test'}
        >>> normalized = normalize_item(item)
        >>> 'content' in normalized
        True
    """
    if text_fields is None:
        # Default text fields to normalize
        text_fields = ['content', 'text', 'cleaned_title', 'title']

    # Create copy to avoid mutating original
    normalized_item = item.copy()

    for field in text_fields:
        if field in normalized_item and isinstance(normalized_item[field], str):
            normalized_item[field] = normalize_unicode(normalized_item[field])

    return normalized_item


def normalize_batch(items: List[Dict[str, Any]], text_fields: List[str] = None) -> List[Dict[str, Any]]:
    """
    Normalize text fields in a batch of items.

    Args:
        items: List of data item dictionaries
        text_fields: List of field names to normalize (optional)

    Returns:
        List of items with normalized text fields

    Example:
        >>> items = [
        ...     {'content': 'Text 1', 'id': 1},
        ...     {'content': 'Text 2', 'id': 2}
        ... ]
        >>> normalized = normalize_batch(items)
        >>> len(normalized)
        2
    """
    return [normalize_item(item, text_fields) for item in items]


def get_normalization_stats(original: str, normalized: str) -> Dict[str, Any]:
    """
    Get statistics about normalization changes.

    Args:
        original: Original text
        normalized: Normalized text

    Returns:
        Dictionary with statistics:
        - changed: Whether text was modified
        - length_diff: Difference in character count
        - original_length: Original character count
        - normalized_length: Normalized character count

    Example:
        >>> stats = get_normalization_stats("test", "test")
        >>> 'changed' in stats
        True
    """
    changed = original != normalized
    original_len = len(original)
    normalized_len = len(normalized)

    return {
        'changed': changed,
        'length_diff': normalized_len - original_len,
        'original_length': original_len,
        'normalized_length': normalized_len
    }


if __name__ == "__main__":
    # Test normalization with examples
    print("Text Normalization Test")
    print("=" * 60)

    test_cases = [
        ("Regular ASCII text", "No change expected"),
        ("Café with accent", "Café normalized"),
        ("Text with em-dash", "Em-dash example"),
    ]

    for original, description in test_cases:
        normalized = normalize_unicode(original)
        stats = get_normalization_stats(original, normalized)

        print(f"\n{description}:")
        print(f"  Original:   {repr(original)}")
        print(f"  Normalized: {repr(normalized)}")
        print(f"  Changed:    {stats['changed']}")
        if stats['length_diff'] != 0:
            print(f"  Length diff: {stats['length_diff']}")

    # Test batch normalization
    print("\n" + "=" * 60)
    print("Batch Normalization Test")
    print("=" * 60)

    items = [
        {'content': 'Test text 1', 'source': 'test1'},
        {'content': 'Regular text', 'source': 'test2'},
        {'text': 'Different field name', 'source': 'test3'}
    ]

    normalized_items = normalize_batch(items)
    print(f"\nNormalized {len(normalized_items)} items")
    for i, item in enumerate(normalized_items):
        text = item.get('content') or item.get('text')
        print(f"  Item {i + 1}: {repr(text)}")
