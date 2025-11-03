#!/usr/bin/env python3
"""
Language Filter Module

Detects and filters text by language using fasttext and langdetect
as fallback. Filters for English-only content.

Usage:
    from scripts.language_filter import detect_language, filter_english_only

    # Detect language of a single text
    lang = detect_language("Hello world")  # Returns 'en'

    # Filter a batch of items
    english_items, stats = filter_english_only(items)
"""

from typing import List, Dict, Any, Tuple
import langdetect
from langdetect import detect, DetectorFactory

# Set seed for reproducible results
DetectorFactory.seed = 0


def detect_language_langdetect(text: str) -> str:
    """
    Detect language using langdetect library (fallback method).

    Args:
        text: Input text

    Returns:
        ISO 639-1 language code (e.g., 'en', 'fr', 'es')
        Returns 'unknown' if detection fails

    Example:
        >>> detect_language_langdetect("Hello world")
        'en'
    """
    if not text or not isinstance(text, str) or len(text.strip()) < 3:
        return 'unknown'

    try:
        lang = detect(text)
        return lang
    except Exception:
        return 'unknown'


def detect_language_fasttext(text: str, model_path: str = None) -> str:
    """
    Detect language using fasttext (preferred method).

    Args:
        text: Input text
        model_path: Path to fasttext model (optional)

    Returns:
        ISO 639-1 language code

    Note:
        Falls back to langdetect if fasttext is unavailable or fails.
        FastText model download: https://fasttext.cc/docs/en/language-identification.html
    """
    try:
        import fasttext
        # TODO: Implement fasttext detection when model is available
        # For now, fall back to langdetect
        return detect_language_langdetect(text)
    except ImportError:
        # FastText not installed, use langdetect
        return detect_language_langdetect(text)


def detect_language(text: str, method: str = 'auto') -> str:
    """
    Detect language of text.

    Args:
        text: Input text
        method: Detection method ('auto', 'fasttext', 'langdetect')

    Returns:
        ISO 639-1 language code

    Example:
        >>> lang = detect_language("Bonjour le monde")
        >>> lang in ['fr', 'unknown']
        True
    """
    if method == 'fasttext':
        return detect_language_fasttext(text)
    elif method == 'langdetect':
        return detect_language_langdetect(text)
    else:  # auto
        # Try fasttext first, fall back to langdetect
        return detect_language_fasttext(text)


def is_english(text: str, confidence_threshold: float = 0.7) -> bool:
    """
    Check if text is in English.

    Args:
        text: Input text
        confidence_threshold: Minimum confidence (0.0-1.0)
                             Note: langdetect doesn't provide confidence,
                             so this is primarily for future fasttext integration

    Returns:
        True if detected language is English, False otherwise

    Example:
        >>> is_english("Hello world")
        True
        >>> is_english("Bonjour")
        False
    """
    lang = detect_language(text)
    return lang == 'en'


def filter_english_only(
    items: List[Dict[str, Any]],
    text_field: str = 'content'
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Filter items to keep only English content.

    Args:
        items: List of item dictionaries
        text_field: Field name containing text to check

    Returns:
        Tuple of (filtered items, statistics dictionary)

    Statistics include:
        - original_count: Number of input items
        - english_count: Number of English items
        - filtered_count: Number of non-English items removed
        - language_distribution: Dict of language counts before filtering

    Example:
        >>> items = [
        ...     {'content': 'Hello world', 'id': 1},
        ...     {'content': 'Bonjour le monde', 'id': 2},
        ... ]
        >>> english_items, stats = filter_english_only(items)
        >>> stats['english_count'] >= 1
        True
    """
    if not items:
        return [], {
            'original_count': 0,
            'english_count': 0,
            'filtered_count': 0,
            'language_distribution': {}
        }

    # Track language distribution
    language_counts = {}
    english_items = []

    for item in items:
        text = item.get(text_field, '')
        if not isinstance(text, str):
            text = str(text)

        # Detect language
        lang = detect_language(text)

        # Count languages
        language_counts[lang] = language_counts.get(lang, 0) + 1

        # Keep English items
        if lang == 'en':
            english_items.append(item)

    # Calculate statistics
    original_count = len(items)
    english_count = len(english_items)
    filtered_count = original_count - english_count

    stats = {
        'original_count': original_count,
        'english_count': english_count,
        'filtered_count': filtered_count,
        'language_distribution': language_counts
    }

    return english_items, stats


def get_language_distribution(
    items: List[Dict[str, Any]],
    text_field: str = 'content'
) -> Dict[str, int]:
    """
    Get distribution of languages in a dataset.

    Args:
        items: List of item dictionaries
        text_field: Field name containing text

    Returns:
        Dictionary mapping language codes to counts

    Example:
        >>> items = [
        ...     {'content': 'Hello', 'id': 1},
        ...     {'content': 'World', 'id': 2},
        ... ]
        >>> dist = get_language_distribution(items)
        >>> 'en' in dist
        True
    """
    language_counts = {}

    for item in items:
        text = item.get(text_field, '')
        if not isinstance(text, str):
            text = str(text)

        lang = detect_language(text)
        language_counts[lang] = language_counts.get(lang, 0) + 1

    return language_counts


if __name__ == "__main__":
    # Test language detection
    print("Language Filter Test")
    print("=" * 60)

    test_texts = [
        ("Hello, how are you today?", "en"),
        ("Bonjour, comment allez-vous?", "fr"),
        ("Hola, como estas?", "es"),
        ("The weather is nice", "en"),
        ("Il fait beau aujourd'hui", "fr"),
    ]

    print("\nLanguage Detection:")
    for text, expected in test_texts:
        detected = detect_language(text)
        match = "✓" if detected == expected else "✗"
        print(f"  {match} {repr(text[:30])} -> {detected} (expected {expected})")

    # Test filtering
    print("\n" + "=" * 60)
    print("English Filtering Test")
    print("=" * 60)

    items = [
        {'content': 'The weather is nice today', 'id': 1},
        {'content': 'Bonjour le monde', 'id': 2},
        {'content': 'Rain expected tomorrow', 'id': 3},
        {'content': 'Il fait beau', 'id': 4},
        {'content': 'Sunny skies ahead', 'id': 5},
    ]

    print(f"\nOriginal items: {len(items)}")
    english_items, stats = filter_english_only(items)

    print(f"\nAfter filtering: {len(english_items)}")
    print(f"\nStatistics:")
    print(f"  Original count: {stats['original_count']}")
    print(f"  English count: {stats['english_count']}")
    print(f"  Filtered count: {stats['filtered_count']}")
    print(f"\nLanguage distribution:")
    for lang, count in sorted(stats['language_distribution'].items()):
        print(f"  {lang}: {count}")
