#!/usr/bin/env python3
"""
Reddit Text Processing Module for Humor Dataset

Provides expanded weather keyword matching and Reddit-specific text cleaning
for processing r/TheOnion and r/nottheonion datasets.

Usage:
    from reddit_text_processing import (
        EXPANDED_WEATHER_KEYWORDS,
        clean_reddit_text,
        matches_weather_keywords
    )
"""

import re
import unicodedata
from typing import List, Tuple, Set


# Expanded weather keywords (40+ terms)
# Combines original 21 terms from keyword_matcher.py with additional seasonal,
# extreme weather, and metaphorical terms
EXPANDED_WEATHER_KEYWORDS = [
    # Original keywords (21 terms)
    'weather', 'rain', 'storm', 'thunder', 'lightning',
    'cloud', 'sun', 'wind', 'climate', 'temperature',
    'snow', 'fog', 'drought', 'hurricane', 'tornado',
    'flood', 'heat', 'cold', 'frost', 'dew', 'hail',

    # Seasonal terms (6 terms)
    'winter', 'summer', 'spring', 'fall', 'autumn', 'seasonal',

    # Extreme weather (10 terms)
    'heatwave', 'blizzard', 'wildfire', 'avalanche', 'monsoon',
    'typhoon', 'cyclone', 'tsunami', 'thunderstorm', 'snowstorm',

    # Metaphorical/contextual terms (8 terms)
    'forecast', 'outlook', 'weathering', 'sunny', 'cloudy',
    'rainy', 'stormy', 'breezy'
]


def build_expanded_weather_pattern() -> re.Pattern:
    """
    Build regex pattern for expanded weather keyword matching.

    Returns:
        Compiled regex pattern for case-insensitive whole-word matching

    Notes:
        - Uses word boundaries (\\b) to match whole words only
        - Case-insensitive matching
        - Avoids partial matches
    """
    # Escape special regex characters and join with OR
    escaped_keywords = [re.escape(kw) for kw in EXPANDED_WEATHER_KEYWORDS]
    pattern_str = r'\b(' + '|'.join(escaped_keywords) + r')\b'
    return re.compile(pattern_str, re.IGNORECASE)


# Pre-compiled pattern for efficiency
EXPANDED_WEATHER_PATTERN = build_expanded_weather_pattern()


def find_weather_keywords(text: str) -> List[Tuple[int, int, str]]:
    """
    Find all weather keyword matches in text.

    Args:
        text: Text to search for weather keywords

    Returns:
        List of tuples (start_pos, end_pos, matched_keyword)

    Example:
        >>> matches = find_weather_keywords("Storm warning for today")
        >>> for start, end, keyword in matches:
        ...     print(f"Found '{keyword}' at position {start}")
    """
    matches = []
    for match in EXPANDED_WEATHER_PATTERN.finditer(text):
        start_pos = match.start()
        end_pos = match.end()
        matched_text = match.group(0).lower()
        matches.append((start_pos, end_pos, matched_text))

    return matches


def get_unique_weather_keywords(text: str) -> List[str]:
    """
    Get unique weather keywords found in text.

    Args:
        text: Text to search

    Returns:
        Sorted list of unique matched keywords (lowercase)
    """
    matches = find_weather_keywords(text)
    unique_keywords = set(keyword for _, _, keyword in matches)
    return sorted(list(unique_keywords))


def matches_weather_keywords(text: str) -> bool:
    """
    Check if text contains at least one weather keyword.

    Args:
        text: Text to check

    Returns:
        True if text contains weather keywords, False otherwise
    """
    return EXPANDED_WEATHER_PATTERN.search(text) is not None


def clean_reddit_text(text: str) -> str:
    """
    Clean Reddit-specific artifacts and normalize text.

    Args:
        text: Raw Reddit post title or text

    Returns:
        Cleaned text with artifacts removed and Unicode normalized

    Cleaning steps:
        1. Remove Reddit artifacts: [removed], [deleted], [AutoModerator]
        2. Strip markdown formatting: **, *, [links](urls)
        3. Normalize Unicode: smart quotes → straight quotes, em-dashes → hyphens
        4. Trim excessive whitespace
        5. Normalize line breaks

    Example:
        >>> text = "**Weather alert**—heavy rain [removed]"
        >>> clean_reddit_text(text)
        'Weather alert-heavy rain'
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove Reddit-specific artifacts
    text = re.sub(r'\[removed\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[deleted\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[AutoModerator\]', '', text, flags=re.IGNORECASE)

    # Remove markdown formatting
    # Bold: **text** or __text__
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)

    # Italic: *text* or _text_ (careful not to remove legitimately needed asterisks)
    text = re.sub(r'\*([^*\s][^*]*[^*\s])\*', r'\1', text)
    text = re.sub(r'_([^_\s][^_]*[^_\s])_', r'\1', text)

    # Markdown links: [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove remaining URL patterns
    text = re.sub(r'https?://[^\s]+', '', text)

    # Normalize Unicode characters
    # Smart quotes to straight quotes
    text = text.replace('\u2018', "'")  # Left single quote
    text = text.replace('\u2019', "'")  # Right single quote
    text = text.replace('\u201c', '"')  # Left double quote
    text = text.replace('\u201d', '"')  # Right double quote

    # Em-dashes and en-dashes to hyphens
    text = text.replace('\u2014', '-')  # Em dash
    text = text.replace('\u2013', '-')  # En dash

    # Ellipsis
    text = text.replace('\u2026', '...')  # Horizontal ellipsis

    # Other common Unicode normalizations
    text = text.replace('\u00a0', ' ')  # Non-breaking space to regular space

    # Normalize to NFKD (decomposed form) and remove combining marks if needed
    # This helps with accented characters
    text = unicodedata.normalize('NFKD', text)

    # Trim excessive whitespace
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = text.strip()

    return text


def is_valid_cleaned_text(text: str, min_length: int = 10) -> bool:
    """
    Validate that cleaned text meets minimum quality standards.

    Args:
        text: Cleaned text to validate
        min_length: Minimum character length (default: 10)

    Returns:
        True if text is valid, False otherwise

    Validation checks:
        - Text is not empty
        - Text meets minimum length requirement
        - Text contains at least some alphanumeric characters
    """
    if not text or not isinstance(text, str):
        return False

    # Check minimum length
    if len(text.strip()) < min_length:
        return False

    # Check that text contains at least some alphanumeric content
    if not re.search(r'[a-zA-Z0-9]', text):
        return False

    return True


def process_reddit_title(title: str) -> Tuple[str, bool]:
    """
    Process a Reddit title: clean and validate.

    Args:
        title: Raw Reddit post title

    Returns:
        Tuple of (cleaned_text, is_valid)

    Example:
        >>> cleaned, valid = process_reddit_title("**Storm** warning [removed]")
        >>> if valid:
        ...     print(f"Cleaned: {cleaned}")
    """
    cleaned = clean_reddit_text(title)
    is_valid = is_valid_cleaned_text(cleaned)
    return cleaned, is_valid


if __name__ == "__main__":
    # Test with sample Reddit titles
    print("Reddit Text Processing Test")
    print("=" * 60)

    sample_titles = [
        "**Weather Update**: Heavy rain expected—stay safe!",
        "[removed]",
        "Political climate reaches new low as storm brewing",
        "Trump's forecast: “Tremendous” weather ahead",
        "Hurricane season starts with a bang [AutoModerator]"
    ]

    for title in sample_titles:
        print(f"\nOriginal: {title}")
        cleaned, valid = process_reddit_title(title)
        print(f"Cleaned:  {cleaned}")
        print(f"Valid:    {valid}")

        if valid and matches_weather_keywords(cleaned):
            keywords = get_unique_weather_keywords(cleaned)
            print(f"Weather keywords: {', '.join(keywords)}")
