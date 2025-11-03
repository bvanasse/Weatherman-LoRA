#!/usr/bin/env python3
"""
Keyword Matching Module for Literary Corpus Collection

Provides case-insensitive whole-word keyword matching for weather and humor terms.
Used to identify relevant passages in literary texts.

Usage:
    from keyword_matcher import find_keyword_matches, WEATHER_KEYWORDS, HUMOR_KEYWORDS
"""

import re
from typing import Dict, List, Set, Tuple


# Weather-related keywords (20 terms)
WEATHER_KEYWORDS = [
    'weather', 'rain', 'storm', 'thunder', 'lightning',
    'cloud', 'sun', 'wind', 'climate', 'temperature',
    'snow', 'fog', 'drought', 'hurricane', 'tornado',
    'flood', 'heat', 'cold', 'frost', 'dew', 'hail'
]

# Humor-related keywords (10 terms)
HUMOR_KEYWORDS = [
    'joke', 'wit', 'laugh', 'humor', 'comic',
    'amusing', 'funny', 'satire', 'irony', 'jest'
]


def build_keyword_pattern(keywords: List[str]) -> re.Pattern:
    """
    Build a regex pattern for whole-word keyword matching.

    Args:
        keywords: List of keywords to match

    Returns:
        Compiled regex pattern for case-insensitive whole-word matching

    Notes:
        - Uses word boundaries (\b) to match whole words only
        - Case-insensitive matching
        - Avoids partial matches (e.g., "weather" won't match in "leathern")
    """
    # Escape special regex characters and join with OR
    escaped_keywords = [re.escape(kw) for kw in keywords]
    pattern_str = r'\b(' + '|'.join(escaped_keywords) + r')\b'
    return re.compile(pattern_str, re.IGNORECASE)


# Pre-compiled patterns for efficiency
WEATHER_PATTERN = build_keyword_pattern(WEATHER_KEYWORDS)
HUMOR_PATTERN = build_keyword_pattern(HUMOR_KEYWORDS)


def find_keyword_matches(
    text: str,
    keyword_pattern: re.Pattern,
    keyword_list: List[str]
) -> List[Tuple[int, int, str]]:
    """
    Find all matches of keywords in text.

    Args:
        text: Text to search for keywords
        keyword_pattern: Compiled regex pattern
        keyword_list: Original list of keywords (for reference)

    Returns:
        List of tuples (start_pos, end_pos, matched_keyword)

    Example:
        >>> matches = find_keyword_matches(text, WEATHER_PATTERN, WEATHER_KEYWORDS)
        >>> for start, end, keyword in matches:
        ...     print(f"Found '{keyword}' at position {start}")
    """
    matches = []
    for match in keyword_pattern.finditer(text):
        start_pos = match.start()
        end_pos = match.end()
        matched_text = match.group(0).lower()
        matches.append((start_pos, end_pos, matched_text))

    return matches


def classify_passage_context(
    weather_matches: List[Tuple[int, int, str]],
    humor_matches: List[Tuple[int, int, str]]
) -> str:
    """
    Classify a passage based on which keywords it contains.

    Args:
        weather_matches: List of weather keyword matches
        humor_matches: List of humor keyword matches

    Returns:
        Context type: "both", "weather", "humor", or "none"

    Notes:
        - "both": Contains both weather and humor keywords (highest priority)
        - "weather": Contains only weather keywords
        - "humor": Contains only humor keywords
        - "none": Contains neither (should be filtered out)
    """
    has_weather = len(weather_matches) > 0
    has_humor = len(humor_matches) > 0

    if has_weather and has_humor:
        return "both"
    elif has_weather:
        return "weather"
    elif has_humor:
        return "humor"
    else:
        return "none"


def calculate_relevance_score(
    weather_matches: List[Tuple[int, int, str]],
    humor_matches: List[Tuple[int, int, str]]
) -> int:
    """
    Calculate relevance score for a passage.

    Args:
        weather_matches: List of weather keyword matches
        humor_matches: List of humor keyword matches

    Returns:
        Relevance score (higher is better)

    Scoring:
        - Passages with both weather + humor keywords: 2 points
        - Passages with single keyword type: 1 point
        - Passages with neither: 0 points

    Notes:
        This prioritizes passages that combine both themes,
        which are most relevant for training a weather-focused humor assistant.
    """
    has_weather = len(weather_matches) > 0
    has_humor = len(humor_matches) > 0

    if has_weather and has_humor:
        return 2
    elif has_weather or has_humor:
        return 1
    else:
        return 0


def get_matched_keywords(
    weather_matches: List[Tuple[int, int, str]],
    humor_matches: List[Tuple[int, int, str]]
) -> List[str]:
    """
    Extract unique matched keywords from match results.

    Args:
        weather_matches: List of weather keyword matches
        humor_matches: List of humor keyword matches

    Returns:
        Sorted list of unique matched keywords

    Example:
        >>> keywords = get_matched_keywords(weather, humor)
        >>> print(keywords)
        ['cloud', 'rain', 'satire', 'thunder']
    """
    matched_set: Set[str] = set()

    for _, _, keyword in weather_matches:
        matched_set.add(keyword.lower())

    for _, _, keyword in humor_matches:
        matched_set.add(keyword.lower())

    return sorted(list(matched_set))


def find_all_matches(text: str) -> Dict[str, any]:
    """
    Find all weather and humor keyword matches in text.

    Args:
        text: Text to search

    Returns:
        Dictionary containing:
            - weather_matches: List of weather keyword matches
            - humor_matches: List of humor keyword matches
            - context_type: Classification ("both", "weather", "humor", "none")
            - relevance_score: Numeric score (0-2)
            - matched_keywords: Sorted list of unique matched keywords

    Example:
        >>> result = find_all_matches(passage_text)
        >>> if result['relevance_score'] > 0:
        ...     print(f"Found {len(result['matched_keywords'])} keywords")
    """
    weather_matches = find_keyword_matches(text, WEATHER_PATTERN, WEATHER_KEYWORDS)
    humor_matches = find_keyword_matches(text, HUMOR_PATTERN, HUMOR_KEYWORDS)

    return {
        'weather_matches': weather_matches,
        'humor_matches': humor_matches,
        'context_type': classify_passage_context(weather_matches, humor_matches),
        'relevance_score': calculate_relevance_score(weather_matches, humor_matches),
        'matched_keywords': get_matched_keywords(weather_matches, humor_matches)
    }


if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    The weather was frightful that day. Thunder rolled across the sky,
    and the rain came down in sheets. But Tom, with his usual humor,
    made a joke about it that set everyone laughing.
    """

    result = find_all_matches(sample_text)

    print("Keyword Matching Test")
    print("=" * 60)
    print(f"Sample text: {sample_text.strip()}")
    print()
    print(f"Weather matches: {len(result['weather_matches'])}")
    for start, end, kw in result['weather_matches']:
        print(f"  - '{kw}' at position {start}")
    print()
    print(f"Humor matches: {len(result['humor_matches'])}")
    for start, end, kw in result['humor_matches']:
        print(f"  - '{kw}' at position {start}")
    print()
    print(f"Context type: {result['context_type']}")
    print(f"Relevance score: {result['relevance_score']}")
    print(f"Matched keywords: {', '.join(result['matched_keywords'])}")
