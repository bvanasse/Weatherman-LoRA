#!/usr/bin/env python3
"""
Domain Tagger Module for Instructionalization Pipeline

Determines domain tags based on matched keywords and tone.
Supports multiple domain tags per entry (array format).

Usage:
    from scripts.domain_tagger import determine_domains

    domains = determine_domains(["storm", "lightning", "funny"], "humorous")
    # Returns: ["weather", "humor"]
"""

from typing import List, Dict, Any


# Keyword sets for domain classification
WEATHER_KEYWORDS = {
    'storm', 'rain', 'snow', 'wind', 'thunder', 'lightning', 'cloud', 'fog',
    'hail', 'tornado', 'hurricane', 'blizzard', 'drought', 'flood', 'freeze',
    'heat', 'cold', 'warm', 'cool', 'temperature', 'forecast', 'weather',
    'climate', 'sunny', 'cloudy', 'overcast', 'precipitation', 'humidity',
    'barometer', 'pressure', 'front', 'cyclone', 'typhoon', 'monsoon',
    'drizzle', 'sleet', 'breeze', 'gale', 'tempest', 'downpour', 'shower'
}

HUMOR_KEYWORDS = {
    'funny', 'joke', 'hilarious', 'amusing', 'witty', 'satirical', 'ironic',
    'comedy', 'humor', 'humorous', 'laugh', 'ridiculous', 'absurd', 'silly',
    'comical', 'facetious', 'sarcastic', 'parody', 'mockery', 'jest'
}


def determine_domains(matched_keywords: List[str], tone: str = None) -> List[str]:
    """
    Determine domain tags from matched keywords and tone.

    Args:
        matched_keywords: List of keywords matched in the content
        tone: Tone tag (used to infer humor domain)

    Returns:
        List of domain tags (e.g., ["weather", "humor"])

    Domain rules:
        - Weather keywords → include "weather" in domains
        - Humor keywords or humor/satirical/ironic tone → include "humor"
        - Default to ["weather"] if no keywords available

    Examples:
        >>> determine_domains(["storm", "funny"], "humorous")
        ['weather', 'humor']
        >>> determine_domains(["lightning", "rain"], "didactic")
        ['weather']
        >>> determine_domains([], None)
        ['weather']
        >>> determine_domains(None, "satirical")
        ['weather', 'humor']
    """
    domains = []

    # Check matched keywords
    if matched_keywords:
        keywords_lower = {kw.lower() for kw in matched_keywords}

        # Check for weather keywords
        if keywords_lower & WEATHER_KEYWORDS:
            domains.append('weather')

        # Check for humor keywords
        if keywords_lower & HUMOR_KEYWORDS:
            if 'humor' not in domains:
                domains.append('humor')

    # Check tone for humor indication
    if tone and tone in ['satirical', 'ironic', 'humorous']:
        if 'humor' not in domains:
            domains.append('humor')

    # Default to weather if no domains identified
    if not domains:
        domains.append('weather')

    return domains


def determine_domains_from_item(item: Dict[str, Any]) -> List[str]:
    """
    Determine domains from a full data item.

    Args:
        item: Dictionary containing item metadata

    Returns:
        List of domain tags

    Checks:
        - matched_keywords field
        - Existing tone tag or inferred tone
        - Genre tags for additional context

    Examples:
        >>> item = {"matched_keywords": ["storm", "lightning"], "genre_tags": ["humor"]}
        >>> determine_domains_from_item(item)
        ['weather', 'humor']
    """
    # Get matched keywords
    matched_keywords = item.get('matched_keywords', [])

    # Get tone (from tags or top-level)
    tone = None
    if 'tags' in item and 'tone' in item['tags']:
        tone = item['tags']['tone']
    elif 'tone' in item:
        tone = item['tone']

    # Start with keyword-based domains
    domains = determine_domains(matched_keywords, tone)

    # Check genre tags for additional hints
    if 'genre_tags' in item and item['genre_tags']:
        genres_lower = {g.lower() for g in item['genre_tags']}

        # Add humor if genre indicates it
        if ('humor' in genres_lower or 'satire' in genres_lower or 'comedy' in genres_lower):
            if 'humor' not in domains:
                domains.append('humor')

    # Ensure weather is always present (primary domain)
    if 'weather' not in domains:
        domains.insert(0, 'weather')

    return domains


if __name__ == "__main__":
    # Test domain tagger
    print("Domain Tagger Test")
    print("=" * 60)

    # Test basic cases
    test_cases = [
        (["storm", "funny"], "humorous", ["weather", "humor"]),
        (["lightning", "rain"], "didactic", ["weather"]),
        ([], None, ["weather"]),
        (None, "satirical", ["weather", "humor"]),
        (["joke", "ridiculous"], None, ["humor", "weather"]),
        (["temperature", "forecast"], "humorous", ["weather", "humor"])
    ]

    print("\nTesting keyword and tone combinations:")
    for keywords, tone, expected in test_cases:
        result = determine_domains(keywords, tone)
        # Sort for comparison
        result_sorted = sorted(result)
        expected_sorted = sorted(expected)
        status = "✓" if result_sorted == expected_sorted else "✗"
        print(f"  {status} {str(keywords):40s} + {str(tone):12s} → {result}")
        if result_sorted != expected_sorted:
            print(f"      Expected: {expected}")

    # Test full items
    test_items = [
        {
            'matched_keywords': ['storm', 'lightning', 'funny'],
            'genre_tags': ['humor', 'satire']
        },
        {
            'matched_keywords': ['rain', 'wind'],
            'tone': 'didactic'
        },
        {
            'matched_keywords': [],
            'genre_tags': []
        },
        {
            'matched_keywords': ['joke', 'hilarious'],
            'tags': {'tone': 'satirical'}
        }
    ]

    print("\nTesting full item detection:")
    for i, item in enumerate(test_items, 1):
        domains = determine_domains_from_item(item)
        print(f"  Item {i}: {domains}")
        print(f"    {item}")
