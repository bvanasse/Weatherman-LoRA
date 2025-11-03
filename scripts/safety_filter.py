#!/usr/bin/env python3
"""
Safety Filter Module using OpenAI Moderation API

Detects and filters unsafe content including toxicity, hate speech,
violence, and NSFW content using OpenAI's moderation API.

Usage:
    from scripts.safety_filter import filter_unsafe_content

    # Filter items with safety checks
    safe_items, stats = filter_unsafe_content(items, api_key=api_key)

    # Skip safety checks (for testing)
    safe_items, stats = filter_unsafe_content(items, skip=True)
"""

import os
import time
from typing import List, Dict, Any, Tuple
from openai import OpenAI


def check_safety_batch(
    texts: List[str],
    api_key: str = None,
    max_retries: int = 3,
    backoff_factor: int = 2
) -> List[Dict[str, Any]]:
    """
    Check safety of a batch of texts using OpenAI moderation API.

    Args:
        texts: List of texts to check
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        max_retries: Maximum retry attempts on failure
        backoff_factor: Exponential backoff multiplier

    Returns:
        List of moderation results (one per text)

    Each result contains:
        - flagged: bool indicating if content is unsafe
        - categories: dict of specific category flags
        - category_scores: dict of confidence scores

    Example:
        >>> # Requires valid API key
        >>> results = check_safety_batch(["Hello world"], api_key="sk-...")
        >>> len(results) == 1
        True
    """
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key parameter")

    client = OpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.moderations.create(input=texts)
            return [result.model_dump() for result in response.results]

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                print(f"  Moderation API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Moderation API failed after {max_retries} attempts: {e}")
                raise


def filter_unsafe_content(
    items: List[Dict[str, Any]],
    text_field: str = 'content',
    batch_size: int = 20,
    api_key: str = None,
    skip: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Filter items to remove unsafe content.

    Args:
        items: List of item dictionaries
        text_field: Field name containing text to check
        batch_size: Number of items per API batch call
        api_key: OpenAI API key (optional if set in environment)
        skip: If True, skip safety checks (for testing without API)

    Returns:
        Tuple of (safe items, statistics dictionary)

    Statistics include:
        - original_count: Number of input items
        - safe_count: Number of safe items
        - flagged_count: Number of unsafe items removed
        - flagged_categories: Dict of category counts

    Example:
        >>> items = [
        ...     {'content': 'Hello world', 'id': 1},
        ...     {'content': 'Nice weather today', 'id': 2},
        ... ]
        >>> safe_items, stats = filter_unsafe_content(items, skip=True)
        >>> stats['safe_count'] == 2
        True
    """
    if not items:
        return [], {
            'original_count': 0,
            'safe_count': 0,
            'flagged_count': 0,
            'flagged_categories': {}
        }

    # If skip flag is set, return all items as safe
    if skip:
        return items, {
            'original_count': len(items),
            'safe_count': len(items),
            'flagged_count': 0,
            'flagged_categories': {},
            'skipped': True
        }

    safe_items = []
    flagged_categories = {}
    total_flagged = 0

    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        texts = [item.get(text_field, '') for item in batch]

        try:
            # Check safety
            results = check_safety_batch(texts, api_key=api_key)

            # Filter based on results
            for item, result in zip(batch, results):
                if not result['flagged']:
                    safe_items.append(item)
                else:
                    total_flagged += 1
                    # Track flagged categories
                    for category, is_flagged in result['categories'].items():
                        if is_flagged:
                            flagged_categories[category] = flagged_categories.get(category, 0) + 1

        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            # On error, skip this batch (conservative approach)
            total_flagged += len(batch)

        # Progress update
        if (i + batch_size) % 100 == 0:
            print(f"  Processed {min(i + batch_size, len(items))}/{len(items)} items...")

    # Calculate statistics
    stats = {
        'original_count': len(items),
        'safe_count': len(safe_items),
        'flagged_count': total_flagged,
        'flagged_categories': flagged_categories
    }

    return safe_items, stats


def is_safe(text: str, api_key: str = None) -> bool:
    """
    Check if a single text is safe.

    Args:
        text: Text to check
        api_key: OpenAI API key

    Returns:
        True if safe, False if flagged

    Example:
        >>> # Requires valid API key
        >>> is_safe("Hello world", api_key="sk-...")
        True
    """
    try:
        results = check_safety_batch([text], api_key=api_key)
        return not results[0]['flagged']
    except Exception:
        # Conservative: treat errors as unsafe
        return False


if __name__ == "__main__":
    # Test safety filter (requires API key)
    print("Safety Filter Test")
    print("=" * 60)

    items = [
        {'content': 'The weather is nice today', 'id': 1},
        {'content': 'Rain expected tomorrow', 'id': 2},
        {'content': 'Sunny skies ahead', 'id': 3},
    ]

    # Test with skip=True (no API calls)
    print("\nTest Mode (skip=True):")
    safe_items, stats = filter_unsafe_content(items, skip=True)

    print(f"  Original: {stats['original_count']}")
    print(f"  Safe: {stats['safe_count']}")
    print(f"  Flagged: {stats['flagged_count']}")
    print(f"  Skipped: {stats.get('skipped', False)}")

    # Test with API (requires key)
    if os.getenv('OPENAI_API_KEY'):
        print("\n" + "=" * 60)
        print("API Test Mode:")
        try:
            safe_items, stats = filter_unsafe_content(items[:1], batch_size=1)
            print(f"  Original: {stats['original_count']}")
            print(f"  Safe: {stats['safe_count']}")
            print(f"  Flagged: {stats['flagged_count']}")
        except Exception as e:
            print(f"  API test failed: {e}")
    else:
        print("\nNote: Set OPENAI_API_KEY to test with real API")
