#!/usr/bin/env python3
"""
Deduplication Module using MinHash and LSH

Implements near-duplicate detection using Jaccard similarity with
MinHash signatures and Locality Sensitive Hashing for efficient lookup.

Uses datasketch library for MinHash/LSH implementation.

Usage:
    from scripts.deduplication import remove_duplicates, build_dedup_index

    # Remove duplicates from a list of items
    unique_items, stats = remove_duplicates(items, threshold=0.8)

    # Or build index separately for more control
    lsh_index = build_dedup_index(texts, threshold=0.8)
"""

from typing import List, Dict, Any, Set, Tuple
from datasketch import MinHash, MinHashLSH


def text_to_shingles(text: str, k: int = 3) -> Set[str]:
    """
    Convert text to character-level k-shingles (n-grams).

    Args:
        text: Input text
        k: Size of shingles (default: 3)

    Returns:
        Set of k-length character sequences

    Example:
        >>> shingles = text_to_shingles("hello", k=3)
        >>> 'hel' in shingles and 'ell' in shingles
        True
    """
    text = text.lower().strip()
    if len(text) < k:
        return {text}

    shingles = set()
    for i in range(len(text) - k + 1):
        shingle = text[i:i + k]
        shingles.add(shingle)

    return shingles


def create_minhash(text: str, num_perm: int = 128) -> MinHash:
    """
    Create MinHash signature for a text.

    Args:
        text: Input text
        num_perm: Number of permutations (higher = more accurate, slower)

    Returns:
        MinHash object with signature

    Example:
        >>> mh = create_minhash("example text")
        >>> mh is not None
        True
    """
    minhash = MinHash(num_perm=num_perm)
    shingles = text_to_shingles(text)

    for shingle in shingles:
        minhash.update(shingle.encode('utf8'))

    return minhash


def build_dedup_index(
    texts: List[str],
    threshold: float = 0.8,
    num_perm: int = 128
) -> Tuple[MinHashLSH, List[MinHash]]:
    """
    Build LSH index for efficient duplicate detection.

    Args:
        texts: List of text strings to index
        threshold: Jaccard similarity threshold (0.0-1.0)
        num_perm: Number of MinHash permutations

    Returns:
        Tuple of (LSH index, list of MinHash signatures)

    Example:
        >>> texts = ["hello world", "hello world!", "goodbye"]
        >>> lsh, minhashes = build_dedup_index(texts, threshold=0.8)
        >>> lsh is not None
        True
    """
    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Create MinHash signatures and add to index
    minhashes = []
    for idx, text in enumerate(texts):
        mh = create_minhash(text, num_perm=num_perm)
        minhashes.append(mh)
        lsh.insert(str(idx), mh)

    return lsh, minhashes


def find_duplicates(
    texts: List[str],
    threshold: float = 0.8,
    num_perm: int = 128
) -> Set[int]:
    """
    Find indices of duplicate texts.

    Args:
        texts: List of text strings
        threshold: Jaccard similarity threshold
        num_perm: Number of MinHash permutations

    Returns:
        Set of indices to remove (keeps first occurrence)

    Example:
        >>> texts = ["hello", "hello world", "hello"]
        >>> duplicates = find_duplicates(texts, threshold=0.8)
        >>> 2 in duplicates  # Third item is duplicate of first
        True
    """
    lsh, minhashes = build_dedup_index(texts, threshold=threshold, num_perm=num_perm)

    duplicates_to_remove = set()
    seen_groups = set()

    for idx, mh in enumerate(minhashes):
        # Query for similar items
        similar = lsh.query(mh)

        # Convert to integers and sort
        similar_indices = sorted([int(x) for x in similar])

        # Skip if we've already processed this group
        group_key = tuple(similar_indices)
        if group_key in seen_groups:
            continue

        seen_groups.add(group_key)

        # Keep first occurrence, mark rest as duplicates
        if len(similar_indices) > 1:
            for dup_idx in similar_indices[1:]:
                duplicates_to_remove.add(dup_idx)

    return duplicates_to_remove


def remove_duplicates(
    items: List[Dict[str, Any]],
    threshold: float = 0.8,
    num_perm: int = 128,
    text_field: str = 'content'
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Remove duplicate items based on text similarity.

    Args:
        items: List of item dictionaries
        threshold: Jaccard similarity threshold (0.0-1.0)
        num_perm: Number of MinHash permutations
        text_field: Field name containing text to compare

    Returns:
        Tuple of (unique items, statistics dictionary)

    Statistics include:
        - original_count: Number of input items
        - unique_count: Number of unique items
        - duplicates_removed: Number of duplicates removed
        - duplicate_rate: Percentage of duplicates

    Example:
        >>> items = [
        ...     {'content': 'hello world', 'id': 1},
        ...     {'content': 'hello world!', 'id': 2},
        ...     {'content': 'goodbye', 'id': 3}
        ... ]
        >>> unique, stats = remove_duplicates(items, threshold=0.8)
        >>> stats['duplicates_removed'] >= 0
        True
    """
    if not items:
        return [], {
            'original_count': 0,
            'unique_count': 0,
            'duplicates_removed': 0,
            'duplicate_rate': 0.0
        }

    # Extract texts
    texts = []
    for item in items:
        text = item.get(text_field, '')
        if not isinstance(text, str):
            text = str(text)
        texts.append(text)

    # Find duplicates
    duplicate_indices = find_duplicates(texts, threshold=threshold, num_perm=num_perm)

    # Filter out duplicates
    unique_items = [
        item for idx, item in enumerate(items)
        if idx not in duplicate_indices
    ]

    # Calculate statistics
    original_count = len(items)
    unique_count = len(unique_items)
    duplicates_removed = original_count - unique_count
    duplicate_rate = (duplicates_removed / original_count * 100) if original_count > 0 else 0.0

    stats = {
        'original_count': original_count,
        'unique_count': unique_count,
        'duplicates_removed': duplicates_removed,
        'duplicate_rate': round(duplicate_rate, 2)
    }

    return unique_items, stats


if __name__ == "__main__":
    # Test deduplication
    print("Deduplication Test")
    print("=" * 60)

    # Test case 1: Exact duplicates
    items = [
        {'content': 'The weather is nice today', 'id': 1},
        {'content': 'The weather is nice today', 'id': 2},  # Exact duplicate
        {'content': 'Rain expected tomorrow', 'id': 3},
        {'content': 'The weather is nice today!', 'id': 4},  # Near duplicate
        {'content': 'Sunny skies ahead', 'id': 5},
    ]

    print(f"\nOriginal items: {len(items)}")
    for item in items:
        print(f"  [{item['id']}] {item['content']}")

    unique_items, stats = remove_duplicates(items, threshold=0.8)

    print(f"\nAfter deduplication: {len(unique_items)}")
    for item in unique_items:
        print(f"  [{item['id']}] {item['content']}")

    print(f"\nStatistics:")
    print(f"  Original count: {stats['original_count']}")
    print(f"  Unique count: {stats['unique_count']}")
    print(f"  Duplicates removed: {stats['duplicates_removed']}")
    print(f"  Duplicate rate: {stats['duplicate_rate']}%")

    # Test case 2: No duplicates
    print("\n" + "=" * 60)
    print("Test Case 2: No Duplicates")
    print("=" * 60)

    items2 = [
        {'content': 'Completely different text', 'id': 1},
        {'content': 'Another unique passage', 'id': 2},
        {'content': 'Third distinct item', 'id': 3},
    ]

    unique_items2, stats2 = remove_duplicates(items2, threshold=0.8)
    print(f"\nOriginal: {stats2['original_count']}, Unique: {stats2['unique_count']}")
    print(f"Duplicates removed: {stats2['duplicates_removed']}")
