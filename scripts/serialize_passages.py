#!/usr/bin/env python3
"""
Passage Serialization Module for Literary Corpus Collection

Serializes extracted passages to JSON format with metadata.
Creates structured output for downstream training pipeline.

Usage:
    from serialize_passages import serialize_to_json, generate_passage_id
"""

import json
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from paths import DATA_PROCESSED


def generate_passage_id(author_id: str, book_slug: str, sequence: int) -> str:
    """
    Generate unique passage ID.

    Args:
        author_id: Author identifier (e.g., "twain", "franklin")
        book_slug: Book slug (e.g., "tom_sawyer")
        sequence: Sequence number (0-indexed)

    Returns:
        Formatted passage ID (e.g., "twain_tom_sawyer_0001")

    Example:
        >>> generate_passage_id("twain", "tom_sawyer", 0)
        'twain_tom_sawyer_0001'
    """
    return f"{author_id}_{book_slug}_{sequence:04d}"


def calculate_statistics(passages: List[Dict]) -> Dict:
    """
    Calculate summary statistics for collected passages.

    Args:
        passages: List of passage dictionaries

    Returns:
        Dictionary of statistics

    Statistics:
        - total_passages: Total number of passages
        - books_processed: List of unique books
        - authors: List of unique authors
        - context_type_distribution: Count by context type
        - keyword_distribution: Top 10 most common keywords
        - word_count_stats: Min/max/average word counts
    """
    # Basic counts
    total_passages = len(passages)

    # Unique books and authors
    books = list(set(p['book_title'] for p in passages))
    authors = list(set(p['author_name'] for p in passages))

    # Context type distribution
    context_types = Counter(p['context_type'] for p in passages)

    # Keyword distribution (flatten all matched_keywords lists)
    all_keywords = []
    for p in passages:
        all_keywords.extend(p['matched_keywords'])
    keyword_counts = Counter(all_keywords)
    top_keywords = dict(keyword_counts.most_common(10))

    # Word count statistics
    word_counts = [p['word_count'] for p in passages]
    word_count_stats = {
        'min': min(word_counts) if word_counts else 0,
        'max': max(word_counts) if word_counts else 0,
        'average': sum(word_counts) / len(word_counts) if word_counts else 0
    }

    # Passages per book
    passages_per_book = Counter(p['book_title'] for p in passages)

    # Passages per author
    passages_per_author = Counter(p['author_name'] for p in passages)

    return {
        'total_passages': total_passages,
        'extraction_date': datetime.utcnow().isoformat() + 'Z',
        'books_processed': sorted(books),
        'authors': sorted(authors),
        'context_type_distribution': dict(context_types),
        'keyword_distribution': top_keywords,
        'word_count_stats': word_count_stats,
        'passages_per_book': dict(passages_per_book),
        'passages_per_author': dict(passages_per_author)
    }


def serialize_to_json(
    passages: List,
    output_path: Path,
    book_info_map: Dict
) -> None:
    """
    Serialize passages to JSON file.

    Args:
        passages: List of Passage objects
        output_path: Path to output JSON file
        book_info_map: Dictionary mapping (author_id, book_slug) to book_info

    Process:
        1. Convert Passage objects to dictionaries
        2. Assign unique passage IDs
        3. Calculate summary statistics
        4. Write JSON with atomic operation (temp file + rename)

    Output format:
        {
            "passages": [ ... ],
            "metadata": { ... }
        }
    """
    print(f"\nSerializing {len(passages)} passages to JSON...")

    # Convert passages to dictionaries with unique IDs
    passage_dicts = []
    passage_counter = {}  # Track sequence numbers per book

    for passage in passages:
        # Get book info
        book_key = (passage.to_dict({})['author_name'], passage.to_dict({})['book_title'])

        # Find matching book_info
        book_info = None
        for key, info in book_info_map.items():
            if info['author'] == book_key[0] and info['title'] == book_key[1]:
                book_info = info
                break

        if not book_info:
            print(f"WARNING: Book info not found for {book_key}")
            continue

        # Get or initialize sequence counter for this book
        book_id = f"{book_info['author_id']}_{book_info['slug']}"
        if book_id not in passage_counter:
            passage_counter[book_id] = 0

        # Convert to dictionary
        passage_dict = passage.to_dict(book_info)

        # Assign unique passage ID
        passage_dict['passage_id'] = generate_passage_id(
            book_info['author_id'],
            book_info['slug'],
            passage_counter[book_id]
        )

        passage_counter[book_id] += 1
        passage_dicts.append(passage_dict)

    print(f"  Converted {len(passage_dicts)} passages")

    # Calculate statistics
    statistics = calculate_statistics(passage_dicts)
    print(f"  Calculated statistics")

    # Create output structure
    output_data = {
        'passages': passage_dicts,
        'metadata': statistics
    }

    # Write to temporary file first (atomic operation)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=output_path.parent,
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        json.dump(output_data, tmp_file, indent=2, ensure_ascii=False)

    # Rename temp file to final path (atomic on Unix)
    tmp_path.replace(output_path)

    print(f"  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    # Validate JSON can be loaded
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        print(f"  Validation: JSON file valid, contains {len(loaded_data['passages'])} passages")
    except Exception as e:
        print(f"  WARNING: JSON validation failed: {e}")


def print_statistics(output_path: Path) -> None:
    """
    Print summary statistics from saved JSON file.

    Args:
        output_path: Path to JSON file
    """
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data['metadata']

    print()
    print("=" * 60)
    print("Passage Collection Statistics")
    print("=" * 60)
    print()
    print(f"Total passages: {metadata['total_passages']}")
    print(f"Extraction date: {metadata['extraction_date']}")
    print()

    print("Authors:")
    for author in metadata['authors']:
        count = metadata['passages_per_author'][author]
        print(f"  - {author}: {count} passages")
    print()

    print("Books:")
    for book in metadata['books_processed']:
        count = metadata['passages_per_book'][book]
        print(f"  - {book}: {count} passages")
    print()

    print("Context type distribution:")
    for context_type, count in sorted(metadata['context_type_distribution'].items()):
        print(f"  - {context_type}: {count}")
    print()

    print("Top 10 keywords:")
    for keyword, count in sorted(
        metadata['keyword_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  - {keyword}: {count} occurrences")
    print()

    stats = metadata['word_count_stats']
    print(f"Word count statistics:")
    print(f"  - Minimum: {stats['min']}")
    print(f"  - Maximum: {stats['max']}")
    print(f"  - Average: {stats['average']:.1f}")
    print()


if __name__ == "__main__":
    # Test with example data
    print("Serialization Module Test")
    print("=" * 60)

    # Example passage dictionary
    example_passage = {
        'passage_id': 'twain_tom_sawyer_0001',
        'author_name': 'Mark Twain',
        'book_title': 'The Adventures of Tom Sawyer',
        'book_id': 74,
        'publication_year': 1876,
        'chapter_section': 'CHAPTER I',
        'text': 'Sample passage text with weather and humor references...',
        'word_count': 250,
        'genre_tags': ['humor', 'satire', 'adventure'],
        'keywords_matched': ['weather', 'humor'],
        'context_type': 'both',
        'relevance_score': 2,
        'character_names': ['Tom', 'Huck'],
        'source_url': 'https://www.gutenberg.org/ebooks/74',
        'extraction_date': datetime.utcnow().isoformat() + 'Z'
    }

    passages = [example_passage]
    stats = calculate_statistics(passages)

    print(f"Example statistics:")
    print(json.dumps(stats, indent=2))
