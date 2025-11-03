#!/usr/bin/env python3
"""
Passage Extraction Module for Literary Corpus Collection

Extracts relevant passages from literary texts using keyword matching
and context window approach. Preserves paragraph boundaries and metadata.

Usage:
    python scripts/extract_passages.py
    python scripts/extract_passages.py --book-file twain_tom_sawyer.txt
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk

# Import project paths and keyword matcher
from paths import DATA_RAW, CONFIGS_DIR
from keyword_matcher import find_all_matches, WEATHER_KEYWORDS, HUMOR_KEYWORDS


# Target passage length constraints
MIN_WORD_COUNT = 75
MAX_WORD_COUNT = 600
TARGET_MIN = 150
TARGET_MAX = 450

# Context window (paragraphs to include around match)
CONTEXT_BEFORE = 1
CONTEXT_AFTER_MIN = 1
CONTEXT_AFTER_MAX = 2

# Overlap threshold (allow passages that overlap <50% of paragraphs)
OVERLAP_THRESHOLD = 0.2


def ensure_nltk_data():
    """
    Ensure required NLTK data is downloaded.

    Downloads punkt tokenizer if not already present.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs.

    Args:
        text: Full book text

    Returns:
        List of paragraphs (non-empty)

    Notes:
        - Splits on double newlines
        - Strips whitespace from each paragraph
        - Filters out empty paragraphs
    """
    # Split on double newlines (or more)
    paragraphs = re.split(r'\n\s*\n', text)

    # Strip whitespace and filter empty
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def count_words(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Text to count words in

    Returns:
        Number of words

    Notes:
        - Simple whitespace-based counting
        - Considers punctuation attached to words as part of the word
    """
    return len(text.split())


def find_chapter_marker(paragraph: str) -> Optional[str]:
    """
    Extract chapter marker from paragraph if present.

    Args:
        paragraph: Paragraph text

    Returns:
        Chapter marker string (e.g., "Chapter I") or None

    Patterns:
        - "CHAPTER I", "Chapter 1", "CHAPTER ONE"
        - "I. CHAPTER TITLE", "Chapter I: Title"
    """
    # Common chapter patterns
    patterns = [
        r'^(CHAPTER\s+[IVXLCDM]+)',  # CHAPTER I, CHAPTER XIV
        r'^(Chapter\s+\d+)',           # Chapter 1, Chapter 23
        r'^(CHAPTER\s+\w+)',           # CHAPTER ONE, CHAPTER FIRST
        r'^([IVXLCDM]+\.)',            # I., XIV.
    ]

    for pattern in patterns:
        match = re.search(pattern, paragraph)
        if match:
            return match.group(1)

    return None


def extract_character_names(paragraph: str) -> List[str]:
    """
    Extract character names from dialogue in paragraph.

    Args:
        paragraph: Paragraph text

    Returns:
        List of character names found in dialogue

    Notes:
        - Looks for quoted speech with attribution
        - Simple heuristic: captures words before "said", "replied", etc.
        - Limited to common patterns in 19th century literature
    """
    names = []

    # Common dialogue attribution patterns
    patterns = [
        r'(\w+)\s+said',
        r'said\s+(\w+)',
        r'(\w+)\s+replied',
        r'(\w+)\s+asked',
        r'(\w+)\s+exclaimed',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, paragraph, re.IGNORECASE)
        names.extend(matches)

    # Deduplicate and filter common words
    common_words = {'he', 'she', 'i', 'we', 'they', 'it', 'the', 'a', 'an'}
    names = [n.title() for n in names if n.lower() not in common_words]

    return list(set(names))


def extract_passage_around_match(
    paragraphs: List[str],
    match_para_idx: int,
    target_words: int = TARGET_MIN
) -> Tuple[str, int, int]:
    """
    Extract passage context around a keyword match.

    Args:
        paragraphs: List of all paragraphs
        match_para_idx: Index of paragraph containing match
        target_words: Target word count for passage

    Returns:
        Tuple of (passage_text, start_idx, end_idx)

    Strategy:
        - Include 1 paragraph before match
        - Include match paragraph
        - Include 1-2 paragraphs after match
        - Adjust context to reach target word count (150-450 words)
        - Stay within min/max bounds (100-600 words)
    """
    # Start with context window
    start_idx = max(0, match_para_idx - CONTEXT_BEFORE)
    end_idx = min(len(paragraphs), match_para_idx + CONTEXT_AFTER_MIN + 1)

    # Build initial passage
    passage = '\n\n'.join(paragraphs[start_idx:end_idx])
    word_count = count_words(passage)

    # Expand if too short and we can add more paragraphs
    while word_count < target_words and end_idx < len(paragraphs):
        end_idx += 1
        passage = '\n\n'.join(paragraphs[start_idx:end_idx])
        word_count = count_words(passage)

        # Stop if we exceed max
        if word_count > MAX_WORD_COUNT:
            end_idx -= 1
            passage = '\n\n'.join(paragraphs[start_idx:end_idx])
            break

    return passage, start_idx, end_idx


def scan_for_broader_context(
    paragraphs: List[str],
    window_size: int = 5
) -> List[Tuple[int, Dict]]:
    """
    Scan text with a sliding window to find keyword-rich sections.

    This allows us to capture passages where keywords are spread across
    multiple paragraphs rather than concentrated in one.

    Args:
        paragraphs: List of all paragraphs
        window_size: Number of paragraphs to scan together

    Returns:
        List of (center_para_idx, match_info) tuples
    """
    matches = []

    for i in range(len(paragraphs)):
        # Create a window of paragraphs
        window_start = max(0, i - window_size // 2)
        window_end = min(len(paragraphs), i + window_size // 2 + 1)
        window_text = ' '.join(paragraphs[window_start:window_end])

        # Check for keywords in window
        match_info = find_all_matches(window_text)

        if match_info['relevance_score'] > 0:
            matches.append((i, match_info))

    return matches


class Passage:
    """
    Represents an extracted passage with metadata.

    Attributes:
        text: Full passage text
        start_para_idx: Starting paragraph index in source
        end_para_idx: Ending paragraph index in source
        word_count: Number of words in passage
        chapter_section: Chapter marker if present
        character_names: Character names extracted from dialogue
        matched_keywords: Keywords that triggered extraction
        context_type: "weather", "humor", or "both"
        relevance_score: Numeric score (0-2)
    """

    def __init__(
        self,
        text: str,
        start_para_idx: int,
        end_para_idx: int,
        paragraphs: List[str],
        match_info: Dict
    ):
        self.text = text
        self.start_para_idx = start_para_idx
        self.end_para_idx = end_para_idx
        self.word_count = count_words(text)
        self.matched_keywords = match_info['matched_keywords']
        self.context_type = match_info['context_type']
        self.relevance_score = match_info['relevance_score']

        # Extract chapter marker (check first paragraph of passage)
        self.chapter_section = None
        for i in range(start_para_idx, min(start_para_idx + 3, end_para_idx)):
            if i < len(paragraphs):
                marker = find_chapter_marker(paragraphs[i])
                if marker:
                    self.chapter_section = marker
                    break

        # Extract character names from passage
        self.character_names = []
        for para in paragraphs[start_para_idx:end_para_idx]:
            self.character_names.extend(extract_character_names(para))
        self.character_names = list(set(self.character_names))[:5]  # Limit to 5

    def overlap_percent(self, other: 'Passage') -> float:
        """
        Calculate overlap percentage with another passage.

        Args:
            other: Another Passage object

        Returns:
            Percentage of overlap (0.0 to 1.0)
        """
        # Calculate intersection of paragraph ranges
        overlap_start = max(self.start_para_idx, other.start_para_idx)
        overlap_end = min(self.end_para_idx, other.end_para_idx)

        if overlap_start >= overlap_end:
            return 0.0  # No overlap

        overlap_size = overlap_end - overlap_start
        self_size = self.end_para_idx - self.start_para_idx
        other_size = other.end_para_idx - other.start_para_idx

        # Return overlap as percentage of smaller passage
        min_size = min(self_size, other_size)
        return overlap_size / min_size if min_size > 0 else 0.0

    def to_dict(self, book_info: Dict) -> Dict:
        """
        Convert passage to dictionary for serialization.

        Args:
            book_info: Book metadata from config

        Returns:
            Dictionary with all passage fields
        """
        return {
            'text': self.text,
            'word_count': self.word_count,
            'chapter_section': self.chapter_section,
            'character_names': self.character_names if self.character_names else None,
            'matched_keywords': self.matched_keywords,
            'context_type': self.context_type,
            'relevance_score': self.relevance_score,
            # Metadata placeholders (filled in by caller)
            'passage_id': None,
            'author_name': book_info['author'],
            'book_title': book_info['title'],
            'book_id': book_info['book_id'],
            'publication_year': book_info['publication_year'],
            'genre_tags': book_info['genre_tags'],
            'source_url': f"https://www.gutenberg.org/ebooks/{book_info['book_id']}",
            'extraction_date': datetime.utcnow().isoformat() + 'Z'
        }


def extract_passages_from_book(
    book_file: Path,
    book_info: Dict,
    max_passages: int = 1000
) -> List[Passage]:
    """
    Extract passages from a single book file.

    Args:
        book_file: Path to book text file
        book_info: Book metadata from config
        max_passages: Maximum passages to extract from this book

    Returns:
        List of Passage objects

    Process:
        1. Load and split text into paragraphs
        2. Use sliding window to find keyword-rich sections
        3. Extract context around matches
        4. Filter by word count
        5. Remove heavily overlapping passages (>50% overlap)
        6. Sort by relevance score
    """
    print(f"\nExtracting from: {book_info['title']}")
    print(f"  File: {book_file.name}")

    # Load book text
    text = book_file.read_text(encoding='utf-8')

    # Split into paragraphs
    paragraphs = split_into_paragraphs(text)
    print(f"  Paragraphs: {len(paragraphs)}")

    # Scan with sliding window for better coverage
    matched_sections = scan_for_broader_context(paragraphs, window_size=5)
    print(f"  Keyword-rich sections found: {len(matched_sections)}")

    # Extract passages around matches
    passages = []

    for para_idx, match_info in matched_sections:
        passage_text, start_idx, end_idx = extract_passage_around_match(
            paragraphs, para_idx
        )

        # Apply word count filter
        word_count = count_words(passage_text)
        if word_count < MIN_WORD_COUNT or word_count > MAX_WORD_COUNT:
            continue

        # Re-check that the passage itself contains keywords
        # (not just the window)
        passage_matches = find_all_matches(passage_text)
        if passage_matches['relevance_score'] == 0:
            continue

        # Create passage object
        passage = Passage(
            passage_text,
            start_idx,
            end_idx,
            paragraphs,
            passage_matches
        )

        passages.append(passage)

    print(f"  Candidate passages: {len(passages)}")

    # Remove heavily overlapping passages (>50% overlap)
    # Keep higher scored ones
    passages.sort(key=lambda p: p.relevance_score, reverse=True)
    filtered = []

    for passage in passages:
        # Check overlap with already selected passages
        has_heavy_overlap = any(
            passage.overlap_percent(p) > OVERLAP_THRESHOLD
            for p in filtered
        )

        if not has_heavy_overlap:
            filtered.append(passage)

    print(f"  After overlap filtering: {len(filtered)}")

    # Limit to max_passages
    selected = filtered[:max_passages]

    print(f"  Selected passages: {len(selected)}")
    print(f"    - Context 'both': {sum(1 for p in selected if p.context_type == 'both')}")
    print(f"    - Context 'weather': {sum(1 for p in selected if p.context_type == 'weather')}")
    print(f"    - Context 'humor': {sum(1 for p in selected if p.context_type == 'humor')}")

    return selected


def main():
    """
    Main execution: extract passages from all configured books.
    """
    parser = argparse.ArgumentParser(
        description='Extract passages from Gutenberg books'
    )
    parser.add_argument(
        '--book-file',
        help='Process single book file (for testing)'
    )
    parser.add_argument(
        '--max-per-book',
        type=int,
        default=1000,
        help='Maximum passages per book'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Literary Passage Extractor")
    print("=" * 60)

    # Ensure NLTK data available
    ensure_nltk_data()

    # Load book configuration
    config_path = CONFIGS_DIR / "gutenberg_books.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    books = config['books']

    # Filter to single book if specified
    if args.book_file:
        books = [b for b in books if f"{b['author_id']}_{b['slug']}.txt" == args.book_file]
        if not books:
            print(f"ERROR: Book file not found in config: {args.book_file}")
            sys.exit(1)

    # Process each book
    all_passages = []
    gutenberg_dir = DATA_RAW / "gutenberg"

    for book_info in books:
        book_filename = f"{book_info['author_id']}_{book_info['slug']}.txt"
        book_file = gutenberg_dir / book_filename

        if not book_file.exists():
            print(f"\nWARNING: Book file not found: {book_filename}")
            continue

        passages = extract_passages_from_book(
            book_file,
            book_info,
            max_passages=args.max_per_book
        )

        all_passages.extend(passages)

    # Generate summary
    print()
    print("=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"Total passages extracted: {len(all_passages)}")
    print(f"  - Context 'both': {sum(1 for p in all_passages if p.context_type == 'both')}")
    print(f"  - Context 'weather': {sum(1 for p in all_passages if p.context_type == 'weather')}")
    print(f"  - Context 'humor': {sum(1 for p in all_passages if p.context_type == 'humor')}")
    print()

    if all_passages:
        word_counts = [p.word_count for p in all_passages]
        print(f"Word count statistics:")
        print(f"  - Min: {min(word_counts)}")
        print(f"  - Max: {max(word_counts)}")
        print(f"  - Average: {sum(word_counts) / len(word_counts):.1f}")

    return all_passages


if __name__ == "__main__":
    main()
