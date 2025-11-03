#!/usr/bin/env python3
"""
Literary Corpus Collection - Main Orchestration Script

End-to-end pipeline for collecting literary passages from Project Gutenberg.
Downloads books, extracts passages, and serializes to JSON.

Usage:
    python scripts/collect_literary_corpus.py
    python scripts/collect_literary_corpus.py --max-passages 5000
    python scripts/collect_literary_corpus.py --books twain --output custom_output.json
"""

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# Import project paths
from paths import DATA_RAW, DATA_PROCESSED, CONFIGS_DIR

# Import pipeline modules
import download_gutenberg
import extract_passages


def validate_environment() -> bool:
    """
    Validate that environment is ready for corpus collection.

    Checks:
        - NLTK data is available
        - Config file exists
        - Output directories exist

    Returns:
        True if environment is valid, False otherwise
    """
    print("Validating environment...")

    # Check NLTK data
    try:
        extract_passages.ensure_nltk_data()
        print("  NLTK data: OK")
    except Exception as e:
        print(f"  NLTK data: FAILED - {e}")
        return False

    # Check config file
    config_path = CONFIGS_DIR / "gutenberg_books.json"
    if not config_path.exists():
        print(f"  Config file: FAILED - {config_path} not found")
        return False
    print("  Config file: OK")

    # Ensure directories exist
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print("  Directories: OK")

    return True


def validate_downloads(books: list) -> bool:
    """
    Validate that all books have been downloaded.

    Args:
        books: List of book metadata dictionaries

    Returns:
        True if all books are present, False otherwise
    """
    print("\nValidating downloads...")

    gutenberg_dir = DATA_RAW / "gutenberg"
    missing = []

    for book in books:
        book_filename = f"{book['author_id']}_{book['slug']}.txt"
        book_file = gutenberg_dir / book_filename

        if not book_file.exists():
            missing.append(book_filename)
        else:
            print(f"  Found: {book_filename}")

    if missing:
        print(f"\n  WARNING: Missing {len(missing)} books:")
        for filename in missing:
            print(f"    - {filename}")
        return False

    print("  All books present")
    return True


def run_downloads(force: bool = False) -> bool:
    """
    Run the download stage.

    Args:
        force: If True, re-download even if files exist

    Returns:
        True if downloads successful, False otherwise
    """
    print()
    print("=" * 60)
    print("Stage 1: Download Books from Project Gutenberg")
    print("=" * 60)

    try:
        download_gutenberg.main()
        return True
    except SystemExit as e:
        if e.code == 0:
            return True
        elif e.code == 2:
            print("\nPartial success in downloads")
            return True  # Continue anyway
        else:
            print(f"\nDownload stage failed with exit code {e.code}")
            return False
    except Exception as e:
        print(f"\nDownload stage failed: {e}")
        return False


def run_extraction(books: list, max_per_book: int = 1000) -> tuple:
    """
    Run the passage extraction stage.

    Args:
        books: List of book metadata dictionaries
        max_per_book: Maximum passages to extract per book

    Returns:
        Tuple of (passages, book_map) where book_map maps passages to book info
    """
    print()
    print("=" * 60)
    print("Stage 2: Extract Passages")
    print("=" * 60)

    gutenberg_dir = DATA_RAW / "gutenberg"
    all_passages = []
    passage_to_book = {}  # Map passage to book info

    for book_info in books:
        book_filename = f"{book_info['author_id']}_{book_info['slug']}.txt"
        book_file = gutenberg_dir / book_filename

        if not book_file.exists():
            print(f"\nWARNING: Skipping {book_filename} (not found)")
            continue

        passages = extract_passages.extract_passages_from_book(
            book_file,
            book_info,
            max_passages=max_per_book
        )

        # Track which book each passage came from
        for passage in passages:
            passage_to_book[id(passage)] = book_info

        all_passages.extend(passages)

    return all_passages, passage_to_book


def serialize_passages(
    passages: list,
    passage_to_book: dict,
    output_path: Path
) -> bool:
    """
    Serialize passages to JSON file.

    Args:
        passages: List of Passage objects
        passage_to_book: Dictionary mapping passage IDs to book info
        output_path: Path to output JSON file

    Returns:
        True if successful, False otherwise
    """
    print()
    print("=" * 60)
    print("Stage 3: Serialize to JSON")
    print("=" * 60)

    print(f"\nSerializing {len(passages)} passages to JSON...")

    try:
        # Convert passages to dictionaries
        passage_dicts = []
        passage_counter = {}  # Track sequence numbers per book

        for passage in passages:
            # Get book info for this passage
            book_info = passage_to_book[id(passage)]

            # Get or initialize sequence counter for this book
            book_id = f"{book_info['author_id']}_{book_info['slug']}"
            if book_id not in passage_counter:
                passage_counter[book_id] = 0

            # Convert to dictionary
            passage_dict = passage.to_dict(book_info)

            # Assign unique passage ID
            passage_dict['passage_id'] = f"{book_id}_{passage_counter[book_id]:04d}"
            passage_counter[book_id] += 1

            passage_dicts.append(passage_dict)

        print(f"  Converted {len(passage_dicts)} passages")

        # Calculate statistics
        stats = calculate_statistics(passage_dicts)
        print(f"  Calculated statistics")

        # Create output structure
        output_data = {
            'passages': passage_dicts,
            'metadata': stats
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
        with open(output_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        print(f"  Validation: JSON file valid, contains {len(loaded_data['passages'])} passages")

        return True

    except Exception as e:
        print(f"\nSerialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def calculate_statistics(passages: list) -> dict:
    """Calculate summary statistics for collected passages."""
    total_passages = len(passages)

    # Unique books and authors
    books = list(set(p['book_title'] for p in passages))
    authors = list(set(p['author_name'] for p in passages))

    # Context type distribution
    context_types = Counter(p['context_type'] for p in passages)

    # Keyword distribution
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

    # Passages per book and author
    passages_per_book = Counter(p['book_title'] for p in passages)
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


def generate_report(output_path: Path, output_dir: Path) -> None:
    """Generate a text report of the collection process."""
    print()
    print("=" * 60)
    print("Stage 4: Generate Collection Report")
    print("=" * 60)

    report_path = output_dir / "gutenberg_collection_report.txt"

    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data['metadata']

    # Generate report content
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("Literary Corpus Collection Report")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(f"Collection Date: {metadata['extraction_date']}")
    report_lines.append(f"Total Passages: {metadata['total_passages']}")
    report_lines.append("")

    report_lines.append("Books Processed:")
    for book in metadata['books_processed']:
        count = metadata['passages_per_book'][book]
        report_lines.append(f"  - {book}: {count} passages")
    report_lines.append("")

    report_lines.append("Authors:")
    for author in metadata['authors']:
        count = metadata['passages_per_author'][author]
        report_lines.append(f"  - {author}: {count} passages")
    report_lines.append("")

    report_lines.append("Context Type Distribution:")
    for context_type, count in sorted(metadata['context_type_distribution'].items()):
        percentage = (count / metadata['total_passages']) * 100
        report_lines.append(f"  - {context_type}: {count} ({percentage:.1f}%)")
    report_lines.append("")

    report_lines.append("Top 10 Keywords:")
    for keyword, count in sorted(
        metadata['keyword_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        report_lines.append(f"  - {keyword}: {count} occurrences")
    report_lines.append("")

    stats = metadata['word_count_stats']
    report_lines.append("Word Count Statistics:")
    report_lines.append(f"  - Minimum: {stats['min']} words")
    report_lines.append(f"  - Maximum: {stats['max']} words")
    report_lines.append(f"  - Average: {stats['average']:.1f} words")
    report_lines.append("")

    report_lines.append("Output Files:")
    report_lines.append(f"  - JSON: {output_path}")
    report_lines.append(f"  - Report: {report_path}")
    report_lines.append("")

    # Write report
    report_content = "\n".join(report_lines)
    report_path.write_text(report_content, encoding='utf-8')

    print(f"\nReport saved to: {report_path}")
    print()
    print(report_content)


def main():
    """Main execution: run end-to-end corpus collection pipeline."""
    parser = argparse.ArgumentParser(
        description='Collect literary corpus from Project Gutenberg'
    )
    parser.add_argument(
        '--books',
        help='Filter to specific author (e.g., "twain" or "franklin")'
    )
    parser.add_argument(
        '--max-passages',
        type=int,
        default=5000,
        help='Maximum total passages to collect (default: 5000)'
    )
    parser.add_argument(
        '--max-per-book',
        type=int,
        default=1000,
        help='Maximum passages per book (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Custom output path (default: data/processed/gutenberg_passages.json)'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of books'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download stage (use existing files)'
    )
    args = parser.parse_args()

    # Start timing
    start_time = time.time()

    print("=" * 60)
    print("Literary Corpus Collection Pipeline")
    print("=" * 60)
    print()

    # Validate environment
    if not validate_environment():
        print("\nEnvironment validation failed")
        sys.exit(1)

    # Load configuration
    config_path = CONFIGS_DIR / "gutenberg_books.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    books = config['books']

    # Filter books if requested
    if args.books:
        books = [b for b in books if b['author_id'] == args.books.lower()]
        if not books:
            print(f"ERROR: No books found for author '{args.books}'")
            sys.exit(1)
        print(f"Filtering to {len(books)} books by {args.books}")

    print(f"Processing {len(books)} books")
    print()

    # Stage 1: Download
    if not args.skip_download:
        if not run_downloads(force=args.force_download):
            print("\nDownload stage failed")
            sys.exit(1)

    # Validate downloads
    if not validate_downloads(books):
        print("\nWARNING: Some books are missing, continuing anyway...")

    # Stage 2: Extract
    passages, passage_to_book = run_extraction(books, max_per_book=args.max_per_book)

    if not passages:
        print("\nERROR: No passages extracted")
        sys.exit(1)

    # Limit to max_passages if needed
    if len(passages) > args.max_passages:
        print(f"\nLimiting from {len(passages)} to {args.max_passages} passages")
        # Sort by relevance score and take top N
        passages.sort(key=lambda p: p.relevance_score, reverse=True)
        # Update mapping for limited passages
        limited_passages = passages[:args.max_passages]
        new_mapping = {id(p): passage_to_book[id(p)] for p in limited_passages}
        passages = limited_passages
        passage_to_book = new_mapping

    # Stage 3: Serialize
    output_path = args.output or (DATA_PROCESSED / "gutenberg_passages.json")
    if not serialize_passages(passages, passage_to_book, output_path):
        print("\nSerialization stage failed")
        sys.exit(1)

    # Stage 4: Generate report
    generate_report(output_path, output_path.parent)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print()
    print("=" * 60)
    print("Collection Complete!")
    print("=" * 60)
    print()
    print(f"Total passages collected: {len(passages)}")
    print(f"Output file: {output_path}")
    print(f"Time elapsed: {minutes}m {seconds}s")
    print()
    print("Next steps:")
    print("  1. Process Reddit humor data (Roadmap Item 3)")
    print("  2. Run data normalization pipeline (Roadmap Item 4)")
    print("  3. Convert to training format (Roadmap Item 5)")
    print()

    sys.exit(0)


if __name__ == "__main__":
    main()
