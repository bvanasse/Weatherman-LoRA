#!/usr/bin/env python3
"""
Project Gutenberg Download Script for Weatherman-LoRA

Downloads literary texts from Project Gutenberg with automatic header/footer removal
and caching support. Implements retry logic for network resilience.

Usage:
    python scripts/download_gutenberg.py
    python scripts/download_gutenberg.py --config configs/gutenberg_books.json
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Import project paths
from paths import DATA_RAW, CONFIGS_DIR


# Gutenberg URL template
GUTENBERG_URL_TEMPLATE = "https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]  # Exponential backoff in seconds


def remove_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg header and footer boilerplate.

    Args:
        text: Raw text from Gutenberg download

    Returns:
        Clean book text without header/footer

    Notes:
        - Looks for standard START/END markers
        - Handles variations in marker format
        - Preserves actual book content between markers
    """
    # Common start markers
    start_patterns = [
        r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK .+? \*\*\*',
        r'\*\*\*START OF (THIS|THE) PROJECT GUTENBERG EBOOK .+?\*\*\*',
    ]

    # Common end markers
    end_patterns = [
        r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK .+? \*\*\*',
        r'\*\*\*END OF (THIS|THE) PROJECT GUTENBERG EBOOK .+?\*\*\*',
    ]

    # Try to find start marker
    start_pos = 0
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_pos = match.end()
            break

    # Try to find end marker
    end_pos = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            end_pos = match.start()
            break

    # Extract content between markers
    clean_text = text[start_pos:end_pos].strip()

    # If no markers found, return original (with warning logged)
    if start_pos == 0 and end_pos == len(text):
        print("  WARNING: No Gutenberg markers found, returning full text")
        return text.strip()

    return clean_text


def download_book(
    book_id: int,
    output_path: Path,
    force: bool = False
) -> bool:
    """
    Download a single book from Project Gutenberg.

    Args:
        book_id: Gutenberg book ID number
        output_path: Path where to save the downloaded text
        force: If True, re-download even if file exists

    Returns:
        True if download successful, False otherwise

    Notes:
        - Implements exponential backoff retry logic
        - Skips download if file exists (unless force=True)
        - Removes Gutenberg header/footer automatically
    """
    # Check if file already exists
    if output_path.exists() and not force:
        print(f"  File already exists, skipping: {output_path.name}")
        return True

    # Construct download URL
    url = GUTENBERG_URL_TEMPLATE.format(book_id=book_id)

    # Attempt download with retries
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Downloading from {url} (attempt {attempt + 1}/{MAX_RETRIES})...")

            # Make HTTP request
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Get text content
            text = response.text

            # Remove Gutenberg boilerplate
            clean_text = remove_gutenberg_boilerplate(text)

            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(clean_text, encoding='utf-8')

            print(f"  Successfully downloaded: {output_path.name} ({len(clean_text):,} chars)")
            return True

        except requests.exceptions.RequestException as e:
            print(f"  Download failed: {e}")

            # If not last attempt, wait and retry
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"  Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"  ERROR: Failed to download after {MAX_RETRIES} attempts")
                return False

        except Exception as e:
            print(f"  ERROR: Unexpected error during download: {e}")
            return False

    return False


def load_book_config(config_path: Path) -> List[Dict]:
    """
    Load book metadata from JSON configuration file.

    Args:
        config_path: Path to gutenberg_books.json

    Returns:
        List of book metadata dictionaries

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is malformed
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config.get('books', [])


def main():
    """
    Main execution: download all configured Gutenberg books.

    Process:
        1. Load book configuration
        2. Create output directory
        3. Download each book with retry logic
        4. Generate summary report
    """
    print("=" * 60)
    print("Project Gutenberg Book Downloader")
    print("=" * 60)
    print()

    # Load configuration
    config_path = CONFIGS_DIR / "gutenberg_books.json"
    print(f"Loading configuration from: {config_path}")

    try:
        books = load_book_config(config_path)
        print(f"Found {len(books)} books to download")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)

    # Create output directory
    output_dir = DATA_RAW / "gutenberg"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Download each book
    print("-" * 60)
    print("Downloading books:")
    print("-" * 60)
    print()

    success_count = 0
    failed_books = []

    for book in books:
        author_id = book['author_id']
        slug = book['slug']
        book_id = book['book_id']
        title = book['title']

        print(f"Book: {title} by {book['author']}")
        print(f"  ID: {book_id}")

        # Construct output filename
        output_filename = f"{author_id}_{slug}.txt"
        output_path = output_dir / output_filename

        # Download book
        success = download_book(book_id, output_path)

        if success:
            success_count += 1
        else:
            failed_books.append(title)

        print()

    # Generate summary report
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print()
    print(f"Total books:       {len(books)}")
    print(f"Downloaded:        {success_count}")
    print(f"Failed:            {len(failed_books)}")
    print()

    if failed_books:
        print("Failed downloads:")
        for title in failed_books:
            print(f"  - {title}")
        print()
        print("Status: PARTIAL SUCCESS")
        sys.exit(2)  # Partial success exit code
    else:
        print("Status: SUCCESS")
        print()
        print(f"All books downloaded to: {output_dir}")
        sys.exit(0)


if __name__ == "__main__":
    main()
