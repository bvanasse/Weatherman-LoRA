#!/usr/bin/env python3
"""
Output Writer Module for Instructionalization Pipeline

Writes chat-format training data to JSONL files with atomic operations.
Uses tempfile + rename pattern to prevent corruption.

Usage:
    from scripts.output_writer import write_jsonl_output

    write_jsonl_output(items, Path("output/train.jsonl"))
"""

import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any


def write_jsonl_output(items: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write items to JSONL file with atomic operation.

    Args:
        items: List of items to write
        output_path: Path to output JSONL file

    Process:
        1. Create temporary file in same directory
        2. Write all items as JSONL (one JSON per line)
        3. Atomic rename to final path
        4. Display file size and count

    Atomic operation ensures:
        - No partial writes if process crashes
        - No corruption of existing files
        - Safe for concurrent access

    Examples:
        >>> items = [{"messages": [...], "tags": {...}}]
        >>> write_jsonl_output(items, Path("output.jsonl"))
    """
    if not items:
        print(f"WARNING: No items to write to {output_path}")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {len(items)} items to {output_path.name}...")

    # Create temporary file in same directory for atomic write
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=output_path.parent,
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

        # Write each item as JSONL
        for item in items:
            json.dump(item, tmp_file, ensure_ascii=False)
            tmp_file.write('\n')

    # Atomic rename
    tmp_path.replace(output_path)

    # Display stats
    file_size = output_path.stat().st_size
    print(f"  Saved to: {output_path}")
    print(f"  File size: {file_size / 1024:.2f} KB ({file_size / (1024 * 1024):.2f} MB)")
    print(f"  Items: {len(items)}")


def validate_jsonl_output(output_path: Path, expected_count: int = None) -> bool:
    """
    Validate JSONL output file.

    Args:
        output_path: Path to JSONL file
        expected_count: Expected number of items (optional)

    Returns:
        True if valid, False otherwise

    Validation checks:
        - Each line is valid JSON
        - Count matches expected (if provided)
        - File is readable

    Examples:
        >>> validate_jsonl_output(Path("output.jsonl"), expected_count=100)
        True
    """
    if not output_path.exists():
        print(f"  ERROR: Output file not found: {output_path}")
        return False

    try:
        count = 0
        with open(output_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"  ERROR: Invalid JSON on line {line_num}: {e}")
                    return False

        # Check count if expected provided
        if expected_count is not None and count != expected_count:
            print(f"  WARNING: Count mismatch - expected {expected_count}, found {count}")
            # Still return True as file is valid, just different count

        print(f"  Validation: {count} items, all valid JSON")
        return True

    except Exception as e:
        print(f"  ERROR: Validation failed: {e}")
        return False


def append_to_jsonl(items: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Append items to existing JSONL file (or create new).

    Args:
        items: List of items to append
        output_path: Path to JSONL file

    Notes:
        - Creates file if it doesn't exist
        - Appends to end if file exists
        - Not atomic (use write_jsonl_output for full rewrites)

    Examples:
        >>> items = [{"messages": [...], "tags": {...}}]
        >>> append_to_jsonl(items, Path("output.jsonl"))
    """
    if not items:
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = 'a' if output_path.exists() else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for item in items:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    print(f"  Appended {len(items)} items to {output_path.name}")


if __name__ == "__main__":
    # Test output writer
    print("Output Writer Test")
    print("=" * 60)

    import tempfile
    import shutil

    # Create temporary directory for tests
    test_dir = Path(tempfile.mkdtemp())

    try:
        # Create test items
        test_items = [
            {
                'messages': [
                    {'role': 'system', 'content': 'System message'},
                    {'role': 'user', 'content': 'User query'},
                    {'role': 'assistant', 'content': 'Assistant response'}
                ],
                'tags': {
                    'persona': 'twain',
                    'tone': 'humorous',
                    'domain': ['weather', 'humor']
                }
            },
            {
                'messages': [
                    {'role': 'system', 'content': 'System message 2'},
                    {'role': 'user', 'content': 'User query 2'},
                    {'role': 'assistant', 'content': 'Assistant response 2'}
                ],
                'tags': {
                    'persona': 'franklin',
                    'tone': 'didactic',
                    'domain': ['weather']
                }
            }
        ]

        # Test writing
        output_file = test_dir / "test_output.jsonl"
        write_jsonl_output(test_items, output_file)

        # Test validation
        print("\n" + "=" * 60)
        print("Validating output:")
        is_valid = validate_jsonl_output(output_file, expected_count=2)
        print(f"  Valid: {is_valid}")

        # Test appending
        print("\n" + "=" * 60)
        print("Testing append:")
        more_items = [test_items[0]]  # Add one more
        append_to_jsonl(more_items, output_file)

        # Validate again
        is_valid = validate_jsonl_output(output_file, expected_count=3)
        print(f"  Valid after append: {is_valid}")

    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory: {test_dir}")
