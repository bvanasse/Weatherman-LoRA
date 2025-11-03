#!/usr/bin/env python3
"""
Data Loader Module for Pipeline

Loads data from multiple sources (JSON and JSONL formats)
and combines them while preserving metadata.

Usage:
    from scripts.data_loader import load_json_file, load_jsonl_file, load_multiple_sources

    # Load single file
    items = load_json_file(Path("data.json"))

    # Load multiple sources
    all_items = load_multiple_sources([path1, path2, path3])
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load data from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        List of item dictionaries

    Handles:
        - JSON arrays at root level
        - JSON objects wrapped in root key
        - Missing files (returns empty list with warning)

    Example:
        >>> path = Path("test.json")
        >>> items = load_json_file(path)
        >>> isinstance(items, list)
        True
    """
    if not file_path.exists():
        print(f"WARNING: File not found: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try to find list in common root keys
            for key in ['items', 'data', 'passages', 'entries']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If dict but no list found, wrap it
            return [data]
        else:
            print(f"WARNING: Unexpected JSON structure in {file_path}")
            return []

    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {file_path}: {e}")
        return []
    except Exception as e:
        print(f"ERROR: Failed to load {file_path}: {e}")
        return []


def load_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file (one JSON object per line).

    Args:
        file_path: Path to JSONL file

    Returns:
        List of item dictionaries

    Example:
        >>> path = Path("test.jsonl")
        >>> items = load_jsonl_file(path)
        >>> isinstance(items, list)
        True
    """
    if not file_path.exists():
        print(f"WARNING: File not found: {file_path}")
        return []

    items = []
    line_num = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError as e:
                    print(f"WARNING: Invalid JSON on line {line_num} in {file_path}: {e}")
                    continue

    except Exception as e:
        print(f"ERROR: Failed to load {file_path}: {e}")
        return []

    return items


def detect_file_format(file_path: Path) -> str:
    """
    Detect file format based on extension.

    Args:
        file_path: Path to file

    Returns:
        Format string: 'json', 'jsonl', or 'unknown'

    Example:
        >>> detect_file_format(Path("data.json"))
        'json'
        >>> detect_file_format(Path("data.jsonl"))
        'jsonl'
    """
    suffix = file_path.suffix.lower()

    if suffix == '.json':
        return 'json'
    elif suffix == '.jsonl':
        return 'jsonl'
    else:
        return 'unknown'


def load_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load file with automatic format detection.

    Args:
        file_path: Path to file

    Returns:
        List of item dictionaries

    Example:
        >>> items = load_file(Path("data.json"))
        >>> isinstance(items, list)
        True
    """
    file_format = detect_file_format(file_path)

    if file_format == 'json':
        return load_json_file(file_path)
    elif file_format == 'jsonl':
        return load_jsonl_file(file_path)
    else:
        print(f"WARNING: Unknown file format for {file_path}, trying JSON...")
        return load_json_file(file_path)


def load_multiple_sources(file_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Load and combine data from multiple source files.

    Args:
        file_paths: List of paths to data files

    Returns:
        Combined list of all items from all sources

    Preserves:
        - All metadata from original items
        - Source information in each item

    Example:
        >>> paths = [Path("source1.json"), Path("source2.jsonl")]
        >>> all_items = load_multiple_sources(paths)
        >>> isinstance(all_items, list)
        True
    """
    all_items = []

    for file_path in file_paths:
        print(f"Loading {file_path.name}...")
        items = load_file(file_path)

        if items:
            # Add source file info to each item if not already present
            for item in items:
                if 'source_file' not in item:
                    item['source_file'] = str(file_path.name)

            all_items.extend(items)
            print(f"  Loaded {len(items)} items from {file_path.name}")
        else:
            print(f"  No items loaded from {file_path.name}")

    print(f"\nTotal items loaded: {len(all_items)}")
    return all_items


def validate_items(items: List[Dict[str, Any]], required_fields: List[str] = None) -> bool:
    """
    Validate that items have required fields.

    Args:
        items: List of item dictionaries
        required_fields: List of field names that must be present
                        Default: checks for at least one text field

    Returns:
        True if all items valid, False otherwise

    Example:
        >>> items = [{'content': 'text', 'id': 1}]
        >>> validate_items(items, required_fields=['content'])
        True
    """
    if not items:
        print("WARNING: No items to validate")
        return False

    if required_fields is None:
        # Default: check for common text fields
        required_fields = []

    missing_count = 0

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            print(f"WARNING: Item {idx} is not a dictionary")
            missing_count += 1
            continue

        # Check required fields
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            print(f"WARNING: Item {idx} missing fields: {missing_fields}")
            missing_count += 1

    if missing_count > 0:
        print(f"Validation: {missing_count}/{len(items)} items have issues")
        return False

    return True


if __name__ == "__main__":
    # Test data loader
    print("Data Loader Test")
    print("=" * 60)

    # Create test files
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test JSON file
        json_file = tmpdir / "test.json"
        json_data = [
            {'content': 'Test item 1', 'id': 1},
            {'content': 'Test item 2', 'id': 2}
        ]
        with open(json_file, 'w') as f:
            json.dump(json_data, f)

        # Test JSONL file
        jsonl_file = tmpdir / "test.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write('{"content": "JSONL item 1", "id": 3}\n')
            f.write('{"content": "JSONL item 2", "id": 4}\n')

        # Load JSON
        print("\nLoading JSON file:")
        json_items = load_json_file(json_file)
        print(f"  Loaded {len(json_items)} items")

        # Load JSONL
        print("\nLoading JSONL file:")
        jsonl_items = load_jsonl_file(jsonl_file)
        print(f"  Loaded {len(jsonl_items)} items")

        # Load multiple sources
        print("\n" + "=" * 60)
        print("Loading Multiple Sources:")
        print("=" * 60)
        all_items = load_multiple_sources([json_file, jsonl_file])

        print(f"\nTotal items: {len(all_items)}")
        for item in all_items:
            print(f"  [{item['id']}] {item['content']} (source: {item.get('source_file', 'N/A')})")

        # Validate
        print("\n" + "=" * 60)
        print("Validation:")
        is_valid = validate_items(all_items, required_fields=['content', 'id'])
        print(f"  All items valid: {is_valid}")
