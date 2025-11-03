#!/usr/bin/env python3
"""
Tests for Text Processing Functions

Tests Task Group 2: Text Processing Functions
- Unicode normalization
- MinHash/LSH deduplication
- Language detection
- Safety filtering (mocked)
- Metadata preservation
"""

import os
import sys
import pytest
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.text_normalization import normalize_unicode, normalize_item, normalize_batch
from scripts.deduplication import remove_duplicates, find_duplicates, text_to_shingles
from scripts.language_filter import detect_language, filter_english_only, is_english
from scripts.safety_filter import filter_unsafe_content
from scripts.data_loader import load_json_file, load_jsonl_file, load_multiple_sources


class TestTextNormalization:
    """Test Unicode normalization functions."""

    def test_normalize_unicode_basic(self):
        """Test basic Unicode normalization."""
        text = "Hello world"
        normalized = normalize_unicode(text)
        assert isinstance(normalized, str)
        assert len(normalized) > 0

    def test_normalize_unicode_empty(self):
        """Test normalization of empty string."""
        normalized = normalize_unicode("")
        assert normalized == ""

    def test_normalize_unicode_preserves_text(self):
        """Test that regular ASCII text is preserved."""
        text = "The weather is nice today"
        normalized = normalize_unicode(text)
        assert normalized == text

    def test_normalize_item_preserves_metadata(self):
        """Test that normalization preserves metadata fields."""
        item = {
            'content': 'Test text',
            'id': 123,
            'source': 'test',
            'tags': ['weather', 'humor']
        }
        normalized = normalize_item(item)

        assert normalized['id'] == 123
        assert normalized['source'] == 'test'
        assert normalized['tags'] == ['weather', 'humor']
        assert 'content' in normalized

    def test_normalize_batch(self):
        """Test batch normalization."""
        items = [
            {'content': 'Text 1', 'id': 1},
            {'content': 'Text 2', 'id': 2},
            {'content': 'Text 3', 'id': 3}
        ]
        normalized = normalize_batch(items)

        assert len(normalized) == 3
        assert all('content' in item for item in normalized)
        assert all('id' in item for item in normalized)


class TestDeduplication:
    """Test MinHash/LSH deduplication."""

    def test_text_to_shingles(self):
        """Test conversion of text to character shingles."""
        shingles = text_to_shingles("hello", k=3)
        assert isinstance(shingles, set)
        assert 'hel' in shingles or 'ell' in shingles

    def test_remove_duplicates_exact(self):
        """Test removal of exact duplicates."""
        items = [
            {'content': 'Exact same text', 'id': 1},
            {'content': 'Exact same text', 'id': 2},
            {'content': 'Different text', 'id': 3}
        ]
        unique_items, stats = remove_duplicates(items, threshold=0.8)

        assert stats['duplicates_removed'] >= 1
        assert stats['unique_count'] <= stats['original_count']
        assert stats['original_count'] == 3

    def test_remove_duplicates_preserves_metadata(self):
        """Test that deduplication preserves metadata."""
        items = [
            {'content': 'Unique text 1', 'id': 1, 'source': 'test'},
            {'content': 'Unique text 2', 'id': 2, 'source': 'test'}
        ]
        unique_items, stats = remove_duplicates(items, threshold=0.8)

        for item in unique_items:
            assert 'id' in item
            assert 'source' in item

    def test_remove_duplicates_threshold(self):
        """Test deduplication with different thresholds."""
        items = [
            {'content': 'The weather is nice', 'id': 1},
            {'content': 'The weather is nice!', 'id': 2},
            {'content': 'Completely different', 'id': 3}
        ]

        # High threshold (0.9) - more strict, fewer duplicates
        unique_strict, stats_strict = remove_duplicates(items, threshold=0.9)

        # Lower threshold (0.7) - less strict, more duplicates
        unique_lenient, stats_lenient = remove_duplicates(items, threshold=0.7)

        # Should have at least the completely different item
        assert len(unique_strict) >= 1
        assert len(unique_lenient) >= 1

    def test_find_duplicates_empty(self):
        """Test deduplication with empty list."""
        items, stats = remove_duplicates([], threshold=0.8)
        assert len(items) == 0
        assert stats['original_count'] == 0


class TestLanguageFilter:
    """Test language detection and filtering."""

    def test_detect_language_english(self):
        """Test detection of English text."""
        text = "Hello world, how are you today?"
        lang = detect_language(text)
        assert lang == 'en'

    def test_is_english(self):
        """Test English check function."""
        assert is_english("Hello world") == True
        assert is_english("The weather is nice") == True

    def test_filter_english_only(self):
        """Test filtering for English content."""
        items = [
            {'content': 'Hello world', 'id': 1},
            {'content': 'The weather is nice today', 'id': 2},
            {'content': 'Another English text', 'id': 3}
        ]
        english_items, stats = filter_english_only(items)

        assert stats['original_count'] == 3
        assert stats['english_count'] >= 1
        assert 'language_distribution' in stats

    def test_filter_english_preserves_metadata(self):
        """Test that language filtering preserves metadata."""
        items = [
            {'content': 'English text', 'id': 1, 'source': 'test', 'tags': ['a']}
        ]
        filtered, stats = filter_english_only(items)

        if len(filtered) > 0:
            assert 'id' in filtered[0]
            assert 'source' in filtered[0]
            assert 'tags' in filtered[0]


class TestSafetyFilter:
    """Test safety filtering (mocked - no API calls)."""

    def test_filter_unsafe_skip_mode(self):
        """Test safety filter in skip mode (no API calls)."""
        items = [
            {'content': 'Safe text', 'id': 1},
            {'content': 'Another safe text', 'id': 2}
        ]
        safe_items, stats = filter_unsafe_content(items, skip=True)

        assert stats['original_count'] == 2
        assert stats['safe_count'] == 2
        assert stats['flagged_count'] == 0
        assert stats.get('skipped') == True

    def test_filter_unsafe_preserves_metadata(self):
        """Test that safety filtering preserves metadata."""
        items = [
            {'content': 'Safe text', 'id': 1, 'source': 'test'}
        ]
        safe_items, stats = filter_unsafe_content(items, skip=True)

        assert len(safe_items) == 1
        assert 'id' in safe_items[0]
        assert 'source' in safe_items[0]

    def test_filter_unsafe_empty_list(self):
        """Test safety filter with empty list."""
        safe_items, stats = filter_unsafe_content([], skip=True)
        assert len(safe_items) == 0
        assert stats['original_count'] == 0


class TestDataLoader:
    """Test data loading functions."""

    def test_load_json_file_missing(self):
        """Test loading non-existent JSON file."""
        items = load_json_file(Path("/nonexistent/file.json"))
        assert isinstance(items, list)
        assert len(items) == 0

    def test_load_jsonl_file_missing(self):
        """Test loading non-existent JSONL file."""
        items = load_jsonl_file(Path("/nonexistent/file.jsonl"))
        assert isinstance(items, list)
        assert len(items) == 0

    def test_load_multiple_sources_empty(self):
        """Test loading from empty source list."""
        items = load_multiple_sources([])
        assert isinstance(items, list)
        assert len(items) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
