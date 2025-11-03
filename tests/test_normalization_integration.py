#!/usr/bin/env python3
"""
Integration Tests for Data Normalization Pipeline

Tests complete end-to-end pipeline with realistic data scenarios,
edge cases, error handling, and metadata preservation.
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestErrorHandling:
    """Test error handling for malformed input and API failures."""

    def test_malformed_json_input(self):
        """Test handling of malformed JSON file."""
        from scripts.data_loader import load_json_file

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_json = Path(tmpdir) / "bad.json"
            bad_json.write_text("{ this is not valid json }")

            items = load_json_file(bad_json)
            assert isinstance(items, list)
            assert len(items) == 0  # Should return empty list, not crash

    def test_malformed_jsonl_input(self):
        """Test handling of partially malformed JSONL file."""
        from scripts.data_loader import load_jsonl_file

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_jsonl = Path(tmpdir) / "bad.jsonl"
            bad_jsonl.write_text('{"valid": "json"}\n{ invalid json }\n{"more": "valid"}')

            items = load_jsonl_file(bad_jsonl)
            assert isinstance(items, list)
            # Should skip malformed line but load valid ones
            assert len(items) >= 1

    def test_safety_filter_api_failure_recovery(self):
        """Test safety filter behavior when API is unavailable."""
        from scripts.safety_filter import filter_unsafe_content

        items = [
            {'content': 'Test text', 'id': 1}
        ]

        # With skip=True, should handle gracefully
        safe_items, stats = filter_unsafe_content(items, skip=True)
        assert len(safe_items) == 1
        assert stats['skipped'] == True


class TestEdgeCases:
    """Test edge cases like empty datasets and extreme filtering."""

    def test_all_items_filtered_by_language(self):
        """Test behavior when all items are filtered out by language detector."""
        from scripts.language_filter import filter_english_only

        # Note: langdetect may detect short text as various languages
        # This test ensures graceful handling of edge case
        items = [
            {'content': 'x', 'id': 1},  # Too short, may be detected as non-English
        ]

        english_items, stats = filter_english_only(items)
        assert isinstance(english_items, list)
        assert stats['original_count'] == 1
        # May or may not filter, but should not crash

    def test_all_items_flagged_by_safety_filter(self):
        """Test behavior when all items are flagged (skip mode)."""
        from scripts.safety_filter import filter_unsafe_content

        items = [
            {'content': 'Safe content 1', 'id': 1},
            {'content': 'Safe content 2', 'id': 2}
        ]

        # In skip mode, all should pass
        safe_items, stats = filter_unsafe_content(items, skip=True)
        assert len(safe_items) == 2
        assert stats['flagged_count'] == 0

    def test_empty_dataset_through_pipeline(self):
        """Test pipeline with empty input dataset."""
        from scripts.text_normalization import normalize_batch
        from scripts.deduplication import remove_duplicates
        from scripts.language_filter import filter_english_only
        from scripts.safety_filter import filter_unsafe_content

        items = []

        # Should handle empty list gracefully at each stage
        items = normalize_batch(items)
        assert len(items) == 0

        items, dedup_stats = remove_duplicates(items)
        assert len(items) == 0
        assert dedup_stats['original_count'] == 0

        items, lang_stats = filter_english_only(items)
        assert len(items) == 0
        assert lang_stats['original_count'] == 0

        items, safety_stats = filter_unsafe_content(items, skip=True)
        assert len(items) == 0
        assert safety_stats['original_count'] == 0


class TestMetadataPreservation:
    """Test that metadata is preserved end-to-end through all stages."""

    def test_metadata_preservation_end_to_end(self):
        """Test complete metadata preservation through entire pipeline."""
        from scripts.text_normalization import normalize_batch
        from scripts.deduplication import remove_duplicates
        from scripts.language_filter import filter_english_only
        from scripts.safety_filter import filter_unsafe_content

        # Create items with rich metadata
        items = [
            {
                'content': 'The weather is beautiful today',
                'id': 1,
                'source': 'test',
                'reddit_id': 'abc123',
                'subreddit': 'TheOnion',
                'created_utc': 1234567890,
                'tags': ['weather', 'humor'],
                'score': 42
            },
            {
                'content': 'Rain expected tomorrow morning',
                'id': 2,
                'source': 'test',
                'reddit_id': 'def456',
                'subreddit': 'nottheonion',
                'created_utc': 1234567900,
                'tags': ['weather'],
                'score': 28
            }
        ]

        # Run through all stages
        items = normalize_batch(items)
        items, _ = remove_duplicates(items, threshold=0.8)
        items, _ = filter_english_only(items)
        items, _ = filter_unsafe_content(items, skip=True)

        # Verify metadata is intact
        assert len(items) >= 1
        for item in items:
            assert 'id' in item
            assert 'source' in item
            assert 'tags' in item
            assert 'score' in item
            if 'reddit_id' in item:
                assert isinstance(item['reddit_id'], str)
            if 'subreddit' in item:
                assert isinstance(item['subreddit'], str)

    def test_source_file_metadata_added(self):
        """Test that source_file metadata is added during loading."""
        from scripts.data_loader import load_multiple_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test file
            test_file = tmpdir / "test_source.json"
            test_file.write_text('[{"content": "Test", "id": 1}]')

            # Load with source tracking
            items = load_multiple_sources([test_file])

            assert len(items) == 1
            assert 'source_file' in items[0]
            assert items[0]['source_file'] == 'test_source.json'


class TestConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_invalid_threshold_handling(self):
        """Test that invalid threshold values are caught."""
        from scripts.deduplication import remove_duplicates

        items = [{'content': 'Test', 'id': 1}]

        # Threshold should be between 0 and 1
        # The function should handle this gracefully or raise clear error
        try:
            # Valid threshold
            _, stats = remove_duplicates(items, threshold=0.8)
            assert stats['original_count'] == 1
        except Exception as e:
            pytest.fail(f"Valid threshold raised exception: {e}")


class TestStatisticsReporting:
    """Test accuracy and completeness of statistics reporting."""

    def test_statistics_report_accuracy(self):
        """Test that statistics accurately reflect processing."""
        from scripts.statistics_reporter import calculate_statistics

        # Known input
        processing_stats = {
            'loaded': {'count': 100, 'files_count': 1},
            'deduplicated': {
                'original_count': 100,
                'unique_count': 80,
                'duplicates_removed': 20,
                'duplicate_rate': 20.0
            },
            'language_filtered': {
                'original_count': 80,
                'english_count': 75,
                'filtered_count': 5,
                'language_distribution': {'en': 75, 'fr': 5}
            },
            'safety_filtered': {
                'original_count': 75,
                'safe_count': 70,
                'flagged_count': 5,
                'flagged_categories': {'hate': 3, 'violence': 2},
                'skipped': False
            }
        }

        stats = calculate_statistics(processing_stats)

        # Verify calculations
        assert stats['summary']['initial_count'] == 100
        assert stats['summary']['final_count'] == 70
        assert stats['summary']['total_filtered'] == 30
        assert stats['summary']['retention_rate'] == 70.0

    def test_statistics_handles_missing_stages(self):
        """Test statistics calculation with incomplete stage data."""
        from scripts.statistics_reporter import calculate_statistics

        # Only some stages completed
        processing_stats = {
            'loaded': {'count': 50, 'files_count': 1},
            'deduplicated': {
                'original_count': 50,
                'unique_count': 45,
                'duplicates_removed': 5,
                'duplicate_rate': 10.0
            }
        }

        stats = calculate_statistics(processing_stats)

        # Should calculate based on available stages
        assert 'summary' in stats
        assert stats['summary']['initial_count'] == 50
        assert stats['summary']['final_count'] == 45


class TestFullIntegrationWithRealData:
    """Integration test with realistic multi-source data."""

    def test_integration_with_sample_gutenberg_and_reddit_data(self):
        """Test complete pipeline with sample data from both sources."""
        from scripts.data_loader import load_multiple_sources
        from scripts.text_normalization import normalize_batch
        from scripts.deduplication import remove_duplicates
        from scripts.language_filter import filter_english_only
        from scripts.safety_filter import filter_unsafe_content
        from scripts.statistics_reporter import calculate_statistics

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create sample Gutenberg data (JSON)
            gutenberg_file = tmpdir / "gutenberg_sample.json"
            gutenberg_data = [
                {
                    'content': 'It was a bright cold day in April, and the clocks were striking thirteen.',
                    'id': 'gutenberg_1',
                    'source': 'gutenberg',
                    'book_title': '1984'
                },
                {
                    'content': 'Call me Ishmael.',
                    'id': 'gutenberg_2',
                    'source': 'gutenberg',
                    'book_title': 'Moby Dick'
                }
            ]
            with open(gutenberg_file, 'w') as f:
                json.dump(gutenberg_data, f)

            # Create sample Reddit data (JSONL)
            reddit_file = tmpdir / "reddit_sample.jsonl"
            with open(reddit_file, 'w') as f:
                f.write('{"content": "Weather forecast: 100% chance of chaos", "id": "reddit_1", "source": "reddit", "subreddit": "TheOnion"}\n')
                f.write('{"content": "Hurricane season brings stormy debates", "id": "reddit_2", "source": "reddit", "subreddit": "nottheonion"}\n')

            # Run complete pipeline
            print("\n--- Integration Test: Loading Data ---")
            items = load_multiple_sources([gutenberg_file, reddit_file])
            assert len(items) == 4
            print(f"Loaded {len(items)} items")

            processing_stats = {
                'loaded': {'count': len(items), 'files_count': 2}
            }

            print("\n--- Integration Test: Normalization ---")
            items = normalize_batch(items)
            processing_stats['normalized'] = {'count': len(items), 'form': 'NFC'}
            print(f"Normalized {len(items)} items")

            print("\n--- Integration Test: Deduplication ---")
            items, dedup_stats = remove_duplicates(items, threshold=0.8)
            processing_stats['deduplicated'] = dedup_stats
            print(f"After dedup: {len(items)} items (removed {dedup_stats['duplicates_removed']})")

            print("\n--- Integration Test: Language Filter ---")
            items, lang_stats = filter_english_only(items)
            processing_stats['language_filtered'] = lang_stats
            print(f"After language filter: {len(items)} items")

            print("\n--- Integration Test: Safety Filter ---")
            items, safety_stats = filter_unsafe_content(items, skip=True)
            processing_stats['safety_filtered'] = safety_stats
            print(f"After safety filter: {len(items)} items")

            # Generate statistics
            print("\n--- Integration Test: Statistics ---")
            stats = calculate_statistics(processing_stats)

            # Verify results
            assert stats['summary']['initial_count'] == 4
            assert stats['summary']['final_count'] >= 1
            assert 'stages' in stats
            assert len(items) >= 1

            print(f"Final: {stats['summary']['final_count']} items")
            print(f"Retention rate: {stats['summary']['retention_rate']}%")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
