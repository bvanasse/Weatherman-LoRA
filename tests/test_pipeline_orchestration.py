#!/usr/bin/env python3
"""
Tests for Pipeline Orchestration

Tests Task Group 3: Pipeline Orchestration and Statistics
- End-to-end pipeline execution
- Statistics report generation
- Idempotency
- Atomic file writes
- Output validation
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

from scripts.statistics_reporter import calculate_statistics, write_json_report, write_markdown_report
from scripts.normalization_pipeline_orchestrator import (
    validate_environment, write_output_atomic, validate_output
)


class TestStatisticsReporter:
    """Test statistics calculation and reporting."""

    def test_calculate_statistics_basic(self):
        """Test basic statistics calculation."""
        processing_stats = {
            'loaded': {'count': 100, 'files_count': 2},
            'deduplicated': {
                'original_count': 100,
                'unique_count': 90,
                'duplicates_removed': 10,
                'duplicate_rate': 10.0
            }
        }

        stats = calculate_statistics(processing_stats)

        assert 'timestamp' in stats
        assert 'pipeline_version' in stats
        assert 'stages' in stats
        assert 'summary' in stats
        assert stats['summary']['initial_count'] == 100
        assert stats['summary']['final_count'] == 90

    def test_calculate_statistics_all_stages(self):
        """Test statistics with all pipeline stages."""
        processing_stats = {
            'loaded': {'count': 1000, 'files_count': 2},
            'normalized': {'count': 1000, 'form': 'NFC'},
            'deduplicated': {
                'original_count': 1000,
                'unique_count': 850,
                'duplicates_removed': 150,
                'duplicate_rate': 15.0
            },
            'language_filtered': {
                'original_count': 850,
                'english_count': 800,
                'filtered_count': 50,
                'language_distribution': {'en': 800, 'fr': 50}
            },
            'safety_filtered': {
                'original_count': 800,
                'safe_count': 780,
                'flagged_count': 20,
                'flagged_categories': {},
                'skipped': False
            }
        }

        stats = calculate_statistics(processing_stats)

        assert stats['summary']['initial_count'] == 1000
        assert stats['summary']['final_count'] == 780
        assert stats['summary']['total_filtered'] == 220
        assert 'retention_rate' in stats['summary']

    def test_write_json_report(self):
        """Test writing JSON statistics report."""
        stats = {
            'timestamp': '2025-11-02T12:00:00',
            'pipeline_version': '1.0',
            'summary': {
                'initial_count': 100,
                'final_count': 90
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "stats.json"
            write_json_report(stats, output_path)

            assert output_path.exists()
            with open(output_path, 'r') as f:
                loaded = json.load(f)
                assert loaded['pipeline_version'] == '1.0'
                assert loaded['summary']['initial_count'] == 100

    def test_write_markdown_report(self):
        """Test writing Markdown statistics report."""
        stats = {
            'timestamp': '2025-11-02T12:00:00',
            'pipeline_version': '1.0',
            'summary': {
                'initial_count': 100,
                'final_count': 90,
                'total_filtered': 10,
                'retention_rate': 90.0
            },
            'stages': {
                'deduplication': {
                    'original_count': 100,
                    'unique_count': 90,
                    'duplicates_removed': 10,
                    'duplicate_rate': 10.0
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "stats.md"
            write_markdown_report(stats, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert '# Data Normalization Pipeline Statistics' in content
            assert 'Summary' in content
            assert 'Pipeline Stages' in content


class TestPipelineOrchestration:
    """Test pipeline orchestration functions."""

    def test_write_output_atomic(self):
        """Test atomic file writing with temp file + rename."""
        items = [
            {'content': 'Text 1', 'id': 1},
            {'content': 'Text 2', 'id': 2},
            {'content': 'Text 3', 'id': 3}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            write_output_atomic(items, output_path)

            assert output_path.exists()

            # Validate content
            with open(output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3
                for line in lines:
                    item = json.loads(line)
                    assert 'content' in item
                    assert 'id' in item

    def test_validate_output_success(self):
        """Test output validation with correct data."""
        items = [
            {'content': 'Text 1', 'id': 1},
            {'content': 'Text 2', 'id': 2}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            write_output_atomic(items, output_path)

            is_valid = validate_output(output_path, expected_count=2)
            assert is_valid == True

    def test_validate_output_wrong_count(self):
        """Test output validation with count mismatch."""
        items = [{'content': 'Text 1', 'id': 1}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            write_output_atomic(items, output_path)

            # Still returns True but with warning
            is_valid = validate_output(output_path, expected_count=2)
            assert is_valid == True  # Doesn't fail, just warns

    def test_end_to_end_pipeline_with_sample_data(self):
        """Test end-to-end pipeline with small sample dataset."""
        # Create sample data files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create sample input
            input_file = tmpdir / "input.json"
            sample_data = [
                {'content': 'The weather is nice today', 'id': 1, 'source': 'test'},
                {'content': 'The weather is nice today', 'id': 2, 'source': 'test'},  # Duplicate
                {'content': 'Rain expected tomorrow', 'id': 3, 'source': 'test'},
                {'content': 'Sunny skies ahead', 'id': 4, 'source': 'test'}
            ]
            with open(input_file, 'w') as f:
                json.dump(sample_data, f)

            # Import pipeline components
            from scripts.text_normalization import normalize_batch
            from scripts.deduplication import remove_duplicates
            from scripts.language_filter import filter_english_only
            from scripts.safety_filter import filter_unsafe_content
            from scripts.data_loader import load_json_file

            # Load data
            items = load_json_file(input_file)
            assert len(items) == 4

            # Normalize
            items = normalize_batch(items)
            assert len(items) == 4

            # Deduplicate
            items, dedup_stats = remove_duplicates(items, threshold=0.8)
            assert dedup_stats['duplicates_removed'] >= 1
            assert len(items) <= 4

            # Language filter
            items, lang_stats = filter_english_only(items)
            assert lang_stats['english_count'] >= 1

            # Safety filter (skip mode)
            items, safety_stats = filter_unsafe_content(items, skip=True)
            assert safety_stats['skipped'] == True

            # Final check
            assert len(items) >= 1


class TestIdempotency:
    """Test pipeline idempotency."""

    def test_atomic_write_prevents_corruption(self):
        """Test that atomic writes prevent file corruption."""
        items = [{'content': 'Test', 'id': 1}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"

            # First write
            write_output_atomic(items, output_path)
            first_content = output_path.read_text()

            # Second write (should replace cleanly)
            write_output_atomic(items, output_path)
            second_content = output_path.read_text()

            assert first_content == second_content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
