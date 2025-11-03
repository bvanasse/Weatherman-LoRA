#!/usr/bin/env python3
"""
Tests for Stratified Splitting and File Generation (Task Group 3)

Tests stratified sampling, atomic file writes, and statistics generation.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import json

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import unittest
from stratified_splitter import stratified_split, create_stratification_key, get_split_statistics
from output_writer import write_jsonl_output, validate_jsonl_output
from instructionalization_stats import calculate_instructionalization_stats


class TestStratifiedSplitter(unittest.TestCase):
    """Test stratified sampling maintains tag distributions."""

    def test_stratification_key_creation(self):
        """Test stratification key combines persona-tone-domain."""
        item = {
            'tags': {
                'persona': 'twain',
                'tone': 'humorous',
                'domain': ['weather', 'humor']
            }
        }
        key = create_stratification_key(item)
        self.assertEqual(key, 'twain-humorous-weather')

    def test_split_ratio(self):
        """Test 90/10 split ratio."""
        items = [
            {'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather']}}
            for _ in range(100)
        ]

        train, val = stratified_split(items, ratio=0.9, seed=42)

        self.assertEqual(len(train), 90)
        self.assertEqual(len(val), 10)

    def test_stratification_maintains_distribution(self):
        """Test stratified sampling maintains tag distributions."""
        # Create items with two strata (60/40 split)
        items = []
        for i in range(60):
            items.append({
                'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather']}
            })
        for i in range(40):
            items.append({
                'tags': {'persona': 'franklin', 'tone': 'didactic', 'domain': ['weather']}
            })

        train, val = stratified_split(items, ratio=0.9, seed=42)

        # Count personas in train
        train_twain = sum(1 for item in train if item['tags']['persona'] == 'twain')
        train_franklin = sum(1 for item in train if item['tags']['persona'] == 'franklin')

        # Should maintain roughly 60/40 ratio
        train_twain_pct = train_twain / len(train)
        self.assertAlmostEqual(train_twain_pct, 0.6, delta=0.05)

    def test_random_shuffle(self):
        """Test random shuffle removes temporal bias."""
        items = [
            {'id': i, 'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather']}}
            for i in range(100)
        ]

        train, val = stratified_split(items, ratio=0.9, seed=42)

        # Check that items are not in sequential order
        train_ids = [item['id'] for item in train[:10]]
        is_sequential = train_ids == list(range(10))
        self.assertFalse(is_sequential, "Items should not be in sequential order")

    def test_split_statistics(self):
        """Test split statistics calculation."""
        train = [{'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather']}}] * 90
        val = [{'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather']}}] * 10

        stats = get_split_statistics(train, val)

        self.assertEqual(stats['train_count'], 90)
        self.assertEqual(stats['val_count'], 10)
        self.assertEqual(stats['total_count'], 100)
        self.assertAlmostEqual(stats['split_ratio'], 0.9, places=2)


class TestOutputWriter(unittest.TestCase):
    """Test atomic file writes and validation."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_atomic_write(self):
        """Test atomic file write operation."""
        items = [
            {
                'messages': [
                    {'role': 'system', 'content': 'System'},
                    {'role': 'user', 'content': 'User'},
                    {'role': 'assistant', 'content': 'Assistant'}
                ],
                'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather']}
            }
        ]

        output_file = self.test_dir / "test_output.jsonl"
        write_jsonl_output(items, output_file)

        # Verify file exists
        self.assertTrue(output_file.exists())

        # Verify content
        with open(output_file, 'r') as f:
            line = f.readline()
            loaded = json.loads(line)
            self.assertEqual(loaded['tags']['persona'], 'twain')

    def test_output_validation(self):
        """Test output validation."""
        items = [
            {
                'messages': [
                    {'role': 'system', 'content': 'System'},
                    {'role': 'user', 'content': 'User'},
                    {'role': 'assistant', 'content': 'Assistant'}
                ],
                'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather']}
            },
            {
                'messages': [
                    {'role': 'system', 'content': 'System 2'},
                    {'role': 'user', 'content': 'User 2'},
                    {'role': 'assistant', 'content': 'Assistant 2'}
                ],
                'tags': {'persona': 'franklin', 'tone': 'didactic', 'domain': ['weather']}
            }
        ]

        output_file = self.test_dir / "test_output.jsonl"
        write_jsonl_output(items, output_file)

        # Validate
        is_valid = validate_jsonl_output(output_file, expected_count=2)
        self.assertTrue(is_valid)

    def test_jsonl_parsing(self):
        """Test JSONL format correctness."""
        items = [
            {
                'messages': [
                    {'role': 'system', 'content': 'System'},
                    {'role': 'user', 'content': 'User'},
                    {'role': 'assistant', 'content': 'Assistant'}
                ],
                'tags': {'persona': 'neutral', 'tone': 'humorous', 'domain': ['weather']}
            }
        ] * 5

        output_file = self.test_dir / "test_output.jsonl"
        write_jsonl_output(items, output_file)

        # Read back and verify each line is valid JSON
        with open(output_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 5)

            for line in lines:
                # Should not raise exception
                loaded = json.loads(line)
                self.assertIn('messages', loaded)
                self.assertIn('tags', loaded)


class TestInstructionalizationStats(unittest.TestCase):
    """Test statistics report generation."""

    def test_statistics_calculation(self):
        """Test comprehensive statistics calculation."""
        train = [
            {
                'messages': [
                    {'role': 'system', 'content': 'System'},
                    {'role': 'user', 'content': 'User'},
                    {'role': 'assistant', 'content': 'A' * 100}
                ],
                'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather', 'humor']}
            }
        ] * 90

        val = [
            {
                'messages': [
                    {'role': 'system', 'content': 'System'},
                    {'role': 'user', 'content': 'User'},
                    {'role': 'assistant', 'content': 'B' * 100}
                ],
                'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather', 'humor']}
            }
        ] * 10

        stats = calculate_instructionalization_stats(train, val)

        # Check basic structure
        self.assertIn('split', stats)
        self.assertIn('persona_distribution', stats)
        self.assertIn('tone_distribution', stats)
        self.assertIn('domain_distribution', stats)
        self.assertIn('message_format', stats)
        self.assertIn('stratification_quality', stats)

        # Check counts
        self.assertEqual(stats['split']['train_count'], 90)
        self.assertEqual(stats['split']['val_count'], 10)

    def test_tag_distribution_counting(self):
        """Test tag distribution counting."""
        train = [
            {'messages': [], 'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather']}},
            {'messages': [], 'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather']}},
            {'messages': [], 'tags': {'persona': 'franklin', 'tone': 'didactic', 'domain': ['weather']}}
        ]

        val = [
            {'messages': [], 'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather']}}
        ]

        stats = calculate_instructionalization_stats(train, val)

        # Check persona counts
        self.assertEqual(stats['persona_distribution']['train']['twain'], 2)
        self.assertEqual(stats['persona_distribution']['train']['franklin'], 1)

    def test_multi_turn_detection(self):
        """Test multi-turn conversation detection."""
        train = [
            {
                'messages': [
                    {'role': 'system', 'content': 'S'},
                    {'role': 'user', 'content': 'U'},
                    {'role': 'assistant', 'content': 'A'}
                ],  # Single-turn (3 messages)
                'tags': {}
            },
            {
                'messages': [
                    {'role': 'system', 'content': 'S'},
                    {'role': 'user', 'content': 'U1'},
                    {'role': 'assistant', 'content': 'A1'},
                    {'role': 'user', 'content': 'U2'},
                    {'role': 'assistant', 'content': 'A2'}
                ],  # Multi-turn (5 messages)
                'tags': {}
            }
        ]

        val = []

        stats = calculate_instructionalization_stats(train, val)

        self.assertEqual(stats['message_format']['train_single_turn'], 1)
        self.assertEqual(stats['message_format']['train_multi_turn'], 1)

    def test_stratification_quality_metrics(self):
        """Test stratification quality calculation."""
        # Perfect balance
        train = [
            {'messages': [], 'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather']}}
        ] * 90

        val = [
            {'messages': [], 'tags': {'persona': 'twain', 'tone': 'humorous', 'domain': ['weather']}}
        ] * 10

        stats = calculate_instructionalization_stats(train, val)

        # Should have high similarity (perfect distribution match)
        self.assertGreater(stats['stratification_quality']['average_similarity'], 0.95)


if __name__ == '__main__':
    unittest.main()
