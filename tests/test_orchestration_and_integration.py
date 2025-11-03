#!/usr/bin/env python3
"""
Tests for Pipeline Orchestration and Integration (Task Group 4)

Tests end-to-end pipeline execution and integration workflows.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import json

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import unittest
from instructionalization_orchestrator import (
    validate_environment,
    apply_tags,
    convert_item_to_chat,
    run_pipeline
)
from data_loader import load_jsonl_file


class TestOrchestration(unittest.TestCase):
    """Test orchestration logic."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_environment_validation(self):
        """Test environment validation."""
        # Create test input file
        input_file = self.test_dir / "input.jsonl"
        input_file.write_text('{"test": "data"}\n')

        output_train = self.test_dir / "train.jsonl"
        output_val = self.test_dir / "val.jsonl"

        # Should pass
        is_valid = validate_environment(input_file, output_train, output_val)
        self.assertTrue(is_valid)

    def test_environment_validation_missing_input(self):
        """Test validation fails with missing input."""
        input_file = self.test_dir / "missing.jsonl"
        output_train = self.test_dir / "train.jsonl"
        output_val = self.test_dir / "val.jsonl"

        # Should fail
        is_valid = validate_environment(input_file, output_train, output_val)
        self.assertFalse(is_valid)

    def test_apply_tags_to_item(self):
        """Test tag application to item."""
        item = {
            'text': 'Test passage',
            'author_name': 'Mark Twain',
            'matched_keywords': ['storm', 'funny'],
            'genre_tags': ['humor']
        }

        tagged_item = apply_tags(item)

        self.assertIn('tags', tagged_item)
        self.assertEqual(tagged_item['tags']['persona'], 'twain')
        self.assertEqual(tagged_item['tags']['tone'], 'humorous')
        self.assertIn('weather', tagged_item['tags']['domain'])
        self.assertIn('humor', tagged_item['tags']['domain'])

    def test_convert_item_to_chat(self):
        """Test chat format conversion."""
        item = {
            'text': 'Weather passage content',
            'tags': {
                'persona': 'franklin',
                'tone': 'didactic',
                'domain': ['weather']
            }
        }

        chat_item = convert_item_to_chat(item)

        self.assertIn('messages', chat_item)
        self.assertIn('tags', chat_item)
        self.assertEqual(len(chat_item['messages']), 3)
        self.assertEqual(chat_item['messages'][0]['role'], 'system')

    def test_dry_run_mode(self):
        """Test pipeline in dry-run mode (no file writes)."""
        # Create test input file
        input_file = self.test_dir / "input.jsonl"
        test_data = [
            {
                'text': 'Test passage',
                'author_name': 'Mark Twain',
                'matched_keywords': ['storm'],
                'word_count': 10
            }
        ]

        with open(input_file, 'w') as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')

        output_train = self.test_dir / "train.jsonl"
        output_val = self.test_dir / "val.jsonl"

        # Run in dry-run mode
        success = run_pipeline(
            input_path=input_file,
            output_train=output_train,
            output_val=output_val,
            split_ratio=0.9,
            seed=42,
            dry_run=True
        )

        self.assertTrue(success)
        # Files should not be created
        self.assertFalse(output_train.exists())
        self.assertFalse(output_val.exists())


class TestIntegration(unittest.TestCase):
    """Test end-to-end integration workflows."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_end_to_end_pipeline(self):
        """Test complete pipeline with synthetic data."""
        # Create synthetic input data (20 items)
        input_file = self.test_dir / "input.jsonl"

        test_items = []
        # 10 Twain items
        for i in range(10):
            test_items.append({
                'text': f'Twain passage {i} about weather.',
                'author_name': 'Mark Twain',
                'book_title': 'The Adventures of Tom Sawyer',
                'matched_keywords': ['weather', 'storm'],
                'genre_tags': ['humor', 'satire'],
                'word_count': 50 + i*10
            })

        # 5 Franklin items
        for i in range(5):
            test_items.append({
                'text': f'Franklin passage {i} about weather wisdom.',
                'author_name': 'Benjamin Franklin',
                'book_title': 'Autobiography',
                'matched_keywords': ['weather', 'forecast'],
                'genre_tags': ['biography'],
                'word_count': 60 + i*10
            })

        # 5 Reddit items
        for i in range(5):
            test_items.append({
                'text': f'Satirical weather headline {i}.',
                'source_file': 'reddit_humor_weather.jsonl',
                'subreddit': 'TheOnion',
                'matched_keywords': ['weather', 'funny'],
                'word_count': 40 + i*5
            })

        # Write input file
        with open(input_file, 'w') as f:
            for item in test_items:
                json.dump(item, f)
                f.write('\n')

        output_train = self.test_dir / "train.jsonl"
        output_val = self.test_dir / "val.jsonl"

        # Run pipeline
        success = run_pipeline(
            input_path=input_file,
            output_train=output_train,
            output_val=output_val,
            split_ratio=0.9,
            seed=42,
            dry_run=False
        )

        self.assertTrue(success)

        # Verify output files exist
        self.assertTrue(output_train.exists())
        self.assertTrue(output_val.exists())

        # Load and verify output
        train_items = load_jsonl_file(output_train)
        val_items = load_jsonl_file(output_val)

        # Check counts
        self.assertEqual(len(train_items) + len(val_items), 20)
        self.assertAlmostEqual(len(train_items) / 20, 0.9, delta=0.1)

        # Verify structure of first train item
        first_item = train_items[0]
        self.assertIn('messages', first_item)
        self.assertIn('tags', first_item)
        self.assertIn('persona', first_item['tags'])
        self.assertIn('tone', first_item['tags'])
        self.assertIn('domain', first_item['tags'])

    def test_tag_distribution_balance(self):
        """Test stratification preserves tag distributions."""
        # Create input with known distribution
        input_file = self.test_dir / "input.jsonl"

        test_items = []
        # 60% Twain
        for i in range(60):
            test_items.append({
                'text': f'Twain {i}',
                'author_name': 'Mark Twain',
                'matched_keywords': ['storm'],
                'genre_tags': ['humor'],
                'word_count': 50
            })

        # 40% Franklin
        for i in range(40):
            test_items.append({
                'text': f'Franklin {i}',
                'author_name': 'Benjamin Franklin',
                'matched_keywords': ['weather'],
                'genre_tags': ['biography'],
                'word_count': 50
            })

        # Write input
        with open(input_file, 'w') as f:
            for item in test_items:
                json.dump(item, f)
                f.write('\n')

        output_train = self.test_dir / "train.jsonl"
        output_val = self.test_dir / "val.jsonl"

        # Run pipeline
        success = run_pipeline(
            input_path=input_file,
            output_train=output_train,
            output_val=output_val,
            split_ratio=0.9,
            seed=42,
            dry_run=False
        )

        self.assertTrue(success)

        # Load output
        train_items = load_jsonl_file(output_train)
        val_items = load_jsonl_file(output_val)

        # Count personas in train
        train_twain = sum(1 for item in train_items if item['tags']['persona'] == 'twain')
        train_total = len(train_items)

        # Should maintain ~60% ratio
        train_twain_pct = train_twain / train_total
        self.assertAlmostEqual(train_twain_pct, 0.6, delta=0.05)

    def test_multi_turn_for_long_passages(self):
        """Test multi-turn conversion for long passages."""
        input_file = self.test_dir / "input.jsonl"

        # Create multiple long passages to ensure proper split
        long_text = "This is a long weather passage. " * 150  # ~450 words

        test_items = []
        for i in range(10):  # Create 10 items so split works properly
            test_items.append({
                'text': long_text,
                'author_name': 'Mark Twain',
                'matched_keywords': ['weather'],
                'genre_tags': ['humor'],
                'word_count': 450
            })

        with open(input_file, 'w') as f:
            for item in test_items:
                json.dump(item, f)
                f.write('\n')

        output_train = self.test_dir / "train.jsonl"
        output_val = self.test_dir / "val.jsonl"

        # Run pipeline
        success = run_pipeline(
            input_path=input_file,
            output_train=output_train,
            output_val=output_val,
            split_ratio=0.9,
            seed=42,
            dry_run=False
        )

        self.assertTrue(success)

        # Check if items have multi-turn (5 messages)
        all_items = load_jsonl_file(output_train) + load_jsonl_file(output_val)
        item = all_items[0]

        # Should have 5 messages (system, user, assistant, user, assistant)
        self.assertEqual(len(item['messages']), 5)

    def test_output_files_are_valid_jsonl(self):
        """Test output files are valid JSONL and loadable."""
        input_file = self.test_dir / "input.jsonl"

        test_items = [
            {
                'text': f'Test passage {i}',
                'author_name': 'Mark Twain',
                'matched_keywords': ['storm'],
                'word_count': 50
            }
            for i in range(10)
        ]

        with open(input_file, 'w') as f:
            for item in test_items:
                json.dump(item, f)
                f.write('\n')

        output_train = self.test_dir / "train.jsonl"
        output_val = self.test_dir / "val.jsonl"

        # Run pipeline
        success = run_pipeline(
            input_path=input_file,
            output_train=output_train,
            output_val=output_val,
            split_ratio=0.9,
            seed=42,
            dry_run=False
        )

        self.assertTrue(success)

        # Verify each line is valid JSON
        for output_file in [output_train, output_val]:
            with open(output_file, 'r') as f:
                for line in f:
                    # Should not raise exception
                    loaded = json.loads(line)
                    self.assertIn('messages', loaded)
                    self.assertIn('tags', loaded)


if __name__ == '__main__':
    unittest.main()
