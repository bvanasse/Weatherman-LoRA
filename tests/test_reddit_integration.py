#!/usr/bin/env python3
"""
Tests for Reddit Humor Dataset Processing - Task Group 5
Integration tests for full pipeline flow and edge cases
"""

import sys
from pathlib import Path
import unittest
import pandas as pd
import tempfile
import json

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from reddit_pipeline_orchestrator import run_pipeline
from reddit_csv_processor import load_reddit_csv
from reddit_text_processing import clean_reddit_text
from reddit_jsonl_converter import validate_jsonl_output


class TestFullPipelineIntegration(unittest.TestCase):
    """Test complete pipeline with realistic data"""

    def setUp(self):
        """Create test CSV with realistic Reddit data"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.csv_file = self.test_dir / "reddit_test.csv"
        self.output_file = self.test_dir / "output.jsonl"

        # Create realistic test data
        data = {
            'title': [
                'Storm warning: Heavy rain expected tonight',
                'Hurricane season forecast looks grim',
                'Political climate worsens as temperatures rise',
                'Weather service predicts 100% chance of confusion',
                'Snow day cancelled due to lack of snow',
                '[removed]',  # Should be filtered
                'Tech',  # Too short, should be filtered
                'Scientists baffled by unusual weather patterns in Washington',
            ],
            'id': ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'],
            'created_utc': [1545089481] * 8,
            'url': [f'http://reddit.com/test{i}' for i in range(8)],
            'num_comments': [50, 42, 38, 35, 30, 5, 2, 20],
            'subreddit': ['TheOnion', 'nottheonion', 'TheOnion', 'nottheonion',
                         'TheOnion', 'TheOnion', 'nottheonion', 'nottheonion'],
            'timestamp': ['2018-12-17 18:31:21'] * 8
        }

        df = pd.DataFrame(data)
        df.to_csv(self.csv_file, index=False)

    def test_end_to_end_processing(self):
        """Test that full pipeline processes data correctly"""
        result = run_pipeline(
            csv_paths=[self.csv_file],
            output_path=self.output_file,
            max_samples=10
        )

        # Should have filtered data
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

        # Output file should exist
        self.assertTrue(self.output_file.exists())

    def test_filters_out_invalid_entries(self):
        """Test that invalid entries are filtered out"""
        result = run_pipeline(
            csv_paths=[self.csv_file],
            output_path=self.output_file,
            max_samples=10
        )

        # [removed] and 'Tech' should be filtered out
        # So we should have less than 8 entries
        self.assertLess(len(result), 8)

    def test_preserves_metadata_through_pipeline(self):
        """Test that metadata is preserved through entire pipeline"""
        run_pipeline(
            csv_paths=[self.csv_file],
            output_path=self.output_file,
            max_samples=10
        )

        # Load output and check metadata
        with open(self.output_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                self.assertIn('tags', entry)
                self.assertIn('reddit_id', entry['tags'])
                self.assertIn('subreddit', entry['tags'])
                self.assertIn('created_utc', entry['tags'])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def test_empty_csv_handling(self):
        """Test handling of empty CSV file"""
        csv_file = self.test_dir / "empty.csv"

        # Create empty CSV with headers only
        df = pd.DataFrame(columns=['title', 'id', 'created_utc', 'url', 'num_comments', 'subreddit'])
        df.to_csv(csv_file, index=False)

        output_file = self.test_dir / "output.jsonl"

        result = run_pipeline(
            csv_paths=[csv_file],
            output_path=output_file,
            max_samples=10
        )

        # Should handle gracefully
        self.assertTrue(result is None or len(result) == 0)

    def test_all_titles_filtered_out(self):
        """Test handling when all titles are filtered out (no weather keywords)"""
        csv_file = self.test_dir / "no_weather.csv"

        data = {
            'title': ['Politics update', 'Tech news', 'Sports scores'],
            'id': ['a1', 'a2', 'a3'],
            'created_utc': [1545089481] * 3,
            'url': ['http://test.com'] * 3,
            'num_comments': [5] * 3,
            'subreddit': ['TheOnion'] * 3,
            'timestamp': ['2018-12-17 18:31:21'] * 3
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        output_file = self.test_dir / "output.jsonl"

        result = run_pipeline(
            csv_paths=[csv_file],
            output_path=output_file,
            max_samples=10
        )

        # Should return None or empty DataFrame
        self.assertTrue(result is None or len(result) == 0)

    def test_malformed_csv_rows(self):
        """Test handling of CSV with malformed rows"""
        csv_file = self.test_dir / "malformed.csv"

        # Write CSV with some malformed content
        with open(csv_file, 'w') as f:
            f.write('title,id,created_utc,url,num_comments,subreddit,timestamp\n')
            f.write('Weather update,a1,1545089481,http://test.com,5,TheOnion,2018-12-17 18:31:21\n')
            # This should be handled gracefully by pandas
            f.write('Storm warning,a2,invalid_timestamp,http://test.com,5,TheOnion,2018-12-17 18:31:21\n')

        # Should load without crashing
        df = load_reddit_csv(csv_file)
        self.assertIsNotNone(df)

    def test_unicode_edge_cases(self):
        """Test handling of various Unicode characters"""
        csv_file = self.test_dir / "unicode.csv"

        data = {
            'title': [
                'Weather update: It\u2019s raining \u201ccats and dogs\u201d',
                'Storm warning\u2014take cover!',
                'Temperature: 72\u00b0F with sunny skies'
            ],
            'id': ['a1', 'a2', 'a3'],
            'created_utc': [1545089481] * 3,
            'url': ['http://test.com'] * 3,
            'num_comments': [5] * 3,
            'subreddit': ['TheOnion'] * 3,
            'timestamp': ['2018-12-17 18:31:21'] * 3
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False, encoding='utf-8')

        output_file = self.test_dir / "output.jsonl"

        result = run_pipeline(
            csv_paths=[csv_file],
            output_path=output_file,
            max_samples=10
        )

        # Should process successfully
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)


class TestDataQuality(unittest.TestCase):
    """Test data quality across pipeline"""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def test_all_output_has_weather_keywords(self):
        """Test that all output entries contain weather keywords"""
        csv_file = self.test_dir / "test.csv"

        data = {
            'title': [
                'Storm approaching',
                'Rain forecast',
                'Weather update',
                'No keywords here'  # Should be filtered
            ],
            'id': ['a1', 'a2', 'a3', 'a4'],
            'created_utc': [1545089481] * 4,
            'url': ['http://test.com'] * 4,
            'num_comments': [5] * 4,
            'subreddit': ['TheOnion'] * 4,
            'timestamp': ['2018-12-17 18:31:21'] * 4
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        output_file = self.test_dir / "output.jsonl"

        run_pipeline(
            csv_paths=[csv_file],
            output_path=output_file,
            max_samples=10
        )

        # Check each output entry has weather keywords
        with open(output_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                self.assertIn('matched_keywords', entry['tags'])
                self.assertGreater(len(entry['tags']['matched_keywords']), 0)

    def test_all_output_meets_length_requirement(self):
        """Test that all output titles meet minimum length"""
        csv_file = self.test_dir / "test.csv"

        data = {
            'title': [
                'Storm warning for entire region',
                'Rain',  # Too short
                'Weather forecast predicts sunshine'
            ],
            'id': ['a1', 'a2', 'a3'],
            'created_utc': [1545089481] * 3,
            'url': ['http://test.com'] * 3,
            'num_comments': [5] * 3,
            'subreddit': ['TheOnion'] * 3,
            'timestamp': ['2018-12-17 18:31:21'] * 3
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        output_file = self.test_dir / "output.jsonl"

        run_pipeline(
            csv_paths=[csv_file],
            output_path=output_file,
            max_samples=10
        )

        # Check each output title is at least 10 characters
        with open(output_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                assistant_msg = entry['messages'][2]['content']
                self.assertGreaterEqual(len(assistant_msg), 10)

    def test_output_format_consistency(self):
        """Test that all output entries have consistent format"""
        csv_file = self.test_dir / "test.csv"

        data = {
            'title': ['Weather update'] * 5,
            'id': [f'a{i}' for i in range(5)],
            'created_utc': [1545089481] * 5,
            'url': ['http://test.com'] * 5,
            'num_comments': [5] * 5,
            'subreddit': ['TheOnion', 'nottheonion', 'TheOnion', 'nottheonion', 'TheOnion'],
            'timestamp': ['2018-12-17 18:31:21'] * 5
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        output_file = self.test_dir / "output.jsonl"

        run_pipeline(
            csv_paths=[csv_file],
            output_path=output_file,
            max_samples=10
        )

        # Check format consistency
        with open(output_file, 'r') as f:
            for line in f:
                entry = json.loads(line)

                # Check structure
                self.assertIn('messages', entry)
                self.assertIn('tags', entry)

                # Check messages
                self.assertEqual(len(entry['messages']), 3)
                self.assertEqual(entry['messages'][0]['role'], 'system')
                self.assertEqual(entry['messages'][1]['role'], 'user')
                self.assertEqual(entry['messages'][2]['role'], 'assistant')

                # Check tags
                self.assertIn('persona', entry['tags'])
                self.assertIn('tone', entry['tags'])
                self.assertIn('domain', entry['tags'])
                self.assertIn('source', entry['tags'])


if __name__ == '__main__':
    unittest.main()
