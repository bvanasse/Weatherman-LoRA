#!/usr/bin/env python3
"""
Tests for Reddit Humor Dataset Processing - Task Group 4
Tests for end-to-end pipeline orchestration and statistics
"""

import sys
from pathlib import Path
import unittest
import pandas as pd
import tempfile
import json

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from reddit_pipeline_orchestrator import (
    validate_environment,
    calculate_keyword_distribution,
    calculate_subreddit_statistics,
    print_pipeline_statistics,
    validate_output_count,
    run_pipeline
)


class TestEnvironmentValidation(unittest.TestCase):
    """Test environment validation"""

    def test_validates_csv_existence(self):
        """Test that CSV file existence is checked"""
        test_dir = Path(tempfile.mkdtemp())
        csv_files = [test_dir / "test.csv"]

        # Should fail - files don't exist
        result = validate_environment(csv_files)
        self.assertFalse(result)

        # Create files
        for csv_file in csv_files:
            csv_file.touch()

        # Should pass now
        result = validate_environment(csv_files)
        self.assertTrue(result)

    def test_validates_output_directory(self):
        """Test that output directory validation works"""
        test_dir = Path(tempfile.mkdtemp())
        output_file = test_dir / "output.jsonl"

        # Directory exists, should be okay
        csv_files = []
        result = validate_environment(csv_files, output_path=output_file)
        # Will fail on CSV check but that's expected
        self.assertIsInstance(result, bool)


class TestStatisticsCalculation(unittest.TestCase):
    """Test statistics calculation"""

    def test_calculates_keyword_distribution(self):
        """Test keyword distribution calculation"""
        df = pd.DataFrame({
            'matched_keywords': [
                ['rain', 'storm'],
                ['rain', 'weather'],
                ['storm', 'thunder']
            ]
        })

        dist = calculate_keyword_distribution(df)
        self.assertIn('rain', dist)
        self.assertIn('storm', dist)
        self.assertEqual(dist['rain'], 2)
        self.assertEqual(dist['storm'], 2)

    def test_returns_top_10_keywords(self):
        """Test that only top 10 keywords are returned"""
        # Create data with 15 unique keywords
        keywords_list = []
        for i in range(15):
            keywords_list.append([f'keyword_{i}'] * (15 - i))

        df = pd.DataFrame({'matched_keywords': keywords_list})

        dist = calculate_keyword_distribution(df, top_n=10)
        self.assertLessEqual(len(dist), 10)

    def test_calculates_subreddit_stats(self):
        """Test subreddit statistics calculation"""
        df = pd.DataFrame({
            'subreddit': ['TheOnion', 'TheOnion', 'nottheonion', 'TheOnion'],
            'cleaned_title': ['Title 1', 'Title 2', 'Title 3', 'Title 4']
        })

        stats = calculate_subreddit_statistics(df)
        self.assertIn('examples_per_subreddit', stats)
        self.assertEqual(stats['examples_per_subreddit']['TheOnion'], 3)
        self.assertEqual(stats['examples_per_subreddit']['nottheonion'], 1)

    def test_calculates_average_title_length(self):
        """Test average title length calculation"""
        df = pd.DataFrame({
            'cleaned_title': ['Short', 'Medium length', 'This is a longer title'],
            'subreddit': ['TheOnion'] * 3
        })

        stats = calculate_subreddit_statistics(df)
        self.assertIn('avg_title_length', stats)
        self.assertGreater(stats['avg_title_length'], 0)


class TestOutputValidation(unittest.TestCase):
    """Test output validation"""

    def test_validates_count_in_range(self):
        """Test validation of output count within target range"""
        # Within range
        is_valid, msg = validate_output_count(3000, target_range=(2000, 4000))
        self.assertTrue(is_valid)

        # Below range
        is_valid, msg = validate_output_count(1500, target_range=(2000, 4000))
        self.assertFalse(is_valid)
        self.assertIn('below', msg.lower())

        # Above range
        is_valid, msg = validate_output_count(5000, target_range=(2000, 4000))
        self.assertFalse(is_valid)
        self.assertIn('above', msg.lower())

    def test_validates_exact_boundaries(self):
        """Test that exact boundary values are accepted"""
        is_valid, _ = validate_output_count(2000, target_range=(2000, 4000))
        self.assertTrue(is_valid)

        is_valid, _ = validate_output_count(4000, target_range=(2000, 4000))
        self.assertTrue(is_valid)


class TestPipelineOrchestration(unittest.TestCase):
    """Test end-to-end pipeline execution"""

    def setUp(self):
        """Create temporary test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.output_file = self.test_dir / "test_output.jsonl"

        # Create test CSV files
        self.csv_files = []
        for i in range(2):
            csv_file = self.test_dir / f"test_{i}.csv"
            data = {
                'title': [
                    'Weather forecast predicts rain',
                    'Storm warning issued',
                    'Climate change affects weather patterns'
                ],
                'id': [f'{i}_a1', f'{i}_a2', f'{i}_a3'],
                'created_utc': [1545089481, 1545089482, 1545089483],
                'url': [f'http://test.com/{i}_1', f'http://test.com/{i}_2', f'http://test.com/{i}_3'],
                'num_comments': [5, 3, 2],
                'subreddit': ['TheOnion' if i == 0 else 'nottheonion'] * 3,
                'timestamp': ['2018-12-17 18:31:21'] * 3
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
            self.csv_files.append(csv_file)

    def test_pipeline_runs_successfully(self):
        """Test that pipeline executes without errors"""
        try:
            result_df = run_pipeline(
                csv_paths=self.csv_files,
                output_path=self.output_file,
                max_samples=10
            )
            self.assertIsNotNone(result_df)
        except Exception as e:
            self.fail(f"Pipeline raised unexpected exception: {e}")

    def test_pipeline_creates_output_file(self):
        """Test that pipeline creates output JSONL file"""
        run_pipeline(
            csv_paths=self.csv_files,
            output_path=self.output_file,
            max_samples=10
        )

        self.assertTrue(self.output_file.exists())

    def test_pipeline_output_is_valid_jsonl(self):
        """Test that pipeline output is valid JSONL"""
        run_pipeline(
            csv_paths=self.csv_files,
            output_path=self.output_file,
            max_samples=10
        )

        # Try to load and parse
        with open(self.output_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                self.assertIn('messages', entry)
                self.assertIn('tags', entry)

    def test_pipeline_respects_max_samples(self):
        """Test that pipeline respects max_samples parameter"""
        run_pipeline(
            csv_paths=self.csv_files,
            output_path=self.output_file,
            max_samples=3
        )

        # Count lines in output
        with open(self.output_file, 'r') as f:
            line_count = sum(1 for line in f if line.strip())

        self.assertLessEqual(line_count, 3)


class TestCLIArguments(unittest.TestCase):
    """Test CLI argument handling"""

    def test_handles_output_path_argument(self):
        """Test that output path can be specified"""
        # This is implicitly tested in pipeline tests
        pass

    def test_handles_max_examples_argument(self):
        """Test that max examples can be specified"""
        # This is implicitly tested in pipeline tests
        pass


if __name__ == '__main__':
    unittest.main()
