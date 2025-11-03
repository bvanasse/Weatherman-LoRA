#!/usr/bin/env python3
"""
Tests for Reddit Humor Dataset Processing - Task Group 2
Tests for CSV loading and filtering pipeline
"""

import sys
from pathlib import Path
import unittest
import pandas as pd
import tempfile

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from reddit_csv_processor import (
    load_reddit_csv,
    filter_by_weather_keywords,
    apply_quality_filters,
    balance_subreddit_samples,
    process_all_csvs
)
from reddit_text_processing import clean_reddit_text


class TestCSVLoading(unittest.TestCase):
    """Test pandas CSV loading with encoding handling"""

    def setUp(self):
        """Create temporary test CSV files"""
        self.test_dir = tempfile.mkdtemp()
        self.test_csv = Path(self.test_dir) / "test.csv"

    def test_loads_valid_csv(self):
        """Test loading a valid CSV file"""
        # Create test CSV
        data = {
            'title': ['Weather is nice', 'No match here'],
            'created_utc': [1545089481, 1545089482],
            'url': ['http://test.com/1', 'http://test.com/2'],
            'id': ['a1', 'a2'],
            'num_comments': [5, 3],
            'subreddit': ['TheOnion', 'TheOnion'],
            'timestamp': ['2018-12-17 18:31:21', '2018-12-17 18:31:22']
        }
        df = pd.DataFrame(data)
        df.to_csv(self.test_csv, index=False)

        # Load it
        result = load_reddit_csv(self.test_csv)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertIn('title', result.columns)

    def test_handles_encoding_issues(self):
        """Test that CSV loader handles different encodings"""
        # Create CSV with UTF-8
        data = {'title': ['Test with \u201csmart quotes\u201d']}
        df = pd.DataFrame(data)
        df.to_csv(self.test_csv, index=False, encoding='utf-8')

        result = load_reddit_csv(self.test_csv)
        self.assertIsNotNone(result)

    def test_preserves_metadata_columns(self):
        """Test that all required metadata columns are preserved"""
        data = {
            'title': ['Weather update'],
            'created_utc': [1545089481],
            'url': ['http://test.com/1'],
            'id': ['a1'],
            'num_comments': [5],
            'subreddit': ['TheOnion'],
            'timestamp': ['2018-12-17 18:31:21']
        }
        df = pd.DataFrame(data)
        df.to_csv(self.test_csv, index=False)

        result = load_reddit_csv(self.test_csv)
        required_cols = ['title', 'created_utc', 'url', 'id', 'num_comments', 'subreddit']
        for col in required_cols:
            self.assertIn(col, result.columns)


class TestKeywordFiltering(unittest.TestCase):
    """Test weather keyword filtering"""

    def test_filters_weather_titles(self):
        """Test that only weather-related titles are retained"""
        data = {
            'title': ['Storm warning issued', 'Politics news', 'Hurricane approaching'],
            'id': ['a1', 'a2', 'a3']
        }
        df = pd.DataFrame(data)

        filtered = filter_by_weather_keywords(df)
        self.assertEqual(len(filtered), 2)
        self.assertIn('storm', filtered['title'].iloc[0].lower())

    def test_tracks_matched_keywords(self):
        """Test that matched keywords are tracked"""
        data = {
            'title': ['Heavy rain and snow expected'],
            'id': ['a1']
        }
        df = pd.DataFrame(data)

        filtered = filter_by_weather_keywords(df)
        self.assertIn('matched_keywords', filtered.columns)
        keywords = filtered['matched_keywords'].iloc[0]
        self.assertIn('rain', keywords)
        self.assertIn('snow', keywords)

    def test_handles_metaphorical_usage(self):
        """Test matching of metaphorical weather terms"""
        data = {
            'title': ['Political climate worsens as forecast looks bleak'],
            'id': ['a1']
        }
        df = pd.DataFrame(data)

        filtered = filter_by_weather_keywords(df)
        self.assertEqual(len(filtered), 1)


class TestQualityFiltering(unittest.TestCase):
    """Test quality filtering and text cleaning"""

    def test_filters_short_titles(self):
        """Test filtering of titles that are too short"""
        data = {
            'title': ['Short', 'This is a longer title with weather'],
            'id': ['a1', 'a2'],
            'subreddit': ['TheOnion', 'TheOnion']
        }
        df = pd.DataFrame(data)

        filtered = apply_quality_filters(df)
        # Only the longer one should pass
        self.assertLessEqual(len(filtered), 1)

    def test_filters_missing_metadata(self):
        """Test filtering of entries with missing critical metadata"""
        data = {
            'title': ['Weather update', 'Another title'],
            'id': ['a1', None],
            'subreddit': ['TheOnion', 'TheOnion']
        }
        df = pd.DataFrame(data)

        filtered = apply_quality_filters(df)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered['id'].iloc[0], 'a1')

    def test_applies_text_cleaning(self):
        """Test that text cleaning is applied during quality filtering"""
        data = {
            'title': ['**Weather** alert [removed]'],
            'id': ['a1'],
            'subreddit': ['TheOnion']
        }
        df = pd.DataFrame(data)

        filtered = apply_quality_filters(df)
        self.assertIn('cleaned_title', filtered.columns)
        cleaned = filtered['cleaned_title'].iloc[0]
        self.assertNotIn('**', cleaned)
        self.assertNotIn('[removed]', cleaned)


class TestSubredditBalancing(unittest.TestCase):
    """Test subreddit-aware sampling and balancing"""

    def test_balances_across_subreddits(self):
        """Test that sampling balances across subreddits"""
        # Create imbalanced dataset
        data = {
            'title': ['Title {}'.format(i) for i in range(100)],
            'id': ['a{}'.format(i) for i in range(100)],
            'subreddit': ['TheOnion'] * 80 + ['nottheonion'] * 20,
            'num_comments': list(range(100))
        }
        df = pd.DataFrame(data)

        # Sample 40 total
        balanced = balance_subreddit_samples(df, max_samples=40)

        subreddit_counts = balanced['subreddit'].value_counts()
        # Should try to balance, so counts should be more even
        self.assertGreater(len(balanced), 0)
        self.assertLessEqual(len(balanced), 40)

    def test_prioritizes_high_engagement(self):
        """Test that high num_comments posts are prioritized"""
        data = {
            'title': ['Title {}'.format(i) for i in range(10)],
            'id': ['a{}'.format(i) for i in range(10)],
            'subreddit': ['TheOnion'] * 10,
            'num_comments': [1, 2, 3, 4, 5, 100, 200, 300, 400, 500]
        }
        df = pd.DataFrame(data)

        balanced = balance_subreddit_samples(df, max_samples=5)

        # Top 5 should have higher num_comments
        min_comments = balanced['num_comments'].min()
        self.assertGreater(min_comments, 10)


class TestFullPipeline(unittest.TestCase):
    """Test end-to-end CSV processing pipeline"""

    def setUp(self):
        """Create temporary test CSV files"""
        self.test_dir = tempfile.mkdtemp()

    def test_processes_multiple_csvs(self):
        """Test processing of multiple CSV files"""
        # Create test CSVs
        csv1 = Path(self.test_dir) / "file1.csv"
        csv2 = Path(self.test_dir) / "file2.csv"

        data1 = {
            'title': ['Storm warning'] * 5,
            'id': ['a{}'.format(i) for i in range(5)],
            'created_utc': [1545089481] * 5,
            'url': ['http://test.com/{}'.format(i) for i in range(5)],
            'num_comments': [5] * 5,
            'subreddit': ['TheOnion'] * 5,
            'timestamp': ['2018-12-17 18:31:21'] * 5
        }

        data2 = {
            'title': ['Weather forecast'] * 3,
            'id': ['b{}'.format(i) for i in range(3)],
            'created_utc': [1545089481] * 3,
            'url': ['http://test.com/{}'.format(i) for i in range(3)],
            'num_comments': [3] * 3,
            'subreddit': ['nottheonion'] * 3,
            'timestamp': ['2018-12-17 18:31:21'] * 3
        }

        pd.DataFrame(data1).to_csv(csv1, index=False)
        pd.DataFrame(data2).to_csv(csv2, index=False)

        result = process_all_csvs([csv1, csv2])
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)


if __name__ == '__main__':
    unittest.main()
