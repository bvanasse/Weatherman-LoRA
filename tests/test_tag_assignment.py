#!/usr/bin/env python3
"""
Tests for Tag Assignment Modules (Task Group 1)

Tests persona, tone, and domain tag assignment logic.
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import unittest
from persona_tagger import determine_persona, determine_persona_from_item
from tone_tagger import determine_tone, determine_tone_from_item
from domain_tagger import determine_domains, determine_domains_from_item
from metadata_filter import filter_metadata, merge_tags


class TestPersonaTagger(unittest.TestCase):
    """Test persona tag extraction from source metadata."""

    def test_twain_detection(self):
        """Test Twain persona detection."""
        self.assertEqual(determine_persona("gutenberg-twain-tom_sawyer"), "twain")
        self.assertEqual(determine_persona("Mark Twain's Adventures"), "twain")
        self.assertEqual(determine_persona("TWAIN"), "twain")

    def test_franklin_detection(self):
        """Test Franklin persona detection."""
        self.assertEqual(determine_persona("gutenberg-franklin-autobiography"), "franklin")
        self.assertEqual(determine_persona("Benjamin Franklin"), "franklin")
        self.assertEqual(determine_persona("FRANKLIN"), "franklin")

    def test_reddit_neutral(self):
        """Test Reddit sources map to neutral."""
        self.assertEqual(determine_persona("reddit-theonion"), "neutral")
        self.assertEqual(determine_persona("reddit-nottheonion"), "neutral")

    def test_fallback_neutral(self):
        """Test fallback to neutral for unknown sources."""
        self.assertEqual(determine_persona(""), "neutral")
        self.assertEqual(determine_persona("unknown-source"), "neutral")
        self.assertEqual(determine_persona(None), "neutral")

    def test_persona_from_item(self):
        """Test persona extraction from full item."""
        # Test with author_name
        item1 = {'author_name': 'Mark Twain'}
        self.assertEqual(determine_persona_from_item(item1), "twain")

        # Test with source_file
        item2 = {'source_file': 'reddit_humor_weather.jsonl'}
        self.assertEqual(determine_persona_from_item(item2), "neutral")

        # Test with book_title
        item3 = {'book_title': 'The Adventures of Tom Sawyer'}
        self.assertEqual(determine_persona_from_item(item3), "twain")


class TestToneTagger(unittest.TestCase):
    """Test tone tag mapping from source."""

    def test_theonion_satirical(self):
        """Test r/TheOnion maps to satirical."""
        self.assertEqual(determine_tone("reddit-theonion", {}), "satirical")

    def test_nottheonion_ironic(self):
        """Test r/nottheonion maps to ironic."""
        self.assertEqual(determine_tone("reddit-nottheonion", {}), "ironic")

    def test_franklin_didactic(self):
        """Test Franklin sources map to didactic."""
        self.assertEqual(determine_tone("gutenberg-franklin", {}), "didactic")

    def test_default_humorous(self):
        """Test default tone is humorous."""
        self.assertEqual(determine_tone("gutenberg-twain", {}), "humorous")
        self.assertEqual(determine_tone("", {}), "humorous")

    def test_preserve_existing_tone(self):
        """Test existing tone tags are preserved."""
        self.assertEqual(determine_tone("any-source", {"tone": "existing"}), "existing")

    def test_tone_from_item(self):
        """Test tone extraction from full item."""
        # Test with subreddit
        item1 = {'subreddit': 'TheOnion'}
        self.assertEqual(determine_tone_from_item(item1), "satirical")

        # Test with author
        item2 = {'author_name': 'Benjamin Franklin'}
        self.assertEqual(determine_tone_from_item(item2), "didactic")

        # Test with genre tags
        item3 = {'genre_tags': ['humor', 'satire']}
        self.assertEqual(determine_tone_from_item(item3), "humorous")


class TestDomainTagger(unittest.TestCase):
    """Test domain tag determination from keywords."""

    def test_weather_keywords(self):
        """Test weather keywords produce weather domain."""
        domains = determine_domains(["storm", "rain"], None)
        self.assertIn("weather", domains)

    def test_humor_keywords(self):
        """Test humor keywords produce humor domain."""
        domains = determine_domains(["funny", "joke"], None)
        self.assertIn("humor", domains)

    def test_multiple_domains(self):
        """Test multiple domains from mixed keywords."""
        domains = determine_domains(["storm", "funny"], "humorous")
        self.assertIn("weather", domains)
        self.assertIn("humor", domains)

    def test_tone_adds_humor(self):
        """Test satirical/ironic/humorous tone adds humor domain."""
        domains = determine_domains(["rain"], "satirical")
        self.assertIn("humor", domains)

    def test_default_weather(self):
        """Test default to weather domain."""
        domains = determine_domains([], None)
        self.assertEqual(domains, ["weather"])

    def test_domains_from_item(self):
        """Test domain extraction from full item."""
        # Test with keywords
        item1 = {'matched_keywords': ['storm', 'lightning']}
        domains1 = determine_domains_from_item(item1)
        self.assertIn("weather", domains1)

        # Test with genre tags
        item2 = {'genre_tags': ['humor', 'satire'], 'matched_keywords': ['rain']}
        domains2 = determine_domains_from_item(item2)
        self.assertIn("weather", domains2)
        self.assertIn("humor", domains2)


class TestMetadataFilter(unittest.TestCase):
    """Test metadata filtering and tag merging."""

    def test_preserve_essential_fields(self):
        """Test essential fields are preserved."""
        item = {
            'author_name': 'Mark Twain',
            'source_file': 'gutenberg.json',
            'matched_keywords': ['storm']
        }
        filtered = filter_metadata(item)
        self.assertIn('author_name', filtered)
        self.assertIn('source_file', filtered)
        self.assertIn('matched_keywords', filtered)

    def test_exclude_unnecessary_fields(self):
        """Test unnecessary fields are excluded."""
        item = {
            'author_name': 'Mark Twain',
            'reddit_id': 'abc123',
            'created_utc': 1234567890,
            'word_count': 150
        }
        filtered = filter_metadata(item)
        self.assertIn('author_name', filtered)
        self.assertNotIn('reddit_id', filtered)
        self.assertNotIn('created_utc', filtered)
        self.assertNotIn('word_count', filtered)

    def test_filter_empty_values(self):
        """Test empty values are filtered out."""
        item = {
            'author_name': 'Mark Twain',
            'matched_keywords': [],
            'source_file': ''
        }
        filtered = filter_metadata(item)
        self.assertIn('author_name', filtered)
        self.assertNotIn('matched_keywords', filtered)
        self.assertNotIn('source_file', filtered)

    def test_merge_tags(self):
        """Test tag merging with metadata."""
        metadata = {'source_file': 'gutenberg.json', 'author_name': 'Mark Twain'}
        merged = merge_tags('twain', 'humorous', ['weather', 'humor'], metadata)

        self.assertEqual(merged['persona'], 'twain')
        self.assertEqual(merged['tone'], 'humorous')
        self.assertEqual(merged['domain'], ['weather', 'humor'])
        self.assertEqual(merged['source_file'], 'gutenberg.json')
        self.assertEqual(merged['author_name'], 'Mark Twain')

    def test_json_serializable(self):
        """Test merged tags are JSON serializable."""
        import json
        metadata = {'source_file': 'gutenberg.json'}
        merged = merge_tags('twain', 'humorous', ['weather'], metadata)

        # Should not raise exception
        json_str = json.dumps(merged)
        self.assertIsInstance(json_str, str)


if __name__ == '__main__':
    unittest.main()
