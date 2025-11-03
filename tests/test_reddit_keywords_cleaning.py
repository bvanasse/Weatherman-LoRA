#!/usr/bin/env python3
"""
Tests for Reddit Humor Dataset Processing - Task Group 1
Tests for keyword expansion and text cleaning utilities
"""

import sys
from pathlib import Path
import unittest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from reddit_text_processing import (
    EXPANDED_WEATHER_KEYWORDS,
    build_expanded_weather_pattern,
    clean_reddit_text,
    is_valid_cleaned_text
)


class TestExpandedWeatherKeywords(unittest.TestCase):
    """Test expanded weather keyword list"""

    def test_keyword_list_size(self):
        """Test that expanded keyword list has 40+ terms"""
        self.assertGreaterEqual(
            len(EXPANDED_WEATHER_KEYWORDS),
            40,
            "Expanded keyword list should contain 40+ terms"
        )

    def test_includes_seasonal_terms(self):
        """Test that seasonal terms are included"""
        seasonal_terms = ['winter', 'summer', 'spring', 'fall', 'autumn']
        for term in seasonal_terms:
            self.assertIn(
                term.lower(),
                [kw.lower() for kw in EXPANDED_WEATHER_KEYWORDS],
                f"Seasonal term '{term}' should be in keyword list"
            )

    def test_includes_extreme_weather(self):
        """Test that extreme weather terms are included"""
        extreme_terms = ['heatwave', 'blizzard', 'wildfire', 'avalanche', 'monsoon', 'typhoon']
        for term in extreme_terms:
            self.assertIn(
                term.lower(),
                [kw.lower() for kw in EXPANDED_WEATHER_KEYWORDS],
                f"Extreme weather term '{term}' should be in keyword list"
            )

    def test_includes_metaphorical_terms(self):
        """Test that metaphorical weather terms are included"""
        metaphorical_terms = ['forecast', 'outlook', 'climate']
        for term in metaphorical_terms:
            self.assertIn(
                term.lower(),
                [kw.lower() for kw in EXPANDED_WEATHER_KEYWORDS],
                f"Metaphorical term '{term}' should be in keyword list"
            )


class TestWholeWordMatching(unittest.TestCase):
    """Test whole-word regex matching avoids partial matches"""

    def setUp(self):
        self.pattern = build_expanded_weather_pattern()

    def test_matches_whole_word(self):
        """Test that pattern matches whole words"""
        text = "The weather was terrible"
        matches = self.pattern.findall(text)
        self.assertIn('weather', [m.lower() for m in matches])

    def test_avoids_partial_matches(self):
        """Test that pattern avoids partial word matches"""
        """Test that pattern avoids partial word matches"""
        text = "The leathern coat protects from weather"
        matches = self.pattern.findall(text)
        # Should match "weather" but not "ather" within "leathern"
        self.assertEqual(len(matches), 1, "Should find exactly one weather keyword")
        self.assertIn("weather", [m.lower() for m in matches])
    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive"""
        text = "WEATHER Weather weather"
        matches = self.pattern.findall(text)
        self.assertEqual(len(matches), 3, "Should match all case variations")

    def test_metaphorical_usage(self):
        """Test matching metaphorical weather usage"""
        text = "The political climate is changing"
        matches = self.pattern.findall(text)
        self.assertIn('climate', [m.lower() for m in matches])


class TestRedditTextCleaning(unittest.TestCase):
    """Test Reddit-specific text cleaning"""

    def test_removes_deleted_marker(self):
        """Test removal of [deleted] marker"""
        text = "This is a title [deleted]"
        cleaned = clean_reddit_text(text)
        self.assertNotIn('[deleted]', cleaned)

    def test_removes_removed_marker(self):
        """Test removal of [removed] marker"""
        text = "[removed] This is a title"
        cleaned = clean_reddit_text(text)
        self.assertNotIn('[removed]', cleaned)

    def test_removes_automoderator(self):
        """Test removal of AutoModerator references"""
        text = "[AutoModerator] Weather update"
        cleaned = clean_reddit_text(text)
        self.assertNotIn('[AutoModerator]', cleaned.lower())

    def test_normalizes_smart_quotes(self):
        """Test normalization of smart quotes to straight quotes"""
        text = "It\u2019s a \u201cbeautiful\u201d day"  # Using Unicode escape sequences
        cleaned = clean_reddit_text(text)
        self.assertNotIn('\u201c', cleaned)  # Left double quote
        self.assertNotIn('\u201d', cleaned)  # Right double quote
        self.assertIn('"', cleaned)

    def test_normalizes_em_dashes(self):
        """Test normalization of em-dashes to hyphens"""
        text = "Rain\u2014lots of it\u2014is coming"  # Using Unicode escape sequence for em-dash
        cleaned = clean_reddit_text(text)
        self.assertNotIn('\u2014', cleaned)  # Em dash
        self.assertIn('-', cleaned)

    def test_strips_markdown_bold(self):
        """Test removal of markdown bold formatting"""
        text = "**Weather alert** for today"
        cleaned = clean_reddit_text(text)
        self.assertNotIn('**', cleaned)
        self.assertIn('Weather alert', cleaned)

    def test_trims_whitespace(self):
        """Test trimming of excessive whitespace"""
        text = "  Weather    update  "
        cleaned = clean_reddit_text(text)
        self.assertEqual(cleaned.strip(), cleaned)
        self.assertNotIn('    ', cleaned)  # No excessive spaces

    def test_preserves_meaning(self):
        """Test that cleaning preserves the original meaning"""
        text = "Storm **warning**: Heavy rain expected—stay safe!"
        cleaned = clean_reddit_text(text)
        self.assertIn('Storm', cleaned)
        self.assertIn('warning', cleaned)
        self.assertIn('Heavy rain expected', cleaned)


class TestTextValidation(unittest.TestCase):
    """Test minimum length validation"""

    def test_rejects_empty_text(self):
        """Test that empty text is rejected"""
        self.assertFalse(is_valid_cleaned_text(""))
        self.assertFalse(is_valid_cleaned_text("   "))

    def test_rejects_too_short_text(self):
        """Test that text shorter than 10 chars is rejected"""
        self.assertFalse(is_valid_cleaned_text("Short"))
        self.assertFalse(is_valid_cleaned_text("12345"))

    def test_accepts_valid_text(self):
        """Test that text with 10+ characters is accepted"""
        self.assertTrue(is_valid_cleaned_text("This is valid text"))
        self.assertTrue(is_valid_cleaned_text("Ten chars!"))

    def test_rejects_only_artifacts(self):
        """Test that text with only artifacts is rejected"""
        text = "[removed]"
        cleaned = clean_reddit_text(text)
        self.assertFalse(is_valid_cleaned_text(cleaned))


if __name__ == '__main__':
    unittest.main()
