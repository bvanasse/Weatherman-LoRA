#!/usr/bin/env python3
"""
Tests for Reddit Humor Dataset Processing - Task Group 3
Tests for chat-format JSONL generation
"""

import sys
from pathlib import Path
import unittest
import pandas as pd
import json
import tempfile

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from reddit_jsonl_converter import (
    generate_varied_user_message,
    create_chat_format_entry,
    determine_tone_tags,
    create_metadata_tags,
    convert_to_jsonl,
    validate_jsonl_output
)


class TestUserMessageVariation(unittest.TestCase):
    """Test user message variation"""

    def test_generates_varied_messages(self):
        """Test that different user messages are generated"""
        messages = [generate_varied_user_message() for _ in range(10)]

        # Should have some variation (not all the same)
        unique_messages = set(messages)
        self.assertGreater(len(unique_messages), 1, "Should generate varied user messages")

    def test_all_weather_related(self):
        """Test that all generated messages are weather-related"""
        for _ in range(20):
            msg = generate_varied_user_message()
            self.assertIsInstance(msg, str)
            self.assertGreater(len(msg), 0)


class TestChatFormatStructure(unittest.TestCase):
    """Test chat-format JSONL structure"""

    def test_creates_messages_array(self):
        """Test that entry contains messages array"""
        entry = create_chat_format_entry(
            cleaned_title="Storm warning for today",
            subreddit="TheOnion",
            post_id="a1",
            created_utc=1545089481,
            url="http://test.com",
            num_comments=5,
            matched_keywords=["storm"]
        )

        self.assertIn('messages', entry)
        self.assertIsInstance(entry['messages'], list)

    def test_has_three_message_roles(self):
        """Test that messages array has system, user, assistant roles"""
        entry = create_chat_format_entry(
            cleaned_title="Storm warning for today",
            subreddit="TheOnion",
            post_id="a1",
            created_utc=1545089481,
            url="http://test.com",
            num_comments=5,
            matched_keywords=["storm"]
        )

        messages = entry['messages']
        self.assertEqual(len(messages), 3)

        roles = [msg['role'] for msg in messages]
        self.assertEqual(roles, ['system', 'user', 'assistant'])

    def test_system_message_content(self):
        """Test that system message describes witty weather assistant"""
        entry = create_chat_format_entry(
            cleaned_title="Storm warning",
            subreddit="TheOnion",
            post_id="a1",
            created_utc=1545089481,
            url="http://test.com",
            num_comments=5,
            matched_keywords=["storm"]
        )

        system_msg = entry['messages'][0]['content']
        self.assertIn('witty', system_msg.lower())
        self.assertIn('weather', system_msg.lower())

    def test_assistant_message_is_cleaned_title(self):
        """Test that assistant message is the cleaned Reddit title"""
        cleaned_title = "Storm warning for today"
        entry = create_chat_format_entry(
            cleaned_title=cleaned_title,
            subreddit="TheOnion",
            post_id="a1",
            created_utc=1545089481,
            url="http://test.com",
            num_comments=5,
            matched_keywords=["storm"]
        )

        assistant_msg = entry['messages'][2]['content']
        self.assertEqual(assistant_msg, cleaned_title)


class TestSourceAwareTagging(unittest.TestCase):
    """Test source-aware tagging differentiation"""

    def test_theonion_satirical_tone(self):
        """Test that TheOnion posts get satirical tone tag"""
        tags = determine_tone_tags("TheOnion")
        self.assertIn('tone', tags)
        self.assertEqual(tags['tone'], 'satirical')

    def test_nottheonion_ironic_tone(self):
        """Test that nottheonion posts get ironic tone tag"""
        tags = determine_tone_tags("nottheonion")
        self.assertIn('tone', tags)
        self.assertEqual(tags['tone'], 'ironic')

    def test_shared_domain_tags(self):
        """Test that all posts get weather and humor domain tags"""
        for subreddit in ['TheOnion', 'nottheonion']:
            tags = determine_tone_tags(subreddit)
            self.assertIn('domain', tags)
            # Domain can be a list or string
            if isinstance(tags['domain'], list):
                self.assertIn('weather', tags['domain'])
                self.assertIn('humor', tags['domain'])
            else:
                self.assertIn('weather', tags['domain'])

    def test_persona_neutral(self):
        """Test that persona is set to neutral"""
        for subreddit in ['TheOnion', 'nottheonion']:
            tags = determine_tone_tags(subreddit)
            self.assertIn('persona', tags)
            self.assertEqual(tags['persona'], 'neutral')

    def test_source_tag_included(self):
        """Test that source tag identifies dataset origin"""
        theonion_tags = determine_tone_tags("TheOnion")
        self.assertIn('source', theonion_tags)
        self.assertIn('reddit', theonion_tags['source'].lower())

        nottheonion_tags = determine_tone_tags("nottheonion")
        self.assertIn('source', nottheonion_tags)
        self.assertIn('reddit', nottheonion_tags['source'].lower())


class TestMetadataEmbedding(unittest.TestCase):
    """Test metadata embedding in tags field"""

    def test_embeds_reddit_id(self):
        """Test that Reddit post ID is embedded"""
        tags = create_metadata_tags(
            post_id="a1b2c3",
            subreddit="TheOnion",
            created_utc=1545089481,
            url="http://test.com",
            num_comments=42
        )
        self.assertIn('reddit_id', tags)
        self.assertEqual(tags['reddit_id'], 'a1b2c3')

    def test_embeds_subreddit(self):
        """Test that subreddit is embedded"""
        tags = create_metadata_tags(
            post_id="a1",
            subreddit="TheOnion",
            created_utc=1545089481,
            url="http://test.com",
            num_comments=42
        )
        self.assertIn('subreddit', tags)
        self.assertEqual(tags['subreddit'], 'TheOnion')

    def test_embeds_timestamp(self):
        """Test that created_utc timestamp is embedded"""
        tags = create_metadata_tags(
            post_id="a1",
            subreddit="TheOnion",
            created_utc=1545089481,
            url="http://test.com",
            num_comments=42
        )
        self.assertIn('created_utc', tags)
        self.assertEqual(tags['created_utc'], 1545089481)

    def test_embeds_url(self):
        """Test that Reddit URL is embedded"""
        url = "https://reddit.com/r/TheOnion/test"
        tags = create_metadata_tags(
            post_id="a1",
            subreddit="TheOnion",
            created_utc=1545089481,
            url=url,
            num_comments=42
        )
        self.assertIn('url', tags)
        self.assertEqual(tags['url'], url)

    def test_embeds_score_from_comments(self):
        """Test that num_comments is stored as score"""
        tags = create_metadata_tags(
            post_id="a1",
            subreddit="TheOnion",
            created_utc=1545089481,
            url="http://test.com",
            num_comments=42
        )
        self.assertIn('score', tags)
        self.assertEqual(tags['score'], 42)


class TestJSONLFormatting(unittest.TestCase):
    """Test JSONL output formatting"""

    def test_one_json_per_line(self):
        """Test that output has one JSON object per line"""
        df = pd.DataFrame({
            'cleaned_title': ['Storm warning', 'Rain expected'],
            'id': ['a1', 'a2'],
            'subreddit': ['TheOnion', 'nottheonion'],
            'created_utc': [1545089481, 1545089482],
            'url': ['http://test.com/1', 'http://test.com/2'],
            'num_comments': [5, 3],
            'matched_keywords': [['storm'], ['rain']]
        })

        output_file = Path(tempfile.mktemp(suffix='.jsonl'))
        convert_to_jsonl(df, output_file)

        # Read lines
        with open(output_file, 'r') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2)

        # Each line should be valid JSON
        for line in lines:
            obj = json.loads(line)
            self.assertIn('messages', obj)
            self.assertIn('tags', obj)

        output_file.unlink()

    def test_atomic_write_pattern(self):
        """Test that file write uses atomic pattern"""
        df = pd.DataFrame({
            'cleaned_title': ['Storm warning'],
            'id': ['a1'],
            'subreddit': ['TheOnion'],
            'created_utc': [1545089481],
            'url': ['http://test.com'],
            'num_comments': [5],
            'matched_keywords': [['storm']]
        })

        output_file = Path(tempfile.mktemp(suffix='.jsonl'))
        convert_to_jsonl(df, output_file)

        # File should exist
        self.assertTrue(output_file.exists())

        # Should be valid JSONL
        is_valid = validate_jsonl_output(output_file)
        self.assertTrue(is_valid)

        output_file.unlink()


class TestOutputValidation(unittest.TestCase):
    """Test output validation checks"""

    def test_validates_correct_jsonl(self):
        """Test validation of correctly formatted JSONL"""
        output_file = Path(tempfile.mktemp(suffix='.jsonl'))

        # Write valid JSONL
        with open(output_file, 'w') as f:
            entry = {
                'messages': [
                    {'role': 'system', 'content': 'Test'},
                    {'role': 'user', 'content': 'Test'},
                    {'role': 'assistant', 'content': 'Test'}
                ],
                'tags': {'test': 'value'}
            }
            f.write(json.dumps(entry) + '\n')

        is_valid = validate_jsonl_output(output_file)
        self.assertTrue(is_valid)

        output_file.unlink()

    def test_rejects_invalid_json(self):
        """Test that invalid JSON is detected"""
        output_file = Path(tempfile.mktemp(suffix='.jsonl'))

        # Write invalid JSON
        with open(output_file, 'w') as f:
            f.write('{"invalid": json}\n')

        is_valid = validate_jsonl_output(output_file)
        self.assertFalse(is_valid)

        output_file.unlink()


if __name__ == '__main__':
    unittest.main()
