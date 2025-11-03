#!/usr/bin/env python3
"""
Tests for Chat Message Generation (Task Group 2)

Tests system/user message generation and chat format conversion.
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import unittest
from system_message_generator import generate_system_message, get_system_content
from user_message_generator import generate_user_message, get_user_content, get_all_templates
from chat_converter import convert_to_single_turn, convert_to_multi_turn, convert_to_chat_format
from chat_format_validator import validate_chat_entry, validate_role_order


class TestSystemMessageGenerator(unittest.TestCase):
    """Test persona-aware system message generation."""

    def test_twain_system_message(self):
        """Test Twain persona system message."""
        msg = generate_system_message('twain')
        self.assertEqual(msg['role'], 'system')
        self.assertIn('Mark Twain', msg['content'])
        self.assertIn('witty', msg['content'].lower())

    def test_franklin_system_message(self):
        """Test Franklin persona system message."""
        msg = generate_system_message('franklin')
        self.assertEqual(msg['role'], 'system')
        self.assertIn('Benjamin Franklin', msg['content'])
        self.assertIn('wise', msg['content'].lower())

    def test_neutral_system_message(self):
        """Test neutral persona system message."""
        msg = generate_system_message('neutral')
        self.assertEqual(msg['role'], 'system')
        self.assertIn('helpful', msg['content'].lower())

    def test_unknown_persona_defaults_to_neutral(self):
        """Test unknown persona defaults to neutral."""
        msg = generate_system_message('unknown')
        self.assertEqual(msg['role'], 'system')
        # Should get neutral message
        neutral_msg = generate_system_message('neutral')
        self.assertEqual(msg['content'], neutral_msg['content'])

    def test_get_system_content(self):
        """Test getting system content string only."""
        content = get_system_content('twain')
        self.assertIsInstance(content, str)
        self.assertIn('Mark Twain', content)


class TestUserMessageGenerator(unittest.TestCase):
    """Test user message template variation."""

    def test_user_message_structure(self):
        """Test user message has correct structure."""
        msg = generate_user_message()
        self.assertEqual(msg['role'], 'user')
        self.assertIsInstance(msg['content'], str)
        self.assertGreater(len(msg['content']), 0)

    def test_user_message_from_templates(self):
        """Test user message comes from template list."""
        templates = get_all_templates()
        msg = generate_user_message()
        self.assertIn(msg['content'], templates)

    def test_template_count(self):
        """Test we have 15 user message templates."""
        templates = get_all_templates()
        self.assertEqual(len(templates), 15)

    def test_user_message_variation(self):
        """Test user messages show variation."""
        # Generate multiple messages
        messages = [generate_user_message()['content'] for _ in range(50)]
        unique_messages = set(messages)

        # Should have multiple unique messages (not all the same)
        self.assertGreater(len(unique_messages), 1)


class TestChatConverter(unittest.TestCase):
    """Test chat format conversion."""

    def test_single_turn_conversion(self):
        """Test single-turn conversation creation."""
        item = {'text': 'Weather passage about storms.'}
        entry = convert_to_single_turn(item, 'twain')

        self.assertIn('messages', entry)
        self.assertEqual(len(entry['messages']), 3)

        # Check roles
        self.assertEqual(entry['messages'][0]['role'], 'system')
        self.assertEqual(entry['messages'][1]['role'], 'user')
        self.assertEqual(entry['messages'][2]['role'], 'assistant')

        # Check assistant content is the passage
        self.assertEqual(entry['messages'][2]['content'], 'Weather passage about storms.')

    def test_multi_turn_conversion(self):
        """Test multi-turn conversation for long passages."""
        long_text = "This is a long passage about weather. " * 50
        item = {'text': long_text}
        entry = convert_to_multi_turn(item, 'franklin')

        self.assertIn('messages', entry)
        self.assertEqual(len(entry['messages']), 5)

        # Check roles
        roles = [msg['role'] for msg in entry['messages']]
        self.assertEqual(roles, ['system', 'user', 'assistant', 'user', 'assistant'])

    def test_automatic_format_selection(self):
        """Test automatic single vs multi-turn selection."""
        # Short passage → single-turn
        short_item = {'text': 'Short passage.', 'word_count': 2}
        short_entry = convert_to_chat_format(short_item, 'neutral')
        self.assertEqual(len(short_entry['messages']), 3)

        # Long passage → multi-turn
        long_item = {'text': 'Long passage. ' * 400, 'word_count': 400}
        long_entry = convert_to_chat_format(long_item, 'neutral')
        self.assertEqual(len(long_entry['messages']), 5)

    def test_persona_in_system_message(self):
        """Test persona appears in system message."""
        item = {'text': 'Test passage'}

        twain_entry = convert_to_single_turn(item, 'twain')
        self.assertIn('Mark Twain', twain_entry['messages'][0]['content'])

        franklin_entry = convert_to_single_turn(item, 'franklin')
        self.assertIn('Benjamin Franklin', franklin_entry['messages'][0]['content'])


class TestChatFormatValidator(unittest.TestCase):
    """Test chat format schema validation."""

    def test_validate_valid_entry(self):
        """Test validation of valid entry."""
        entry = {
            'messages': [
                {'role': 'system', 'content': 'System prompt'},
                {'role': 'user', 'content': 'User query'},
                {'role': 'assistant', 'content': 'Assistant response'}
            ],
            'tags': {
                'persona': 'neutral',
                'tone': 'humorous',
                'domain': ['weather']
            }
        }

        is_valid, error = validate_chat_entry(entry)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_missing_messages_field(self):
        """Test detection of missing messages field."""
        entry = {'tags': {}}
        is_valid, error = validate_chat_entry(entry)
        self.assertFalse(is_valid)
        self.assertIn('messages', error.lower())

    def test_missing_tags_field(self):
        """Test detection of missing tags field."""
        entry = {
            'messages': [
                {'role': 'system', 'content': 'test'},
                {'role': 'user', 'content': 'test'},
                {'role': 'assistant', 'content': 'test'}
            ]
        }
        is_valid, error = validate_chat_entry(entry)
        self.assertFalse(is_valid)
        self.assertIn('tags', error.lower())

    def test_invalid_role_order(self):
        """Test detection of invalid role ordering."""
        # User before system
        is_valid, error = validate_role_order(['user', 'assistant'])
        self.assertFalse(is_valid)
        self.assertIn('system', error.lower())

        # Assistant before user
        is_valid, error = validate_role_order(['system', 'assistant'])
        self.assertFalse(is_valid)

    def test_valid_role_orders(self):
        """Test validation of valid role orders."""
        # Single-turn
        is_valid, error = validate_role_order(['system', 'user', 'assistant'])
        self.assertTrue(is_valid)

        # Multi-turn
        is_valid, error = validate_role_order(['system', 'user', 'assistant', 'user', 'assistant'])
        self.assertTrue(is_valid)


if __name__ == '__main__':
    unittest.main()
