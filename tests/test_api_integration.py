#!/usr/bin/env python3
"""
Tests for Task Group 2: Claude Haiku 4.5 API Integration and Prompt Engineering

Tests cover:
- API client initialization with valid/invalid keys
- Retry logic with exponential backoff on failures
- Rate limiting throttle mechanism
- API response parsing and extraction
- Prompt template generation for different scenarios and personas
"""

import unittest
import time
from unittest.mock import Mock, patch, MagicMock
from scripts.conversation_parser import ConversationParser
from scripts.prompt_templates import PromptTemplateEngine
from scripts.geographic_database import get_location_by_city


class TestAPIClientInitialization(unittest.TestCase):
    """Test API client initialization."""

    @patch('scripts.claude_api_client.ANTHROPIC_AVAILABLE', True)
    @patch('scripts.claude_api_client.anthropic')
    def test_client_initialization_with_api_key(self, mock_anthropic):
        """Test client initializes successfully with provided API key."""
        from scripts.claude_api_client import ClaudeAPIClient

        mock_anthropic.Anthropic.return_value = Mock()

        client = ClaudeAPIClient(api_key='test_key_123')

        self.assertEqual(client.api_key, 'test_key_123')
        self.assertEqual(client.max_retries, 3)
        self.assertEqual(client.retry_delays, [1.0, 2.0, 4.0])
        self.assertIsNotNone(client.metrics)

    @patch('scripts.claude_api_client.ANTHROPIC_AVAILABLE', True)
    @patch('scripts.claude_api_client.anthropic')
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'env_key_456'})
    def test_client_initialization_from_environment(self, mock_anthropic):
        """Test client reads API key from environment variable."""
        from scripts.claude_api_client import ClaudeAPIClient

        mock_anthropic.Anthropic.return_value = Mock()

        client = ClaudeAPIClient()

        self.assertEqual(client.api_key, 'env_key_456')

    def test_client_import_error_when_anthropic_unavailable(self):
        """Test appropriate error when anthropic package not installed."""
        with patch('scripts.claude_api_client.ANTHROPIC_AVAILABLE', False):
            from scripts.claude_api_client import ClaudeAPIClient

            with self.assertRaises(ImportError) as context:
                ClaudeAPIClient(api_key='test_key')

            self.assertIn('anthropic', str(context.exception).lower())


class TestRetryLogic(unittest.TestCase):
    """Test retry logic with exponential backoff."""

    @patch('scripts.claude_api_client.ANTHROPIC_AVAILABLE', True)
    @patch('scripts.claude_api_client.anthropic')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_retry_on_failure(self, mock_sleep, mock_anthropic):
        """Test client retries on API failure with exponential backoff."""
        from scripts.claude_api_client import ClaudeAPIClient

        # Create mock client that fails twice then succeeds
        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = 'msg_123'
        mock_response.model = 'claude-3-5-haiku-20241022'
        mock_response.role = 'assistant'
        mock_response.content = [Mock(text='Test response')]
        mock_response.stop_reason = 'end_turn'
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)

        # Fail twice, then succeed
        mock_client.messages.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            mock_response
        ]

        mock_anthropic.Anthropic.return_value = mock_client

        client = ClaudeAPIClient(api_key='test_key')

        # Make request
        messages = [{'role': 'user', 'content': 'Test'}]
        response = client.generate_message(messages)

        # Verify retries occurred
        self.assertEqual(mock_client.messages.create.call_count, 3)

        # Verify exponential backoff delays
        self.assertEqual(mock_sleep.call_count, 2)  # Slept twice before success
        mock_sleep.assert_any_call(1.0)  # First retry delay
        mock_sleep.assert_any_call(2.0)  # Second retry delay

        # Verify successful response
        self.assertEqual(response['id'], 'msg_123')

    @patch('scripts.claude_api_client.ANTHROPIC_AVAILABLE', True)
    @patch('scripts.claude_api_client.anthropic')
    @patch('time.sleep')
    def test_max_retries_exceeded(self, mock_sleep, mock_anthropic):
        """Test client raises exception after max retries exceeded."""
        from scripts.claude_api_client import ClaudeAPIClient

        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("Persistent API Error")

        mock_anthropic.Anthropic.return_value = mock_client

        client = ClaudeAPIClient(api_key='test_key', max_retries=3)

        messages = [{'role': 'user', 'content': 'Test'}]

        with self.assertRaises(Exception) as context:
            client.generate_message(messages)

        # Verify all retries attempted
        self.assertEqual(mock_client.messages.create.call_count, 3)
        self.assertIn('Persistent API Error', str(context.exception))


class TestRateLimiting(unittest.TestCase):
    """Test rate limiting mechanism."""

    @patch('scripts.claude_api_client.ANTHROPIC_AVAILABLE', True)
    @patch('scripts.claude_api_client.anthropic')
    def test_rate_limit_delay_between_calls(self, mock_anthropic):
        """Test that rate limiting enforces minimum delay between calls."""
        from scripts.claude_api_client import ClaudeAPIClient

        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = 'msg_123'
        mock_response.model = 'claude-3-5-haiku-20241022'
        mock_response.role = 'assistant'
        mock_response.content = [Mock(text='Test')]
        mock_response.stop_reason = 'end_turn'
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)

        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        client = ClaudeAPIClient(api_key='test_key', rate_limit_delay=0.1)

        messages = [{'role': 'user', 'content': 'Test'}]

        # Make first call
        start_time = time.time()
        client.generate_message(messages)
        first_call_time = time.time()

        # Make second call immediately
        client.generate_message(messages)
        second_call_time = time.time()

        # Verify minimum delay was enforced
        actual_delay = second_call_time - first_call_time
        # Should be at least rate_limit_delay (0.1s), accounting for some margin
        self.assertGreaterEqual(actual_delay, 0.08)


class TestAPIResponseParsing(unittest.TestCase):
    """Test API response parsing and extraction."""

    @patch('scripts.claude_api_client.ANTHROPIC_AVAILABLE', True)
    @patch('scripts.claude_api_client.anthropic')
    def test_extract_text_from_response(self, mock_anthropic):
        """Test extracting text content from API response."""
        from scripts.claude_api_client import ClaudeAPIClient

        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = 'msg_123'
        mock_response.model = 'claude-3-5-haiku-20241022'
        mock_response.role = 'assistant'
        mock_response.content = [
            Mock(text='First part'),
            Mock(text='Second part')
        ]
        mock_response.stop_reason = 'end_turn'
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)

        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        client = ClaudeAPIClient(api_key='test_key')

        text = client.generate_conversation(prompt='Test')

        # Verify text extraction
        self.assertIn('First part', text)
        self.assertIn('Second part', text)

    @patch('scripts.claude_api_client.ANTHROPIC_AVAILABLE', True)
    @patch('scripts.claude_api_client.anthropic')
    def test_metrics_tracking(self, mock_anthropic):
        """Test that API metrics are tracked correctly."""
        from scripts.claude_api_client import ClaudeAPIClient

        mock_client = Mock()
        mock_response = Mock()
        mock_response.id = 'msg_123'
        mock_response.model = 'claude-3-5-haiku-20241022'
        mock_response.role = 'assistant'
        mock_response.content = [Mock(text='Test')]
        mock_response.stop_reason = 'end_turn'
        mock_response.usage = Mock(input_tokens=50, output_tokens=100)

        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        client = ClaudeAPIClient(api_key='test_key')

        # Make some calls
        messages = [{'role': 'user', 'content': 'Test'}]
        client.generate_message(messages)
        client.generate_message(messages)

        metrics = client.get_metrics()

        # Verify metrics
        self.assertEqual(metrics['total_calls'], 2)
        self.assertEqual(metrics['successful_calls'], 2)
        self.assertEqual(metrics['failed_calls'], 0)
        self.assertEqual(metrics['total_input_tokens'], 100)
        self.assertEqual(metrics['total_output_tokens'], 200)
        self.assertEqual(metrics['success_rate'], 100.0)


class TestConversationParsing(unittest.TestCase):
    """Test conversation parsing functionality."""

    def test_parse_valid_conversation(self):
        """Test parsing a valid tool-calling conversation."""
        parser = ConversationParser()

        conversation = {
            'messages': [
                {'role': 'system', 'content': 'You are a weather assistant.'},
                {'role': 'user', 'content': 'What is the weather?'},
                {
                    'role': 'assistant',
                    'content': 'Let me check.',
                    'tool_calls': [{
                        'id': 'call_123',
                        'type': 'function',
                        'function': {
                            'name': 'get_current_weather',
                            'arguments': '{"latitude": 40.7, "longitude": -74.0}'
                        }
                    }]
                },
                {
                    'role': 'tool',
                    'tool_call_id': 'call_123',
                    'content': '{"temperature": 22}'
                },
                {'role': 'assistant', 'content': 'It is 22°C.'}
            ]
        }

        result, error = parser.parse_conversation_from_json(conversation)

        self.assertIsNotNone(result)
        self.assertEqual(error, "")

        metadata = parser.get_conversation_metadata(result)
        self.assertEqual(metadata['message_count'], 5)
        self.assertEqual(metadata['tool_call_count'], 1)
        self.assertIn('get_current_weather', metadata['function_names'])

    def test_extract_json_from_markdown(self):
        """Test extracting JSON from markdown code blocks."""
        parser = ConversationParser()

        text = '''```json
{
  "messages": [
    {"role": "system", "content": "Test"}
  ]
}
```'''

        result = parser.extract_json_from_text(text)

        self.assertIsNotNone(result)
        self.assertIn('messages', result)


class TestPromptTemplates(unittest.TestCase):
    """Test prompt template generation."""

    def test_persona_instructions(self):
        """Test that different personas have distinct instructions."""
        engine = PromptTemplateEngine()

        neutral = engine.get_persona_instructions('neutral')
        twain = engine.get_persona_instructions('twain')
        franklin = engine.get_persona_instructions('franklin')

        # Verify all are unique
        self.assertNotEqual(neutral, twain)
        self.assertNotEqual(neutral, franklin)
        self.assertNotEqual(twain, franklin)

        # Verify content mentions persona characteristics
        self.assertIn('professional', neutral.lower())
        self.assertIn('twain', twain.lower())
        self.assertIn('franklin', franklin.lower())

    def test_system_prompt_generation(self):
        """Test system prompt generation for different scenarios."""
        engine = PromptTemplateEngine()

        success_prompt = engine.get_system_prompt('neutral', 'success')
        error_prompt = engine.get_system_prompt('neutral', 'error')
        multi_turn_prompt = engine.get_system_prompt('neutral', 'multi_turn')

        # Verify all contain base instructions
        for prompt in [success_prompt, error_prompt, multi_turn_prompt]:
            self.assertIn('weather', prompt.lower())
            self.assertIn('tool', prompt.lower())

        # Verify error-specific content
        self.assertIn('error', error_prompt.lower())

        # Verify multi-turn specific content
        self.assertIn('multi-turn', multi_turn_prompt.lower())

    def test_user_query_generation(self):
        """Test user query generation for different scenarios."""
        engine = PromptTemplateEngine()
        nyc = get_location_by_city('New York')

        current_query = engine.generate_user_query(nyc, 'success', 'current')
        forecast_query = engine.generate_user_query(nyc, 'success', 'forecast')

        # Verify queries mention the location
        self.assertIn('New York', current_query)
        self.assertIn('New York', forecast_query)

        # Verify query type differences
        self.assertTrue(
            'weather' in current_query.lower() or 'current' in current_query.lower()
        )
        # Forecast query should mention forecast or outlook
        self.assertTrue(
            'forecast' in forecast_query.lower() or 'outlook' in forecast_query.lower()
        )

    def test_complete_prompt_generation(self):
        """Test complete prompt generation with all components."""
        engine = PromptTemplateEngine()
        london = get_location_by_city('London')

        prompt_data = engine.create_generation_prompt(
            location=london,
            persona='twain',
            scenario='success',
            query_type='current'
        )

        # Verify all components present
        self.assertIn('system', prompt_data)
        self.assertIn('user_query', prompt_data)
        self.assertIn('location', prompt_data)
        self.assertIn('tools', prompt_data)

        # Verify correct persona
        self.assertEqual(prompt_data['persona'], 'twain')

        # Verify location context
        self.assertEqual(prompt_data['location'].city, 'London')


if __name__ == '__main__':
    unittest.main(verbosity=2)
