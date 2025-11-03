#!/usr/bin/env python3
"""
Tests for Task Group 4: Validation Pipeline and Quality Checks

Tests cover:
- Tool_calls schema validation (function name, arguments structure)
- JSON argument parsing and schema compliance
- Semantic weather data validation (temperature ranges, valid conditions)
- Groundedness check (assistant references tool output)
- Role ordering validation (system, user, assistant, tool, assistant)
"""

import unittest
import json
from scripts.chat_format_validator import validate_chat_entry, validate_tool_call, validate_role_order
from scripts.synthetic_data_validators import (
    validate_json_schema,
    validate_semantic_weather_data,
    validate_groundedness,
    validate_conversation_full,
    ValidationReporter
)
from scripts.conversation_orchestrator import ConversationOrchestrator
from scripts.conversation_assembly import ConversationAssembler


class TestToolCallsSchemaValidation(unittest.TestCase):
    """Test tool_calls array validation."""

    def test_valid_tool_call_structure(self):
        """Test that valid tool_call structure passes validation."""
        tool_call = {
            'id': 'call_123',
            'type': 'function',
            'function': {
                'name': 'get_current_weather',
                'arguments': '{"latitude": 40.7, "longitude": -74.0}'
            }
        }

        is_valid, error = validate_tool_call(tool_call)
        self.assertTrue(is_valid, f"Should be valid: {error}")

    def test_missing_tool_call_fields(self):
        """Test that missing fields are caught."""
        # Missing 'id'
        tool_call = {
            'type': 'function',
            'function': {'name': 'test', 'arguments': '{}'}
        }
        is_valid, error = validate_tool_call(tool_call)
        self.assertFalse(is_valid)
        self.assertIn('id', error.lower())

        # Missing 'function'
        tool_call = {
            'id': 'call_123',
            'type': 'function'
        }
        is_valid, error = validate_tool_call(tool_call)
        self.assertFalse(is_valid)
        self.assertIn('function', error.lower())

    def test_invalid_json_in_arguments(self):
        """Test that invalid JSON in arguments is caught."""
        tool_call = {
            'id': 'call_123',
            'type': 'function',
            'function': {
                'name': 'get_current_weather',
                'arguments': '{invalid json}'
            }
        }

        is_valid, error = validate_tool_call(tool_call)
        self.assertFalse(is_valid)
        self.assertIn('json', error.lower())


class TestJSONArgumentParsing(unittest.TestCase):
    """Test JSON argument parsing and schema compliance."""

    def test_json_schema_validation_with_valid_arguments(self):
        """Test that valid arguments pass schema validation."""
        orchestrator = ConversationOrchestrator()
        assembler = ConversationAssembler(mock_mode=True)

        config = orchestrator.get_next_generation_config()
        conv = assembler.assemble_conversation(config)

        is_valid, error = validate_json_schema(conv)
        self.assertTrue(is_valid, f"Should be valid: {error}")

    def test_parameter_range_validation(self):
        """Test that out-of-range parameters are caught."""
        conversation = {
            'messages': [
                {'role': 'system', 'content': 'Test'},
                {'role': 'user', 'content': 'Test'},
                {
                    'role': 'assistant',
                    'content': 'Test',
                    'tool_calls': [{
                        'id': 'call_123',
                        'type': 'function',
                        'function': {
                            'name': 'get_current_weather',
                            'arguments': '{"latitude": 100, "longitude": -74.0}'  # Invalid lat
                        }
                    }]
                }
            ]
        }

        is_valid, error = validate_json_schema(conversation)
        self.assertFalse(is_valid)
        self.assertIn('latitude', error.lower())


class TestSemanticWeatherValidation(unittest.TestCase):
    """Test semantic weather data validation."""

    def test_realistic_temperature_ranges(self):
        """Test that temperatures are in reasonable ranges."""
        orchestrator = ConversationOrchestrator()
        assembler = ConversationAssembler(mock_mode=True)

        # Generate and validate multiple conversations
        for _ in range(10):
            config = orchestrator.get_next_generation_config()
            conv = assembler.assemble_conversation(config)

            is_valid, error = validate_semantic_weather_data(conv)
            self.assertTrue(is_valid, f"Should be valid: {error}")

    def test_invalid_temperature_rejected(self):
        """Test that unrealistic temperatures are rejected."""
        # Create conversation with unrealistic temperature
        conversation = {
            'messages': [
                {'role': 'system', 'content': 'Test'},
                {'role': 'user', 'content': 'Weather?'},
                {
                    'role': 'assistant',
                    'content': 'Checking',
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
                    'content': '{"temperature": 150, "condition": "clear"}'  # Unrealistic!
                },
                {'role': 'assistant', 'content': 'It is 150°C'}
            ],
            'tags': {'persona': 'neutral', 'tone': 'neutral', 'domain': ['weather']},
            'metadata': {'conversation_id': 'test'}
        }

        is_valid, error = validate_semantic_weather_data(conversation)
        self.assertFalse(is_valid)
        self.assertIn('temperature', error.lower())

    def test_forecast_day_count_validation(self):
        """Test that forecast day counts match requests."""
        conversation = {
            'messages': [
                {'role': 'system', 'content': 'Test'},
                {'role': 'user', 'content': 'Forecast?'},
                {
                    'role': 'assistant',
                    'content': 'Getting forecast',
                    'tool_calls': [{
                        'id': 'call_123',
                        'type': 'function',
                        'function': {
                            'name': 'get_forecast',
                            'arguments': '{"latitude": 40.7, "longitude": -74.0, "days": 5}'
                        }
                    }]
                },
                {
                    'role': 'tool',
                    'tool_call_id': 'call_123',
                    'content': json.dumps({
                        'forecast_days': 5,
                        'daily': [
                            {'date': '2025-01-01', 'temperature_min': 10, 'temperature_max': 20, 'condition': 'clear'}
                            for _ in range(5)  # Correct: 5 days
                        ]
                    })
                },
                {'role': 'assistant', 'content': 'Here is the forecast'}
            ],
            'tags': {'persona': 'neutral', 'tone': 'neutral', 'domain': ['weather']},
            'metadata': {'conversation_id': 'test'}
        }

        is_valid, error = validate_semantic_weather_data(conversation)
        self.assertTrue(is_valid, f"Should be valid: {error}")


class TestGroundednessValidation(unittest.TestCase):
    """Test groundedness validation."""

    def test_grounded_response(self):
        """Test that responses referencing tool data are marked as grounded."""
        conversation = {
            'messages': [
                {'role': 'system', 'content': 'Test'},
                {'role': 'user', 'content': 'Weather in NYC?'},
                {
                    'role': 'assistant',
                    'content': 'Checking',
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
                    'content': '{"temperature": 22, "condition": "partly_cloudy", "location": "New York, USA"}'
                },
                {'role': 'assistant', 'content': 'The weather in New York is 22°C with partly cloudy conditions.'}
            ],
            'tags': {'persona': 'neutral', 'tone': 'neutral', 'domain': ['weather']},
            'metadata': {'conversation_id': 'test'}
        }

        is_grounded, score = validate_groundedness(conversation)
        self.assertTrue(is_grounded)
        self.assertGreater(score, 0.5)

    def test_hallucinated_response(self):
        """Test that responses not referencing tool data are flagged."""
        conversation = {
            'messages': [
                {'role': 'system', 'content': 'Test'},
                {'role': 'user', 'content': 'Weather?'},
                {
                    'role': 'assistant',
                    'content': 'Checking',
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
                    'content': '{"temperature": 22, "condition": "partly_cloudy"}'
                },
                {'role': 'assistant', 'content': 'It is sunny and 30°C'}  # Hallucinated!
            ],
            'tags': {'persona': 'neutral', 'tone': 'neutral', 'domain': ['weather']},
            'metadata': {'conversation_id': 'test'}
        }

        is_grounded, score = validate_groundedness(conversation)
        # Should have low groundedness score
        self.assertLess(score, 0.7)


class TestRoleOrderingValidation(unittest.TestCase):
    """Test role ordering validation with tool messages."""

    def test_valid_tool_calling_sequence(self):
        """Test that valid tool-calling sequence passes."""
        roles = ['system', 'user', 'assistant', 'tool', 'assistant']
        is_valid, error = validate_role_order(roles)
        self.assertTrue(is_valid, f"Should be valid: {error}")

    def test_tool_without_prior_assistant_fails(self):
        """Test that tool message without prior assistant fails."""
        roles = ['system', 'user', 'tool', 'assistant']
        is_valid, error = validate_role_order(roles)
        self.assertFalse(is_valid)

    def test_tool_without_following_assistant_fails(self):
        """Test that tool message without following assistant fails."""
        roles = ['system', 'user', 'assistant', 'tool']
        is_valid, error = validate_role_order(roles)
        self.assertFalse(is_valid)

    def test_multi_turn_with_tools(self):
        """Test multi-turn conversation with tool calls."""
        roles = ['system', 'user', 'assistant', 'tool', 'assistant', 'user', 'assistant']
        is_valid, error = validate_role_order(roles)
        self.assertTrue(is_valid, f"Should be valid: {error}")


class TestValidationReporter(unittest.TestCase):
    """Test validation reporting functionality."""

    def test_batch_validation(self):
        """Test validating a batch of conversations."""
        orchestrator = ConversationOrchestrator()
        assembler = ConversationAssembler(mock_mode=True)

        # Generate test conversations
        conversations = []
        for _ in range(20):
            config = orchestrator.get_next_generation_config()
            conv = assembler.assemble_conversation(config)
            conversations.append(conv)

        # Validate batch
        reporter = ValidationReporter()
        report = reporter.validate_batch(conversations)

        # Verify report
        self.assertEqual(report.total_validated, 20)
        self.assertGreaterEqual(report.passed, 0)
        self.assertGreaterEqual(report.get_pass_rate(), 0)
        self.assertLessEqual(report.get_pass_rate(), 100)

    def test_full_conversation_validation(self):
        """Test comprehensive conversation validation."""
        orchestrator = ConversationOrchestrator()
        assembler = ConversationAssembler(mock_mode=True)

        config = orchestrator.get_next_generation_config()
        conv = assembler.assemble_conversation(config)

        is_valid, errors = validate_conversation_full(conv)
        self.assertTrue(is_valid, f"Errors: {errors}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
