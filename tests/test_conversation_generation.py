#!/usr/bin/env python3
"""
Tests for Task Group 3: Synthetic Conversation Generation Pipeline

Tests cover:
- Scenario distribution (60-70% success, 15-20% error, 15-20% multi-turn)
- Persona distribution (60% neutral, 25% Twain, 15% Franklin)
- Conversation structure (system, user, assistant with tool_calls, tool, assistant)
- Unique conversation ID generation
- Metadata tag assignment (persona, tone, domain, source)
"""

import unittest
import json
from scripts.conversation_orchestrator import ConversationOrchestrator, GenerationStats
from scripts.conversation_assembly import ConversationAssembler
from scripts.geographic_database import get_location_by_city


class TestScenarioDistribution(unittest.TestCase):
    """Test that scenario distribution matches target ranges."""

    def test_scenario_distribution_across_1000_generations(self):
        """Test scenario distribution over 1000 generations."""
        orchestrator = ConversationOrchestrator()

        # Generate 1000 configs
        for _ in range(1000):
            orchestrator.get_next_generation_config()

        stats = orchestrator.get_statistics()
        distributions = stats['distributions']
        scenario_dist = distributions['scenario']

        # Verify success: 60-70%
        self.assertGreaterEqual(scenario_dist['success'], 60.0,
            f"Success rate too low: {scenario_dist['success']:.1f}%")
        self.assertLessEqual(scenario_dist['success'], 70.0,
            f"Success rate too high: {scenario_dist['success']:.1f}%")

        # Verify error: 15-20%
        self.assertGreaterEqual(scenario_dist['error'], 12.0,  # Allow some variance
            f"Error rate too low: {scenario_dist['error']:.1f}%")
        self.assertLessEqual(scenario_dist['error'], 23.0,
            f"Error rate too high: {scenario_dist['error']:.1f}%")

        # Verify multi_turn: 15-20%
        self.assertGreaterEqual(scenario_dist['multi_turn'], 12.0,
            f"Multi-turn rate too low: {scenario_dist['multi_turn']:.1f}%")
        self.assertLessEqual(scenario_dist['multi_turn'], 23.0,
            f"Multi-turn rate too high: {scenario_dist['multi_turn']:.1f}%")


class TestPersonaDistribution(unittest.TestCase):
    """Test that persona distribution matches target ranges."""

    def test_persona_distribution_across_1000_generations(self):
        """Test persona distribution over 1000 generations."""
        orchestrator = ConversationOrchestrator()

        # Generate 1000 configs
        for _ in range(1000):
            orchestrator.get_next_generation_config()

        stats = orchestrator.get_statistics()
        distributions = stats['distributions']
        persona_dist = distributions['persona']

        # Verify neutral: ~60% (allow 55-65%)
        self.assertGreaterEqual(persona_dist['neutral'], 55.0,
            f"Neutral rate too low: {persona_dist['neutral']:.1f}%")
        self.assertLessEqual(persona_dist['neutral'], 65.0,
            f"Neutral rate too high: {persona_dist['neutral']:.1f}%")

        # Verify twain: ~25% (allow 20-30%)
        self.assertGreaterEqual(persona_dist['twain'], 20.0,
            f"Twain rate too low: {persona_dist['twain']:.1f}%")
        self.assertLessEqual(persona_dist['twain'], 30.0,
            f"Twain rate too high: {persona_dist['twain']:.1f}%")

        # Verify franklin: ~15% (allow 10-20%)
        self.assertGreaterEqual(persona_dist['franklin'], 10.0,
            f"Franklin rate too low: {persona_dist['franklin']:.1f}%")
        self.assertLessEqual(persona_dist['franklin'], 20.0,
            f"Franklin rate too high: {persona_dist['franklin']:.1f}%")


class TestConversationStructure(unittest.TestCase):
    """Test that generated conversations have correct structure."""

    def test_conversation_has_correct_message_sequence(self):
        """Test that conversation follows correct role sequence."""
        orchestrator = ConversationOrchestrator()
        assembler = ConversationAssembler(mock_mode=True)

        config = orchestrator.get_next_generation_config()
        conversation = assembler.assemble_conversation(config)

        messages = conversation['messages']

        # Verify minimum message count (system, user, assistant, tool, assistant)
        self.assertGreaterEqual(len(messages), 5)

        # Verify role sequence
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')
        self.assertEqual(messages[2]['role'], 'assistant')
        self.assertIn('tool_calls', messages[2])
        self.assertEqual(messages[3]['role'], 'tool')
        self.assertEqual(messages[4]['role'], 'assistant')

    def test_assistant_message_has_tool_calls(self):
        """Test that assistant message contains valid tool_calls."""
        orchestrator = ConversationOrchestrator()
        assembler = ConversationAssembler(mock_mode=True)

        config = orchestrator.get_next_generation_config()
        conversation = assembler.assemble_conversation(config)

        # Find assistant message with tool_calls
        assistant_msg = conversation['messages'][2]

        self.assertEqual(assistant_msg['role'], 'assistant')
        self.assertIn('tool_calls', assistant_msg)
        self.assertIsInstance(assistant_msg['tool_calls'], list)
        self.assertGreater(len(assistant_msg['tool_calls']), 0)

        # Validate tool_call structure
        tool_call = assistant_msg['tool_calls'][0]
        self.assertIn('id', tool_call)
        self.assertEqual(tool_call['type'], 'function')
        self.assertIn('function', tool_call)
        self.assertIn('name', tool_call['function'])
        self.assertIn('arguments', tool_call['function'])

        # Verify arguments is valid JSON
        args = json.loads(tool_call['function']['arguments'])
        self.assertIsInstance(args, dict)

    def test_tool_message_references_tool_call(self):
        """Test that tool message has correct tool_call_id."""
        orchestrator = ConversationOrchestrator()
        assembler = ConversationAssembler(mock_mode=True)

        config = orchestrator.get_next_generation_config()
        conversation = assembler.assemble_conversation(config)

        assistant_msg = conversation['messages'][2]
        tool_msg = conversation['messages'][3]

        # Get tool_call_id from assistant
        tool_call_id = assistant_msg['tool_calls'][0]['id']

        # Verify tool message references it
        self.assertEqual(tool_msg['role'], 'tool')
        self.assertEqual(tool_msg['tool_call_id'], tool_call_id)
        self.assertIn('content', tool_msg)

        # Verify content is valid JSON
        content = json.loads(tool_msg['content'])
        self.assertIsInstance(content, dict)


class TestConversationIDGeneration(unittest.TestCase):
    """Test unique conversation ID generation."""

    def test_conversation_ids_are_unique(self):
        """Test that generated conversation IDs are unique."""
        orchestrator = ConversationOrchestrator()

        ids = set()
        for _ in range(100):
            config = orchestrator.get_next_generation_config()
            conv_id = config['conversation_id']

            # Check format (should be UUID)
            self.assertIsInstance(conv_id, str)
            self.assertGreater(len(conv_id), 20)

            # Check uniqueness
            self.assertNotIn(conv_id, ids, "Duplicate conversation ID generated")
            ids.add(conv_id)

    def test_conversation_id_in_metadata(self):
        """Test that conversation ID appears in metadata."""
        orchestrator = ConversationOrchestrator()
        assembler = ConversationAssembler(mock_mode=True)

        config = orchestrator.get_next_generation_config()
        conversation = assembler.assemble_conversation(config)

        self.assertIn('metadata', conversation)
        self.assertIn('conversation_id', conversation['metadata'])
        self.assertEqual(
            conversation['metadata']['conversation_id'],
            config['conversation_id']
        )


class TestMetadataTagAssignment(unittest.TestCase):
    """Test metadata and tag assignment."""

    def test_tags_field_present_and_complete(self):
        """Test that tags field contains all required fields."""
        orchestrator = ConversationOrchestrator()
        assembler = ConversationAssembler(mock_mode=True)

        config = orchestrator.get_next_generation_config()
        conversation = assembler.assemble_conversation(config)

        # Verify tags field exists
        self.assertIn('tags', conversation)
        tags = conversation['tags']

        # Verify required tag fields
        required_tags = ['persona', 'tone', 'domain', 'source']
        for tag_name in required_tags:
            self.assertIn(tag_name, tags, f"Missing required tag: {tag_name}")

        # Verify tag values
        self.assertIn(tags['persona'], ['neutral', 'twain', 'franklin'])
        self.assertIn(tags['tone'], ['neutral', 'humorous', 'didactic'])
        self.assertIsInstance(tags['domain'], list)
        self.assertIn('weather', tags['domain'])
        self.assertIn('tool_use', tags['domain'])
        self.assertEqual(tags['source'], 'synthetic')

    def test_persona_tone_consistency(self):
        """Test that persona and tone tags are consistent."""
        orchestrator = ConversationOrchestrator()
        assembler = ConversationAssembler(mock_mode=True)

        # Generate multiple conversations
        for _ in range(20):
            config = orchestrator.get_next_generation_config()
            conversation = assembler.assemble_conversation(config)

            persona = conversation['tags']['persona']
            tone = conversation['tags']['tone']

            # Verify persona-tone mapping
            expected_tone = {
                'neutral': 'neutral',
                'twain': 'humorous',
                'franklin': 'didactic'
            }

            self.assertEqual(
                tone,
                expected_tone[persona],
                f"Persona '{persona}' should have tone '{expected_tone[persona]}', got '{tone}'"
            )

    def test_metadata_includes_location_info(self):
        """Test that metadata includes location information."""
        orchestrator = ConversationOrchestrator()
        assembler = ConversationAssembler(mock_mode=True)

        config = orchestrator.get_next_generation_config()
        conversation = assembler.assemble_conversation(config)

        # Verify metadata structure
        self.assertIn('metadata', conversation)
        metadata = conversation['metadata']

        self.assertIn('location', metadata)
        location = metadata['location']

        # Verify location fields
        self.assertIn('city', location)
        self.assertIn('country', location)
        self.assertIn('climate_zone', location)

        # Verify values are not empty
        self.assertGreater(len(location['city']), 0)
        self.assertGreater(len(location['country']), 0)
        self.assertGreater(len(location['climate_zone']), 0)

    def test_metadata_includes_generation_info(self):
        """Test that metadata includes generation timestamp and info."""
        orchestrator = ConversationOrchestrator()
        assembler = ConversationAssembler(mock_mode=True)

        config = orchestrator.get_next_generation_config()
        conversation = assembler.assemble_conversation(config)

        metadata = conversation['metadata']

        # Verify generation info
        self.assertIn('generated_at', metadata)
        self.assertIn('query_type', metadata)
        self.assertIn('generator', metadata)

        # Verify query_type is valid
        self.assertIn(metadata['query_type'], ['current', 'forecast', 'geocode'])


class TestGeographicDiversity(unittest.TestCase):
    """Test that generated conversations use diverse locations."""

    def test_location_diversity_across_100_generations(self):
        """Test that at least 50 unique locations are used in 100 generations."""
        orchestrator = ConversationOrchestrator()

        # Generate 100 configs
        for _ in range(100):
            orchestrator.get_next_generation_config()

        stats = orchestrator.get_statistics()

        # Should use at least 50 unique locations (as per spec)
        self.assertGreaterEqual(
            stats['unique_locations'],
            50,
            f"Expected at least 50 unique locations, got {stats['unique_locations']}"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
