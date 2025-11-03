#!/usr/bin/env python3
"""
Conversation Assembly Pipeline Module

Assembles complete tool-calling conversations from generation configs.
Coordinates prompt generation, API calls, tool responses, and metadata tagging.

Usage:
    from scripts.conversation_assembly import ConversationAssembler

    assembler = ConversationAssembler(claude_client)
    conversation = assembler.assemble_conversation(config)
"""

import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from scripts.prompt_templates import PromptTemplateEngine
from scripts.mock_weather_responses import generate_weather_response
from scripts.geographic_database import Location


class ConversationAssembler:
    """
    Assemble complete tool-calling conversations.

    Creates structured conversations with:
    - System message with persona instructions
    - User query
    - Assistant message with tool_calls
    - Tool response message
    - Final assistant response
    - Metadata and tags
    """

    def __init__(self, claude_client=None, mock_mode: bool = False):
        """
        Initialize conversation assembler.

        Args:
            claude_client: ClaudeAPIClient instance (optional if mock_mode=True)
            mock_mode: If True, generate conversations without API calls (for testing)
        """
        self.claude_client = claude_client
        self.mock_mode = mock_mode
        self.prompt_engine = PromptTemplateEngine()

    def create_system_message(self, persona: str, scenario: str) -> Dict[str, str]:
        """
        Create system message with persona and scenario instructions.

        Args:
            persona: Persona name
            scenario: Scenario type

        Returns:
            System message dictionary
        """
        system_prompt = self.prompt_engine.get_system_prompt(persona, scenario)
        return {
            'role': 'system',
            'content': system_prompt
        }

    def create_user_message(
        self,
        location: Location,
        scenario: str,
        query_type: str
    ) -> Dict[str, str]:
        """
        Create user query message.

        Args:
            location: Location object
            scenario: Scenario type
            query_type: Query type

        Returns:
            User message dictionary
        """
        user_query = self.prompt_engine.generate_user_query(location, scenario, query_type)
        return {
            'role': 'user',
            'content': user_query
        }

    def create_tool_call(
        self,
        function_name: str,
        arguments: Dict[str, Any],
        call_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create tool_call object in OpenAI format.

        Args:
            function_name: Name of the function to call
            arguments: Function arguments dictionary
            call_id: Tool call ID (generated if not provided)

        Returns:
            Tool call dictionary
        """
        if call_id is None:
            call_id = f"call_{uuid.uuid4().hex[:12]}"

        return {
            'id': call_id,
            'type': 'function',
            'function': {
                'name': function_name,
                'arguments': json.dumps(arguments)
            }
        }

    def create_assistant_tool_call_message(
        self,
        location: Location,
        query_type: str,
        scenario: str
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        Create assistant message with tool_calls.

        Args:
            location: Location object
            query_type: Query type
            scenario: Scenario type

        Returns:
            Tuple of (assistant message, list of tool_call_ids)
        """
        tool_calls = []
        call_ids = []

        if scenario == 'error':
            # Create intentionally invalid tool calls for error scenarios
            if query_type == 'current':
                # Invalid latitude
                tool_call = self.create_tool_call(
                    'get_current_weather',
                    {'latitude': 100, 'longitude': location.longitude}
                )
            elif query_type == 'forecast':
                # Days out of range
                tool_call = self.create_tool_call(
                    'get_forecast',
                    {'latitude': location.latitude, 'longitude': location.longitude, 'days': 20}
                )
            else:
                # Unknown city
                tool_call = self.create_tool_call(
                    'geocode_location',
                    {'city': 'Nonexistent City', 'country': 'Fake Country'}
                )
        else:
            # Create valid tool calls for success/multi-turn scenarios
            if query_type == 'current':
                tool_call = self.create_tool_call(
                    'get_current_weather',
                    {'latitude': location.latitude, 'longitude': location.longitude}
                )
            elif query_type == 'forecast':
                import random
                days = random.randint(3, 7)
                tool_call = self.create_tool_call(
                    'get_forecast',
                    {'latitude': location.latitude, 'longitude': location.longitude, 'days': days}
                )
            else:  # geocode
                tool_call = self.create_tool_call(
                    'geocode_location',
                    {'city': location.city, 'country': location.country}
                )

        tool_calls.append(tool_call)
        call_ids.append(tool_call['id'])

        assistant_message = {
            'role': 'assistant',
            'content': self._get_assistant_intro(query_type),
            'tool_calls': tool_calls
        }

        return assistant_message, call_ids

    def _get_assistant_intro(self, query_type: str) -> str:
        """Get brief intro message before tool call."""
        intros = {
            'current': "Let me check the current weather for you.",
            'forecast': "I'll get the forecast for you.",
            'geocode': "Let me find those coordinates for you."
        }
        return intros.get(query_type, "Let me look that up.")

    def create_tool_response_message(
        self,
        tool_call_id: str,
        function_name: str,
        arguments: Dict[str, Any],
        scenario: str
    ) -> Dict[str, Any]:
        """
        Create tool response message with mock weather data.

        Args:
            tool_call_id: ID of the tool call this responds to
            function_name: Name of the function
            arguments: Function arguments
            scenario: Scenario type (affects whether response is error)

        Returns:
            Tool message dictionary
        """
        if scenario == 'error':
            # Return error response
            from scripts.mock_weather_responses import generate_error_response
            content = json.dumps(generate_error_response(
                'out_of_range' if 'days' in arguments else 'invalid_location',
                function_name
            ))
        else:
            # Return successful mock response
            content = generate_weather_response(function_name, arguments)

        return {
            'role': 'tool',
            'tool_call_id': tool_call_id,
            'content': content
        }

    def create_final_assistant_message(
        self,
        tool_responses: List[Dict[str, Any]],
        persona: str,
        scenario: str,
        location: Location
    ) -> Dict[str, str]:
        """
        Create final assistant response incorporating tool data.

        Args:
            tool_responses: List of tool response messages
            persona: Persona name
            scenario: Scenario type
            location: Location object

        Returns:
            Assistant message dictionary
        """
        # Parse tool response data
        tool_data = json.loads(tool_responses[0]['content'])

        if scenario == 'error':
            # Handle error response
            content = self._generate_error_response_text(tool_data, persona)
        else:
            # Handle success response with persona
            content = self._generate_success_response_text(tool_data, persona, location)

        return {
            'role': 'assistant',
            'content': content
        }

    def _generate_error_response_text(self, error_data: Dict[str, Any], persona: str) -> str:
        """Generate error response text."""
        error_msg = error_data.get('message', 'An error occurred')

        if persona == 'neutral':
            return f"I apologize, but I encountered an error: {error_msg}"
        elif persona == 'twain':
            return f"Well, I'd tell you about the weather, but it seems the weather gods are playing tricks on us. Error: {error_msg}. As they say, if you don't like the data, wait five minutes!"
        else:  # franklin
            return f"As Poor Richard would say, 'He that waits upon fortune is never sure of a dinner.' In this case, the weather service has let us down: {error_msg}"

    def _generate_success_response_text(
        self,
        weather_data: Dict[str, Any],
        persona: str,
        location: Location
    ) -> str:
        """Generate successful response text with persona style."""
        # Extract key weather info
        if 'temperature' in weather_data:
            # Current weather
            temp = weather_data['temperature']
            condition = weather_data.get('condition', 'unknown')

            if persona == 'neutral':
                return f"The current weather in {location.city} is {temp}°C with {condition.replace('_', ' ')} conditions."
            elif persona == 'twain':
                return f"Ah, {location.city}! Where it's currently {temp}°C and {condition.replace('_', ' ')}. Climate is what we expect, weather is what we get – and today, you're getting {condition.replace('_', ' ')} whether you like it or not!"
            else:  # franklin
                return f"In {location.city}, the temperature stands at {temp}°C with {condition.replace('_', ' ')} skies. Early to bed and early to rise makes a man healthy, wealthy, and wise – especially with weather like this!"

        elif 'daily' in weather_data:
            # Forecast
            days = weather_data.get('forecast_days', len(weather_data['daily']))
            first_day = weather_data['daily'][0]
            temp_range = f"{first_day['temperature_min']}°C to {first_day['temperature_max']}°C"

            if persona == 'neutral':
                return f"The {days}-day forecast for {location.city} shows temperatures ranging from {temp_range} tomorrow, with {first_day['condition'].replace('_', ' ')} conditions."
            elif persona == 'twain':
                return f"The forecast for {location.city} looks like {temp_range} tomorrow. But remember, weather forecasts and promises are alike – they're not always kept!"
            else:  # franklin
                return f"The weather prognostication for {location.city} suggests {temp_range} on the morrow. As I always say, 'Some are weatherwise, some are otherwise.'"

        else:
            # Geocode or other
            return f"Here's the information you requested for {location.city}."

    def assemble_conversation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assemble complete conversation from generation config.

        Args:
            config: Generation configuration dictionary

        Returns:
            Complete conversation dictionary with messages and metadata
        """
        scenario = config['scenario']
        persona = config['persona']
        query_type = config['query_type']
        location = config['location']

        # Build messages
        messages = []

        # 1. System message
        messages.append(self.create_system_message(persona, scenario))

        # 2. User message
        messages.append(self.create_user_message(location, scenario, query_type))

        # 3. Assistant message with tool_calls
        assistant_msg, call_ids = self.create_assistant_tool_call_message(
            location, query_type, scenario
        )
        messages.append(assistant_msg)

        # 4. Tool response message(s)
        tool_responses = []
        for call_id, tool_call in zip(call_ids, assistant_msg['tool_calls']):
            function_name = tool_call['function']['name']
            arguments = json.loads(tool_call['function']['arguments'])

            tool_msg = self.create_tool_response_message(
                call_id, function_name, arguments, scenario
            )
            messages.append(tool_msg)
            tool_responses.append(tool_msg)

        # 5. Final assistant response
        final_msg = self.create_final_assistant_message(
            tool_responses, persona, scenario, location
        )
        messages.append(final_msg)

        # Create conversation object with metadata
        conversation = {
            'messages': messages,
            'tags': {
                'persona': persona,
                'tone': self._get_tone_tag(persona),
                'domain': ['weather', 'tool_use'],
                'source': 'synthetic',
                'scenario': scenario
            },
            'metadata': {
                'conversation_id': config['conversation_id'],
                'location': {
                    'city': location.city,
                    'country': location.country,
                    'climate_zone': location.climate_zone
                },
                'query_type': query_type,
                'generated_at': config['timestamp'],
                'generator': 'conversation_assembly_pipeline'
            }
        }

        return conversation

    def _get_tone_tag(self, persona: str) -> str:
        """Get tone tag for persona."""
        tone_map = {
            'neutral': 'neutral',
            'twain': 'humorous',
            'franklin': 'didactic'
        }
        return tone_map.get(persona, 'neutral')


if __name__ == "__main__":
    # Test conversation assembler
    print("Conversation Assembly Test")
    print("=" * 60)

    from scripts.conversation_orchestrator import ConversationOrchestrator
    from scripts.geographic_database import get_location_by_city

    # Create assembler in mock mode (no API calls)
    assembler = ConversationAssembler(mock_mode=True)
    orchestrator = ConversationOrchestrator()

    # Generate test conversations
    print("\nGenerating test conversations...")

    for i in range(3):
        config = orchestrator.get_next_generation_config()
        conversation = assembler.assemble_conversation(config)

        print(f"\n{'='*60}")
        print(f"Conversation {i+1}:")
        print(f"  Persona: {conversation['tags']['persona']}")
        print(f"  Scenario: {conversation['tags']['scenario']}")
        print(f"  Location: {conversation['metadata']['location']['city']}")
        print(f"  Messages: {len(conversation['messages'])}")

        # Show conversation structure
        for j, msg in enumerate(conversation['messages']):
            role = msg['role']
            content_preview = msg.get('content', '')[:50] if msg.get('content') else '[tool_calls]'
            print(f"    {j+1}. {role:10s}: {content_preview}...")

    print(f"\n{'='*60}")
    print("Test complete!")
