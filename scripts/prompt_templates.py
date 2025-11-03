#!/usr/bin/env python3
"""
Prompt Template Module for Synthetic Conversation Generation

Creates prompts for Claude API to generate weather tool-use conversations.
Supports multiple personas (neutral, Twain, Franklin) and scenarios.

Usage:
    from scripts.prompt_templates import PromptTemplateEngine

    engine = PromptTemplateEngine()
    prompt = engine.generate_prompt(scenario='success', persona='twain', location=nyc)
"""

import random
from typing import Dict, Any, Optional, List
from scripts.geographic_database import Location
from scripts.weather_tool_schema import get_tool_schemas


class PromptTemplateEngine:
    """
    Generate prompts for synthetic conversation generation.

    Supports:
    - Multiple personas (neutral, Twain, Franklin)
    - Different scenarios (success, error, multi-turn)
    - Dynamic location context injection
    """

    def __init__(self):
        """Initialize prompt template engine."""
        self.tool_schemas = get_tool_schemas()

    def get_persona_instructions(self, persona: str) -> str:
        """
        Get persona-specific instructions for assistant responses.

        Args:
            persona: Persona name ('neutral', 'twain', 'franklin')

        Returns:
            Persona instruction string
        """
        persona_instructions = {
            'neutral': """You are a professional weather assistant. Provide accurate, helpful weather information in a clear and direct manner. Focus on delivering precise data from the weather tools.""",

            'twain': """You are a weather assistant with Mark Twain's wit and style. After using the weather tools, respond with Twain's characteristic humor, irony, and folksy wisdom. Make clever observations about the weather data. Channel his voice: "Everybody complains about the weather, but nobody does anything about it." Be amusing while staying grounded in the actual tool output.""",

            'franklin': """You are a weather assistant in the style of Benjamin Franklin. After using the weather tools, respond with Franklin's didactic, almanac-style wisdom. Turn weather observations into practical advice and moral lessons. Channel Poor Richard's style: "Early to bed and early to rise, makes a man healthy, wealthy and wise." Be instructive while staying accurate to the tool data."""
        }

        return persona_instructions.get(persona, persona_instructions['neutral'])

    def get_system_prompt(
        self,
        persona: str = 'neutral',
        scenario: str = 'success'
    ) -> str:
        """
        Generate system prompt for conversation generation.

        Args:
            persona: Persona name ('neutral', 'twain', 'franklin')
            scenario: Scenario type ('success', 'error', 'multi_turn')

        Returns:
            System prompt string
        """
        persona_instruction = self.get_persona_instructions(persona)

        base_prompt = f"""{persona_instruction}

You have access to weather tools to answer user queries. Always use the appropriate tool before providing your response.

Important guidelines:
1. Use the tool_calls format correctly - call the function with proper JSON arguments
2. Wait for the tool response before giving your final answer
3. Reference the actual data from the tool response - do not make up weather data
4. The tool calling JSON structure should be identical regardless of persona
5. Only apply persona style to your final response AFTER receiving tool results

Available weather tools:
- get_current_weather(latitude, longitude): Get current weather conditions
- get_forecast(latitude, longitude, days): Get multi-day forecast (1-14 days)
- geocode_location(city, country): Convert city name to coordinates
"""

        if scenario == 'error':
            base_prompt += """
For this conversation, you should encounter an error with the tool call (invalid parameters, missing data, or location not found). Handle the error gracefully and explain the issue to the user."""

        elif scenario == 'multi_turn':
            base_prompt += """
This should be a multi-turn conversation where the user asks follow-up questions or requests additional information after the initial weather query."""

        return base_prompt

    def generate_user_query(
        self,
        location: Location,
        scenario: str = 'success',
        query_type: str = 'current'
    ) -> str:
        """
        Generate realistic user query for weather information.

        Args:
            location: Location object for context
            scenario: Scenario type ('success', 'error', 'multi_turn')
            query_type: Type of query ('current', 'forecast', 'geocode')

        Returns:
            User query string
        """
        if scenario == 'error':
            # Generate queries that will cause errors
            error_queries = [
                f"What's the weather at latitude 100 and longitude 50?",  # Invalid lat
                f"Can you get the forecast for 20 days?",  # Days out of range
                f"What's the weather in Nonexistent City, Fake Country?",  # Unknown location
                f"Get me the current weather",  # Missing location
            ]
            return random.choice(error_queries)

        # Success case queries
        if query_type == 'current':
            current_queries = [
                f"What's the current weather in {location.city}, {location.country}?",
                f"Can you tell me the weather conditions in {location.city} right now?",
                f"How's the weather in {location.city} today?",
                f"I need the current weather for {location.city}, {location.country}",
                f"What's it like outside in {location.city}?",
            ]
            return random.choice(current_queries)

        elif query_type == 'forecast':
            days = random.randint(3, 7)
            forecast_queries = [
                f"What's the {days}-day forecast for {location.city}, {location.country}?",
                f"Can you give me the weather forecast for {location.city} over the next {days} days?",
                f"I'm traveling to {location.city} - what's the forecast for the next {days} days?",
                f"Show me the {days}-day weather outlook for {location.city}",
            ]
            return random.choice(forecast_queries)

        elif query_type == 'geocode':
            geocode_queries = [
                f"Can you find the coordinates for {location.city}, {location.country}?",
                f"What are the latitude and longitude of {location.city}?",
                f"I need the geographic coordinates of {location.city}, {location.country}",
            ]
            return random.choice(geocode_queries)

        return f"What's the weather in {location.city}?"

    def generate_multi_turn_followup(
        self,
        location: Location,
        initial_query_type: str
    ) -> str:
        """
        Generate follow-up query for multi-turn conversation.

        Args:
            location: Location object for context
            initial_query_type: Type of initial query

        Returns:
            Follow-up query string
        """
        if initial_query_type == 'current':
            followups = [
                f"What about the forecast for the next few days?",
                f"How does that compare to tomorrow's weather?",
                f"What's the extended forecast looking like?",
                f"Is it going to rain later this week?",
            ]
        elif initial_query_type == 'forecast':
            followups = [
                f"What's the current weather right now?",
                f"Which day looks best for outdoor activities?",
                f"When should I expect the warmest temperatures?",
            ]
        else:
            followups = [
                f"Now can you tell me the current weather there?",
                f"What's the forecast for that location?",
            ]

        return random.choice(followups)

    def create_generation_prompt(
        self,
        location: Location,
        persona: str = 'neutral',
        scenario: str = 'success',
        query_type: str = 'current'
    ) -> Dict[str, Any]:
        """
        Create complete prompt for conversation generation.

        Args:
            location: Location object
            persona: Persona name
            scenario: Scenario type
            query_type: Type of weather query

        Returns:
            Dictionary with system prompt and user query
        """
        system_prompt = self.get_system_prompt(persona, scenario)
        user_query = self.generate_user_query(location, scenario, query_type)

        return {
            'system': system_prompt,
            'user_query': user_query,
            'location': location,
            'persona': persona,
            'scenario': scenario,
            'query_type': query_type,
            'tools': self.tool_schemas
        }

    def get_instruction_for_claude(
        self,
        location: Location,
        persona: str = 'neutral',
        scenario: str = 'success',
        query_type: str = 'current'
    ) -> str:
        """
        Generate instruction for Claude to create a synthetic conversation.

        This is a meta-prompt that asks Claude to generate a complete conversation.

        Args:
            location: Location object
            persona: Persona name
            scenario: Scenario type
            query_type: Query type

        Returns:
            Meta-prompt string for Claude
        """
        persona_instruction = self.get_persona_instructions(persona)

        prompt = f"""Generate a realistic weather assistant conversation in OpenAI tool calling format.

Context:
- Location: {location.city}, {location.country} (lat: {location.latitude}, lon: {location.longitude})
- Climate: {location.climate_zone}
- Persona: {persona}
- Scenario: {scenario}
- Query type: {query_type}

Persona Instructions:
{persona_instruction}

Generate a complete conversation with these messages in order:
1. System message: Weather assistant instructions (include persona style)
2. User message: "{self.generate_user_query(location, scenario, query_type)}"
3. Assistant message with tool_calls: Call the appropriate weather tool(s) with correct JSON arguments
4. Tool message: Realistic weather data response (based on {location.city}'s climate)
5. Assistant message: Final response using the tool data (apply persona style here)

Output as a JSON object with this structure:
{{
  "messages": [
    {{"role": "system", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "I'll check the weather for you.", "tool_calls": [{{"id": "call_abc123", "type": "function", "function": {{"name": "get_current_weather", "arguments": "{{\\"latitude\\": {location.latitude}, \\"longitude\\": {location.longitude}}}"}}}}]}},
    {{"role": "tool", "tool_call_id": "call_abc123", "content": "{{...realistic weather JSON...}}"}},
    {{"role": "assistant", "content": "...final response with persona style..."}}
  ]
}}

Important:
- Make weather data realistic for {location.city}'s {location.climate_zone} climate
- Tool call JSON must be valid and match tool schemas
- Only apply persona to final assistant response, NOT to tool calls
- Keep responses natural and conversational"""

        if scenario == 'multi_turn':
            prompt += f"""
- Add 2-3 turns of follow-up conversation with additional tool calls"""

        return prompt


if __name__ == "__main__":
    # Test prompt template engine
    print("Prompt Template Engine Test")
    print("=" * 60)

    from scripts.geographic_database import get_location_by_city

    engine = PromptTemplateEngine()

    # Test different personas
    print("\n1. Testing persona instructions:")
    for persona in ['neutral', 'twain', 'franklin']:
        instruction = engine.get_persona_instructions(persona)
        print(f"\n{persona.upper()}:")
        print(f"  {instruction[:100]}...")

    # Test system prompts
    print("\n" + "=" * 60)
    print("2. Testing system prompts:")
    for scenario in ['success', 'error', 'multi_turn']:
        system = engine.get_system_prompt('neutral', scenario)
        print(f"\n{scenario.upper()}: {len(system)} chars")

    # Test user queries
    print("\n" + "=" * 60)
    print("3. Testing user queries:")
    nyc = get_location_by_city('New York')
    for query_type in ['current', 'forecast', 'geocode']:
        query = engine.generate_user_query(nyc, 'success', query_type)
        print(f"\n{query_type}: {query}")

    # Test complete prompt
    print("\n" + "=" * 60)
    print("4. Testing complete prompt generation:")
    london = get_location_by_city('London')
    prompt_data = engine.create_generation_prompt(london, 'twain', 'success', 'current')
    print(f"  Location: {prompt_data['location'].city}")
    print(f"  Persona: {prompt_data['persona']}")
    print(f"  Scenario: {prompt_data['scenario']}")
    print(f"  System prompt length: {len(prompt_data['system'])} chars")
    print(f"  User query: {prompt_data['user_query']}")
