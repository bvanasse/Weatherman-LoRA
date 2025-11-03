#!/usr/bin/env python3
"""
Conversation Parser Module

Parses Claude API responses into structured conversation objects.
Validates tool_calls format and extracts relevant data.

Usage:
    from scripts.conversation_parser import ConversationParser

    parser = ConversationParser()
    conversation = parser.parse_response(claude_response)
"""

import json
import re
from typing import Dict, Any, Optional, List, Tuple


class ConversationParser:
    """
    Parse Claude API responses into structured conversations.

    Handles:
    - Extracting tool_calls from assistant messages
    - Validating JSON structure
    - Converting to OpenAI chat format
    """

    def __init__(self):
        """Initialize conversation parser."""
        pass

    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from text (handles markdown code blocks).

        Args:
            text: Text potentially containing JSON

        Returns:
            Parsed JSON dictionary or None if not found

        Examples:
            >>> parser = ConversationParser()
            >>> text = '```json\\n{"key": "value"}\\n```'
            >>> result = parser.extract_json_from_text(text)
            >>> result['key']
            'value'
        """
        # Try to find JSON in markdown code block
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(json_pattern, text, re.DOTALL)

        if match:
            json_str = match.group(1).strip()
        else:
            # Try treating entire text as JSON
            json_str = text.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def validate_tool_call_structure(
        self,
        tool_call: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validate tool call structure matches OpenAI format.

        Args:
            tool_call: Tool call dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if 'id' not in tool_call:
            return False, "Missing 'id' field in tool_call"

        if 'type' not in tool_call:
            return False, "Missing 'type' field in tool_call"

        if tool_call['type'] != 'function':
            return False, f"Invalid type: {tool_call['type']} (expected 'function')"

        if 'function' not in tool_call:
            return False, "Missing 'function' field in tool_call"

        function = tool_call['function']

        if not isinstance(function, dict):
            return False, "'function' must be a dictionary"

        if 'name' not in function:
            return False, "Missing 'name' in function"

        if 'arguments' not in function:
            return False, "Missing 'arguments' in function"

        # Validate arguments is valid JSON string
        try:
            if isinstance(function['arguments'], str):
                json.loads(function['arguments'])
            elif isinstance(function['arguments'], dict):
                # Convert dict to JSON string
                function['arguments'] = json.dumps(function['arguments'])
        except json.JSONDecodeError:
            return False, "Invalid JSON in function arguments"

        return True, ""

    def validate_message_structure(
        self,
        message: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validate message structure.

        Args:
            message: Message dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check role
        if 'role' not in message:
            return False, "Missing 'role' field"

        valid_roles = ['system', 'user', 'assistant', 'tool']
        if message['role'] not in valid_roles:
            return False, f"Invalid role: {message['role']}"

        # Check content (tool messages might not have content)
        if message['role'] != 'assistant' or 'tool_calls' not in message:
            if 'content' not in message:
                if message['role'] != 'tool':
                    return False, "Missing 'content' field"

        # Validate tool_calls if present
        if 'tool_calls' in message:
            if not isinstance(message['tool_calls'], list):
                return False, "'tool_calls' must be a list"

            for tool_call in message['tool_calls']:
                is_valid, error = self.validate_tool_call_structure(tool_call)
                if not is_valid:
                    return False, f"Invalid tool_call: {error}"

        # Validate tool message specific fields
        if message['role'] == 'tool':
            if 'tool_call_id' not in message:
                return False, "Tool message missing 'tool_call_id'"
            if 'content' not in message:
                return False, "Tool message missing 'content'"

        return True, ""

    def parse_conversation_from_json(
        self,
        conversation_json: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Parse and validate conversation JSON.

        Args:
            conversation_json: Conversation dictionary with 'messages' array

        Returns:
            Tuple of (parsed_conversation, error_message)
            - (conversation_dict, "") if valid
            - (None, "error description") if invalid
        """
        # Check messages field
        if 'messages' not in conversation_json:
            return None, "Missing 'messages' field"

        messages = conversation_json['messages']

        if not isinstance(messages, list):
            return None, "'messages' must be a list"

        if len(messages) == 0:
            return None, "'messages' array is empty"

        # Validate each message
        for i, message in enumerate(messages):
            is_valid, error = self.validate_message_structure(message)
            if not is_valid:
                return None, f"Message {i} invalid: {error}"

        # Check role ordering
        roles = [msg['role'] for msg in messages]

        # First must be system
        if roles[0] != 'system':
            return None, f"First role must be 'system', got '{roles[0]}'"

        # Last must be assistant
        if roles[-1] != 'assistant':
            return None, f"Last role must be 'assistant', got '{roles[-1]}'"

        return conversation_json, ""

    def parse_claude_response(
        self,
        response_text: str
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Parse Claude API response text into conversation.

        Args:
            response_text: Raw text response from Claude

        Returns:
            Tuple of (parsed_conversation, error_message)
        """
        # Extract JSON from response
        conversation_json = self.extract_json_from_text(response_text)

        if conversation_json is None:
            return None, "Could not extract valid JSON from response"

        # Parse and validate conversation
        return self.parse_conversation_from_json(conversation_json)

    def extract_tool_calls_from_conversation(
        self,
        conversation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract all tool_calls from conversation messages.

        Args:
            conversation: Conversation dictionary

        Returns:
            List of tool call dictionaries
        """
        tool_calls = []

        for message in conversation.get('messages', []):
            if message.get('role') == 'assistant' and 'tool_calls' in message:
                tool_calls.extend(message['tool_calls'])

        return tool_calls

    def get_conversation_metadata(
        self,
        conversation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract metadata from conversation.

        Args:
            conversation: Conversation dictionary

        Returns:
            Metadata dictionary
        """
        messages = conversation.get('messages', [])
        tool_calls = self.extract_tool_calls_from_conversation(conversation)

        return {
            'message_count': len(messages),
            'has_tool_calls': len(tool_calls) > 0,
            'tool_call_count': len(tool_calls),
            'function_names': [tc['function']['name'] for tc in tool_calls],
            'turn_count': sum(1 for msg in messages if msg['role'] == 'user')
        }


if __name__ == "__main__":
    # Test conversation parser
    print("Conversation Parser Test")
    print("=" * 60)

    parser = ConversationParser()

    # Test valid conversation
    test_conversation = {
        'messages': [
            {
                'role': 'system',
                'content': 'You are a weather assistant.'
            },
            {
                'role': 'user',
                'content': 'What is the weather in New York?'
            },
            {
                'role': 'assistant',
                'content': 'Let me check the weather for you.',
                'tool_calls': [
                    {
                        'id': 'call_abc123',
                        'type': 'function',
                        'function': {
                            'name': 'get_current_weather',
                            'arguments': '{"latitude": 40.7128, "longitude": -74.0060}'
                        }
                    }
                ]
            },
            {
                'role': 'tool',
                'tool_call_id': 'call_abc123',
                'content': '{"temperature": 22, "condition": "partly_cloudy"}'
            },
            {
                'role': 'assistant',
                'content': 'The current weather in New York is 22°C and partly cloudy.'
            }
        ]
    }

    print("\n1. Testing valid conversation:")
    conversation, error = parser.parse_conversation_from_json(test_conversation)
    if conversation:
        print("  ✓ Valid conversation")
        metadata = parser.get_conversation_metadata(conversation)
        print(f"    Messages: {metadata['message_count']}")
        print(f"    Tool calls: {metadata['tool_call_count']}")
        print(f"    Functions: {metadata['function_names']}")
    else:
        print(f"  ✗ Invalid: {error}")

    # Test invalid conversations
    print("\n" + "=" * 60)
    print("2. Testing invalid conversations:")

    invalid_cases = [
        ({}, "Empty conversation"),
        ({'messages': []}, "Empty messages"),
        ({'messages': [{'role': 'user', 'content': 'test'}]}, "Wrong first role"),
        ({'messages': [
            {'role': 'system', 'content': 'test'},
            {'role': 'user', 'content': 'test'}
        ]}, "Wrong last role"),
    ]

    for test_conv, description in invalid_cases:
        conversation, error = parser.parse_conversation_from_json(test_conv)
        status = "✗" if conversation is None else "✓"
        print(f"  {status} {description:30s}: {error if error else 'Valid'}")

    # Test JSON extraction
    print("\n" + "=" * 60)
    print("3. Testing JSON extraction:")

    json_texts = [
        ('```json\n{"key": "value"}\n```', "Markdown code block"),
        ('{"key": "value"}', "Plain JSON"),
        ('Some text\n```\n{"key": "value"}\n```\nMore text', "JSON in text"),
    ]

    for text, description in json_texts:
        result = parser.extract_json_from_text(text)
        status = "✓" if result else "✗"
        print(f"  {status} {description:30s}: {result}")
