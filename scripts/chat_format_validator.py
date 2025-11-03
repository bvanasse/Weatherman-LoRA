#!/usr/bin/env python3
"""
Chat Format Validator Module for Instructionalization Pipeline

Validates chat-format entries for schema compliance.
Checks messages array structure, roles, and tag fields.

Usage:
    from scripts.chat_format_validator import validate_chat_entry

    is_valid = validate_chat_entry(entry)
"""

from typing import Dict, Any, List, Tuple


def validate_chat_entry(entry: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate chat-format entry for schema compliance.

    Args:
        entry: Chat-format entry dictionary

    Returns:
        Tuple of (is_valid, error_message)
        - (True, "") if valid
        - (False, "error description") if invalid

    Validation checks:
        - Entry has 'messages' array
        - Each message has 'role' and 'content' fields (or tool_calls for assistant)
        - Roles are in valid order (system, user, assistant, [tool, assistant]*, [user, assistant]*)
        - Entry has 'tags' field
        - Tags contain persona, tone, domain
        - Tool messages validated for tool_call_id
        - Assistant tool_calls validated for structure

    Examples:
        >>> entry = {
        ...     "messages": [
        ...         {"role": "system", "content": "System prompt"},
        ...         {"role": "user", "content": "User query"},
        ...         {"role": "assistant", "content": "Assistant response"}
        ...     ],
        ...     "tags": {"persona": "twain", "tone": "humorous", "domain": ["weather"]}
        ... }
        >>> is_valid, error = validate_chat_entry(entry)
        >>> is_valid
        True
    """
    # Check messages field exists
    if 'messages' not in entry:
        return False, "Missing 'messages' field"

    messages = entry['messages']

    # Check messages is a list
    if not isinstance(messages, list):
        return False, "'messages' is not a list"

    # Check messages is not empty
    if len(messages) == 0:
        return False, "'messages' array is empty"

    # Validate each message
    for i, msg in enumerate(messages):
        # Check message is dict
        if not isinstance(msg, dict):
            return False, f"Message {i} is not a dictionary"

        # Check role field
        if 'role' not in msg:
            return False, f"Message {i} missing 'role' field"

        role = msg['role']

        # Check content field (not required for assistant with tool_calls, or tool)
        if role == 'assistant' and 'tool_calls' in msg:
            # Assistant with tool_calls may have empty or no content
            pass
        elif role == 'tool':
            # Tool messages must have content and tool_call_id
            if 'tool_call_id' not in msg:
                return False, f"Message {i} (tool) missing 'tool_call_id' field"
            if 'content' not in msg:
                return False, f"Message {i} (tool) missing 'content' field"
        else:
            # All other roles must have content
            if 'content' not in msg:
                return False, f"Message {i} missing 'content' field"
            # Check content is string
            if not isinstance(msg['content'], str):
                return False, f"Message {i} 'content' is not a string"

        # Validate tool_calls if present
        if 'tool_calls' in msg:
            if role != 'assistant':
                return False, f"Message {i} has tool_calls but role is '{role}' (must be 'assistant')"

            tool_calls = msg['tool_calls']
            if not isinstance(tool_calls, list):
                return False, f"Message {i} 'tool_calls' must be a list"

            for j, tc in enumerate(tool_calls):
                is_valid, error = validate_tool_call(tc)
                if not is_valid:
                    return False, f"Message {i} tool_call {j} invalid: {error}"

    # Validate role ordering
    roles = [msg['role'] for msg in messages]
    is_valid_order, order_error = validate_role_order(roles)
    if not is_valid_order:
        return False, order_error

    # Check tags field exists
    if 'tags' not in entry:
        return False, "Missing 'tags' field"

    tags = entry['tags']

    # Check tags is dict
    if not isinstance(tags, dict):
        return False, "'tags' is not a dictionary"

    # Check required tag fields
    required_tags = ['persona', 'tone', 'domain']
    for tag_name in required_tags:
        if tag_name not in tags:
            return False, f"Missing required tag: '{tag_name}'"

    # Validate domain is list
    if not isinstance(tags['domain'], list):
        return False, "'domain' tag must be a list"

    return True, ""


def validate_tool_call(tool_call: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate tool_call structure.

    Args:
        tool_call: Tool call dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    if 'id' not in tool_call:
        return False, "Missing 'id' field"

    if 'type' not in tool_call:
        return False, "Missing 'type' field"

    if tool_call['type'] != 'function':
        return False, f"Invalid type: {tool_call['type']}"

    if 'function' not in tool_call:
        return False, "Missing 'function' field"

    function = tool_call['function']
    if not isinstance(function, dict):
        return False, "'function' must be a dict"

    if 'name' not in function:
        return False, "Missing 'name' in function"

    if 'arguments' not in function:
        return False, "Missing 'arguments' in function"

    # Validate arguments is valid JSON (string or dict)
    import json
    try:
        if isinstance(function['arguments'], str):
            json.loads(function['arguments'])
        elif not isinstance(function['arguments'], dict):
            return False, "'arguments' must be JSON string or dict"
    except json.JSONDecodeError:
        return False, "Invalid JSON in 'arguments'"

    return True, ""


def validate_role_order(roles: List[str]) -> Tuple[bool, str]:
    """
    Validate that message roles are in valid order.

    Args:
        roles: List of role strings

    Returns:
        Tuple of (is_valid, error_message)

    Valid patterns:
        - [system, user, assistant] (single-turn)
        - [system, user, assistant, user, assistant] (multi-turn)
        - [system, user, assistant, tool, assistant] (tool-calling)
        - [system, user, assistant, tool, assistant, user, assistant] (multi-turn with tools)
        - First role must be 'system'
        - After assistant with tool_calls, must have tool message(s), then assistant
        - User and assistant alternate (with optional tool sequences)

    Examples:
        >>> validate_role_order(['system', 'user', 'assistant'])
        (True, '')
        >>> validate_role_order(['system', 'user', 'assistant', 'tool', 'assistant'])
        (True, '')
        >>> validate_role_order(['user', 'assistant'])
        (False, 'First role must be system')
    """
    if len(roles) == 0:
        return False, "No roles to validate"

    # First role must be system
    if roles[0] != 'system':
        return False, f"First role must be 'system', got '{roles[0]}'"

    # After system, should alternate user/assistant with optional tool sequences
    expected = 'user'
    i = 1
    while i < len(roles):
        role = roles[i]

        if role not in ['user', 'assistant', 'tool']:
            return False, f"Invalid role at position {i}: '{role}'"

        if role == 'user':
            if expected != 'user':
                return False, f"Unexpected 'user' role at position {i} (expected '{expected}')"
            expected = 'assistant'
            i += 1

        elif role == 'assistant':
            if expected not in ['assistant', 'tool']:
                return False, f"Unexpected 'assistant' role at position {i} (expected '{expected}')"

            # Check if next is tool (indicating tool-calling flow)
            if i + 1 < len(roles) and roles[i + 1] == 'tool':
                expected = 'tool'
            else:
                expected = 'user'
            i += 1

        elif role == 'tool':
            if expected != 'tool':
                return False, f"Unexpected 'tool' role at position {i} (must follow assistant with tool_calls)"

            # After tool(s), expect assistant
            # Consume all sequential tool messages
            while i < len(roles) and roles[i] == 'tool':
                i += 1

            # Must have assistant after tool(s)
            if i >= len(roles):
                return False, "Tool message(s) must be followed by assistant message"

            expected = 'assistant'

    # Must end with assistant
    if roles[-1] != 'assistant':
        return False, f"Last role must be 'assistant', got '{roles[-1]}'"

    return True, ""


def validate_jsonl_file(file_path: str) -> Tuple[bool, int, List[str]]:
    """
    Validate entire JSONL file of chat-format entries.

    Args:
        file_path: Path to JSONL file

    Returns:
        Tuple of (all_valid, valid_count, errors)

    Examples:
        >>> # Assuming valid file exists
        >>> is_valid, count, errors = validate_jsonl_file("test.jsonl")
        >>> is_valid
        True
    """
    import json
    from pathlib import Path

    file_path = Path(file_path)
    if not file_path.exists():
        return False, 0, [f"File not found: {file_path}"]

    valid_count = 0
    errors = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    is_valid, error = validate_chat_entry(entry)

                    if is_valid:
                        valid_count += 1
                    else:
                        errors.append(f"Line {line_num}: {error}")

                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")

    except Exception as e:
        errors.append(f"File read error: {e}")
        return False, 0, errors

    all_valid = len(errors) == 0
    return all_valid, valid_count, errors


if __name__ == "__main__":
    # Test chat format validator
    print("Chat Format Validator Test")
    print("=" * 60)

    # Test valid entry
    valid_entry = {
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is the weather?'},
            {'role': 'assistant', 'content': 'The weather is sunny.'}
        ],
        'tags': {
            'persona': 'neutral',
            'tone': 'humorous',
            'domain': ['weather']
        }
    }

    print("\nValidating valid entry:")
    is_valid, error = validate_chat_entry(valid_entry)
    print(f"  Valid: {is_valid}")
    if not is_valid:
        print(f"  Error: {error}")

    # Test invalid entries
    test_cases = [
        ({}, "Empty entry"),
        ({'messages': []}, "Empty messages"),
        ({'messages': [{'role': 'user'}]}, "Missing content"),
        ({'messages': [{'role': 'user', 'content': 'test'}]}, "Wrong first role"),
        ({'messages': [{'role': 'system', 'content': 'test'}]}, "No user/assistant"),
        ({
            'messages': [
                {'role': 'system', 'content': 'test'},
                {'role': 'user', 'content': 'test'},
                {'role': 'assistant', 'content': 'test'}
            ]
        }, "Missing tags"),
    ]

    print("\nTesting invalid entries:")
    for entry, description in test_cases:
        is_valid, error = validate_chat_entry(entry)
        status = "✗" if not is_valid else "✓"
        print(f"  {status} {description:30s}: {error}")

    # Test role ordering
    print("\n" + "=" * 60)
    print("Testing role ordering:")

    role_tests = [
        (['system', 'user', 'assistant'], True),
        (['system', 'user', 'assistant', 'user', 'assistant'], True),
        (['user', 'assistant'], False),
        (['system', 'assistant'], False),
        (['system', 'user', 'user', 'assistant'], False),
    ]

    for roles, expected_valid in role_tests:
        is_valid, error = validate_role_order(roles)
        status = "✓" if is_valid == expected_valid else "✗"
        result = "Valid" if is_valid else f"Invalid: {error}"
        print(f"  {status} {str(roles):50s} → {result}")
