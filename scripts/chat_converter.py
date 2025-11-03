#!/usr/bin/env python3
"""
Chat Converter Module for Instructionalization Pipeline

Converts normalized passages to OpenAI-style chat format with
system/user/assistant roles. Supports single-turn and multi-turn conversations.

Usage:
    from scripts.chat_converter import convert_to_single_turn, convert_to_multi_turn

    entry = convert_to_single_turn(item, "twain")
"""

from typing import Dict, Any, List
from system_message_generator import generate_system_message
from user_message_generator import generate_user_message


def count_words(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Text string

    Returns:
        Word count

    Examples:
        >>> count_words("Hello world")
        2
    """
    return len(text.split())


def convert_to_single_turn(item: Dict[str, Any], persona: str) -> Dict[str, Any]:
    """
    Convert item to single-turn chat format.

    Args:
        item: Data item with text content
        persona: Persona tag for system message

    Returns:
        Dictionary with messages array

    Chat format structure:
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }

    Examples:
        >>> item = {"text": "Some passage content"}
        >>> entry = convert_to_single_turn(item, "twain")
        >>> len(entry["messages"])
        3
        >>> entry["messages"][0]["role"]
        'system'
    """
    # Get text content (try multiple field names)
    text = item.get('text') or item.get('content') or item.get('cleaned_title', '')

    # Generate messages
    system_message = generate_system_message(persona)
    user_message = generate_user_message()
    assistant_message = {
        'role': 'assistant',
        'content': text
    }

    return {
        'messages': [system_message, user_message, assistant_message]
    }


def convert_to_multi_turn(item: Dict[str, Any], persona: str) -> Dict[str, Any]:
    """
    Convert item to multi-turn chat format for long passages.

    Args:
        item: Data item with text content
        persona: Persona tag for system message

    Returns:
        Dictionary with messages array (multiple exchanges)

    Multi-turn logic:
        - Used for passages >300 words
        - Splits content into initial response + follow-up
        - Adds additional user query and assistant continuation

    Examples:
        >>> item = {"text": "Very long passage content... " * 100}
        >>> entry = convert_to_multi_turn(item, "twain")
        >>> len(entry["messages"]) > 3
        True
    """
    # Get text content
    text = item.get('text') or item.get('content') or item.get('cleaned_title', '')

    # Split text into two parts (roughly 60/40)
    words = text.split()
    split_point = int(len(words) * 0.6)

    first_part = ' '.join(words[:split_point])
    second_part = ' '.join(words[split_point:])

    # Generate initial exchange
    system_message = generate_system_message(persona)
    user_message_1 = generate_user_message()
    assistant_message_1 = {
        'role': 'assistant',
        'content': first_part
    }

    # Generate follow-up exchange
    user_message_2 = {
        'role': 'user',
        'content': 'Tell me more'
    }
    assistant_message_2 = {
        'role': 'assistant',
        'content': second_part
    }

    return {
        'messages': [
            system_message,
            user_message_1,
            assistant_message_1,
            user_message_2,
            assistant_message_2
        ]
    }


def convert_to_chat_format(item: Dict[str, Any], persona: str) -> Dict[str, Any]:
    """
    Convert item to chat format (automatically choose single/multi-turn).

    Args:
        item: Data item with text content
        persona: Persona tag for system message

    Returns:
        Dictionary with messages array

    Logic:
        - If text >300 words → multi-turn
        - Otherwise → single-turn

    Examples:
        >>> item = {"text": "Short passage"}
        >>> entry = convert_to_chat_format(item, "neutral")
        >>> len(entry["messages"])
        3
    """
    # Get text content
    text = item.get('text') or item.get('content') or item.get('cleaned_title', '')

    # Get word count (use pre-computed if available)
    word_count = item.get('word_count')
    if word_count is None:
        word_count = count_words(text)

    # Choose format based on length
    if word_count > 300:
        return convert_to_multi_turn(item, persona)
    else:
        return convert_to_single_turn(item, persona)


if __name__ == "__main__":
    # Test chat converter
    print("Chat Converter Test")
    print("=" * 60)

    # Test single-turn
    short_item = {
        'text': 'This is a short weather passage about storms and rain.',
        'word_count': 10
    }

    print("\nTesting single-turn conversion:")
    entry1 = convert_to_single_turn(short_item, 'twain')
    print(f"  Messages: {len(entry1['messages'])}")
    for msg in entry1['messages']:
        print(f"    {msg['role']:10s}: {msg['content'][:60]}...")

    # Test multi-turn
    long_text = "This is a much longer passage about weather. " * 100
    long_item = {
        'text': long_text,
        'word_count': 800
    }

    print("\nTesting multi-turn conversion:")
    entry2 = convert_to_multi_turn(long_item, 'franklin')
    print(f"  Messages: {len(entry2['messages'])}")
    for msg in entry2['messages']:
        content_preview = msg['content'][:50] + '...' if len(msg['content']) > 50 else msg['content']
        print(f"    {msg['role']:10s}: {content_preview}")

    # Test automatic selection
    print("\n" + "=" * 60)
    print("Testing automatic format selection:")

    test_items = [
        {'text': 'Short text.', 'word_count': 2},
        {'text': 'Medium text. ' * 50, 'word_count': 100},
        {'text': 'Long text. ' * 200, 'word_count': 400}
    ]

    for i, item in enumerate(test_items, 1):
        entry = convert_to_chat_format(item, 'neutral')
        print(f"  Item {i} ({item['word_count']} words): {len(entry['messages'])} messages")
