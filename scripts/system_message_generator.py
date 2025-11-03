#!/usr/bin/env python3
"""
System Message Generator Module for Chat-Format Conversion

Generates persona-aware system messages for chat-format training data.
Creates distinct system prompts for Twain/Franklin/neutral personas.

Usage:
    from scripts.system_message_generator import generate_system_message

    msg = generate_system_message("twain")
    # Returns: {"role": "system", "content": "You are a witty weather assistant..."}
"""

from typing import Dict, Any


# System message templates for each persona
SYSTEM_MESSAGE_TEMPLATES = {
    'twain': "You are a witty weather assistant inspired by Mark Twain. Your responses blend accurate weather information with humorous observations and satirical commentary, using colloquial language and clever turns of phrase.",

    'franklin': "You are a wise weather advisor inspired by Benjamin Franklin. Your responses provide practical weather guidance with aphoristic wisdom, drawing on historical knowledge and common sense to help people make informed decisions.",

    'onion': "You are a satirical weather assistant inspired by The Onion. Your responses provide accurate weather information wrapped in deadpan, absurdist humor with straight-faced delivery. Think 'Area Man Checks Weather App 47 Times, Still Surprised By Rain' style commentary - combining mundane weather facts with satirical observations about human behavior.",

    'neutral': "You are a helpful weather assistant. You provide clear, accurate weather information in a friendly and approachable manner."
}


def generate_system_message(persona: str) -> Dict[str, str]:
    """
    Generate persona-aware system message.

    Args:
        persona: Persona tag ("twain", "franklin", "neutral")

    Returns:
        Dictionary with role and content fields for system message

    System message templates:
        - Twain: "You are a witty weather assistant inspired by Mark Twain..."
        - Franklin: "You are a wise weather advisor inspired by Benjamin Franklin..."
        - Neutral: "You are a helpful weather assistant..."

    Examples:
        >>> msg = generate_system_message("twain")
        >>> msg["role"]
        'system'
        >>> "Mark Twain" in msg["content"]
        True
    """
    # Get template for persona (default to neutral if not found)
    content = SYSTEM_MESSAGE_TEMPLATES.get(persona, SYSTEM_MESSAGE_TEMPLATES['neutral'])

    return {
        'role': 'system',
        'content': content
    }


def get_system_content(persona: str) -> str:
    """
    Get system message content string only.

    Args:
        persona: Persona tag

    Returns:
        System message content string

    Examples:
        >>> content = get_system_content("franklin")
        >>> "Benjamin Franklin" in content
        True
    """
    return SYSTEM_MESSAGE_TEMPLATES.get(persona, SYSTEM_MESSAGE_TEMPLATES['neutral'])


if __name__ == "__main__":
    # Test system message generator
    print("System Message Generator Test")
    print("=" * 60)

    personas = ['twain', 'franklin', 'neutral', 'unknown']

    for persona in personas:
        msg = generate_system_message(persona)
        print(f"\nPersona: {persona}")
        print(f"  Role: {msg['role']}")
        print(f"  Content: {msg['content'][:80]}...")
        print(f"  Content length: {len(msg['content'])} chars")

    # Test content extraction
    print("\n" + "=" * 60)
    print("Testing content extraction:")
    for persona in ['twain', 'franklin', 'neutral']:
        content = get_system_content(persona)
        print(f"\n{persona}: {len(content)} chars")
