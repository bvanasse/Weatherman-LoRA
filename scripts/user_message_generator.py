#!/usr/bin/env python3
"""
User Message Generator Module for Chat-Format Conversion

Generates varied user messages for weather queries to avoid overfitting.
Provides 15 different query templates for training diversity.

Usage:
    from scripts.user_message_generator import generate_user_message

    msg = generate_user_message()
    # Returns: {"role": "user", "content": "What's the weather like?"}
"""

import random
from typing import Dict, List


# User message templates for weather queries (15 variations)
USER_MESSAGE_TEMPLATES = [
    "What's the weather like?",
    "Give me the forecast",
    "How's the weather today?",
    "What's the weather forecast?",
    "Tell me about the weather",
    "What can you tell me about today's weather?",
    "How does the weather look?",
    "What's the forecast looking like?",
    "Give me a weather update",
    "What's happening with the weather?",
    "Tell me what the weather is doing",
    "How's it looking outside?",
    "What's the weather situation?",
    "Can you give me the weather?",
    "What's the weather report?"
]


def generate_user_message() -> Dict[str, str]:
    """
    Generate a varied user message for weather queries.

    Returns:
        Dictionary with role and content fields for user message

    Notes:
        - Provides variety to avoid overfitting on specific phrasing
        - All messages are weather-related queries
        - Random selection ensures distribution across training data

    Examples:
        >>> msg = generate_user_message()
        >>> msg["role"]
        'user'
        >>> isinstance(msg["content"], str)
        True
        >>> len(msg["content"]) > 0
        True
    """
    content = random.choice(USER_MESSAGE_TEMPLATES)

    return {
        'role': 'user',
        'content': content
    }


def get_user_content() -> str:
    """
    Get user message content string only (random template).

    Returns:
        Random user message string

    Examples:
        >>> content = get_user_content()
        >>> content in USER_MESSAGE_TEMPLATES
        True
    """
    return random.choice(USER_MESSAGE_TEMPLATES)


def get_all_templates() -> List[str]:
    """
    Get list of all available user message templates.

    Returns:
        List of all template strings

    Examples:
        >>> templates = get_all_templates()
        >>> len(templates)
        15
    """
    return USER_MESSAGE_TEMPLATES.copy()


if __name__ == "__main__":
    # Test user message generator
    print("User Message Generator Test")
    print("=" * 60)

    print(f"\nTotal templates: {len(USER_MESSAGE_TEMPLATES)}")

    print("\nAll templates:")
    for i, template in enumerate(USER_MESSAGE_TEMPLATES, 1):
        print(f"  {i:2d}. {template}")

    # Test random generation
    print("\n" + "=" * 60)
    print("Testing random generation (10 samples):")

    template_counts = {}
    for _ in range(10):
        msg = generate_user_message()
        content = msg['content']
        template_counts[content] = template_counts.get(content, 0) + 1
        print(f"  {msg['role']:8s}: {msg['content']}")

    print("\nDistribution in 10 samples:")
    for template, count in sorted(template_counts.items()):
        print(f"  {count}x: {template}")
