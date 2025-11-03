#!/usr/bin/env python3
"""
Generate humor-focused synthetic tool-use conversation data.

Custom persona distribution: 70% Twain, 20% Franklin, 10% Neutral
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.conversation_orchestrator import ConversationOrchestrator
from scripts.claude_api_client import ClaudeAPIClient
from scripts.synthetic_conversation_generator import SyntheticConversationGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Generate humor-focused synthetic tool-use conversation data'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=2500,
        help='Number of conversations to generate (default: 2500)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/synthetic/tool_use_examples_humor.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Mock mode (no API calls, for testing)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Anthropic API key (or use ANTHROPIC_API_KEY env var)'
    )

    args = parser.parse_args()

    # Custom persona distribution for humor focus
    humor_persona_weights = {
        'twain': 0.70,      # 70% Twain (witty, humorous)
        'franklin': 0.20,   # 20% Franklin (didactic, almanac wisdom)
        'neutral': 0.10     # 10% Neutral (baseline)
    }

    print("=" * 60)
    print("Humor-Focused Synthetic Data Generation")
    print("=" * 60)
    print(f"Target count: {args.count}")
    print(f"Output: {args.output}")
    print()
    print("Persona Distribution:")
    print(f"  - Twain (witty): 70%")
    print(f"  - Franklin (wisdom): 20%")
    print(f"  - Neutral: 10%")
    print("=" * 60)
    print()

    # Initialize orchestrator with humor-focused weights
    orchestrator = ConversationOrchestrator(
        persona_weights=humor_persona_weights
    )

    # Initialize API client
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    client = ClaudeAPIClient(api_key=api_key, mock_mode=args.mock)

    # Initialize generator
    generator = SyntheticConversationGenerator(
        orchestrator=orchestrator,
        api_client=client
    )

    # Generate conversations
    conversations = generator.generate_batch(
        count=args.count,
        output_path=args.output
    )

    # Print statistics
    stats = orchestrator.stats.get_summary()
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Total Generated: {stats['total_generated']}")
    print()
    print("Persona Distribution:")
    for persona, count in stats['personas'].items():
        pct = stats['distributions']['persona'].get(persona, 0)
        print(f"  - {persona}: {count} ({pct:.1f}%)")
    print()
    print("Scenario Distribution:")
    for scenario, count in stats['scenarios'].items():
        pct = stats['distributions']['scenario'].get(scenario, 0)
        print(f"  - {scenario}: {count} ({pct:.1f}%)")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
