#!/usr/bin/env python3
"""
Fix tool call IDs to be exactly 9 alphanumeric characters for Mistral compatibility.

Mistral's chat template requires tool call IDs to be exactly 9 characters.
Our data has IDs like "call_44cd3e6bb292" which are too long.

This script:
1. Reads JSONL files with tool calls
2. Normalizes tool call IDs to 9 characters
3. Ensures consistency between tool_calls and tool responses
"""

import json
import sys
import argparse
from pathlib import Path


def normalize_tool_id(original_id: str) -> str:
    """
    Convert any tool call ID to a 9-character alphanumeric string.

    Strategy: Take the last 9 alphanumeric characters from the original ID.
    This preserves uniqueness while meeting Mistral's requirements.
    """
    # Remove non-alphanumeric characters
    alphanumeric = ''.join(c for c in original_id if c.isalnum())

    # Take last 9 characters (or pad if shorter)
    if len(alphanumeric) >= 9:
        return alphanumeric[-9:]
    else:
        # Pad with zeros if too short
        return alphanumeric.zfill(9)


def fix_conversation(messages: list) -> list:
    """
    Fix all tool call IDs in a conversation to be 9 characters.
    Maintains consistency between tool_calls and tool responses.
    """
    # Map old IDs to new IDs
    id_mapping = {}

    fixed_messages = []
    for msg in messages:
        fixed_msg = msg.copy()

        # Fix tool_calls in assistant messages
        if 'tool_calls' in msg and msg['tool_calls']:
            fixed_tool_calls = []
            for tool_call in msg['tool_calls']:
                fixed_tool_call = tool_call.copy()
                old_id = tool_call['id']

                # Get or create normalized ID
                if old_id not in id_mapping:
                    id_mapping[old_id] = normalize_tool_id(old_id)

                fixed_tool_call['id'] = id_mapping[old_id]
                fixed_tool_calls.append(fixed_tool_call)

            fixed_msg['tool_calls'] = fixed_tool_calls

        # Fix tool_call_id in tool response messages
        if msg.get('role') == 'tool' and 'tool_call_id' in msg:
            old_id = msg['tool_call_id']

            # Get or create normalized ID
            if old_id not in id_mapping:
                id_mapping[old_id] = normalize_tool_id(old_id)

            fixed_msg['tool_call_id'] = id_mapping[old_id]

        fixed_messages.append(fixed_msg)

    return fixed_messages


def process_file(input_path: Path, output_path: Path):
    """Process a JSONL file and fix all tool call IDs."""
    print(f"Processing: {input_path}")

    total_conversations = 0
    fixed_conversations = 0

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line)
                total_conversations += 1

                # Fix the messages
                if 'messages' in data:
                    original_messages = data['messages']
                    fixed_messages = fix_conversation(original_messages)

                    # Check if anything changed
                    if fixed_messages != original_messages:
                        fixed_conversations += 1

                    data['messages'] = fixed_messages

                # Write fixed data
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

                if line_num % 1000 == 0:
                    print(f"  Processed {line_num} conversations...")

            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue

    print(f"✓ Complete: {total_conversations} conversations processed")
    print(f"✓ Fixed {fixed_conversations} conversations with tool calls")
    print(f"✓ Output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix tool call IDs to be 9 characters for Mistral compatibility"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input JSONL file with conversations'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSONL file with fixed IDs'
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    process_file(args.input, args.output)


if __name__ == '__main__':
    main()
