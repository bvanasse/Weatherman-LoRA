#!/usr/bin/env python3
"""
Reddit JSONL Converter Module for Humor Dataset

Converts processed Reddit data to chat-format JSONL for training.
Implements source-aware tagging and metadata preservation.

Usage:
    from reddit_jsonl_converter import convert_to_jsonl
    from paths import DATA_PROCESSED

    output_file = DATA_PROCESSED / "reddit_humor_weather.jsonl"
    convert_to_jsonl(processed_df, output_file)
"""

import json
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


# User message variations for diverse training examples
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


def generate_varied_user_message() -> str:
    """
    Generate a varied user message for weather queries.

    Returns:
        Random user message from template list

    Notes:
        - Provides variety to avoid overfitting on specific phrasing
        - All messages are weather-related queries
        - Random selection ensures distribution across training data
    """
    return random.choice(USER_MESSAGE_TEMPLATES)


def determine_tone_tags(subreddit: str) -> Dict[str, Any]:
    """
    Determine tone and domain tags based on subreddit source.

    Args:
        subreddit: Source subreddit name (TheOnion or nottheonion)

    Returns:
        Dictionary with tone, domain, persona, and source tags

    Tag schema:
        - TheOnion: tone="satirical", source="reddit-theonion"
        - nottheonion: tone="ironic", source="reddit-nottheonion"
        - All: domain=["weather", "humor"], persona="neutral"
    """
    tags = {
        'persona': 'neutral',
        'domain': ['weather', 'humor']
    }

    subreddit_lower = subreddit.lower()

    if 'theonion' in subreddit_lower and 'not' not in subreddit_lower:
        # r/TheOnion - satirical news from The Onion
        tags['tone'] = 'satirical'
        tags['source'] = 'reddit-theonion'
    elif 'nottheonion' in subreddit_lower:
        # r/nottheonion - absurd real news
        tags['tone'] = 'ironic'
        tags['source'] = 'reddit-nottheonion'
    else:
        # Fallback for other subreddits
        tags['tone'] = 'humorous'
        tags['source'] = f'reddit-{subreddit_lower}'

    return tags


def create_metadata_tags(
    post_id: str,
    subreddit: str,
    created_utc: int,
    url: str,
    num_comments: int
) -> Dict[str, Any]:
    """
    Create metadata tags for provenance tracking.

    Args:
        post_id: Reddit post ID
        subreddit: Source subreddit
        created_utc: Post creation timestamp (Unix epoch)
        url: Reddit post URL
        num_comments: Number of comments (quality proxy)

    Returns:
        Dictionary with metadata fields

    Metadata fields:
        - reddit_id: Original post ID for deduplication
        - subreddit: Source subreddit
        - created_utc: Timestamp for temporal analysis
        - url: Full URL for manual review
        - score: Comment count as engagement proxy
    """
    return {
        'reddit_id': post_id,
        'subreddit': subreddit,
        'created_utc': created_utc,
        'url': url,
        'score': num_comments
    }


def create_chat_format_entry(
    cleaned_title: str,
    subreddit: str,
    post_id: str,
    created_utc: int,
    url: str,
    num_comments: int,
    matched_keywords: List[str]
) -> Dict[str, Any]:
    """
    Create a single chat-format JSONL entry.

    Args:
        cleaned_title: Cleaned Reddit post title
        subreddit: Source subreddit
        post_id: Reddit post ID
        created_utc: Post creation timestamp
        url: Reddit post URL
        num_comments: Number of comments
        matched_keywords: List of matched weather keywords

    Returns:
        Dictionary with messages array and tags field

    Chat format structure:
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ],
            "tags": {
                "persona": "neutral",
                "tone": "satirical",
                "domain": ["weather", "humor"],
                "source": "reddit-theonion",
                "reddit_id": "...",
                "subreddit": "...",
                "created_utc": ...,
                "url": "...",
                "score": ...
            }
        }
    """
    # System message - defines the assistant's persona
    system_message = {
        'role': 'system',
        'content': 'You are a witty weather assistant who responds with humorous observations, inspired by satirical news headlines.'
    }

    # User message - varied weather query
    user_message = {
        'role': 'user',
        'content': generate_varied_user_message()
    }

    # Assistant message - the cleaned Reddit title as humorous response
    assistant_message = {
        'role': 'assistant',
        'content': cleaned_title
    }

    # Combine tone tags and metadata tags
    tone_tags = determine_tone_tags(subreddit)
    metadata_tags = create_metadata_tags(post_id, subreddit, created_utc, url, num_comments)

    # Merge all tags
    all_tags = {**tone_tags, **metadata_tags}

    # Add matched keywords for reference
    all_tags['matched_keywords'] = matched_keywords

    return {
        'messages': [system_message, user_message, assistant_message],
        'tags': all_tags
    }


def convert_to_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    """
    Convert DataFrame to chat-format JSONL file.

    Args:
        df: DataFrame with processed Reddit data
        output_path: Path to output JSONL file

    Required DataFrame columns:
        - cleaned_title: Cleaned post title
        - id: Reddit post ID
        - subreddit: Source subreddit
        - created_utc: Post timestamp
        - url: Reddit URL
        - num_comments: Comment count
        - matched_keywords: List of matched keywords

    Process:
        1. Convert each row to chat-format entry
        2. Write to temporary file (atomic operation)
        3. Rename to final path
        4. Validate output
    """
    print(f"\nConverting {len(df)} entries to chat-format JSONL...")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary file in same directory for atomic write
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=output_path.parent,
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

        # Convert each row to chat-format and write as JSONL
        for idx, row in df.iterrows():
            entry = create_chat_format_entry(
                cleaned_title=row['cleaned_title'],
                subreddit=row['subreddit'],
                post_id=row['id'],
                created_utc=row['created_utc'],
                url=row['url'],
                num_comments=row['num_comments'],
                matched_keywords=row.get('matched_keywords', [])
            )

            # Write as single line JSON
            json.dump(entry, tmp_file, ensure_ascii=False)
            tmp_file.write('\n')

    # Atomic rename
    tmp_path.replace(output_path)

    print(f"  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")

    # Validate output
    if validate_jsonl_output(output_path):
        print(f"  \u2713 Validation: JSONL file valid, contains {len(df)} entries")
    else:
        print(f"  \u26a0 WARNING: JSONL validation failed")


def validate_jsonl_output(output_path: Path) -> bool:
    """
    Validate JSONL output file.

    Args:
        output_path: Path to JSONL file

    Returns:
        True if valid, False otherwise

    Validation checks:
        - Each line is valid JSON
        - Each entry has 'messages' array
        - Each entry has 'tags' field
        - Messages array has 3 elements (system, user, assistant)
    """
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse JSON
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  ERROR: Invalid JSON on line {line_count + 1}: {e}")
                    return False

                # Check required fields
                if 'messages' not in entry:
                    print(f"  ERROR: Missing 'messages' field on line {line_count + 1}")
                    return False

                if 'tags' not in entry:
                    print(f"  ERROR: Missing 'tags' field on line {line_count + 1}")
                    return False

                # Check messages structure
                messages = entry['messages']
                if not isinstance(messages, list):
                    print(f"  ERROR: 'messages' is not a list on line {line_count + 1}")
                    return False

                if len(messages) != 3:
                    print(f"  ERROR: Expected 3 messages, got {len(messages)} on line {line_count + 1}")
                    return False

                # Check message roles
                roles = [msg.get('role') for msg in messages]
                if roles != ['system', 'user', 'assistant']:
                    print(f"  ERROR: Invalid message roles on line {line_count + 1}: {roles}")
                    return False

                line_count += 1

        return True

    except Exception as e:
        print(f"  ERROR: Validation failed: {e}")
        return False


def print_sample_entries(output_path: Path, num_samples: int = 3) -> None:
    """
    Print sample entries from JSONL file.

    Args:
        output_path: Path to JSONL file
        num_samples: Number of samples to print (default: 3)
    """
    print(f"\nSample entries from {output_path.name}:")
    print("=" * 60)

    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            entry = json.loads(line)

            print(f"\nEntry {i + 1}:")
            print(f"  User: {entry['messages'][1]['content']}")
            print(f"  Assistant: {entry['messages'][2]['content']}")
            print(f"  Tags: {entry['tags']['tone']}, {entry['tags']['source']}")
            print(f"  Keywords: {', '.join(entry['tags']['matched_keywords'])}")


if __name__ == "__main__":
    # Test with sample data
    print("Reddit JSONL Converter Test")
    print("=" * 60)

    # Create sample DataFrame
    sample_data = {
        'cleaned_title': [
            'Storm warning: Politicians caught in rain of scandal',
            'Hurricane season forecast predicts 100% chance of chaos',
            'Weather service baffled as climate of political discourse worsens'
        ],
        'id': ['a1', 'a2', 'a3'],
        'subreddit': ['TheOnion', 'nottheonion', 'TheOnion'],
        'created_utc': [1545089481, 1545089482, 1545089483],
        'url': [
            'https://reddit.com/r/TheOnion/test1',
            'https://reddit.com/r/nottheonion/test2',
            'https://reddit.com/r/TheOnion/test3'
        ],
        'num_comments': [42, 28, 15],
        'matched_keywords': [['storm', 'rain'], ['hurricane', 'forecast'], ['weather', 'climate']]
    }

    df = pd.DataFrame(sample_data)

    # Convert to JSONL
    output_file = Path(tempfile.mktemp(suffix='.jsonl'))
    convert_to_jsonl(df, output_file)

    # Print samples
    print_sample_entries(output_file)

    # Cleanup
    output_file.unlink()
