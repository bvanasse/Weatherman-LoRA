#!/usr/bin/env python3
"""
Regenerate Training Data with Diverse Responses

This script takes existing training data with repetitive templates and regenerates
the final assistant responses with more diversity using Claude API.

Key features:
- Preserves tool-calling structure (the good part)
- Regenerates only final assistant responses (the repetitive part)
- Uses Claude Sonnet 4.5 for high-quality diverse output
- Leverages literary corpus for authentic voice examples
- High temperature for creativity
- Anti-template instructions to prevent repetition
- Progress tracking and checkpoint recovery

Usage:
    python scripts/regenerate_diverse_responses.py --input data/synthetic/final_train.jsonl --output data/synthetic/final_train_diverse.jsonl
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed")
    print("Install with: pip install anthropic")
    sys.exit(1)


# Template patterns to identify and replace
TEMPLATE_PATTERNS = {
    'twain_forecast_promise': 'weather forecasts and promises are alike',
    'twain_climate_expect': 'Climate is what we expect, weather is what we get',
    'franklin_early_bed': 'Early to bed and early to rise',
    'neutral_info_requested': "Here's the information you requested for",
}


def has_template(response: str) -> bool:
    """Check if response contains a template pattern."""
    response_lower = response.lower()
    return any(pattern.lower() in response_lower for pattern in TEMPLATE_PATTERNS.values())


def load_style_examples(persona: str, corpus_path: Path) -> List[str]:
    """Load authentic style examples from literary corpus."""
    examples = []

    # Load from Gutenberg passages if available
    gutenberg_file = corpus_path / "gutenberg_passages.json"
    if gutenberg_file.exists():
        try:
            with open(gutenberg_file, 'r') as f:
                data = json.load(f)

                if persona == 'twain':
                    # Extract Mark Twain passages
                    for item in data.get('twain', []):
                        if 'text' in item and len(item['text']) > 50:
                            examples.append(item['text'][:500])

                elif persona == 'franklin':
                    # Extract Benjamin Franklin passages
                    for item in data.get('franklin', []):
                        if 'text' in item and len(item['text']) > 50:
                            examples.append(item['text'][:500])
        except Exception as e:
            print(f"Warning: Could not load Gutenberg passages: {e}")

    # Load from Reddit humor if available for Onion style
    if persona == 'onion':
        reddit_file = corpus_path / "reddit_humor_weather.jsonl"
        if reddit_file.exists():
            try:
                with open(reddit_file, 'r') as f:
                    for line in f:
                        item = json.loads(line)
                        if 'title' in item:
                            examples.append(item['title'])
                        if len(examples) >= 50:
                            break
            except Exception as e:
                print(f"Warning: Could not load Reddit humor: {e}")

    return examples[:20]  # Return up to 20 examples


def create_regeneration_prompt(
    original_conversation: List[Dict],
    persona: str,
    style_examples: List[str],
    previous_responses: List[str]
) -> str:
    """Create a prompt for regenerating the final response with diversity."""

    # Extract the conversation context
    context_messages = []
    tool_result = None

    for msg in original_conversation:
        if msg['role'] == 'user':
            context_messages.append(f"User: {msg['content']}")
        elif msg['role'] == 'assistant' and 'tool_calls' in msg:
            # This is the tool call
            context_messages.append("Assistant called weather tool")
        elif msg['role'] == 'tool':
            # This is the tool result
            tool_result = msg['content']

    conversation_context = "\n".join(context_messages)

    # Persona-specific instructions
    persona_instructions = {
        'twain': """You are responding as Mark Twain - witty, ironic, folksy, and observational.
Use his characteristic humor: clever wordplay, gentle mockery, and keen observations about human nature.
Think like: "The coldest winter I ever spent was a summer in San Francisco."
Vary your approach - sometimes use metaphors, sometimes analogies, sometimes just dry wit.""",

        'franklin': """You are responding as Benjamin Franklin - pragmatic, wise, aphoristic, and scientific.
Use his style: practical wisdom, clever sayings, and rational observations.
Think like: "Some are weather-wise, some are otherwise."
Vary your approach - sometimes offer practical advice, sometimes philosophical observations.""",

        'onion': """You are responding in The Onion's satirical style - absurdist, mock-serious, deadpan.
Use satirical news headline format: over-the-top seriousness about mundane topics.
Think like: "Area Man Consults Weather App For 47th Time Today, Still Doesn't Trust It"
Vary your approach - sometimes absurdist, sometimes mock-outraged, sometimes deadpan.""",

        'neutral': """You are a professional weather assistant - clear, informative, and courteous.
Provide accurate information from the tool results in a helpful manner.
Be natural and conversational, not robotic. Vary your phrasing."""
    }

    # Build style examples section
    style_section = ""
    if style_examples:
        style_section = f"\n\n**Authentic {persona.title()} Style Examples:**\n"
        for i, example in enumerate(random.sample(style_examples, min(3, len(style_examples))), 1):
            style_section += f"{i}. {example}\n"

    # Build anti-template section
    anti_template = ""
    if previous_responses:
        anti_template = f"""
**CRITICAL: AVOID THESE OVERUSED PHRASES:**
You must NOT use any of these repetitive patterns that appear too frequently:
{chr(10).join(f"- {resp[:100]}..." for resp in random.sample(previous_responses, min(5, len(previous_responses))))}

Create something fresh and unique. No recycled phrases!
"""

    prompt = f"""You are generating a weather assistant response in a specific persona.

**CONVERSATION CONTEXT:**
{conversation_context}

**TOOL RESULT DATA:**
{tool_result}

**YOUR PERSONA:**
{persona_instructions.get(persona, persona_instructions['neutral'])}
{style_section}
{anti_template}

**CRITICAL REQUIREMENTS:**
1. Use the actual data from the tool result - reference specific numbers and conditions
2. Stay in character for the persona but BE CREATIVE AND VARIED
3. NEVER use the same phrasing as other responses
4. Keep response concise (1-3 sentences for humor personas, 1-2 for neutral)
5. Make it feel natural and spontaneous, not formulaic
6. Reference the tool data naturally - don't just list numbers

**Generate ONLY the final assistant response text (no JSON, no role labels, just the response):**"""

    return prompt


def regenerate_response(
    client: anthropic.Anthropic,
    original_conversation: List[Dict],
    persona: str,
    style_examples: List[str],
    previous_responses: List[str],
    model: str = "claude-haiku-4-5"
) -> str:
    """Regenerate a single response using Claude API."""

    prompt = create_regeneration_prompt(
        original_conversation,
        persona,
        style_examples,
        previous_responses
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.9,  # High temperature for creativity
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        return response.content[0].text.strip()

    except Exception as e:
        print(f"Error generating response: {e}")
        return None


def process_training_data(
    input_file: Path,
    output_file: Path,
    corpus_path: Path,
    api_key: str,
    max_templates_per_pattern: int = 100,
    checkpoint_interval: int = 100,
    model: str = "claude-haiku-4-5"
):
    """Process training data and regenerate templated responses."""

    client = anthropic.Anthropic(api_key=api_key)

    # Load all examples
    print(f"Loading training data from {input_file}...")
    with open(input_file, 'r') as f:
        examples = [json.loads(line) for line in f]

    print(f"Loaded {len(examples):,} examples")

    # Identify templated examples
    print("Identifying templated responses...")
    template_counts = {pattern: 0 for pattern in TEMPLATE_PATTERNS.keys()}
    templated_indices = []

    for i, example in enumerate(examples):
        messages = example['messages']

        # Find final assistant response
        for msg in reversed(messages):
            if msg['role'] == 'assistant' and 'tool_calls' not in msg:
                response = msg['content']

                # Check if it uses a template
                if has_template(response):
                    # Check which template
                    response_lower = response.lower()
                    for pattern_name, pattern_text in TEMPLATE_PATTERNS.items():
                        if pattern_text.lower() in response_lower:
                            if template_counts[pattern_name] < max_templates_per_pattern:
                                # Keep this one (within limit)
                                template_counts[pattern_name] += 1
                            else:
                                # Mark for regeneration
                                templated_indices.append(i)
                            break
                break

    print(f"Found {len(templated_indices):,} examples to regenerate")
    print(f"Keeping {sum(template_counts.values())} templated examples for diversity")

    # Load style examples for each persona
    print("Loading style examples from literary corpus...")
    style_examples_cache = {
        'twain': load_style_examples('twain', corpus_path),
        'franklin': load_style_examples('franklin', corpus_path),
        'onion': load_style_examples('onion', corpus_path),
    }

    print(f"Loaded style examples: Twain={len(style_examples_cache['twain'])}, "
          f"Franklin={len(style_examples_cache['franklin'])}, "
          f"Onion={len(style_examples_cache['onion'])}")

    # Track previously generated responses to avoid repetition
    previous_responses = []

    # Regenerate templated responses
    print(f"\nRegenerating {len(templated_indices):,} responses...")
    print(f"Using model: {model}")
    print(f"Estimated time: ~{len(templated_indices) * 2 / 60:.1f} minutes")
    print()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_file.with_suffix('.checkpoint.jsonl')

    # Resume from checkpoint if exists
    start_index = 0
    if checkpoint_file.exists():
        print(f"Found checkpoint file, resuming...")
        with open(checkpoint_file, 'r') as f:
            processed = sum(1 for _ in f)
        start_index = processed
        print(f"Resuming from index {start_index}")

    regenerated = 0
    failed = 0

    with open(output_file, 'w') as out_f:
        for idx, example in enumerate(examples):
            # Check if this example needs regeneration
            if idx in templated_indices and idx >= start_index:
                persona = example.get('tags', {}).get('persona', 'neutral')
                style_examples = style_examples_cache.get(persona, [])

                # Regenerate response
                new_response = regenerate_response(
                    client,
                    example['messages'],
                    persona,
                    style_examples,
                    previous_responses,
                    model=model
                )

                if new_response:
                    # Replace final assistant response
                    for msg in reversed(example['messages']):
                        if msg['role'] == 'assistant' and 'tool_calls' not in msg:
                            msg['content'] = new_response
                            break

                    previous_responses.append(new_response)
                    if len(previous_responses) > 100:
                        previous_responses.pop(0)  # Keep last 100

                    regenerated += 1

                    if regenerated % 10 == 0:
                        print(f"Progress: {regenerated:,}/{len(templated_indices):,} regenerated "
                              f"({regenerated/len(templated_indices)*100:.1f}%), "
                              f"{failed} failed")
                else:
                    failed += 1
                    print(f"Warning: Failed to regenerate response for example {idx}")

                # Rate limiting
                time.sleep(1)  # 1 second between API calls

                # Checkpoint
                if regenerated % checkpoint_interval == 0:
                    with open(checkpoint_file, 'a') as cp_f:
                        json.dump(example, cp_f)
                        cp_f.write('\n')

            # Write to output
            json.dump(example, out_f)
            out_f.write('\n')

    # Clean up checkpoint file
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print()
    print("="*60)
    print(f"Regeneration complete!")
    print(f"  Total examples: {len(examples):,}")
    print(f"  Regenerated: {regenerated:,}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_file}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Regenerate training data with diverse responses")
    parser.add_argument('--input', type=str, required=True,
                       help='Input training data file (JSONL)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output training data file (JSONL)')
    parser.add_argument('--corpus-path', type=str, default='data/processed',
                       help='Path to literary corpus data')
    parser.add_argument('--max-templates', type=int, default=100,
                       help='Max examples to keep per template pattern')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='Save checkpoint every N examples')
    parser.add_argument('--model', type=str, default='claude-haiku-4-5',
                       choices=['claude-haiku-4-5', 'claude-3-5-haiku-20241022', 'claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022'],
                       help='Claude model to use')

    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)

    input_file = Path(args.input)
    output_file = Path(args.output)
    corpus_path = Path(args.corpus_path)

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    # Run regeneration
    process_training_data(
        input_file=input_file,
        output_file=output_file,
        corpus_path=corpus_path,
        api_key=api_key,
        max_templates_per_pattern=args.max_templates,
        checkpoint_interval=args.checkpoint_interval,
        model=args.model
    )


if __name__ == '__main__':
    main()
