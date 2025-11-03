#!/usr/bin/env python3
"""
Main Script: Synthetic Tool-Use Data Generation

Generates 1,000-3,000 OpenAI-style function calling conversation examples.
Uses Claude Haiku 4.5 API for realistic tool-use training data.

Usage:
    python scripts/generate_synthetic_tool_data.py --count 1000
    python scripts/generate_synthetic_tool_data.py --count 2000 --output data/synthetic/custom.jsonl
    python scripts/generate_synthetic_tool_data.py --count 1000 --mock  # Test without API
"""

import argparse
import json
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with: pip install tqdm")

from scripts.conversation_orchestrator import ConversationOrchestrator
from scripts.conversation_assembly import ConversationAssembler
from scripts.synthetic_data_validators import validate_conversation_full, ValidationReporter
from scripts.output_writer import write_jsonl_output
from scripts.config_loader import PROJECT_ROOT


# Global flag for graceful shutdown
INTERRUPTED = False


def signal_handler(signum, frame):
    """Handle keyboard interrupt gracefully."""
    global INTERRUPTED
    INTERRUPTED = True
    print("\n\nInterrupt received. Finishing current example...")


class GenerationProgress:
    """Track and display generation progress."""

    def __init__(self, total: int, use_tqdm: bool = True):
        self.total = total
        self.generated = 0
        self.validated = 0
        self.failed_validation = 0
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE

        if self.use_tqdm:
            self.pbar = tqdm(total=total, desc="Generating", unit="conv")
        else:
            self.pbar = None

    def update(self, validated: bool = True):
        """Update progress."""
        self.generated += 1
        if validated:
            self.validated += 1
        else:
            self.failed_validation += 1

        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix({
                'validated': self.validated,
                'failed': self.failed_validation
            })
        elif self.generated % 10 == 0:
            # Print progress every 10 examples if no tqdm
            pct = (self.generated / self.total) * 100
            print(f"Progress: {self.generated}/{self.total} ({pct:.1f}%) - "
                  f"Validated: {self.validated}, Failed: {self.failed_validation}")

    def close(self):
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()


class CheckpointManager:
    """Manage checkpoints for resumption."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        conversations: List[Dict[str, Any]],
        stats: Dict[str, Any],
        count: int
    ):
        """Save checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{count}.json"

        checkpoint_data = {
            'count': count,
            'timestamp': datetime.now().isoformat(),
            'stats': stats,
            'conversation_count': len(conversations)
        }

        # Save checkpoint metadata
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        # Save conversations separately
        conv_file = self.checkpoint_dir / f"conversations_{count}.jsonl"
        write_jsonl_output(conversations, conv_file)

        print(f"\n  Checkpoint saved: {count} conversations")

    def load_latest_checkpoint(self) -> tuple[List[Dict[str, Any]], Dict[str, Any], int]:
        """Load latest checkpoint."""
        checkpoint_files = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))

        if not checkpoint_files:
            return [], {}, 0

        latest = checkpoint_files[-1]
        with open(latest, 'r') as f:
            checkpoint_data = json.load(f)

        count = checkpoint_data['count']
        stats = checkpoint_data['stats']

        # Load conversations
        conv_file = self.checkpoint_dir / f"conversations_{count}.jsonl"
        conversations = []

        if conv_file.exists():
            with open(conv_file, 'r') as f:
                for line in f:
                    conversations.append(json.loads(line.strip()))

        return conversations, stats, count

    def cleanup_checkpoints(self):
        """Remove checkpoint files after successful completion."""
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            checkpoint_file.unlink()
        for conv_file in self.checkpoint_dir.glob("conversations_*.jsonl"):
            conv_file.unlink()


def generate_synthetic_data(
    count: int,
    output_path: Path,
    checkpoint_dir: Path,
    mock_mode: bool = False,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Generate synthetic tool-use conversations.

    Args:
        count: Number of conversations to generate
        output_path: Output JSONL file path
        checkpoint_dir: Directory for checkpoints
        mock_mode: If True, generate without API calls (testing)
        api_key: Anthropic API key (optional)

    Returns:
        Generation metadata dictionary
    """
    print("=" * 60)
    print("Synthetic Tool-Use Data Generation")
    print("=" * 60)
    print(f"Target count: {count}")
    print(f"Output path: {output_path}")
    print(f"Mock mode: {mock_mode}")
    print()

    # Initialize components
    orchestrator = ConversationOrchestrator()
    checkpoint_mgr = CheckpointManager(checkpoint_dir)

    # Initialize API client (if not mock mode)
    claude_client = None
    if not mock_mode:
        try:
            from scripts.claude_api_client import ClaudeAPIClient
            claude_client = ClaudeAPIClient(api_key=api_key)
            print("✓ Claude API client initialized")
        except Exception as e:
            print(f"✗ Failed to initialize API client: {e}")
            print("  Falling back to mock mode")
            mock_mode = True

    assembler = ConversationAssembler(claude_client=claude_client, mock_mode=mock_mode)

    # Check for existing checkpoint
    conversations, checkpoint_stats, checkpoint_count = checkpoint_mgr.load_latest_checkpoint()

    if checkpoint_count > 0:
        response = input(f"\nFound checkpoint with {checkpoint_count} conversations. Resume? (y/n): ")
        if response.lower() == 'y':
            print(f"Resuming from checkpoint ({checkpoint_count} conversations)")
        else:
            conversations = []
            checkpoint_count = 0

    # Initialize progress
    remaining = count - checkpoint_count
    progress = GenerationProgress(remaining)

    # Generation loop
    max_retries = 3
    generated_count = checkpoint_count

    while generated_count < count and not INTERRUPTED:
        retries = 0
        success = False

        while retries < max_retries and not success:
            try:
                # Get generation config
                config = orchestrator.get_next_generation_config()

                # Assemble conversation
                conversation = assembler.assemble_conversation(config)

                # Validate conversation
                is_valid, errors = validate_conversation_full(conversation)

                if is_valid:
                    conversations.append(conversation)
                    generated_count += 1
                    progress.update(validated=True)
                    success = True
                else:
                    if retries < max_retries - 1:
                        retries += 1
                        print(f"\n  Validation failed (retry {retries}/{max_retries}): {errors[0]}")
                    else:
                        print(f"\n  Skipping after {max_retries} retries: {errors[0]}")
                        progress.update(validated=False)
                        success = True  # Skip and continue

            except Exception as e:
                if retries < max_retries - 1:
                    retries += 1
                    print(f"\n  Generation error (retry {retries}/{max_retries}): {e}")
                else:
                    print(f"\n  Skipping after {max_retries} retries: {e}")
                    progress.update(validated=False)
                    success = True  # Skip and continue

        # Save checkpoint every 250 examples
        if generated_count % 250 == 0 and generated_count > checkpoint_count:
            checkpoint_mgr.save_checkpoint(
                conversations,
                orchestrator.get_statistics(),
                generated_count
            )

        # Print batch statistics every 100 examples
        if generated_count % 100 == 0 and generated_count > checkpoint_count:
            stats = orchestrator.get_statistics()
            print(f"\n  Batch stats ({generated_count}/{count}):")
            print(f"    Scenarios: {stats['scenarios']}")
            print(f"    Personas: {stats['personas']}")

    progress.close()

    # Handle interruption
    if INTERRUPTED:
        print("\n\nGeneration interrupted by user.")
        response = input("Save partial results? (y/n): ")
        if response.lower() != 'y':
            print("Discarding results.")
            return {}

    # Write final output
    print("\n" + "=" * 60)
    print("Writing output...")
    write_jsonl_output(conversations, output_path)

    # Generate metadata
    stats = orchestrator.get_statistics()
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'total_conversations': len(conversations),
        'target_count': count,
        'completed': len(conversations) >= count,
        'mock_mode': mock_mode,
        'statistics': stats
    }

    # Save metadata
    metadata_path = output_path.parent / "generation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {metadata_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Total generated: {len(conversations)}")
    print(f"Target: {count}")
    print(f"Success rate: {(len(conversations) / count * 100):.1f}%")
    print(f"\nOutput: {output_path}")

    # Print distribution summary
    orchestrator.print_statistics()

    # Cleanup checkpoints
    checkpoint_mgr.cleanup_checkpoints()

    if claude_client:
        claude_client.print_metrics()

    return metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic tool-use conversation data"
    )
    parser.add_argument(
        '--count',
        type=int,
        default=1000,
        help='Number of conversations to generate (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/synthetic/tool_use_examples.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='data/synthetic/checkpoints',
        help='Directory for checkpoints'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Mock mode (no API calls, for testing)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Anthropic API key (or use ANTHROPIC_API_KEY env var)'
    )

    args = parser.parse_args()

    # Convert paths to absolute
    output_path = (PROJECT_ROOT / args.output).resolve()
    checkpoint_dir = (PROJECT_ROOT / args.checkpoint_dir).resolve()

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    try:
        generate_synthetic_data(
            count=args.count,
            output_path=output_path,
            checkpoint_dir=checkpoint_dir,
            mock_mode=args.mock,
            api_key=args.api_key
        )
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
