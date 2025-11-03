#!/usr/bin/env python3
"""
Claude API Client Module

Wrapper for Anthropic API with retry logic, rate limiting, and metrics tracking.
Uses Claude Haiku 4.5 for synthetic conversation generation.

Usage:
    from scripts.claude_api_client import ClaudeAPIClient

    client = ClaudeAPIClient()
    response = client.generate_conversation(prompt="...")
"""

import os
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


# Try to import anthropic, handle gracefully if not installed
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None


@dataclass
class APIMetrics:
    """Track API usage metrics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    def add_success(self, input_tokens: int, output_tokens: int, latency: float):
        """Record a successful API call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency_seconds += latency

    def add_failure(self, error: str):
        """Record a failed API call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.errors.append(error)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        avg_latency = (self.total_latency_seconds / self.successful_calls
                      if self.successful_calls > 0 else 0)

        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': (self.successful_calls / self.total_calls * 100
                           if self.total_calls > 0 else 0),
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'avg_latency_seconds': round(avg_latency, 2),
            'recent_errors': self.errors[-5:]  # Last 5 errors
        }


class ClaudeAPIClient:
    """
    Wrapper for Anthropic Claude API with retry logic and rate limiting.

    Features:
    - Automatic retry with exponential backoff
    - Rate limiting to respect API limits
    - Metrics tracking (calls, tokens, latency)
    - API key management
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-haiku-20241022",
        max_retries: int = 3,
        retry_delays: List[float] = None,
        rate_limit_delay: float = 0.5
    ):
        """
        Initialize Claude API client.

        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var or prompts)
            model: Model name (default: claude-3-5-haiku-20241022)
            max_retries: Maximum number of retries on failure
            retry_delays: Delays between retries in seconds (default: [1, 2, 4])
            rate_limit_delay: Minimum delay between API calls (seconds)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        # Get or prompt for API key
        self.api_key = api_key or self._get_api_key()
        self.model = model
        self.max_retries = max_retries
        self.retry_delays = retry_delays or [1.0, 2.0, 4.0]
        self.rate_limit_delay = rate_limit_delay

        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Track metrics
        self.metrics = APIMetrics()

        # Track last call time for rate limiting
        self.last_call_time = 0.0

    def _get_api_key(self) -> str:
        """
        Get API key from environment or prompt user.

        Returns:
            API key string

        Raises:
            ValueError: If no API key provided
        """
        # Check environment variable
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if api_key:
            return api_key

        # Prompt user
        print("\nAnthropic API key not found in environment.")
        print("Please enter your API key (or set ANTHROPIC_API_KEY environment variable):")
        api_key = input("API Key: ").strip()

        if not api_key:
            raise ValueError("No API key provided")

        return api_key

    def _apply_rate_limit(self):
        """Apply rate limiting delay."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time

        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            time.sleep(sleep_time)

        self.last_call_time = time.time()

    def generate_message(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate message using Claude API with retry logic.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system: System prompt (optional)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            tools: Optional tool schemas for function calling

        Returns:
            API response dictionary

        Raises:
            Exception: If all retries fail
        """
        # Apply rate limiting
        self._apply_rate_limit()

        # Attempt with retries
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Make API call
                kwargs = {
                    'model': self.model,
                    'max_tokens': max_tokens,
                    'messages': messages,
                    'temperature': temperature
                }

                if system:
                    kwargs['system'] = system

                if tools:
                    kwargs['tools'] = tools

                response = self.client.messages.create(**kwargs)

                # Calculate latency
                latency = time.time() - start_time

                # Extract token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

                # Record success
                self.metrics.add_success(input_tokens, output_tokens, latency)

                # Convert to dict for easier handling
                return {
                    'id': response.id,
                    'model': response.model,
                    'role': response.role,
                    'content': response.content,
                    'stop_reason': response.stop_reason,
                    'usage': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens
                    }
                }

            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}"

                # Record failure
                self.metrics.add_failure(error_msg)

                # If this was the last attempt, raise
                if attempt == self.max_retries - 1:
                    raise

                # Otherwise, wait and retry
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                print(f"  {error_msg}")
                print(f"  Retrying in {delay} seconds...")
                time.sleep(delay)

        # Should never reach here, but just in case
        raise Exception("All retry attempts failed")

    def generate_conversation(
        self,
        prompt: str,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0
    ) -> str:
        """
        Generate conversation from a prompt.

        Convenience method that wraps generate_message for simple text generation.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            tools: Optional tool schemas
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling

        Returns:
            Generated text content

        Examples:
            >>> client = ClaudeAPIClient()
            >>> response = client.generate_conversation(
            ...     prompt="Generate a weather query",
            ...     system="You are a weather assistant"
            ... )
        """
        messages = [{'role': 'user', 'content': prompt}]

        response = self.generate_message(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools
        )

        # Extract text from content blocks
        content_blocks = response['content']
        text_parts = []

        for block in content_blocks:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
            elif isinstance(block, dict) and 'text' in block:
                text_parts.append(block['text'])

        return '\n'.join(text_parts)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get API usage metrics.

        Returns:
            Dictionary with metrics summary
        """
        return self.metrics.get_summary()

    def print_metrics(self):
        """Print formatted metrics summary."""
        metrics = self.get_metrics()

        print("\n" + "=" * 60)
        print("Claude API Metrics")
        print("=" * 60)
        print(f"Total Calls: {metrics['total_calls']}")
        print(f"Successful: {metrics['successful_calls']}")
        print(f"Failed: {metrics['failed_calls']}")
        print(f"Success Rate: {metrics['success_rate']:.1f}%")
        print(f"\nToken Usage:")
        print(f"  Input Tokens: {metrics['total_input_tokens']:,}")
        print(f"  Output Tokens: {metrics['total_output_tokens']:,}")
        print(f"  Total Tokens: {metrics['total_tokens']:,}")
        print(f"\nPerformance:")
        print(f"  Avg Latency: {metrics['avg_latency_seconds']:.2f}s")

        if metrics['recent_errors']:
            print(f"\nRecent Errors ({len(metrics['recent_errors'])}):")
            for error in metrics['recent_errors']:
                print(f"  - {error}")


if __name__ == "__main__":
    # Test Claude API client (requires valid API key)
    print("Claude API Client Test")
    print("=" * 60)

    try:
        client = ClaudeAPIClient()
        print("✓ Client initialized successfully")
        print(f"  Model: {client.model}")
        print(f"  Max Retries: {client.max_retries}")
        print(f"  Rate Limit Delay: {client.rate_limit_delay}s")

    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        print("\nNote: This test requires a valid ANTHROPIC_API_KEY")
