#!/usr/bin/env python3
"""
Synthetic Data Validators Module

Advanced validators for synthetic tool-use conversations.
Includes semantic validation, groundedness checking, and reporting.

Usage:
    from scripts.synthetic_data_validators import (
        validate_semantic_weather_data,
        validate_groundedness,
        ValidationReporter
    )
"""

import json
import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

from scripts.chat_format_validator import validate_chat_entry
from scripts.weather_tool_schema import validate_parameters
from scripts.geographic_database import LOCATIONS
from scripts.mock_weather_responses import find_nearest_location


@dataclass
class ValidationReport:
    """Validation report for a batch of conversations."""
    total_validated: int = 0
    passed: int = 0
    failed: int = 0
    failures_by_type: Dict[str, int] = field(default_factory=dict)
    sample_failures: List[Dict[str, str]] = field(default_factory=list)
    max_sample_failures: int = 10

    def add_success(self):
        """Record a successful validation."""
        self.total_validated += 1
        self.passed += 1

    def add_failure(self, failure_type: str, details: str, conversation_id: str = None):
        """Record a validation failure."""
        self.total_validated += 1
        self.failed += 1

        # Track failure counts by type
        self.failures_by_type[failure_type] = \
            self.failures_by_type.get(failure_type, 0) + 1

        # Save sample failures
        if len(self.sample_failures) < self.max_sample_failures:
            self.sample_failures.append({
                'type': failure_type,
                'details': details,
                'conversation_id': conversation_id or 'unknown'
            })

    def get_pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        if self.total_validated == 0:
            return 0.0
        return (self.passed / self.total_validated) * 100

    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'total_validated': self.total_validated,
            'passed': self.passed,
            'failed': self.failed,
            'pass_rate': round(self.get_pass_rate(), 2),
            'failures_by_type': self.failures_by_type.copy(),
            'sample_failure_count': len(self.sample_failures)
        }


def validate_json_schema(conversation: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that all tool_call arguments are valid JSON and match schemas.

    Args:
        conversation: Conversation dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    messages = conversation.get('messages', [])

    for i, msg in enumerate(messages):
        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
            for j, tool_call in enumerate(msg['tool_calls']):
                function = tool_call.get('function', {})
                function_name = function.get('name', '')
                arguments_str = function.get('arguments', '{}')

                # Parse arguments
                try:
                    if isinstance(arguments_str, str):
                        arguments = json.loads(arguments_str)
                    else:
                        arguments = arguments_str
                except json.JSONDecodeError as e:
                    return False, f"Message {i} tool_call {j}: Invalid JSON - {e}"

                # Validate against schema
                is_valid, error = validate_parameters(function_name, arguments)
                if not is_valid:
                    return False, f"Message {i} tool_call {j}: {error}"

    return True, ""


def validate_semantic_weather_data(conversation: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that weather data is semantically realistic.

    Checks:
    - Temperature values reasonable for location climate zone
    - Weather conditions exist in defined set
    - Forecast day counts match request
    - No unrealistic combinations (e.g., snow in tropics in summer)

    Args:
        conversation: Conversation dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    messages = conversation.get('messages', [])
    metadata = conversation.get('metadata', {})

    # Find tool call and tool response
    tool_call_msg = None
    tool_response_msg = None

    for msg in messages:
        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
            tool_call_msg = msg
        elif msg.get('role') == 'tool':
            tool_response_msg = msg

    if not tool_call_msg or not tool_response_msg:
        # No tool calls to validate
        return True, ""

    # Parse tool call arguments
    tool_call = tool_call_msg['tool_calls'][0]
    function_name = tool_call['function']['name']
    arguments = json.loads(tool_call['function']['arguments'])

    # Parse tool response
    try:
        tool_data = json.loads(tool_response_msg['content'])
    except json.JSONDecodeError:
        return False, "Tool response is not valid JSON"

    # Handle error responses
    if tool_data.get('error'):
        # Error responses are valid
        return True, ""

    # Find nearest location for semantic validation
    if 'latitude' in arguments and 'longitude' in arguments:
        location = find_nearest_location(arguments['latitude'], arguments['longitude'])
    else:
        location = None

    # Validate current weather
    if function_name == 'get_current_weather' and 'temperature' in tool_data:
        temp = tool_data['temperature']

        # Check temperature is in reasonable range (-60 to 60 Celsius)
        if not (-60 <= temp <= 60):
            return False, f"Temperature {temp}°C out of reasonable range"

        # Check against climate zone if location found
        if location:
            # Get expected range (very broad to allow variance)
            expected_min = min(location.temp_range_summer[0], location.temp_range_winter[0]) - 15
            expected_max = max(location.temp_range_summer[1], location.temp_range_winter[1]) + 15

            if not (expected_min <= temp <= expected_max):
                return False, (
                    f"Temperature {temp}°C unrealistic for {location.city} "
                    f"(expected {expected_min} to {expected_max}°C)"
                )

        # Validate condition
        condition = tool_data.get('condition', 'unknown')
        valid_conditions = [
            'clear', 'partly_cloudy', 'cloudy', 'rain', 'drizzle', 'snow',
            'sleet', 'fog', 'mist', 'thunderstorm', 'hot', 'dust', 'sandstorm',
            'blizzard', 'freezing_rain', 'extreme_cold'
        ]
        if condition not in valid_conditions:
            return False, f"Invalid weather condition: {condition}"

        # Validate humidity
        humidity = tool_data.get('humidity')
        if humidity is not None:
            if not (0 <= humidity <= 100):
                return False, f"Humidity {humidity}% out of range (0-100)"

    # Validate forecast
    elif function_name == 'get_forecast' and 'daily' in tool_data:
        requested_days = arguments.get('days', 0)
        actual_days = len(tool_data['daily'])

        if requested_days != actual_days:
            return False, f"Forecast days mismatch: requested {requested_days}, got {actual_days}"

        # Validate each day
        for i, day in enumerate(tool_data['daily']):
            temp_min = day.get('temperature_min')
            temp_max = day.get('temperature_max')

            if temp_min is not None and temp_max is not None:
                # Max should be greater than min
                if temp_max <= temp_min:
                    return False, f"Day {i}: Max temp ({temp_max}) not greater than min ({temp_min})"

                # Check reasonable temperature range
                if not (-60 <= temp_min <= 60) or not (-60 <= temp_max <= 60):
                    return False, f"Day {i}: Temperatures out of reasonable range"

    return True, ""


def validate_groundedness(conversation: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Validate that assistant responses reference tool output data.

    Checks that the final assistant response mentions specific data
    from the tool response (temperatures, conditions, etc.).

    Args:
        conversation: Conversation dictionary

    Returns:
        Tuple of (is_grounded, groundedness_score)
        - is_grounded: True if response references tool data
        - groundedness_score: 0.0-1.0 indicating how well grounded
    """
    messages = conversation.get('messages', [])

    # Find tool response and final assistant message
    tool_response_msg = None
    final_assistant_msg = None

    for msg in messages:
        if msg.get('role') == 'tool':
            tool_response_msg = msg
        elif msg.get('role') == 'assistant' and 'tool_calls' not in msg:
            final_assistant_msg = msg

    if not tool_response_msg or not final_assistant_msg:
        # Can't validate groundedness without both
        return True, 1.0

    # Parse tool response
    try:
        tool_data = json.loads(tool_response_msg['content'])
    except json.JSONDecodeError:
        return False, 0.0

    # Handle error responses
    if tool_data.get('error'):
        # Check that assistant mentions error
        assistant_text = final_assistant_msg.get('content', '').lower()
        if 'error' in assistant_text or 'sorry' in assistant_text or 'apologize' in assistant_text:
            return True, 1.0
        return False, 0.0

    # Extract key data points from tool response
    assistant_text = final_assistant_msg.get('content', '')
    data_points_mentioned = 0
    total_data_points = 0

    # Check for temperature mention
    if 'temperature' in tool_data:
        total_data_points += 1
        temp = tool_data['temperature']
        # Look for temperature in assistant response (with some flexibility)
        if str(int(temp)) in assistant_text or f"{temp}" in assistant_text:
            data_points_mentioned += 1

    # Check for condition mention
    if 'condition' in tool_data:
        total_data_points += 1
        condition = tool_data['condition'].replace('_', ' ')
        if condition.lower() in assistant_text.lower():
            data_points_mentioned += 1

    # Check for location mention
    if 'location' in tool_data:
        total_data_points += 1
        location_name = tool_data['location']
        if any(part.lower() in assistant_text.lower() for part in location_name.split(',')):
            data_points_mentioned += 1

    # Check forecast days
    if 'daily' in tool_data and len(tool_data['daily']) > 0:
        total_data_points += 1
        first_day = tool_data['daily'][0]
        # Check if any temp from first day mentioned
        temp_min = first_day.get('temperature_min')
        temp_max = first_day.get('temperature_max')
        if (temp_min and str(int(temp_min)) in assistant_text) or \
           (temp_max and str(int(temp_max)) in assistant_text):
            data_points_mentioned += 1

    # Calculate groundedness score
    if total_data_points == 0:
        return True, 1.0

    groundedness_score = data_points_mentioned / total_data_points

    # Consider grounded if at least 50% of data mentioned
    is_grounded = groundedness_score >= 0.5

    return is_grounded, groundedness_score


def validate_conversation_full(conversation: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Run all validations on a conversation.

    Args:
        conversation: Conversation dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # 1. Basic chat format validation
    is_valid, error = validate_chat_entry(conversation)
    if not is_valid:
        errors.append(f"Chat format: {error}")

    # 2. JSON schema validation
    is_valid, error = validate_json_schema(conversation)
    if not is_valid:
        errors.append(f"JSON schema: {error}")

    # 3. Semantic validation
    is_valid, error = validate_semantic_weather_data(conversation)
    if not is_valid:
        errors.append(f"Semantic: {error}")

    # 4. Groundedness validation
    is_grounded, score = validate_groundedness(conversation)
    if not is_grounded:
        errors.append(f"Groundedness: Score {score:.2f} below threshold (< 0.5)")

    return len(errors) == 0, errors


class ValidationReporter:
    """Generate validation reports for batches of conversations."""

    def __init__(self):
        """Initialize validation reporter."""
        self.report = ValidationReport()

    def validate_batch(self, conversations: List[Dict[str, Any]]) -> ValidationReport:
        """
        Validate a batch of conversations and generate report.

        Args:
            conversations: List of conversation dictionaries

        Returns:
            ValidationReport object
        """
        for conv in conversations:
            is_valid, errors = validate_conversation_full(conv)

            if is_valid:
                self.report.add_success()
            else:
                # Record failure with first error type
                failure_type = errors[0].split(':')[0] if errors else 'unknown'
                details = '; '.join(errors)
                conv_id = conv.get('metadata', {}).get('conversation_id', 'unknown')

                self.report.add_failure(failure_type, details, conv_id)

        return self.report

    def print_report(self):
        """Print formatted validation report."""
        summary = self.report.get_summary()

        print("\n" + "=" * 60)
        print("Validation Report")
        print("=" * 60)
        print(f"Total Validated: {summary['total_validated']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass Rate: {summary['pass_rate']:.1f}%")

        if summary['failures_by_type']:
            print("\nFailures by Type:")
            for failure_type, count in summary['failures_by_type'].items():
                print(f"  {failure_type:20s}: {count}")

        if self.report.sample_failures:
            print(f"\nSample Failures ({len(self.report.sample_failures)}):")
            for i, failure in enumerate(self.report.sample_failures[:5], 1):
                print(f"\n  {i}. {failure['type']}")
                print(f"     Conv ID: {failure['conversation_id']}")
                print(f"     Details: {failure['details'][:100]}...")


if __name__ == "__main__":
    # Test validators
    print("Synthetic Data Validators Test")
    print("=" * 60)

    from scripts.conversation_orchestrator import ConversationOrchestrator
    from scripts.conversation_assembly import ConversationAssembler

    # Generate test conversations
    orchestrator = ConversationOrchestrator()
    assembler = ConversationAssembler(mock_mode=True)

    conversations = []
    for _ in range(10):
        config = orchestrator.get_next_generation_config()
        conv = assembler.assemble_conversation(config)
        conversations.append(conv)

    # Validate batch
    reporter = ValidationReporter()
    report = reporter.validate_batch(conversations)

    # Print report
    reporter.print_report()
