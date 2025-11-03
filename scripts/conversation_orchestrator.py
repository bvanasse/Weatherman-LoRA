#!/usr/bin/env python3
"""
Conversation Orchestrator Module

Orchestrates synthetic conversation generation with scenario and persona distribution.
Coordinates location selection, prompt generation, and conversation assembly.

Usage:
    from scripts.conversation_orchestrator import ConversationOrchestrator

    orchestrator = ConversationOrchestrator()
    config = orchestrator.get_next_generation_config()
"""

import random
import uuid
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from scripts.geographic_database import get_random_location, Location


@dataclass
class GenerationStats:
    """Track generation statistics."""
    total_generated: int = 0
    by_scenario: Dict[str, int] = field(default_factory=lambda: {
        'success': 0,
        'error': 0,
        'multi_turn': 0
    })
    by_persona: Dict[str, int] = field(default_factory=lambda: {
        'neutral': 0,
        'twain': 0,
        'franklin': 0
    })
    by_query_type: Dict[str, int] = field(default_factory=lambda: {
        'current': 0,
        'forecast': 0,
        'geocode': 0
    })
    locations_used: set = field(default_factory=set)

    def add_generation(
        self,
        scenario: str,
        persona: str,
        query_type: str,
        location_city: str
    ):
        """Record a generated conversation."""
        self.total_generated += 1
        self.by_scenario[scenario] = self.by_scenario.get(scenario, 0) + 1
        self.by_persona[persona] = self.by_persona.get(persona, 0) + 1
        self.by_query_type[query_type] = self.by_query_type.get(query_type, 0) + 1
        self.locations_used.add(location_city)

    def get_distribution_percentages(self) -> Dict[str, Dict[str, float]]:
        """Get distribution percentages."""
        if self.total_generated == 0:
            return {
                'scenario': {},
                'persona': {},
                'query_type': {}
            }

        return {
            'scenario': {
                k: (v / self.total_generated * 100)
                for k, v in self.by_scenario.items()
            },
            'persona': {
                k: (v / self.total_generated * 100)
                for k, v in self.by_persona.items()
            },
            'query_type': {
                k: (v / self.total_generated * 100)
                for k, v in self.by_query_type.items()
            }
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary."""
        distributions = self.get_distribution_percentages()

        return {
            'total_generated': self.total_generated,
            'scenarios': self.by_scenario.copy(),
            'personas': self.by_persona.copy(),
            'query_types': self.by_query_type.copy(),
            'unique_locations': len(self.locations_used),
            'distributions': distributions
        }


class ConversationOrchestrator:
    """
    Orchestrate synthetic conversation generation.

    Manages:
    - Scenario distribution (60-70% success, 15-20% error, 15-20% multi-turn)
    - Persona distribution (60% neutral, 25% Twain, 15% Franklin)
    - Location selection with geographic diversity
    - Generation configuration creation
    """

    def __init__(
        self,
        scenario_weights: Optional[Dict[str, float]] = None,
        persona_weights: Optional[Dict[str, float]] = None,
        query_type_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize conversation orchestrator.

        Args:
            scenario_weights: Custom scenario distribution weights
            persona_weights: Custom persona distribution weights
            query_type_weights: Custom query type distribution weights
        """
        # Default scenario distribution: 65% success, 17.5% error, 17.5% multi-turn
        self.scenario_weights = scenario_weights or {
            'success': 0.65,
            'error': 0.175,
            'multi_turn': 0.175
        }

        # Default persona distribution: 60% neutral, 25% twain, 15% franklin
        self.persona_weights = persona_weights or {
            'neutral': 0.60,
            'twain': 0.25,
            'franklin': 0.15
        }

        # Default query type distribution: 50% current, 40% forecast, 10% geocode
        self.query_type_weights = query_type_weights or {
            'current': 0.50,
            'forecast': 0.40,
            'geocode': 0.10
        }

        # Track statistics
        self.stats = GenerationStats()

        # Track recently used locations to encourage diversity
        self.recent_locations = []
        self.max_recent_locations = 20

    def select_scenario(self) -> str:
        """
        Select scenario based on distribution weights.

        Returns:
            Scenario name ('success', 'error', 'multi_turn')
        """
        scenarios = list(self.scenario_weights.keys())
        weights = list(self.scenario_weights.values())
        return random.choices(scenarios, weights=weights, k=1)[0]

    def select_persona(self) -> str:
        """
        Select persona based on distribution weights.

        Returns:
            Persona name ('neutral', 'twain', 'franklin')
        """
        personas = list(self.persona_weights.keys())
        weights = list(self.persona_weights.values())
        return random.choices(personas, weights=weights, k=1)[0]

    def select_query_type(self, scenario: str) -> str:
        """
        Select query type based on distribution weights and scenario.

        Args:
            scenario: Scenario type

        Returns:
            Query type ('current', 'forecast', 'geocode')
        """
        # For error scenarios, adjust distribution to test more edge cases
        if scenario == 'error':
            # More variety in error scenarios
            return random.choice(['current', 'forecast', 'geocode'])

        query_types = list(self.query_type_weights.keys())
        weights = list(self.query_type_weights.values())
        return random.choices(query_types, weights=weights, k=1)[0]

    def select_location(self) -> Location:
        """
        Select location with geographic diversity.

        Prefers locations not recently used to maximize diversity.

        Returns:
            Location object
        """
        # Try to get a location not recently used
        max_attempts = 10
        for _ in range(max_attempts):
            location = get_random_location()
            if location.city not in self.recent_locations:
                break

        # Track recent locations
        self.recent_locations.append(location.city)
        if len(self.recent_locations) > self.max_recent_locations:
            self.recent_locations.pop(0)

        return location

    def generate_conversation_id(self) -> str:
        """
        Generate unique conversation ID.

        Returns:
            UUID string
        """
        return str(uuid.uuid4())

    def get_next_generation_config(self) -> Dict[str, Any]:
        """
        Get configuration for next conversation generation.

        Returns:
            Configuration dictionary with scenario, persona, location, etc.
        """
        scenario = self.select_scenario()
        persona = self.select_persona()
        query_type = self.select_query_type(scenario)
        location = self.select_location()
        conversation_id = self.generate_conversation_id()

        config = {
            'conversation_id': conversation_id,
            'scenario': scenario,
            'persona': persona,
            'query_type': query_type,
            'location': location,
            'timestamp': datetime.now().isoformat()
        }

        # Record stats
        self.stats.add_generation(scenario, persona, query_type, location.city)

        return config

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get generation statistics.

        Returns:
            Statistics dictionary
        """
        return self.stats.get_summary()

    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        distributions = stats['distributions']

        print("\n" + "=" * 60)
        print("Generation Statistics")
        print("=" * 60)
        print(f"Total Generated: {stats['total_generated']}")
        print(f"Unique Locations: {stats['unique_locations']}")

        print("\nScenario Distribution:")
        for scenario, count in stats['scenarios'].items():
            pct = distributions['scenario'].get(scenario, 0)
            print(f"  {scenario:12s}: {count:4d} ({pct:5.1f}%)")

        print("\nPersona Distribution:")
        for persona, count in stats['personas'].items():
            pct = distributions['persona'].get(persona, 0)
            print(f"  {persona:12s}: {count:4d} ({pct:5.1f}%)")

        print("\nQuery Type Distribution:")
        for query_type, count in stats['query_types'].items():
            pct = distributions['query_type'].get(query_type, 0)
            print(f"  {query_type:12s}: {count:4d} ({pct:5.1f}%)")


if __name__ == "__main__":
    # Test conversation orchestrator
    print("Conversation Orchestrator Test")
    print("=" * 60)

    orchestrator = ConversationOrchestrator()

    # Generate multiple configs to test distribution
    print("\nGenerating 100 test configurations...")
    for _ in range(100):
        config = orchestrator.get_next_generation_config()

    # Print statistics
    orchestrator.print_statistics()

    # Verify distributions are close to targets
    stats = orchestrator.get_statistics()
    distributions = stats['distributions']

    print("\n" + "=" * 60)
    print("Distribution Validation:")

    scenario_dist = distributions['scenario']
    persona_dist = distributions['persona']

    print("\nScenario distribution targets:")
    print(f"  Success: 60-70% (actual: {scenario_dist['success']:.1f}%)")
    print(f"  Error: 15-20% (actual: {scenario_dist['error']:.1f}%)")
    print(f"  Multi-turn: 15-20% (actual: {scenario_dist['multi_turn']:.1f}%)")

    print("\nPersona distribution targets:")
    print(f"  Neutral: ~60% (actual: {persona_dist['neutral']:.1f}%)")
    print(f"  Twain: ~25% (actual: {persona_dist['twain']:.1f}%)")
    print(f"  Franklin: ~15% (actual: {persona_dist['franklin']:.1f}%)")

    print(f"\nGeographic diversity: {stats['unique_locations']} unique locations used")
