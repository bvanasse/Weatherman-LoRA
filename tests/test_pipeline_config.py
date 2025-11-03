#!/usr/bin/env python3
"""
Tests for Pipeline Configuration Loading and Validation

Tests Task Group 1: Configuration and Dependencies
- Configuration file loading
- Required fields validation
- Threshold values
- Environment variable overrides
"""

import os
import sys
import json
import pytest
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config_loader import load_json, validate_required_fields, get_config_value


class TestPipelineConfigLoading:
    """Test loading pipeline configuration files."""

    def test_pipeline_config_exists(self):
        """Test that pipeline_config.json file exists."""
        config_path = Path(__file__).parent.parent / 'configs' / 'pipeline_config.json'
        assert config_path.exists(), f"Pipeline config not found at {config_path}"

    def test_pipeline_config_valid_json(self):
        """Test that pipeline_config.json is valid JSON."""
        config_path = Path(__file__).parent.parent / 'configs' / 'pipeline_config.json'
        config = load_json(config_path)
        assert isinstance(config, dict), "Config should be a dictionary"
        assert len(config) > 0, "Config should not be empty"

    def test_paths_config_has_pipeline_paths(self):
        """Test that paths_config.json includes pipeline output paths."""
        config_path = Path(__file__).parent.parent / 'configs' / 'paths_config.json'
        config = load_json(config_path)

        assert 'data' in config, "Config should have 'data' section"
        assert 'pipeline' in config['data'], "Config should have 'data.pipeline' section"

        pipeline_paths = config['data']['pipeline']
        assert 'output' in pipeline_paths, "Pipeline should have output path"
        assert 'stats_json' in pipeline_paths, "Pipeline should have stats_json path"
        assert 'stats_md' in pipeline_paths, "Pipeline should have stats_md path"

    def test_pipeline_config_required_fields(self):
        """Test that pipeline config contains all required fields."""
        config_path = Path(__file__).parent.parent / 'configs' / 'pipeline_config.json'
        config = load_json(config_path)

        required_fields = [
            'normalization.form',
            'deduplication.threshold',
            'deduplication.num_perm',
            'language_filter.target_language',
            'safety_filter.enabled',
            'safety_filter.batch_size',
            'quality.min_length',
            'quality.max_length',
        ]

        # Should not raise ValueError
        validate_required_fields(config, required_fields, 'pipeline_config.json')

    def test_deduplication_threshold_value(self):
        """Test that deduplication threshold is set to 0.8 as specified."""
        config_path = Path(__file__).parent.parent / 'configs' / 'pipeline_config.json'
        config = load_json(config_path)

        threshold = get_config_value(config, 'deduplication.threshold')
        assert threshold == 0.8, f"Deduplication threshold should be 0.8, got {threshold}"

    def test_normalization_form_is_nfc(self):
        """Test that unicode normalization form is set to NFC."""
        config_path = Path(__file__).parent.parent / 'configs' / 'pipeline_config.json'
        config = load_json(config_path)

        form = get_config_value(config, 'normalization.form')
        assert form == 'NFC', f"Normalization form should be 'NFC', got {form}"

    def test_quality_min_length_positive(self):
        """Test that minimum length is a positive integer."""
        config_path = Path(__file__).parent.parent / 'configs' / 'pipeline_config.json'
        config = load_json(config_path)

        min_length = get_config_value(config, 'quality.min_length')
        assert isinstance(min_length, int), "Min length should be an integer"
        assert min_length > 0, "Min length should be positive"

    def test_safety_filter_batch_size(self):
        """Test that safety filter batch size is reasonable."""
        config_path = Path(__file__).parent.parent / 'configs' / 'pipeline_config.json'
        config = load_json(config_path)

        batch_size = get_config_value(config, 'safety_filter.batch_size')
        assert isinstance(batch_size, int), "Batch size should be an integer"
        assert 1 <= batch_size <= 100, "Batch size should be between 1 and 100"


class TestConfigEnvironmentOverrides:
    """Test environment variable overrides for configuration."""

    def test_config_loads_without_env_vars(self):
        """Test that config loads successfully without environment variables."""
        # Clear any existing env vars
        if 'WEATHERMAN_BASE_DIR' in os.environ:
            del os.environ['WEATHERMAN_BASE_DIR']

        config_path = Path(__file__).parent.parent / 'configs' / 'pipeline_config.json'
        config = load_json(config_path)
        assert isinstance(config, dict), "Config should load without env vars"

    def test_missing_required_field_raises_error(self):
        """Test that missing required fields raise ValueError."""
        config = {'normalization': {'form': 'NFC'}}  # Missing deduplication
        required_fields = ['deduplication.threshold']

        with pytest.raises(ValueError) as exc_info:
            validate_required_fields(config, required_fields, 'test_config')

        assert 'deduplication.threshold' in str(exc_info.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
