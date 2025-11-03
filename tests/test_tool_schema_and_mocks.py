#!/usr/bin/env python3
"""
Tests for Task Group 1: Weather Tool Schema Design and Mock Response Generation

Tests cover:
- Tool schema validation (latitude/longitude bounds, required fields)
- get_forecast schema validation (days parameter 1-14 range)
- geocode_location mock responses (valid city/country mapping)
- Semantic weather data realism (temperature ranges for climate zones)
- Error response generation (invalid parameters, missing data)
"""

import unittest
import json
from scripts.weather_tool_schema import TOOL_SCHEMAS, validate_parameters
from scripts.geographic_database import (
    get_location_by_city, get_locations_by_climate, LOCATIONS
)
from scripts.mock_weather_responses import (
    generate_current_weather, generate_forecast, generate_geocode_response,
    generate_error_response, find_nearest_location, generate_realistic_temperature,
    get_season_at_location
)


class TestToolSchemaValidation(unittest.TestCase):
    """Test tool schema validation for required fields and parameter bounds."""

    def test_get_current_weather_schema_valid_parameters(self):
        """Test valid latitude/longitude parameters for get_current_weather."""
        is_valid, error = validate_parameters('get_current_weather', {
            'latitude': 40.7128,
            'longitude': -74.0060
        })
        self.assertTrue(is_valid, f"Should be valid: {error}")

    def test_get_current_weather_latitude_out_of_bounds(self):
        """Test latitude bounds checking (must be -90 to 90)."""
        # Test latitude > 90
        is_valid, error = validate_parameters('get_current_weather', {
            'latitude': 100,
            'longitude': -74.0
        })
        self.assertFalse(is_valid)
        self.assertIn('latitude', error.lower())

        # Test latitude < -90
        is_valid, error = validate_parameters('get_current_weather', {
            'latitude': -100,
            'longitude': -74.0
        })
        self.assertFalse(is_valid)
        self.assertIn('latitude', error.lower())

    def test_get_current_weather_longitude_out_of_bounds(self):
        """Test longitude bounds checking (must be -180 to 180)."""
        # Test longitude > 180
        is_valid, error = validate_parameters('get_current_weather', {
            'latitude': 40.7,
            'longitude': 200
        })
        self.assertFalse(is_valid)
        self.assertIn('longitude', error.lower())

        # Test longitude < -180
        is_valid, error = validate_parameters('get_current_weather', {
            'latitude': 40.7,
            'longitude': -200
        })
        self.assertFalse(is_valid)
        self.assertIn('longitude', error.lower())

    def test_get_current_weather_missing_required_fields(self):
        """Test that missing latitude or longitude is rejected."""
        # Missing latitude
        is_valid, error = validate_parameters('get_current_weather', {
            'longitude': -74.0
        })
        self.assertFalse(is_valid)
        self.assertIn('latitude', error.lower())

        # Missing longitude
        is_valid, error = validate_parameters('get_current_weather', {
            'latitude': 40.7
        })
        self.assertFalse(is_valid)
        self.assertIn('longitude', error.lower())

    def test_get_forecast_days_parameter_validation(self):
        """Test days parameter must be between 1-14."""
        # Valid days
        for days in [1, 7, 14]:
            is_valid, error = validate_parameters('get_forecast', {
                'latitude': 40.7,
                'longitude': -74.0,
                'days': days
            })
            self.assertTrue(is_valid, f"Days={days} should be valid: {error}")

        # Invalid days (< 1)
        is_valid, error = validate_parameters('get_forecast', {
            'latitude': 40.7,
            'longitude': -74.0,
            'days': 0
        })
        self.assertFalse(is_valid)
        self.assertIn('days', error.lower())

        # Invalid days (> 14)
        is_valid, error = validate_parameters('get_forecast', {
            'latitude': 40.7,
            'longitude': -74.0,
            'days': 20
        })
        self.assertFalse(is_valid)
        self.assertIn('days', error.lower())


class TestGeocodeLocationMocks(unittest.TestCase):
    """Test geocode_location mock responses for valid city/country mapping."""

    def test_geocode_known_cities(self):
        """Test that known cities return valid coordinates."""
        test_cities = [
            ('New York', 'USA', 40.7128, -74.0060),
            ('London', 'United Kingdom', 51.5074, -0.1278),
            ('Tokyo', 'Japan', 35.6762, 139.6503),
            ('Sydney', 'Australia', -33.8688, 151.2093),
        ]

        for city, country, expected_lat, expected_lon in test_cities:
            response = generate_geocode_response(city, country)
            self.assertTrue(response['found'], f"{city}, {country} should be found")
            self.assertAlmostEqual(response['latitude'], expected_lat, places=2)
            self.assertAlmostEqual(response['longitude'], expected_lon, places=2)

    def test_geocode_unknown_city(self):
        """Test that unknown cities return not found response."""
        response = generate_geocode_response('Nonexistent City', 'Fake Country')
        self.assertFalse(response['found'])
        self.assertIn('error', response)


class TestSemanticWeatherRealism(unittest.TestCase):
    """Test that weather data is semantically realistic for climate zones."""

    def test_tropical_climate_temperature_ranges(self):
        """Test tropical locations have appropriate temperature ranges."""
        # Get tropical locations
        tropical_locs = get_locations_by_climate('tropical')
        self.assertGreater(len(tropical_locs), 0, "Should have tropical locations")

        # Test multiple tropical locations
        for loc in tropical_locs[:3]:  # Test first 3
            # Summer temps should be warm
            summer_temp = generate_realistic_temperature(loc, 'summer')
            self.assertGreaterEqual(summer_temp, 20,
                f"{loc.city} tropical summer should be >= 20°C")
            self.assertLessEqual(summer_temp, 40,
                f"{loc.city} tropical summer should be <= 40°C")

            # Winter temps should still be warm in tropics
            winter_temp = generate_realistic_temperature(loc, 'winter')
            self.assertGreaterEqual(winter_temp, 15,
                f"{loc.city} tropical winter should be >= 15°C")

    def test_arctic_climate_temperature_ranges(self):
        """Test arctic/subarctic locations have cold temperatures."""
        # Find arctic/subarctic locations
        cold_locs = [loc for loc in LOCATIONS if loc.climate_zone in ['arctic', 'subarctic']]
        self.assertGreater(len(cold_locs), 0, "Should have arctic/subarctic locations")

        for loc in cold_locs:
            # Winter temps should be freezing or below
            winter_temp = generate_realistic_temperature(loc, 'winter')
            self.assertLessEqual(winter_temp, 0,
                f"{loc.city} arctic winter should be <= 0°C")

    def test_desert_climate_temperature_extremes(self):
        """Test desert locations have high summer temperatures."""
        # Get desert locations
        desert_locs = get_locations_by_climate('desert')
        self.assertGreater(len(desert_locs), 0, "Should have desert locations")

        for loc in desert_locs[:2]:  # Test first 2
            # Summer temps should be very hot
            summer_temp = generate_realistic_temperature(loc, 'summer')
            self.assertGreaterEqual(summer_temp, 25,
                f"{loc.city} desert summer should be >= 25°C")

    def test_generated_weather_has_realistic_ranges(self):
        """Test that generated current weather has realistic values."""
        # Test New York (temperate)
        nyc_weather = generate_current_weather(40.7128, -74.0060)
        self.assertIn('temperature', nyc_weather)
        self.assertGreaterEqual(nyc_weather['temperature'], -20)
        self.assertLessEqual(nyc_weather['temperature'], 40)
        self.assertIn('humidity', nyc_weather)
        self.assertGreaterEqual(nyc_weather['humidity'], 0)
        self.assertLessEqual(nyc_weather['humidity'], 100)

        # Test Phoenix (desert)
        phoenix_weather = generate_current_weather(33.4484, -112.0740)
        # Desert should have low humidity
        self.assertLessEqual(phoenix_weather['humidity'], 60)

    def test_forecast_temperature_consistency(self):
        """Test that forecast max temps are higher than min temps."""
        forecast = generate_forecast(40.7, -74.0, 5)
        self.assertEqual(len(forecast['daily']), 5)

        for day in forecast['daily']:
            self.assertIn('temperature_max', day)
            self.assertIn('temperature_min', day)
            self.assertGreater(day['temperature_max'], day['temperature_min'],
                "Max temp should be higher than min temp")
            # Difference should be reasonable (not more than 20°C)
            diff = day['temperature_max'] - day['temperature_min']
            self.assertLessEqual(diff, 20, "Daily temp range should be <= 20°C")


class TestErrorResponseGeneration(unittest.TestCase):
    """Test error response generation for various failure scenarios."""

    def test_error_response_structure(self):
        """Test that error responses have correct structure."""
        error = generate_error_response('invalid_location', 'get_current_weather')

        self.assertIn('error', error)
        self.assertTrue(error['error'])
        self.assertIn('error_type', error)
        self.assertEqual(error['error_type'], 'invalid_location')
        self.assertIn('message', error)
        self.assertIn('function', error)
        self.assertEqual(error['function'], 'get_current_weather')

    def test_different_error_types(self):
        """Test various error types generate appropriate messages."""
        error_types = [
            'invalid_location',
            'missing_parameter',
            'out_of_range',
            'api_error',
            'rate_limit'
        ]

        for error_type in error_types:
            error = generate_error_response(error_type, 'get_current_weather')
            self.assertEqual(error['error_type'], error_type)
            self.assertIn('message', error)
            self.assertIsInstance(error['message'], str)
            self.assertGreater(len(error['message']), 0)

    def test_error_with_details(self):
        """Test error responses include provided details."""
        details = "Latitude must be between -90 and 90"
        error = generate_error_response('out_of_range', 'get_current_weather',
                                       details=details)
        self.assertIn(details, error['message'])


class TestGeographicDatabaseCoverage(unittest.TestCase):
    """Test that geographic database has sufficient coverage."""

    def test_minimum_location_count(self):
        """Test that we have at least 50 locations."""
        self.assertGreaterEqual(len(LOCATIONS), 50,
            "Should have at least 50 locations in database")

    def test_multiple_continents_represented(self):
        """Test that multiple continents are represented."""
        continents = set(loc.continent for loc in LOCATIONS)
        self.assertGreaterEqual(len(continents), 5,
            "Should have locations from at least 5 continents")

        # Check for specific continents
        expected_continents = ['North America', 'Europe', 'Asia', 'Africa', 'Oceania']
        for continent in expected_continents:
            self.assertIn(continent, continents,
                f"Should have locations from {continent}")

    def test_multiple_climate_zones(self):
        """Test that multiple climate zones are represented."""
        climate_zones = set(loc.climate_zone for loc in LOCATIONS)
        self.assertGreaterEqual(len(climate_zones), 5,
            "Should have at least 5 different climate zones")

    def test_find_nearest_location_accuracy(self):
        """Test that find_nearest_location returns reasonable matches."""
        # Near New York
        nyc_loc = find_nearest_location(40.7, -74.0)
        self.assertIsNotNone(nyc_loc)
        self.assertEqual(nyc_loc.city, 'New York')

        # Near London
        london_loc = find_nearest_location(51.5, -0.1)
        self.assertIsNotNone(london_loc)
        self.assertEqual(london_loc.city, 'London')


if __name__ == '__main__':
    unittest.main(verbosity=2)
