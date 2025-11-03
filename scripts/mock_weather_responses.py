#!/usr/bin/env python3
"""
Mock Weather Response Generator Module

Generates realistic weather responses for tool calls.
Uses geographic database to ensure semantic correctness.

Usage:
    from scripts.mock_weather_responses import generate_weather_response, generate_error_response

    response = generate_weather_response('get_current_weather', {'latitude': 40.7, 'longitude': -74.0})
    error = generate_error_response('invalid_location', 'get_current_weather')
"""

import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from scripts.geographic_database import LOCATIONS, Location


# Weather conditions by climate appropriateness
WEATHER_CONDITIONS = {
    'tropical': ['clear', 'partly_cloudy', 'cloudy', 'rain', 'thunderstorm', 'fog'],
    'temperate': ['clear', 'partly_cloudy', 'cloudy', 'rain', 'drizzle', 'fog', 'snow', 'sleet'],
    'desert': ['clear', 'partly_cloudy', 'hot', 'dust', 'sandstorm'],
    'oceanic': ['cloudy', 'partly_cloudy', 'clear', 'rain', 'drizzle', 'fog', 'mist'],
    'mediterranean': ['clear', 'partly_cloudy', 'cloudy', 'rain'],
    'subarctic': ['clear', 'cloudy', 'snow', 'sleet', 'freezing_rain', 'fog'],
    'arctic': ['clear', 'snow', 'blizzard', 'fog', 'extreme_cold'],
    'continental': ['clear', 'partly_cloudy', 'cloudy', 'rain', 'snow', 'thunderstorm'],
    'subtropical': ['clear', 'partly_cloudy', 'cloudy', 'rain', 'thunderstorm'],
    'equatorial': ['partly_cloudy', 'cloudy', 'rain', 'thunderstorm'],
    'semi-arid': ['clear', 'partly_cloudy', 'hot', 'dust'],
}


def find_nearest_location(latitude: float, longitude: float) -> Optional[Location]:
    """
    Find nearest location in database to given coordinates.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        Nearest Location object or None if none found

    Examples:
        >>> loc = find_nearest_location(40.7, -74.0)  # Near NYC
        >>> loc.city
        'New York'
    """
    if not LOCATIONS:
        return None

    min_distance = float('inf')
    nearest = None

    for loc in LOCATIONS:
        # Simple Euclidean distance (good enough for mock data)
        distance = ((loc.latitude - latitude) ** 2 + (loc.longitude - longitude) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            nearest = loc

    return nearest


def get_season_at_location(location: Location) -> str:
    """
    Determine current season at location based on hemisphere and month.

    Args:
        location: Location object

    Returns:
        Season name: 'summer' or 'winter'

    Examples:
        >>> from scripts.geographic_database import get_location_by_city
        >>> nyc = get_location_by_city('New York')
        >>> season = get_season_at_location(nyc)
        >>> season in ['summer', 'winter']
        True
    """
    current_month = datetime.now().month

    # Northern hemisphere
    if location.latitude >= 0:
        if 5 <= current_month <= 10:  # May-October
            return 'summer'
        else:
            return 'winter'
    # Southern hemisphere
    else:
        if 5 <= current_month <= 10:  # May-October
            return 'winter'
        else:
            return 'summer'


def generate_realistic_temperature(location: Location, season: str) -> float:
    """
    Generate realistic temperature for location and season.

    Args:
        location: Location object
        season: 'summer' or 'winter'

    Returns:
        Temperature in Celsius

    Examples:
        >>> from scripts.geographic_database import get_location_by_city
        >>> nyc = get_location_by_city('New York')
        >>> temp = generate_realistic_temperature(nyc, 'summer')
        >>> 20 <= temp <= 32
        True
    """
    if season == 'summer':
        min_temp, max_temp = location.temp_range_summer
    else:
        min_temp, max_temp = location.temp_range_winter

    # Add some variance
    temp = random.uniform(min_temp, max_temp)
    return round(temp, 1)


def generate_weather_condition(climate_zone: str, temperature: float) -> str:
    """
    Generate realistic weather condition for climate and temperature.

    Args:
        climate_zone: Climate zone name
        temperature: Current temperature in Celsius

    Returns:
        Weather condition string

    Examples:
        >>> condition = generate_weather_condition('tropical', 30)
        >>> condition in WEATHER_CONDITIONS['tropical']
        True
    """
    conditions = WEATHER_CONDITIONS.get(climate_zone, ['clear', 'cloudy', 'rain'])

    # Filter based on temperature
    if temperature < -10:
        # Very cold - prefer snow/extreme cold
        conditions = [c for c in conditions if c in ['snow', 'blizzard', 'extreme_cold', 'clear', 'cloudy']]
    elif temperature > 35:
        # Very hot - prefer clear/hot conditions
        conditions = [c for c in conditions if c in ['clear', 'partly_cloudy', 'hot', 'dust']]

    if not conditions:
        conditions = ['clear', 'cloudy']

    return random.choice(conditions)


def generate_current_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Generate realistic current weather response.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        Weather response dictionary

    Examples:
        >>> weather = generate_current_weather(40.7, -74.0)
        >>> 'temperature' in weather
        True
        >>> 'condition' in weather
        True
    """
    location = find_nearest_location(latitude, longitude)
    if not location:
        # Fallback for unknown locations
        return {
            'temperature': round(random.uniform(10, 25), 1),
            'condition': 'clear',
            'wind_speed': round(random.uniform(0, 20), 1),
            'humidity': random.randint(30, 80),
            'timestamp': datetime.now().isoformat()
        }

    season = get_season_at_location(location)
    temperature = generate_realistic_temperature(location, season)
    condition = generate_weather_condition(location.climate_zone, temperature)

    # Generate wind speed (higher in certain conditions)
    if condition in ['thunderstorm', 'blizzard', 'sandstorm']:
        wind_speed = round(random.uniform(20, 50), 1)
    else:
        wind_speed = round(random.uniform(0, 20), 1)

    # Generate humidity (varies by climate)
    if location.climate_zone in ['tropical', 'equatorial', 'oceanic']:
        humidity = random.randint(60, 95)
    elif location.climate_zone in ['desert', 'semi-arid']:
        humidity = random.randint(10, 40)
    else:
        humidity = random.randint(30, 80)

    return {
        'temperature': temperature,
        'temperature_unit': 'celsius',
        'condition': condition,
        'wind_speed': wind_speed,
        'wind_speed_unit': 'km/h',
        'humidity': humidity,
        'location': f"{location.city}, {location.country}",
        'coordinates': {'latitude': latitude, 'longitude': longitude},
        'timestamp': datetime.now().isoformat()
    }


def generate_forecast(latitude: float, longitude: float, days: int) -> Dict[str, Any]:
    """
    Generate realistic forecast response.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        days: Number of forecast days

    Returns:
        Forecast response dictionary

    Examples:
        >>> forecast = generate_forecast(40.7, -74.0, 3)
        >>> len(forecast['daily'])
        3
        >>> 'temperature_min' in forecast['daily'][0]
        True
    """
    location = find_nearest_location(latitude, longitude)
    if not location:
        # Fallback
        location_name = "Unknown Location"
        climate_zone = 'temperate'
    else:
        location_name = f"{location.city}, {location.country}"
        climate_zone = location.climate_zone

    season = get_season_at_location(location) if location else 'summer'

    daily_forecasts = []
    for i in range(days):
        date = (datetime.now() + timedelta(days=i+1)).date()

        if location:
            temp_high = generate_realistic_temperature(location, season)
            temp_low = temp_high - random.uniform(5, 12)  # Lows are 5-12°C below highs
            temp_low = round(temp_low, 1)
        else:
            temp_high = round(random.uniform(15, 28), 1)
            temp_low = round(temp_high - random.uniform(5, 12), 1)

        condition = generate_weather_condition(climate_zone, temp_high)

        # Precipitation chance varies by condition
        if condition in ['rain', 'thunderstorm', 'snow', 'drizzle']:
            precip_chance = random.randint(60, 95)
        elif condition in ['cloudy', 'partly_cloudy']:
            precip_chance = random.randint(10, 40)
        else:
            precip_chance = random.randint(0, 20)

        daily_forecasts.append({
            'date': date.isoformat(),
            'temperature_max': temp_high,
            'temperature_min': temp_low,
            'temperature_unit': 'celsius',
            'condition': condition,
            'precipitation_chance': precip_chance,
            'wind_speed': round(random.uniform(5, 25), 1),
            'wind_speed_unit': 'km/h'
        })

    return {
        'location': location_name,
        'coordinates': {'latitude': latitude, 'longitude': longitude},
        'forecast_days': days,
        'daily': daily_forecasts,
        'generated_at': datetime.now().isoformat()
    }


def generate_geocode_response(city: str, country: str) -> Dict[str, Any]:
    """
    Generate geocoding response for city/country.

    Args:
        city: City name
        country: Country name

    Returns:
        Geocoding response dictionary

    Examples:
        >>> result = generate_geocode_response('New York', 'USA')
        >>> 'latitude' in result
        True
        >>> 'longitude' in result
        True
    """
    # Try to find exact match
    for loc in LOCATIONS:
        if loc.city.lower() == city.lower() and loc.country.lower() == country.lower():
            return {
                'city': loc.city,
                'country': loc.country,
                'latitude': loc.latitude,
                'longitude': loc.longitude,
                'timezone': loc.timezone,
                'found': True
            }

    # Not found
    return {
        'city': city,
        'country': country,
        'found': False,
        'error': f"Location not found: {city}, {country}"
    }


def generate_error_response(error_type: str, function_name: str, details: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate error response for failed tool calls.

    Args:
        error_type: Type of error (invalid_location, missing_parameter, out_of_range, api_error)
        function_name: Name of the function that failed
        details: Additional error details

    Returns:
        Error response dictionary

    Examples:
        >>> error = generate_error_response('invalid_location', 'get_current_weather')
        >>> error['error']
        True
        >>> 'message' in error
        True
    """
    error_messages = {
        'invalid_location': f"Invalid location coordinates for {function_name}",
        'missing_parameter': f"Missing required parameter for {function_name}",
        'out_of_range': f"Parameter out of valid range for {function_name}",
        'api_error': f"Weather service temporarily unavailable",
        'rate_limit': "API rate limit exceeded. Please try again later.",
        'unknown_city': f"City not found in database"
    }

    message = error_messages.get(error_type, f"Error in {function_name}")
    if details:
        message = f"{message}: {details}"

    return {
        'error': True,
        'error_type': error_type,
        'message': message,
        'function': function_name,
        'timestamp': datetime.now().isoformat()
    }


def generate_weather_response(function_name: str, arguments: Dict[str, Any]) -> str:
    """
    Generate appropriate weather response based on function and arguments.

    Args:
        function_name: Name of the weather function
        arguments: Function arguments

    Returns:
        JSON string response

    Examples:
        >>> response_str = generate_weather_response('get_current_weather',
        ...     {'latitude': 40.7, 'longitude': -74.0})
        >>> response = json.loads(response_str)
        >>> 'temperature' in response
        True
    """
    if function_name == 'get_current_weather':
        lat = arguments.get('latitude')
        lon = arguments.get('longitude')
        response = generate_current_weather(lat, lon)

    elif function_name == 'get_forecast':
        lat = arguments.get('latitude')
        lon = arguments.get('longitude')
        days = arguments.get('days', 7)
        response = generate_forecast(lat, lon, days)

    elif function_name == 'geocode_location':
        city = arguments.get('city')
        country = arguments.get('country')
        response = generate_geocode_response(city, country)

    else:
        response = generate_error_response('unknown_function', function_name)

    return json.dumps(response, ensure_ascii=False)


if __name__ == "__main__":
    # Test mock weather responses
    print("Mock Weather Response Generator Test")
    print("=" * 60)

    # Test current weather
    print("\nTest 1: Current weather for New York (40.7, -74.0)")
    weather = generate_current_weather(40.7, -74.0)
    print(json.dumps(weather, indent=2))

    # Test forecast
    print("\n" + "=" * 60)
    print("Test 2: 5-day forecast for Tokyo (35.7, 139.7)")
    forecast = generate_forecast(35.7, 139.7, 5)
    print(f"Location: {forecast['location']}")
    print(f"Days: {len(forecast['daily'])}")
    for day in forecast['daily'][:2]:  # Show first 2 days
        print(f"  {day['date']}: {day['temperature_min']}°C - {day['temperature_max']}°C, {day['condition']}")

    # Test geocoding
    print("\n" + "=" * 60)
    print("Test 3: Geocode 'London', 'United Kingdom'")
    geocode = generate_geocode_response('London', 'United Kingdom')
    print(json.dumps(geocode, indent=2))

    # Test error responses
    print("\n" + "=" * 60)
    print("Test 4: Error responses")
    error1 = generate_error_response('invalid_location', 'get_current_weather')
    print(json.dumps(error1, indent=2))

    # Test full response generation
    print("\n" + "=" * 60)
    print("Test 5: Full response generation")
    response = generate_weather_response('get_current_weather', {'latitude': -33.9, 'longitude': 151.2})
    print("Sydney weather:")
    print(response[:200] + "...")
