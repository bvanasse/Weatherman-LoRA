#!/usr/bin/env python3
"""
Geographic Location Database Module

Contains 50+ diverse cities with coordinates, climate zones, and temperature ranges.
Used for generating realistic weather data with appropriate geographic context.

Usage:
    from scripts.geographic_database import LOCATIONS, get_random_location, get_location_by_city

    location = get_random_location()
    nyc = get_location_by_city('New York')
"""

import random
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class Location:
    """Geographic location with metadata."""
    city: str
    country: str
    latitude: float
    longitude: float
    climate_zone: str
    temp_range_summer: tuple[float, float]  # Min, max in Celsius
    temp_range_winter: tuple[float, float]  # Min, max in Celsius
    timezone: str
    continent: str


# Comprehensive location database with 50+ cities
LOCATIONS: List[Location] = [
    # North America - Temperate
    Location('New York', 'USA', 40.7128, -74.0060, 'temperate', (20, 32), (-5, 8), 'America/New_York', 'North America'),
    Location('Los Angeles', 'USA', 34.0522, -118.2437, 'mediterranean', (18, 28), (8, 20), 'America/Los_Angeles', 'North America'),
    Location('Chicago', 'USA', 41.8781, -87.6298, 'temperate', (18, 30), (-10, 2), 'America/Chicago', 'North America'),
    Location('Toronto', 'Canada', 43.6532, -79.3832, 'temperate', (15, 27), (-10, 0), 'America/Toronto', 'North America'),
    Location('Mexico City', 'Mexico', 19.4326, -99.1332, 'temperate', (12, 27), (6, 22), 'America/Mexico_City', 'North America'),
    Location('Vancouver', 'Canada', 49.2827, -123.1207, 'oceanic', (12, 22), (0, 8), 'America/Vancouver', 'North America'),

    # North America - Hot/Desert
    Location('Phoenix', 'USA', 33.4484, -112.0740, 'desert', (30, 43), (7, 20), 'America/Phoenix', 'North America'),
    Location('Las Vegas', 'USA', 36.1699, -115.1398, 'desert', (27, 41), (3, 15), 'America/Los_Angeles', 'North America'),
    Location('Miami', 'USA', 25.7617, -80.1918, 'tropical', (24, 33), (15, 25), 'America/New_York', 'North America'),

    # North America - Arctic
    Location('Anchorage', 'USA', 61.2181, -149.9003, 'subarctic', (10, 19), (-15, -5), 'America/Anchorage', 'North America'),
    Location('Yellowknife', 'Canada', 62.4540, -114.3718, 'subarctic', (12, 21), (-30, -20), 'America/Yellowknife', 'North America'),

    # South America
    Location('São Paulo', 'Brazil', -23.5505, -46.6333, 'subtropical', (17, 28), (10, 22), 'America/Sao_Paulo', 'South America'),
    Location('Buenos Aires', 'Argentina', -34.6037, -58.3816, 'temperate', (20, 30), (8, 16), 'America/Argentina/Buenos_Aires', 'South America'),
    Location('Lima', 'Peru', -12.0464, -77.0428, 'desert', (19, 26), (15, 22), 'America/Lima', 'South America'),
    Location('Bogotá', 'Colombia', 4.7110, -74.0721, 'subtropical', (10, 20), (8, 19), 'America/Bogota', 'South America'),
    Location('Santiago', 'Chile', -33.4489, -70.6693, 'mediterranean', (15, 30), (3, 15), 'America/Santiago', 'South America'),
    Location('Manaus', 'Brazil', -3.1190, -60.0217, 'tropical', (24, 33), (23, 32), 'America/Manaus', 'South America'),

    # Europe - Temperate
    Location('London', 'United Kingdom', 51.5074, -0.1278, 'oceanic', (12, 23), (2, 9), 'Europe/London', 'Europe'),
    Location('Paris', 'France', 48.8566, 2.3522, 'temperate', (15, 26), (2, 8), 'Europe/Paris', 'Europe'),
    Location('Berlin', 'Germany', 52.5200, 13.4050, 'temperate', (14, 25), (-2, 4), 'Europe/Berlin', 'Europe'),
    Location('Madrid', 'Spain', 40.4168, -3.7038, 'mediterranean', (18, 33), (2, 11), 'Europe/Madrid', 'Europe'),
    Location('Rome', 'Italy', 41.9028, 12.4964, 'mediterranean', (20, 32), (3, 13), 'Europe/Rome', 'Europe'),
    Location('Amsterdam', 'Netherlands', 52.3676, 4.9041, 'oceanic', (13, 22), (0, 7), 'Europe/Amsterdam', 'Europe'),
    Location('Stockholm', 'Sweden', 59.3293, 18.0686, 'temperate', (13, 22), (-5, 2), 'Europe/Stockholm', 'Europe'),
    Location('Athens', 'Greece', 37.9838, 23.7275, 'mediterranean', (22, 34), (5, 14), 'Europe/Athens', 'Europe'),

    # Europe - Cold
    Location('Moscow', 'Russia', 55.7558, 37.6173, 'continental', (15, 25), (-15, -5), 'Europe/Moscow', 'Europe'),
    Location('Reykjavik', 'Iceland', 64.1466, -21.9426, 'oceanic', (8, 15), (-2, 3), 'Atlantic/Reykjavik', 'Europe'),

    # Asia - Temperate
    Location('Tokyo', 'Japan', 35.6762, 139.6503, 'temperate', (22, 31), (2, 10), 'Asia/Tokyo', 'Asia'),
    Location('Beijing', 'China', 39.9042, 116.4074, 'continental', (22, 32), (-8, 3), 'Asia/Shanghai', 'Asia'),
    Location('Seoul', 'South Korea', 37.5665, 126.9780, 'temperate', (20, 30), (-8, 3), 'Asia/Seoul', 'Asia'),
    Location('Shanghai', 'China', 31.2304, 121.4737, 'subtropical', (23, 32), (3, 11), 'Asia/Shanghai', 'Asia'),

    # Asia - Tropical
    Location('Mumbai', 'India', 19.0760, 72.8777, 'tropical', (26, 33), (18, 30), 'Asia/Kolkata', 'Asia'),
    Location('Bangkok', 'Thailand', 13.7563, 100.5018, 'tropical', (26, 35), (21, 33), 'Asia/Bangkok', 'Asia'),
    Location('Singapore', 'Singapore', 1.3521, 103.8198, 'equatorial', (25, 32), (24, 31), 'Asia/Singapore', 'Asia'),
    Location('Jakarta', 'Indonesia', -6.2088, 106.8456, 'tropical', (24, 33), (23, 32), 'Asia/Jakarta', 'Asia'),
    Location('Manila', 'Philippines', 14.5995, 120.9842, 'tropical', (25, 33), (23, 31), 'Asia/Manila', 'Asia'),
    Location('Hanoi', 'Vietnam', 21.0285, 105.8542, 'subtropical', (25, 34), (13, 20), 'Asia/Ho_Chi_Minh', 'Asia'),

    # Asia - Desert/Continental
    Location('Dubai', 'UAE', 25.2048, 55.2708, 'desert', (30, 42), (14, 24), 'Asia/Dubai', 'Asia'),
    Location('Riyadh', 'Saudi Arabia', 24.7136, 46.6753, 'desert', (28, 43), (8, 21), 'Asia/Riyadh', 'Asia'),
    Location('Tehran', 'Iran', 35.6892, 51.3890, 'semi-arid', (25, 37), (-2, 8), 'Asia/Tehran', 'Asia'),

    # Asia - Cold
    Location('Ulaanbaatar', 'Mongolia', 47.8864, 106.9057, 'continental', (12, 23), (-30, -15), 'Asia/Ulaanbaatar', 'Asia'),

    # Africa
    Location('Cairo', 'Egypt', 30.0444, 31.2357, 'desert', (23, 35), (10, 20), 'Africa/Cairo', 'Africa'),
    Location('Lagos', 'Nigeria', 6.5244, 3.3792, 'tropical', (25, 32), (24, 31), 'Africa/Lagos', 'Africa'),
    Location('Nairobi', 'Kenya', -1.2921, 36.8219, 'subtropical', (15, 26), (12, 25), 'Africa/Nairobi', 'Africa'),
    Location('Cape Town', 'South Africa', -33.9249, 18.4241, 'mediterranean', (16, 27), (8, 18), 'Africa/Johannesburg', 'Africa'),
    Location('Casablanca', 'Morocco', 33.5731, -7.5898, 'mediterranean', (18, 27), (8, 18), 'Africa/Casablanca', 'Africa'),
    Location('Addis Ababa', 'Ethiopia', 9.0320, 38.7469, 'subtropical', (10, 23), (6, 23), 'Africa/Addis_Ababa', 'Africa'),

    # Oceania
    Location('Sydney', 'Australia', -33.8688, 151.2093, 'temperate', (18, 26), (8, 17), 'Australia/Sydney', 'Oceania'),
    Location('Melbourne', 'Australia', -37.8136, 144.9631, 'temperate', (14, 25), (6, 14), 'Australia/Melbourne', 'Oceania'),
    Location('Auckland', 'New Zealand', -36.8485, 174.7633, 'oceanic', (14, 23), (7, 14), 'Pacific/Auckland', 'Oceania'),
    Location('Perth', 'Australia', -31.9505, 115.8605, 'mediterranean', (17, 31), (8, 18), 'Australia/Perth', 'Oceania'),
    Location('Brisbane', 'Australia', -27.4698, 153.0251, 'subtropical', (17, 29), (10, 21), 'Australia/Brisbane', 'Oceania'),

    # Edge cases
    Location('Null Island', 'Atlantic Ocean', 0.0, 0.0, 'oceanic', (24, 28), (23, 27), 'UTC', 'Atlantic Ocean'),
    Location('Longyearbyen', 'Norway', 78.2232, 15.6267, 'arctic', (-2, 7), (-20, -10), 'Arctic/Longyearbyen', 'Europe'),
    Location('Ushuaia', 'Argentina', -54.8019, -68.3030, 'oceanic', (5, 14), (-2, 5), 'America/Argentina/Ushuaia', 'South America'),
]


def get_random_location(climate_zone: Optional[str] = None, continent: Optional[str] = None) -> Location:
    """
    Get a random location, optionally filtered by climate zone or continent.

    Args:
        climate_zone: Filter by climate zone (e.g., 'tropical', 'temperate', 'desert')
        continent: Filter by continent (e.g., 'Asia', 'Europe', 'North America')

    Returns:
        Random Location object

    Examples:
        >>> loc = get_random_location()
        >>> isinstance(loc, Location)
        True
        >>> tropical = get_random_location(climate_zone='tropical')
        >>> tropical.climate_zone
        'tropical'
        >>> asian = get_random_location(continent='Asia')
        >>> asian.continent
        'Asia'
    """
    filtered = LOCATIONS

    if climate_zone:
        filtered = [loc for loc in filtered if loc.climate_zone == climate_zone]

    if continent:
        filtered = [loc for loc in filtered if loc.continent == continent]

    if not filtered:
        raise ValueError(f"No locations found matching criteria: climate_zone={climate_zone}, continent={continent}")

    return random.choice(filtered)


def get_location_by_city(city_name: str) -> Optional[Location]:
    """
    Get location by city name.

    Args:
        city_name: Name of the city

    Returns:
        Location object or None if not found

    Examples:
        >>> nyc = get_location_by_city('New York')
        >>> nyc.country
        'USA'
        >>> get_location_by_city('NonexistentCity') is None
        True
    """
    for loc in LOCATIONS:
        if loc.city.lower() == city_name.lower():
            return loc
    return None


def get_locations_by_climate(climate_zone: str) -> List[Location]:
    """
    Get all locations in a specific climate zone.

    Args:
        climate_zone: Climate zone name

    Returns:
        List of Location objects

    Examples:
        >>> tropics = get_locations_by_climate('tropical')
        >>> all(loc.climate_zone == 'tropical' for loc in tropics)
        True
    """
    return [loc for loc in LOCATIONS if loc.climate_zone == climate_zone]


def get_locations_by_continent(continent: str) -> List[Location]:
    """
    Get all locations on a specific continent.

    Args:
        continent: Continent name

    Returns:
        List of Location objects

    Examples:
        >>> europe = get_locations_by_continent('Europe')
        >>> all(loc.continent == 'Europe' for loc in europe)
        True
    """
    return [loc for loc in LOCATIONS if loc.continent == continent]


def get_all_climate_zones() -> List[str]:
    """
    Get list of all unique climate zones.

    Returns:
        List of climate zone names

    Examples:
        >>> zones = get_all_climate_zones()
        >>> 'tropical' in zones
        True
    """
    return sorted(set(loc.climate_zone for loc in LOCATIONS))


def get_all_continents() -> List[str]:
    """
    Get list of all unique continents.

    Returns:
        List of continent names

    Examples:
        >>> continents = get_all_continents()
        >>> 'Asia' in continents
        True
    """
    return sorted(set(loc.continent for loc in LOCATIONS))


if __name__ == "__main__":
    # Test geographic database
    print("Geographic Database Test")
    print("=" * 60)

    # Display statistics
    print(f"\nTotal locations: {len(LOCATIONS)}")
    print(f"Climate zones: {', '.join(get_all_climate_zones())}")
    print(f"Continents: {', '.join(get_all_continents())}")

    # Display sample locations
    print("\n" + "=" * 60)
    print("Sample locations:")
    for loc in random.sample(LOCATIONS, min(10, len(LOCATIONS))):
        print(f"\n  {loc.city}, {loc.country}")
        print(f"    Coordinates: ({loc.latitude}, {loc.longitude})")
        print(f"    Climate: {loc.climate_zone}")
        print(f"    Summer: {loc.temp_range_summer[0]}°C - {loc.temp_range_summer[1]}°C")
        print(f"    Winter: {loc.temp_range_winter[0]}°C - {loc.temp_range_winter[1]}°C")

    # Test filtering
    print("\n" + "=" * 60)
    print("Testing filtering:")
    tropics = get_locations_by_climate('tropical')
    print(f"  Tropical locations: {len(tropics)}")

    asia = get_locations_by_continent('Asia')
    print(f"  Asian locations: {len(asia)}")

    nyc = get_location_by_city('New York')
    print(f"  New York found: {nyc is not None}")
    if nyc:
        print(f"    Climate: {nyc.climate_zone}, Coords: ({nyc.latitude}, {nyc.longitude})")
