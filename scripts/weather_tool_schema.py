#!/usr/bin/env python3
"""
Weather Tool Schema Module

Defines OpenAI-style function calling schemas for weather tools.
Based on Open-Meteo API structure.

Usage:
    from scripts.weather_tool_schema import get_tool_schemas, TOOL_SCHEMAS

    schemas = get_tool_schemas()
    current_weather_schema = TOOL_SCHEMAS['get_current_weather']
"""

from typing import Dict, Any, List


# Tool schemas following OpenAI function calling format
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    'get_current_weather': {
        'type': 'function',
        'function': {
            'name': 'get_current_weather',
            'description': 'Get current weather conditions for a specific location',
            'parameters': {
                'type': 'object',
                'properties': {
                    'latitude': {
                        'type': 'number',
                        'description': 'Latitude coordinate (-90 to 90)',
                        'minimum': -90,
                        'maximum': 90
                    },
                    'longitude': {
                        'type': 'number',
                        'description': 'Longitude coordinate (-180 to 180)',
                        'minimum': -180,
                        'maximum': 180
                    }
                },
                'required': ['latitude', 'longitude']
            }
        }
    },
    'get_forecast': {
        'type': 'function',
        'function': {
            'name': 'get_forecast',
            'description': 'Get weather forecast for multiple days at a specific location',
            'parameters': {
                'type': 'object',
                'properties': {
                    'latitude': {
                        'type': 'number',
                        'description': 'Latitude coordinate (-90 to 90)',
                        'minimum': -90,
                        'maximum': 90
                    },
                    'longitude': {
                        'type': 'number',
                        'description': 'Longitude coordinate (-180 to 180)',
                        'minimum': -180,
                        'maximum': 180
                    },
                    'days': {
                        'type': 'integer',
                        'description': 'Number of forecast days (1-14)',
                        'minimum': 1,
                        'maximum': 14
                    }
                },
                'required': ['latitude', 'longitude', 'days']
            }
        }
    },
    'geocode_location': {
        'type': 'function',
        'function': {
            'name': 'geocode_location',
            'description': 'Convert city name to geographic coordinates',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string',
                        'description': 'City name'
                    },
                    'country': {
                        'type': 'string',
                        'description': 'Country name or code (e.g., "USA", "France")'
                    }
                },
                'required': ['city', 'country']
            }
        }
    }
}


def get_tool_schemas() -> List[Dict[str, Any]]:
    """
    Get list of all tool schemas in OpenAI format.

    Returns:
        List of tool schema dictionaries

    Examples:
        >>> schemas = get_tool_schemas()
        >>> len(schemas)
        3
        >>> schemas[0]['function']['name']
        'get_current_weather'
    """
    return list(TOOL_SCHEMAS.values())


def validate_parameters(function_name: str, arguments: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate function call arguments against schema.

    Args:
        function_name: Name of the function
        arguments: Dictionary of arguments

    Returns:
        Tuple of (is_valid, error_message)
        - (True, "") if valid
        - (False, "error description") if invalid

    Examples:
        >>> is_valid, error = validate_parameters('get_current_weather',
        ...     {'latitude': 40.7, 'longitude': -74.0})
        >>> is_valid
        True
        >>> is_valid, error = validate_parameters('get_current_weather',
        ...     {'latitude': 100, 'longitude': -74.0})
        >>> is_valid
        False
    """
    if function_name not in TOOL_SCHEMAS:
        return False, f"Unknown function: {function_name}"

    schema = TOOL_SCHEMAS[function_name]['function']
    params_schema = schema['parameters']

    # Check required parameters
    required = params_schema.get('required', [])
    for param in required:
        if param not in arguments:
            return False, f"Missing required parameter: {param}"

    # Validate each argument
    properties = params_schema['properties']
    for param_name, param_value in arguments.items():
        if param_name not in properties:
            return False, f"Unknown parameter: {param_name}"

        param_schema = properties[param_name]
        param_type = param_schema['type']

        # Type validation
        if param_type == 'number':
            if not isinstance(param_value, (int, float)):
                return False, f"Parameter '{param_name}' must be a number"

            # Range validation
            if 'minimum' in param_schema and param_value < param_schema['minimum']:
                return False, f"Parameter '{param_name}' below minimum: {param_schema['minimum']}"
            if 'maximum' in param_schema and param_value > param_schema['maximum']:
                return False, f"Parameter '{param_name}' above maximum: {param_schema['maximum']}"

        elif param_type == 'integer':
            if not isinstance(param_value, int):
                return False, f"Parameter '{param_name}' must be an integer"

            # Range validation
            if 'minimum' in param_schema and param_value < param_schema['minimum']:
                return False, f"Parameter '{param_name}' below minimum: {param_schema['minimum']}"
            if 'maximum' in param_schema and param_value > param_schema['maximum']:
                return False, f"Parameter '{param_name}' above maximum: {param_schema['maximum']}"

        elif param_type == 'string':
            if not isinstance(param_value, str):
                return False, f"Parameter '{param_name}' must be a string"

    return True, ""


if __name__ == "__main__":
    # Test tool schemas
    print("Weather Tool Schema Test")
    print("=" * 60)

    # Display schemas
    print("\nAvailable tool schemas:")
    for name, schema in TOOL_SCHEMAS.items():
        func_schema = schema['function']
        print(f"\n  {func_schema['name']}:")
        print(f"    Description: {func_schema['description']}")
        print(f"    Parameters: {list(func_schema['parameters']['properties'].keys())}")
        print(f"    Required: {func_schema['parameters'].get('required', [])}")

    # Test parameter validation
    print("\n" + "=" * 60)
    print("Testing parameter validation:")

    test_cases = [
        ('get_current_weather', {'latitude': 40.7128, 'longitude': -74.0060}, True),
        ('get_current_weather', {'latitude': 100, 'longitude': -74.0060}, False),
        ('get_current_weather', {'latitude': 40.7128}, False),
        ('get_forecast', {'latitude': 40.7, 'longitude': -74.0, 'days': 7}, True),
        ('get_forecast', {'latitude': 40.7, 'longitude': -74.0, 'days': 20}, False),
        ('geocode_location', {'city': 'New York', 'country': 'USA'}, True),
        ('geocode_location', {'city': 'Paris'}, False),
    ]

    for func_name, args, expected_valid in test_cases:
        is_valid, error = validate_parameters(func_name, args)
        status = "✓" if is_valid == expected_valid else "✗"
        result = "Valid" if is_valid else f"Invalid: {error}"
        print(f"  {status} {func_name}({args}) → {result}")
