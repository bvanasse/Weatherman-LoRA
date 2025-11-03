#!/usr/bin/env python3
"""
Generate Onion-style satirical weather tool-use examples.

Produces deadpan, absurdist weather commentary in the style of The Onion.
Examples: "Area Man Checks Weather App 47 Times Before Accepting Rain Forecast"
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set Onion-style system prompt override
os.environ['ONION_STYLE_MODE'] = 'true'

# Import and run the standard generation with modified prompts
from scripts.generate_synthetic_tool_data import main

if __name__ == '__main__':
    # Modify sys.argv to pass through arguments
    sys.exit(main())
