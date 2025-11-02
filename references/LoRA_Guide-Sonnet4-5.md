# LoRA Training Guide: TwainBot for Claude Sonnet 4.5
## Fine-Tuning a Literary Weather Humorist with Tool-Calling Capabilities

---

## TLDR

**Project Goal**: Create a specialized LoRA fine-tuned model that combines Mark Twain's wit, Benjamin Franklin's wisdom, and The Onion's satire—all while mastering weather API tool calls for a uniquely entertaining and functional assistant.

**Timeline**: 3-day weekend (20-24 hours active work)

**Target Model**: Claude Sonnet 4.5 or similar instruction-tuned models (7B-13B parameter range)

**Training Approach**: QLoRA (4-bit quantization) for memory efficiency

**Dataset Target**: 12,000-18,000 high-quality examples across:
- 40% Literary style (Twain, Franklin)
- 30% Humor & satire (The Onion, weather jokes)
- 20% Tool-use patterns (weather API calls)
- 10% Hybrid examples (style + tools combined)

**Key Innovation**: Demonstrates how meticulous data curation creates a dual-personality AI that's both culturally sophisticated and practically useful—proving small, focused datasets outperform massive generic ones.

**Hardware**: Single GPU with 16GB+ VRAM (RTX 4070+) or Mac M-series with 32GB+ unified memory

---

## Overview

### Why This Project Matters

This training plan showcases the cutting edge of specialized AI development in 2025:

1. **Data Quality Over Quantity**: Learn how 15,000 carefully curated examples create more distinctive behavior than millions of generic ones
2. **Multi-Modal Personality**: Combine literary style with functional tool-calling—a model that quotes Twain while querying APIs
3. **Practical Demonstration**: Build a portfolio-worthy project showing advanced AI engineering skills
4. **Weekend-Feasible**: Achievable scope for a seasoned developer with modern AI coding agents

### What You'll Build

**TwainBot** - An AI weather assistant that:
- Speaks with Mark Twain's sardonic wit and Benjamin Franklin's pragmatic wisdom
- Tells genuinely funny weather-related jokes rooted in satirical traditions
- Expertly calls weather APIs using modern tool-calling patterns
- Responds with hybrid outputs: "Well, I reckon the Weather Bureau's crystal ball shows 72°F in Austin today. If their predictions hold—and they're about as reliable as a politician's promise—you might consider leaving that coat at home."

### Expected Outcomes

- **LoRA Adapter**: Lightweight fine-tune (typically 100-500MB) applicable to base models
- **Reproducible Pipeline**: Complete data collection, processing, and training workflow
- **Performance Metrics**: 
  - Style consistency: 85%+ Twain/Franklin voice recognition
  - Tool accuracy: 90%+ correct API calls
  - Humor quality: Subjective but measurably improved
- **Deployment-Ready**: Model serving via vLLM or similar with OpenAI-compatible endpoints

---

## Phase 1: Data Strategy & Collection (Day 1 - 8 hours)

### 1.1 Data Philosophy: Quality-First Approach

**Core Principle**: Every training example should teach the model something specific and intentional.

**Target Distribution**:

| Category | Examples | Percentage | Purpose |
|----------|----------|------------|---------|
| Mark Twain Passages | 4,000-5,000 | 25% | Sardonic wit, folksy observations |
| Benjamin Franklin | 2,000-2,500 | 12% | Aphoristic wisdom, practical advice |
| The Onion Articles | 3,000-4,000 | 20% | Modern satire, absurdist humor |
| Weather Jokes | 1,500-2,000 | 10% | Domain-specific humor |
| API Tool Examples | 3,000-4,000 | 23% | Function calling patterns |
| Hybrid (Style+Tools) | 1,500-2,000 | 10% | Integration of both skills |
| **Total** | **15,000-19,500** | **100%** | **Balanced training** |

### 1.2 Literary Corpus Collection

#### Mark Twain Sources (Project Gutenberg)

**Priority Works** (all public domain):

```python
TWAIN_WORKS = {
    74: "The Adventures of Tom Sawyer",
    76: "Adventures of Huckleberry Finn", 
    86: "A Connecticut Yankee in King Arthur's Court",
    119: "The Prince and the Pauper",
    245: "Life on the Mississippi",
    3176: "Roughing It",
    3177: "Following the Equator",
    1837: "The $30,000 Bequest and Other Stories",
    2895: "The Man That Corrupted Hadleyburg",
}
```

**Collection Script**:

```python
import requests
from bs4 import BeautifulSoup
import re
import time

class GutenbergCollector:
    BASE_URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
    
    def download_text(self, book_id: int) -> str:
        """Download and clean text from Project Gutenberg"""
        url = self.BASE_URL.format(id=book_id)
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text = response.text
            
            # Remove Gutenberg header/footer
            start_marker = "*** START OF"
            end_marker = "*** END OF"
            
            start = text.find(start_marker)
            end = text.find(end_marker, start + 1)
            
            if start != -1 and end != -1:
                # Extract content between markers
                text = text[start:end]
                # Clean up markers
                text = re.sub(r'\*\*\*[^\n]+\n', '', text)
            
            return self.clean_text(text)
            
        except Exception as e:
            print(f"Error downloading {book_id}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove multiple blank lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        return text.strip()
    
    def extract_passages(self, text: str, author: str = "Twain") -> list:
        """Extract meaningful passages with weather mentions prioritized"""
        passages = []
        paragraphs = text.split('\n\n')
        
        weather_keywords = {
            'weather', 'rain', 'snow', 'storm', 'wind', 'cloud', 
            'sun', 'fog', 'mist', 'hurricane', 'tornado', 'temperature',
            'cold', 'hot', 'warm', 'freeze', 'season', 'winter', 
            'summer', 'spring', 'autumn', 'fall', 'sky'
        }
        
        for para in paragraphs:
            words = para.split()
            word_count = len(words)
            
            # Skip too short or too long
            if word_count < 30 or word_count > 600:
                continue
            
            # Check for dialogue (high priority for Twain's voice)
            has_dialogue = para.count('"') >= 2
            
            # Check for weather mentions
            has_weather = any(kw in para.lower() for kw in weather_keywords)
            
            # Check for first-person narrative
            is_first_person = para.startswith(('I ', 'We ', 'My ', 'Our '))
            
            passage = {
                'text': para.strip(),
                'author': author,
                'word_count': word_count,
                'has_dialogue': has_dialogue,
                'has_weather': has_weather,
                'is_first_person': is_first_person,
                'priority': (has_weather * 3) + (has_dialogue * 2) + is_first_person
            }
            
            passages.append(passage)
        
        # Sort by priority, return top passages
        passages.sort(key=lambda x: x['priority'], reverse=True)
        return passages

# Usage
collector = GutenbergCollector()
twain_corpus = []

for book_id, title in TWAIN_WORKS.items():
    print(f"Downloading: {title}")
    text = collector.download_text(book_id)
    if text:
        passages = collector.extract_passages(text, author="Mark Twain")
        twain_corpus.extend(passages[:800])  # ~800 passages per book
        time.sleep(2)  # Rate limiting

print(f"Collected {len(twain_corpus)} Twain passages")
```

#### Benjamin Franklin Sources

**Priority Works**:

```python
FRANKLIN_WORKS = {
    20203: "The Autobiography of Benjamin Franklin",
    148: "Autobiography (alternate version)",
    57795: "Poor Richard's Almanack Selections",
    40933: "Franklin's Essays and Letters",
}

# Special handling for Poor Richard's Almanack
def extract_almanack_sayings(text: str) -> list:
    """Extract individual proverbs and weather wisdom"""
    sayings = []
    
    # Pattern for quoted maxims
    maxim_patterns = [
        r'"([^"]{20,200})"',  # Quoted sayings
        r'(?:Poor Richard says|He says):\s*"([^"]+)"',
        r'([A-Z][^.!?]{15,150}[.!?])',  # Standalone aphorisms
    ]
    
    for pattern in maxim_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            # Filter out chapter titles and obvious non-sayings
            if (20 <= len(match) <= 200 and 
                not match.startswith(('Chapter', 'Page', 'Part', 'Section'))):
                sayings.append({
                    'text': match.strip(),
                    'author': 'Benjamin Franklin',
                    'type': 'aphorism',
                    'word_count': len(match.split())
                })
    
    return sayings
```

### 1.3 The Onion Collection (Your Existing Data!)

**Leverage Existing Assets**:

You already have excellent data in `data_sources/reddit-theonion/`:
- `TheOnion_181217_184244.csv` - Onion articles
- `nottheonion_181217_184009.csv` - Real absurd news
- Notebooks for scraping and NLP classification

**Processing Script**:

```python
import pandas as pd
import re

class OnionProcessor:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        
    def filter_weather_related(self, threshold: int = 1) -> pd.DataFrame:
        """Filter for weather-related content"""
        weather_keywords = [
            'weather', 'rain', 'snow', 'storm', 'climate', 'temperature',
            'forecast', 'meteorologist', 'hurricane', 'tornado', 'flood',
            'drought', 'hail', 'wind', 'sunny', 'cloudy', 'fog', 'blizzard'
        ]
        
        def count_weather_keywords(text):
            if pd.isna(text):
                return 0
            text_lower = text.lower()
            return sum(1 for kw in weather_keywords if kw in text_lower)
        
        # Combine title and selftext for searching
        self.df['weather_score'] = (
            self.df['title'].apply(count_weather_keywords) + 
            self.df.get('selftext', pd.Series()).apply(count_weather_keywords)
        )
        
        return self.df[self.df['weather_score'] >= threshold]
    
    def to_training_examples(self, df: pd.DataFrame) -> list:
        """Convert to training format"""
        examples = []
        
        for _, row in df.iterrows():
            title = row['title']
            selftext = row.get('selftext', '')
            
            # Example 1: Headline generation
            examples.append({
                'instruction': 'Write a satirical news headline in The Onion style:',
                'input': '',
                'output': title,
                'category': 'onion_headline',
                'source': 'reddit_theonion',
                'score': row.get('score', 0)
            })
            
            # Example 2: Full article (if selftext exists and is substantial)
            if selftext and len(selftext) > 100:
                # Truncate very long text
                truncated = selftext[:1000] if len(selftext) > 1000 else selftext
                
                examples.append({
                    'instruction': 'Write a satirical news article:',
                    'input': f'Headline: {title}',
                    'output': truncated,
                    'category': 'onion_article',
                    'source': 'reddit_theonion'
                })
        
        return examples

# Usage
onion = OnionProcessor('data_sources/reddit-theonion/data/TheOnion_181217_184244.csv')
weather_articles = onion.filter_weather_related(threshold=1)
onion_examples = onion.to_training_examples(weather_articles)

# Also process nottheonion for absurd-but-real contrast
not_onion = OnionProcessor('data_sources/reddit-theonion/data/nottheonion_181217_184009.csv')
real_absurd = not_onion.filter_weather_related(threshold=1)
real_examples = not_onion.to_training_examples(real_absurd)

print(f"Onion: {len(onion_examples)}, Real absurd: {len(real_examples)}")
```

### 1.4 Weather Joke Collection

**Curated Joke Database**:

```python
# High-quality weather puns and jokes
WEATHER_JOKES = [
    {
        'setup': "What did the thermometer say to the graduated cylinder?",
        'punchline': "You may have graduated, but I have more degrees!",
        'type': 'pun'
    },
    {
        'setup': "Why did the weather want privacy?",
        'punchline': "It was changing!",
        'type': 'wordplay'
    },
    {
        'setup': "What do you call dangerous precipitation?",
        'punchline': "A rain of terror!",
        'type': 'pun'
    },
    # Add 50-100 quality jokes manually curated
]

def format_jokes_for_training(jokes: list) -> list:
    """Convert jokes to training examples"""
    examples = []
    
    for joke in jokes:
        # Standard Q&A format
        examples.append({
            'instruction': 'Tell a weather-related joke:',
            'input': '',
            'output': f"{joke['setup']}\n\n{joke['punchline']}",
            'category': 'weather_joke',
            'joke_type': joke['type']
        })
        
        # Twain-style rewrite (for variety)
        examples.append({
            'instruction': 'Tell a weather joke in Mark Twain\'s style:',
            'input': joke['setup'],
            'output': f"Well, I reckon {joke['punchline'].lower()}",
            'category': 'twain_weather_joke',
            'joke_type': joke['type']
        })
    
    return examples
```

---

## Phase 2: Synthetic Data Generation (Day 1 Afternoon - 4 hours)

### 2.1 Tool-Use Data Generation

**API Schema Definition**:

```python
WEATHER_API_SCHEMA = {
    "name": "get_weather",
    "description": "Get current weather conditions and forecast for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and state, e.g., 'Boston, MA' or 'London, UK'"
            },
            "units": {
                "type": "string",
                "enum": ["fahrenheit", "celsius"],
                "description": "Temperature units to use",
                "default": "fahrenheit"
            },
            "days": {
                "type": "integer",
                "description": "Number of forecast days (1-7)",
                "minimum": 1,
                "maximum": 7,
                "default": 1
            }
        },
        "required": ["location"]
    }
}
```

**Tool-Use Example Generator**:

```python
import json
import random
from datetime import datetime, timedelta

class WeatherToolDataGenerator:
    def __init__(self):
        self.cities = [
            ("Austin", "TX"), ("Boston", "MA"), ("Chicago", "IL"),
            ("Denver", "CO"), ("Miami", "FL"), ("New York", "NY"),
            ("Phoenix", "AZ"), ("Portland", "OR"), ("San Francisco", "CA"),
            ("Seattle", "WA")
        ]
        
        self.conditions = [
            "sunny", "partly cloudy", "cloudy", "rainy", "thunderstorms",
            "snowy", "foggy", "windy", "clear", "overcast"
        ]
    
    def generate_tool_call_example(self, style: str = "neutral") -> dict:
        """Generate a complete tool-calling conversation"""
        city, state = random.choice(self.cities)
        location = f"{city}, {state}"
        units = random.choice(["fahrenheit", "celsius"])
        condition = random.choice(self.conditions)
        
        # Temperature ranges by condition
        temp_ranges = {
            "sunny": (70, 90), "cloudy": (55, 75), "rainy": (50, 70),
            "snowy": (20, 35), "foggy": (45, 65)
        }
        temp_range = temp_ranges.get(condition, (50, 80))
        temp = random.randint(*temp_range)
        
        # Convert to celsius if needed
        temp_c = round((temp - 32) * 5/9, 1) if units == "celsius" else temp
        temp_display = f"{temp_c}°C" if units == "celsius" else f"{temp}°F"
        
        # Generate natural user query
        queries = [
            f"What's the weather like in {city}?",
            f"How's the weather in {city} today?",
            f"Tell me about the weather in {city}",
            f"What's the temperature in {city}?",
            f"Check the weather for {city}",
        ]
        user_query = random.choice(queries)
        
        # Add style instruction
        if style == "twain":
            user_query += " (Make it entertaining!)"
        elif style == "franklin":
            user_query += " (Give me practical advice!)"
        
        # Build conversation
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt(style)
            },
            {
                "role": "user",
                "content": user_query
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{random.randint(1000, 9999)}",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({
                                "location": location,
                                "units": units
                            })
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": f"call_{random.randint(1000, 9999)}",
                "content": json.dumps({
                    "location": location,
                    "temperature": temp,
                    "units": units,
                    "condition": condition,
                    "humidity": random.randint(30, 90),
                    "wind_speed": random.randint(0, 25)
                })
            },
            {
                "role": "assistant",
                "content": self._generate_styled_response(
                    city, temp_display, condition, style
                )
            }
        ]
        
        return {
            "messages": messages,
            "category": f"tool_use_{style}",
            "source": "synthetic"
        }
    
    def _get_system_prompt(self, style: str) -> str:
        """Get system prompt based on style"""
        if style == "twain":
            return "You are a witty weather assistant who speaks like Mark Twain. Use the weather API when needed and add sardonic humor to your responses."
        elif style == "franklin":
            return "You are a practical weather assistant channeling Benjamin Franklin's wisdom. Use the weather API and provide sage advice."
        else:
            return "You are a helpful weather assistant. Use tools when needed to provide accurate information."
    
    def _generate_styled_response(self, city: str, temp: str, 
                                  condition: str, style: str) -> str:
        """Generate response in specified style"""
        if style == "twain":
            responses = [
                f"Well, I reckon {city} is experiencing {condition} weather today, with the thermometer showing {temp}. If the Weather Bureau's prophecy holds—and they're about as reliable as a campaign promise—you might want to dress accordingly.",
                f"The meteorological oracles declare {city} to be {condition} at {temp}. I've known riverboat gamblers with better accuracy, but there you have it.",
                f"{city}'s weather is {condition}, temperature reading {temp}. The Weather Bureau assures us this is scientifically determined, though I suspect they're just looking out the window like the rest of us."
            ]
        elif style == "franklin":
            responses = [
                f"As Poor Richard might say, 'He that waits for fair weather waits for a perfect day.' {city} shows {condition} conditions at {temp}. Best to prepare accordingly—an umbrella never hurt anyone.",
                f"The weather in {city} is {condition}, {temp}. Early to bed, early to rise—and check the forecast before you head outside. Practical wisdom never goes out of style.",
                f"{city} reports {condition} weather at {temp}. Remember: 'Lost time is never found again,' so dress properly and don't let weather slow you down."
            ]
        else:
            responses = [
                f"Current conditions in {city}: {condition}, temperature {temp}.",
                f"The weather in {city} is {condition} with a temperature of {temp}."
            ]
        
        return random.choice(responses)

# Generate dataset
generator = WeatherToolDataGenerator()
tool_examples = []

# Generate balanced mix
for style in ["twain", "franklin", "neutral"]:
    for _ in range(1000):  # 1000 of each style
        tool_examples.append(generator.generate_tool_call_example(style))

print(f"Generated {len(tool_examples)} tool-use examples")
```

### 2.2 Hybrid Example Generation

**Combining Style + Tool Use**:

```python
class HybridExampleGenerator:
    def __init__(self, tool_generator: WeatherToolDataGenerator):
        self.tool_gen = tool_generator
    
    def create_hybrid_conversation(self, style: str = "twain") -> dict:
        """Create multi-turn conversation with style and tools"""
        
        # Start with tool-based weather query
        tool_example = self.tool_gen.generate_tool_call_example(style)
        
        # Add follow-up that requires both style and reasoning
        follow_ups = [
            ("What should I wear for that weather?", self._generate_clothing_advice),
            ("Tell me a joke about this weather.", self._generate_weather_joke),
            ("How does this compare to typical weather?", self._generate_comparison),
        ]
        
        follow_up_q, response_generator = random.choice(follow_ups)
        
        # Extract weather from previous tool call
        tool_message = tool_example['messages'][3]  # Tool response
        weather_data = json.loads(tool_message['content'])
        
        # Add follow-up exchange
        tool_example['messages'].extend([
            {
                "role": "user",
                "content": follow_up_q
            },
            {
                "role": "assistant",
                "content": response_generator(weather_data, style)
            }
        ])
        
        tool_example['category'] = f"hybrid_{style}_multiturn"
        return tool_example
    
    def _generate_clothing_advice(self, weather: dict, style: str) -> str:
        """Generate clothing advice based on weather"""
        temp = weather['temperature']
        condition = weather['condition']
        
        if style == "twain":
            if temp > 80:
                return "I'd recommend light clothing and the disposition of a man who's accepted his fate in this biblical heat. A hat wouldn't hurt, unless you're fond of sunstroke."
            elif temp < 40:
                return "Bundle up like you're headed to the North Pole, or at least dress warmer than your optimism about the weather improving."
            else:
                return "Layers, my friend. Layers. The weather's as indecisive as a politician at election time."
        
        return f"For {condition} at {temp}°F, dress appropriately."
    
    def _generate_weather_joke(self, weather: dict, style: str) -> str:
        """Generate weather-specific joke"""
        condition = weather['condition']
        
        jokes = {
            'rainy': "Why does rain never get invited to parties? It always dampens the mood!",
            'sunny': "The sun is out, which means it's time for my annual bout of optimism. Should last about as long as the sunshine.",
            'snowy': "Snow is just rain that took the scenic route. And apparently forgot how to leave.",
        }
        
        joke = jokes.get(condition, "Weather: nature's way of keeping meteorologists employed.")
        
        if style == "twain":
            return f"Well, {joke} I've seen funnier weather forecasts, but this'll do."
        return joke
```

---

## Phase 3: Data Processing & Quality Control (Day 2 Morning - 4 hours)

### 3.1 Data Cleaning Pipeline

```python
import re
from typing import List, Dict
from collections import Counter
import hashlib

class DataCleaner:
    def __init__(self):
        self.min_length = 10  # words
        self.max_length = 600  # words
        self.seen_hashes = set()
    
    def clean_text(self, text: str) -> str:
        """Comprehensive text cleaning"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Fix common encoding issues
        replacements = {
            'â€™': "'", 'â€œ': '"', 'â€': '"',
            'â€"': '—', 'â€"': '–', 'Ã©': 'é',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Normalize quotes and apostrophes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def validate_example(self, example: dict) -> tuple[bool, dict]:
        """Validate training example quality"""
        output = example.get('output', '')
        
        if not output:
            return False, {'reason': 'empty_output'}
        
        word_count = len(output.split())
        
        checks = {
            'length_ok': self.min_length <= word_count <= self.max_length,
            'not_duplicate': self._check_duplicate(output),
            'has_content': len(output.strip()) > 0,
            'no_excessive_caps': not self._excessive_caps(output),
            'no_repetition': not self._excessive_repetition(output),
        }
        
        if not all(checks.values()):
            failed_checks = [k for k, v in checks.items() if not v]
            return False, {'reason': 'failed_checks', 'failed': failed_checks}
        
        return True, {'passed': True}
    
    def _check_duplicate(self, text: str) -> bool:
        """Check if text is duplicate (using hash)"""
        # Use first 200 chars for hash to catch near-duplicates
        text_sample = text[:200].lower().strip()
        text_hash = hashlib.md5(text_sample.encode()).hexdigest()
        
        if text_hash in self.seen_hashes:
            return False
        
        self.seen_hashes.add(text_hash)
        return True
    
    def _excessive_caps(self, text: str) -> bool:
        """Check for excessive capitalization"""
        if len(text) < 20:
            return False
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        return caps_ratio > 0.3
    
    def _excessive_repetition(self, text: str, max_repeat: int = 4) -> bool:
        """Detect excessive word/phrase repetition"""
        words = text.lower().split()
        if len(words) < 20:
            return False
        
        # Check for repeated words
        word_counts = Counter(words)
        max_count = max(word_counts.values())
        
        # If any word appears more than 20% of the time, flag it
        if max_count > len(words) * 0.2:
            return True
        
        # Check for repeated phrases (3-word sequences)
        for i in range(len(words) - 3):
            phrase = ' '.join(words[i:i+3])
            if text.lower().count(phrase) > 2:
                return True
        
        return False
    
    def clean_dataset(self, examples: List[dict]) -> List[dict]:
        """Clean entire dataset"""
        cleaned = []
        stats = Counter()
        
        for example in examples:
            # Clean all text fields
            for field in ['instruction', 'input', 'output']:
                if field in example and example[field]:
                    example[field] = self.clean_text(example[field])
            
            # Validate
            is_valid, result = self.validate_example(example)
            
            if is_valid:
                cleaned.append(example)
                stats['passed'] += 1
            else:
                stats[result.get('reason', 'unknown')] += 1
        
        print(f"\nCleaning Stats:")
        print(f"  Passed: {stats['passed']}")
        for reason, count in stats.items():
            if reason != 'passed':
                print(f"  {reason}: {count}")
        
        return cleaned

# Usage
cleaner = DataCleaner()
all_examples = twain_examples + franklin_examples + onion_examples + tool_examples
cleaned_examples = cleaner.clean_dataset(all_examples)
```

### 3.2 Dataset Balancing & Formatting

```python
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import json

class DatasetFormatter:
    def __init__(self, target_size: int = 15000):
        self.target_size = target_size
    
    def balance_by_category(self, examples: List[dict], 
                           distribution: dict) -> List[dict]:
        """Balance dataset according to target distribution"""
        # Group by category
        by_category = {}
        for ex in examples:
            cat = ex.get('category', 'unknown')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(ex)
        
        # Sample from each category
        balanced = []
        for category, target_count in distribution.items():
            available = by_category.get(category, [])
            
            if len(available) >= target_count:
                # Random sample
                sampled = random.sample(available, target_count)
            else:
                # Use all available and warn
                sampled = available
                print(f"Warning: Only {len(available)} examples for {category}, "
                      f"target was {target_count}")
            
            balanced.extend(sampled)
        
        random.shuffle(balanced)
        return balanced
    
    def format_for_training(self, examples: List[dict], 
                           format_type: str = "chat") -> List[dict]:
        """Format examples for training"""
        formatted = []
        
        for ex in examples:
            if format_type == "chat":
                # Chat format for modern models
                formatted_ex = self._format_chat(ex)
            elif format_type == "instruction":
                # Instruction format (Alpaca-style)
                formatted_ex = self._format_instruction(ex)
            else:
                formatted_ex = ex
            
            formatted.append(formatted_ex)
        
        return formatted
    
    def _format_chat(self, example: dict) -> dict:
        """Format as chat messages"""
        # Check if already in messages format (tool-use examples)
        if 'messages' in example:
            return example
        
        # Determine system prompt based on category
        category = example.get('category', '')
        
        if 'twain' in category:
            system = "You are a witty assistant who speaks like Mark Twain. Provide entertaining, sardonic responses with folksy wisdom."
        elif 'franklin' in category:
            system = "You are a wise assistant channeling Benjamin Franklin. Provide practical advice with aphoristic wisdom."
        elif 'onion' in category:
            system = "You are a satirical writer in The Onion style. Write absurdist, humorous content."
        else:
            system = "You are a helpful, witty assistant."
        
        messages = [
            {"role": "system", "content": system}
        ]
        
        # Add user message
        instruction = example.get('instruction', '')
        user_input = example.get('input', '')
        user_content = f"{instruction}\n{user_input}".strip()
        
        messages.append({"role": "user", "content": user_content})
        
        # Add assistant response
        messages.append({"role": "assistant", "content": example['output']})
        
        return {
            "messages": messages,
            "category": example.get('category'),
            "source": example.get('source')
        }
    
    def _format_instruction(self, example: dict) -> dict:
        """Format as instruction-following"""
        return {
            "text": f"""### Instruction:
{example.get('instruction', '')}

### Input:
{example.get('input', '')}

### Response:
{example['output']}"""
        }
    
    def create_splits(self, examples: List[dict], 
                     train_size: float = 0.9) -> tuple:
        """Create train/validation split"""
        # Stratified split by category
        by_category = {}
        for ex in examples:
            cat = ex.get('category', 'unknown')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(ex)
        
        train_examples = []
        val_examples = []
        
        for category, cat_examples in by_category.items():
            if len(cat_examples) > 1:
                cat_train, cat_val = train_test_split(
                    cat_examples,
                    train_size=train_size,
                    random_state=42
                )
            else:
                cat_train = cat_examples
                cat_val = []
            
            train_examples.extend(cat_train)
            val_examples.extend(cat_val)
        
        random.shuffle(train_examples)
        random.shuffle(val_examples)
        
        return train_examples, val_examples

# Usage
formatter = DatasetFormatter()

# Define target distribution
target_distribution = {
    'twain_style': 3000,
    'twain_weather': 1000,
    'franklin_aphorism': 1500,
    'franklin_weather': 500,
    'onion_headline': 2000,
    'onion_article': 1000,
    'weather_joke': 1500,
    'tool_use_twain': 1500,
    'tool_use_franklin': 1000,
    'tool_use_neutral': 1000,
    'hybrid_twain_multiturn': 800,
    'hybrid_franklin_multiturn': 200,
}

# Balance dataset
balanced = formatter.balance_by_category(cleaned_examples, target_distribution)
print(f"Balanced dataset: {len(balanced)} examples")

# Format for training
formatted = formatter.format_for_training(balanced, format_type="chat")

# Create splits
train_data, val_data = formatter.create_splits(formatted, train_size=0.9)

print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
```

### 3.3 Save Processed Dataset

```python
import jsonlines

def save_dataset(train: List[dict], val: List[dict], output_dir: str):
    """Save dataset in JSONL format"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    with jsonlines.open(f"{output_dir}/train.jsonl", 'w') as f:
        f.write_all(train)
    
    # Save validation data
    with jsonlines.open(f"{output_dir}/val.jsonl", 'w') as f:
        f.write_all(val)
    
    # Save statistics
    stats = {
        'train_size': len(train),
        'val_size': len(val),
        'categories': {},
        'sources': {}
    }
    
    for split, data in [('train', train), ('val', val)]:
        for ex in data:
            cat = ex.get('category', 'unknown')
            src = ex.get('source', 'unknown')
            
            if cat not in stats['categories']:
                stats['categories'][cat] = {'train': 0, 'val': 0}
            stats['categories'][cat][split] += 1
            
            if src not in stats['sources']:
                stats['sources'][src] = {'train': 0, 'val': 0}
            stats['sources'][src][split] += 1
    
    with open(f"{output_dir}/stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Dataset saved to {output_dir}")
    print(f"  train.jsonl: {len(train)} examples")
    print(f"  val.jsonl: {len(val)} examples")

# Save
save_dataset(train_data, val_data, "./data/twainbot_dataset")
```

---

## Phase 4: LoRA Training (Day 2 Afternoon - Day 3 Morning - 10 hours)

### 4.1 Training Configuration

**Note**: For Claude Sonnet 4.5 and similar models, you'll typically use an open-source alternative like Llama 3 or Mistral for local training, as Claude models are API-only. However, the data preparation approach remains the same.

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

# Model selection
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  
# Alternatives: "mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen2-7B-Instruct"

# QLoRA configuration for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank (8, 16, 32, or 64)
    lora_alpha=32,  # Typically 2x rank
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: ~25M / 8B total (0.3%)
```

### 4.2 Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./models/twainbot-lora",
    run_name="twainbot-v1",
    
    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size: 4 * 4 = 16
    gradient_checkpointing=True,
    
    # Learning rate
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    
    # Optimization
    optim="paged_adamw_32bit",
    weight_decay=0.001,
    max_grad_norm=0.3,
    
    # Precision
    bf16=True,  # Use fp16=True if bf16 not supported
    
    # Logging
    logging_steps=10,
    logging_strategy="steps",
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Other
    report_to="wandb",  # or "tensorboard"
    push_to_hub=False,
)
```

### 4.3 Data Loading & Training

```python
from datasets import load_dataset

# Load datasets
train_dataset = load_dataset('json', data_files='./data/twainbot_dataset/train.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='./data/twainbot_dataset/val.jsonl', split='train')

def formatting_func(example):
    """Format messages into training text"""
    messages = example['messages']
    
    # Use model's chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # Fallback formatting
        text = ""
        for msg in messages:
            role = msg['role']
            content = msg.get('content', '')
            
            if role == "system":
                text += f"<|system|>\n{content}\n"
            elif role == "user":
                text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                if 'tool_calls' in msg and msg['tool_calls']:
                    # Format tool calls
                    for call in msg['tool_calls']:
                        func = call['function']
                        text += f"<|assistant|>\n<tool_call>\n{json.dumps(func)}\n</tool_call>\n"
                else:
                    text += f"<|assistant|>\n{content}\n"
            elif role == "tool":
                text += f"<|tool|>\n{content}\n"
        
        text += "<|endoftext|>"
    
    return text

# Create trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    formatting_func=formatting_func,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,  # Set True for better GPU utilization with short sequences
)

# Train
print("🚀 Starting LoRA training...")
trainer.train()

# Save final model
trainer.save_model("./models/twainbot-lora-final")
tokenizer.save_pretrained("./models/twainbot-lora-final")

print("✅ Training complete!")
```

### 4.4 Monitoring Training

```python
# Initialize Weights & Biases (optional but recommended)
import wandb

wandb.init(
    project="twainbot-lora",
    config={
        "base_model": BASE_MODEL,
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "dataset_size": len(train_dataset),
    }
)

# During training, monitor:
# - Training loss (should decrease steadily)
# - Eval loss (should decrease, watch for overfitting)
# - Learning rate (cosine decay)
# - GPU memory usage
# - Training speed (tokens/second)
```

---

## Phase 5: Evaluation & Testing (Day 3 - 6 hours)

### 5.1 Quantitative Evaluation

```python
from peft import AutoPeftModelForCausalLM
import torch

class ModelEvaluator:
    def __init__(self, model_path: str):
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """Generate response for a prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        return response
    
    def evaluate_style_consistency(self, test_prompts: List[dict]) -> dict:
        """Evaluate style consistency across prompts"""
        results = []
        
        # Style markers for Twain
        twain_markers = [
            'reckon', 'well', 'might', 'say', 'suppose',
            'peculiar', 'considerable', 'powerful', 'tolerable'
        ]
        
        # Style markers for Franklin
        franklin_markers = [
            'poor richard', 'early to bed', 'wisdom', 'prudent',
            'diligent', 'thrifty', 'waste', 'virtue'
        ]
        
        for prompt_dict in test_prompts:
            prompt = prompt_dict['prompt']
            expected_style = prompt_dict['style']
            
            response = self.generate_response(prompt)
            
            # Count style markers
            response_lower = response.lower()
            twain_score = sum(1 for m in twain_markers if m in response_lower)
            franklin_score = sum(1 for m in franklin_markers if m in response_lower)
            
            detected_style = "twain" if twain_score > franklin_score else "franklin"
            
            results.append({
                'prompt': prompt,
                'response': response,
                'expected_style': expected_style,
                'detected_style': detected_style,
                'correct': detected_style == expected_style,
                'twain_score': twain_score,
                'franklin_score': franklin_score,
            })
        
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        
        return {
            'accuracy': accuracy,
            'results': results
        }
    
    def evaluate_tool_calling(self, tool_test_cases: List[dict]) -> dict:
        """Evaluate tool calling accuracy"""
        results = []
        
        for test_case in tool_test_cases:
            prompt = test_case['prompt']
            expected_tool = test_case['expected_tool']
            expected_params = test_case.get('expected_params', {})
            
            response = self.generate_response(prompt)
            
            # Check for tool call in response
            has_tool_call = '<tool_call>' in response or 'get_weather' in response
            
            # Extract tool call if present
            tool_call_correct = False
            params_correct = False
            
            if has_tool_call:
                # Parse tool call (simplified)
                if expected_tool in response:
                    tool_call_correct = True
                    
                    # Check parameters
                    params_found = 0
                    for param, value in expected_params.items():
                        if str(value).lower() in response.lower():
                            params_found += 1
                    
                    params_correct = params_found == len(expected_params)
            
            results.append({
                'prompt': prompt,
                'response': response,
                'has_tool_call': has_tool_call,
                'tool_correct': tool_call_correct,
                'params_correct': params_correct,
                'fully_correct': tool_call_correct and params_correct,
            })
        
        accuracy = sum(1 for r in results if r['fully_correct']) / len(results)
        tool_detection = sum(1 for r in results if r['has_tool_call']) / len(results)
        
        return {
            'accuracy': accuracy,
            'tool_detection_rate': tool_detection,
            'results': results
        }

# Test cases
style_test_prompts = [
    {'prompt': 'Describe a sunny day.', 'style': 'twain'},
    {'prompt': 'Give advice about checking the weather.', 'style': 'franklin'},
    {'prompt': 'Tell me about rain.', 'style': 'twain'},
]

tool_test_cases = [
    {
        'prompt': 'What is the weather in Boston?',
        'expected_tool': 'get_weather',
        'expected_params': {'location': 'Boston'}
    },
    {
        'prompt': 'Check the forecast for Seattle.',
        'expected_tool': 'get_weather',
        'expected_params': {'location': 'Seattle'}
    },
]

# Evaluate
evaluator = ModelEvaluator("./models/twainbot-lora-final")
style_results = evaluator.evaluate_style_consistency(style_test_prompts)
tool_results = evaluator.evaluate_tool_calling(tool_test_cases)

print(f"\nStyle Consistency: {style_results['accuracy']:.2%}")
print(f"Tool Calling Accuracy: {tool_results['accuracy']:.2%}")
```

### 5.2 Interactive Testing

```python
def interactive_demo():
    """Interactive demo of the model"""
    evaluator = ModelEvaluator("./models/twainbot-lora-final")
    
    print("TwainBot Demo - Press Ctrl+C to exit")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if not user_input.strip():
                continue
            
            # Generate response
            prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
            response = evaluator.generate_response(prompt, max_length=300)
            
            print(f"\nTwainBot: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

# Run demo
interactive_demo()
```

---

## Phase 6: Deployment & Serving (Day 3 Afternoon - 2 hours)

### 6.1 Model Export

```python
from peft import AutoPeftModelForCausalLM

# Option 1: Keep as LoRA adapter (smaller, flexible)
# Already saved during training at "./models/twainbot-lora-final"

# Option 2: Merge adapter into base model (larger, faster inference)
def merge_and_save(adapter_path: str, output_path: str):
    """Merge LoRA adapter with base model"""
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Merge LoRA weights
    merged_model = model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(output_path)
    
    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Merged model saved to {output_path}")

# Merge if desired
merge_and_save("./models/twainbot-lora-final", "./models/twainbot-merged")
```

### 6.2 Serving with vLLM

```python
# Install vLLM: pip install vllm

from vllm import LLM, SamplingParams

class TwainBotServer:
    def __init__(self, model_path: str):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype="float16",
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
        )
    
    def generate(self, prompt: str) -> str:
        """Generate response"""
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

# FastAPI server
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="TwainBot API")
server = None

class ChatRequest(BaseModel):
    message: str
    style: str = "twain"
    use_tools: bool = True

class ChatResponse(BaseModel):
    response: str
    style: str
    tool_used: bool = False

@app.on_event("startup")
async def startup():
    global server
    server = TwainBotServer("./models/twainbot-merged")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint"""
    # Format prompt with style
    system_prompts = {
        "twain": "You are a witty assistant who speaks like Mark Twain.",
        "franklin": "You are a wise assistant channeling Benjamin Franklin.",
    }
    
    system = system_prompts.get(request.style, system_prompts["twain"])
    
    prompt = f"""<|system|>
{system}
<|user|>
{request.message}
<|assistant|>
"""
    
    response = server.generate(prompt)
    
    # Check if tool was used
    tool_used = '<tool_call>' in response or 'get_weather' in response
    
    return ChatResponse(
        response=response,
        style=request.style,
        tool_used=tool_used
    )

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Resources & References

### Academic Papers

1. **LoRA: Low-Rank Adaptation of Large Language Models**
   - Paper: https://arxiv.org/abs/2106.09685
   - Authors: Hu et al., 2021
   - Key insight: Low-rank updates to attention weights enable efficient fine-tuning

2. **QLoRA: Efficient Finetuning of Quantized LLMs**
   - Paper: https://arxiv.org/abs/2305.14314
   - Authors: Dettmers et al., 2023
   - Key insight: 4-bit quantization + LoRA enables fine-tuning on consumer hardware

3. **Instruction Tuning for Large Language Models: A Survey**
   - Paper: https://arxiv.org/abs/2308.10792
   - Comprehensive overview of instruction-following fine-tuning

### Data Sources

1. **Project Gutenberg**
   - Website: https://www.gutenberg.org
   - API: https://gutendex.com (unofficial API)
   - License: Public domain works
   - Mark Twain works: https://www.gutenberg.org/ebooks/author/53
   - Benjamin Franklin: https://www.gutenberg.org/ebooks/author/1459

2. **Reddit Data**
   - PRAW (Python Reddit API Wrapper): https://praw.readthedocs.io
   - Pushshift API: https://pushshift.io (historical data)
   - Your existing data: `data_sources/reddit-theonion/`

3. **The Onion**
   - Official site: https://www.theonion.com
   - Note: Respect robots.txt and rate limits
   - Consider using existing Reddit submissions instead of direct scraping

4. **Weather APIs**
   - OpenWeatherMap: https://openweathermap.org/api (free tier available)
   - Open-Meteo: https://open-meteo.com (free, no API key required)
   - Weather.gov API: https://www.weather.gov/documentation/services-web-api

### Training Frameworks & Libraries

1. **Hugging Face Ecosystem**
   - Transformers: https://github.com/huggingface/transformers
   - PEFT (Parameter-Efficient Fine-Tuning): https://github.com/huggingface/peft
   - TRL (Transformer Reinforcement Learning): https://github.com/huggingface/trl
   - Datasets: https://github.com/huggingface/datasets

2. **Optimization Libraries**
   - bitsandbytes: https://github.com/TimDettmers/bitsandbytes (quantization)
   - Flash Attention: https://github.com/Dao-AILab/flash-attention
   - vLLM: https://github.com/vllm-project/vllm (serving)

3. **Monitoring & Logging**
   - Weights & Biases: https://wandb.ai
   - TensorBoard: https://www.tensorflow.org/tensorboard
   - MLflow: https://mlflow.org

### Code Examples & Tutorials

1. **Hugging Face LoRA Examples**
   - https://github.com/huggingface/peft/tree/main/examples
   - Comprehensive examples for various use cases

2. **Unsloth (Fast LoRA Training)**
   - https://github.com/unslothai/unsloth
   - 2x faster training, reduced memory usage

3. **LLaMA Recipes**
   - https://github.com/facebookresearch/llama-recipes
   - Meta's official fine-tuning recipes

### Community Resources

1. **r/LocalLLaMA** (Reddit)
   - https://www.reddit.com/r/LocalLLaMA/
   - Active community for local LLM fine-tuning

2. **Hugging Face Forums**
   - https://discuss.huggingface.co/
   - Technical support and discussions

3. **Discord Communities**
   - Hugging Face: https://hf.co/join/discord
   - EleutherAI: https://www.eleuther.ai/get-involved
   - LocalLLaMA: Various community servers

---

## Best Practices & Tips

### Data Quality

1. **Prioritize Quality Over Quantity**
   - 10,000 high-quality examples > 100,000 mediocre ones
   - Every example should teach something specific

2. **Diversity is Key**
   - Balance different styles, tones, and use cases
   - Include edge cases and error handling

3. **Clean Rigorously**
   - Remove duplicates, near-duplicates
   - Fix encoding issues
   - Validate all examples

### Training

1. **Start Small, Scale Up**
   - Begin with 1,000 examples to validate pipeline
   - Gradually increase dataset size
   - Monitor for overfitting

2. **Learning Rate is Critical**
   - LoRA typically uses 2e-4 to 5e-4
   - Use warmup (3-5% of total steps)
   - Cosine decay for smooth convergence

3. **Monitor Everything**
   - Training loss (should decrease steadily)
   - Validation loss (watch for divergence)
   - Sample generations (qualitative check)
   - GPU utilization and memory

### Evaluation

1. **Multi-Faceted Assessment**
   - Quantitative: perplexity, accuracy metrics
   - Qualitative: human evaluation of style
   - Functional: tool-calling correctness

2. **Create Diverse Test Sets**
   - Cover all categories and edge cases
   - Include adversarial examples
   - Test failure modes

### Common Pitfalls

1. **Overfitting**
   - Symptom: Low train loss, high val loss
   - Solution: More data, higher dropout, fewer epochs

2. **Style Drift**
   - Symptom: Model loses distinctive voice
   - Solution: Increase literary examples, reduce generic data

3. **Tool Hallucination**
   - Symptom: Model makes up tool calls or parameters
   - Solution: More tool examples, explicit validation in training

4. **Catastrophic Forgetting**
   - Symptom: Model forgets base capabilities
   - Solution: Lower learning rate, include general examples

---

## Three-Day Implementation Timeline

### Day 1: Data Foundation (8 hours)

**Morning (4 hours)**
- ✅ Set up Python environment and install dependencies
- ✅ Download Mark Twain works from Project Gutenberg
- ✅ Download Benjamin Franklin writings
- ✅ Extract and clean literary passages

**Afternoon (4 hours)**
- ✅ Process existing Reddit Onion data
- ✅ Curate weather joke collection
- ✅ Initial data cleaning and validation
- ✅ Save raw datasets

**Deliverable**: 8,000-12,000 raw examples collected and cleaned

### Day 2: Synthesis & Training (10 hours)

**Morning (4 hours)**
- ✅ Generate tool-use synthetic data (3,000 examples)
- ✅ Create hybrid style+tool examples (1,500 examples)
- ✅ Balance dataset according to target distribution
- ✅ Create train/validation splits

**Afternoon (6 hours)**
- ✅ Configure LoRA training setup
- ✅ Start training (automated, 4-6 hours runtime)
- ✅ Monitor training metrics
- ✅ Generate sample outputs during training

**Deliverable**: Trained LoRA adapter, training logs

### Day 3: Evaluation & Deployment (6 hours)

**Morning (3 hours)**
- ✅ Run quantitative evaluation suite
- ✅ Test style consistency
- ✅ Test tool-calling accuracy
- ✅ Generate evaluation report

**Afternoon (3 hours)**
- ✅ Merge LoRA adapter (if desired)
- ✅ Set up FastAPI serving endpoint
- ✅ Create interactive demo
- ✅ Document results and create examples

**Deliverable**: Production-ready model, API server, documentation

---

## Success Metrics

### Quantitative Targets

- **Training Loss**: < 0.8 final
- **Validation Loss**: < 1.0 final
- **Style Consistency**: > 80% correct voice detection
- **Tool Accuracy**: > 85% correct function calls
- **Inference Speed**: > 30 tokens/second

### Qualitative Targets

- **Voice Authenticity**: Recognizably Twain/Franklin without being caricature
- **Humor Quality**: Genuinely witty, not just trying too hard
- **Tool Integration**: Natural blending of API data with styled responses
- **Robustness**: Handles edge cases gracefully

---

## Conclusion

This guide demonstrates that creating a specialized, personality-driven AI assistant is achievable in a weekend with:

1. **Focused Data Curation**: Quality literary sources + modern humor + functional examples
2. **Efficient Training**: LoRA/QLoRA enables fine-tuning on consumer hardware
3. **Thoughtful Design**: Balancing style, humor, and functionality
4. **Modern Tools**: Leveraging cutting-edge frameworks and libraries

The result—TwainBot—is more than a technical demonstration. It's proof that small, carefully curated datasets can create distinctive AI behaviors that generic training cannot replicate.

**Key Takeaways**:

- Data composition directly determines model personality
- 15,000 curated examples outperform 1M generic ones for specialized tasks
- Hybrid capabilities (style + tools) are achievable with proper data design
- Weekend-scale projects can produce production-worthy results

**Next Steps**:

1. Experiment with other literary voices (Oscar Wilde, Dorothy Parker)
2. Add multi-modal capabilities (weather visualization generation)
3. Expand tool repertoire (news, events, astronomy)
4. Fine-tune for specific domains (travel, sports, politics)

As Mark Twain himself might say: "The secret of getting ahead is getting started." 

Happy training! 🎩☀️

---

**Document Version**: 1.0  
**Last Updated**: November 2, 2025  
**Author**: TwainBot Project Team  
**License**: MIT (for code), respective licenses for data sources

