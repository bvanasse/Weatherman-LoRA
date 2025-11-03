# LoRA Training Plan: Data-Focused Guide
## Comprehensive Training Data Collection, Parsing & Generation for Humor + Tool-Calling Model

---

## TLDR

**Goal**: Create a specialized LoRA fine-tuned model (3-8B parameters) that combines:
- Humorous writing style (Mark Twain, Benjamin Franklin, The Onion)
- Weather API function calling capabilities
- Hybrid responses that use API data with witty commentary

**Timeline**: 3-day weekend project

**Model**: Llama 3.2 3B Instruct (recommended) or Phi-4 (16B)

**Framework**: Unsloth + LLaMA-Factory for efficient LoRA training

**Dataset Target**: 12,000-15,000 high-quality training examples across 5 data categories

**Key Focus**: This guide emphasizes **data collection, parsing, and generation methods** - the foundation of successful fine-tuning.

---

## Overview

This training plan demonstrates how meticulous data curation directly impacts model behavior. By combining literary style datasets (Mark Twain, Benjamin Franklin, The Onion) with functional API training data, we create a unique hybrid model that showcases both personality and utility.

### Why Data Quality Matters

1. **Style Transfer**: Literary excerpts teach distinctive voice patterns
2. **Function Calling**: Structured API examples teach tool usage
3. **Hybrid Capability**: Combined examples teach integration of both skills
4. **Generalization**: Diverse examples improve robustness
5. **Balanced Dataset**: Proper ratios prevent mode collapse

### Data Impact on Training

- **High-quality, diverse data** → Better generalization
- **Consistent formatting** → Faster convergence
- **Balanced categories** → Prevents overfitting to one style
- **Large volume** → More stable training dynamics
- **Clean, parsed data** → Fewer training artifacts

---

## Phase 1: Data Collection Strategy (Day 1 - Morning)
**Duration**: 4-6 hours

### 1.1 Dataset Categories & Targets

| Category | Source | Target Examples | Purpose |
|----------|--------|----------------|---------|
| **The Onion** | Web scraping + HF datasets | 4,000-5,000 | Satirical humor style |
| **Mark Twain** | Project Gutenberg | 3,000-4,000 | Witty 19th-century voice |
| **Benjamin Franklin** | Founders Online + Gutenberg | 1,500-2,000 | Pragmatic wit & maxims |
| **Weather API Calls** | Synthetic generation | 3,000-3,500 | Function calling patterns |
| **Hybrid Examples** | AI-assisted generation | 2,000-2,500 | Combined style + tools |

**Total Target**: 13,500-17,000 examples → Filtered to 12,000-15,000 high-quality

---

## Phase 2: Detailed Data Collection & Parsing Methods

### 2.1 The Onion - Satirical News Articles

#### Collection Methods

**Method A: Hugging Face Dataset (Fastest)**
```python
from datasets import load_dataset

# Pre-existing dataset with ~33,880 articles
onion_dataset = load_dataset("Biddls/Onion_News")

# Inspect structure
print(onion_dataset['train'][0])
# Output: {'text': 'HEADLINE #~# ARTICLE_BODY'}

# Extract ~5,000 high-quality examples
def filter_onion_articles(dataset, min_length=200, max_length=1500):
    """Filter for articles with appropriate length and structure"""
    filtered = []
    for item in dataset['train']:
        parts = item['text'].split(' #~# ')
        if len(parts) == 2:
            headline, body = parts
            if min_length <= len(body) <= max_length:
                # Check for satirical markers
                if any(marker in headline.lower() for marker in 
                       ['area', 'nation', 'report', 'study', 'man', 'woman']):
                    filtered.append({'headline': headline, 'body': body})
    return filtered[:5000]

onion_articles = filter_onion_articles(onion_dataset)
```

**Method B: Direct Web Scraping (More Control)**
```python
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin

class OnionScraper:
    def __init__(self, base_url="https://www.theonion.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (educational-research-bot)'
        })
    
    def scrape_article(self, article_url):
        """Scrape a single Onion article"""
        try:
            response = self.session.get(article_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract headline
            headline_elem = soup.find('h1', class_='sc-1efpnfq-0')
            headline = headline_elem.get_text(strip=True) if headline_elem else None
            
            # Extract article body
            body_elems = soup.find_all('p', class_='sc-77igqf-0')
            body = '\n\n'.join([p.get_text(strip=True) for p in body_elems])
            
            if headline and len(body) > 200:
                return {'headline': headline, 'body': body, 'url': article_url}
        except Exception as e:
            print(f"Error scraping {article_url}: {e}")
        return None
    
    def scrape_category(self, category='news', max_pages=50):
        """Scrape articles from a category page"""
        articles = []
        for page in range(1, max_pages + 1):
            url = f"{self.base_url}/{category}?startIndex={(page-1)*20}"
            try:
                response = self.session.get(url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                article_links = soup.find_all('a', href=True)
                for link in article_links:
                    href = link.get('href')
                    if href and '/news/' in href:
                        full_url = urljoin(self.base_url, href)
                        article = self.scrape_article(full_url)
                        if article:
                            articles.append(article)
                        time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error on page {page}: {e}")
        
        return articles

# Usage
scraper = OnionScraper()
onion_articles = scraper.scrape_category('news', max_pages=30)
```

**Method C: Reddit Data (Already Available)**
```python
import pandas as pd

# Use existing Reddit-scraped data
onion_reddit = pd.read_csv('data_sources/reddit-theonion/data/TheOnion_181217_184244.csv')

# Extract titles and convert to training format
def format_reddit_onion(df, max_examples=2000):
    """Format Reddit submissions as Onion-style training examples"""
    examples = []
    for idx, row in df.head(max_examples).iterrows():
        title = row.get('title', '')
        # Create instruction-output pairs
        examples.append({
            'instruction': 'Write a satirical news headline about:',
            'input': self.extract_topic_from_title(title),
            'output': title
        })
    return examples
```

#### Parsing & Formatting

```python
def parse_onion_to_training_format(articles):
    """Convert Onion articles to standardized training format"""
    training_examples = []
    
    for article in articles:
        headline = article['headline']
        body = article['body']
        
        # Extract topic/keywords for instruction
        topic = extract_main_topic(headline)
        
        # Format 1: Satirical headline generation
        training_examples.append({
            'instruction': f'Write a satirical news headline about {topic}:',
            'input': '',
            'output': headline,
            'category': 'onion_headline'
        })
        
        # Format 2: Full article (truncated to ~500 words)
        truncated_body = truncate_to_sentences(body, max_words=500)
        training_examples.append({
            'instruction': f'Write a satirical news article about {topic}:',
            'input': '',
            'output': f"{headline}\n\n{truncated_body}",
            'category': 'onion_article'
        })
        
        # Format 3: Weather-specific (filtered subset)
        if is_weather_related(headline, body):
            training_examples.append({
                'instruction': 'Write a humorous weather-related news article:',
                'input': '',
                'output': f"{headline}\n\n{truncated_body}",
                'category': 'onion_weather'
            })
    
    return training_examples

def extract_main_topic(headline):
    """Extract main topic/keywords from headline"""
    # Simple keyword extraction
    weather_keywords = ['weather', 'rain', 'snow', 'temperature', 'forecast', 'storm']
    if any(kw in headline.lower() for kw in weather_keywords):
        return 'weather'
    # Add more topic extraction logic
    return 'general'

def is_weather_related(headline, body):
    """Check if article is weather-related"""
    weather_terms = ['weather', 'forecast', 'rain', 'snow', 'temperature', 
                     'hurricane', 'storm', 'sunny', 'cloudy', 'climate']
    text = (headline + ' ' + body).lower()
    return sum(1 for term in weather_terms if term in text) >= 2
```

---

### 2.2 Mark Twain - Literary Works

#### Collection Methods

**Method A: Project Gutenberg (Python Library)**
```python
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_metadata

# Find Mark Twain texts
twain_text_ids = list(get_metadata('author', 'Twain, Mark'))

# Priority texts (by ID - check gutenberg.org for actual IDs)
priority_titles = {
    74: "The Adventures of Tom Sawyer",
    76: "Adventures of Huckleberry Finn",
    86: "A Connecticut Yankee in King Arthur's Court",
    3176: "The Innocents Abroad",
    245: "Life on the Mississippi",
    2895: "Following the Equator",
    1837: "The Prince and the Pauper",
}

def download_twain_text(text_id):
    """Download and clean a Project Gutenberg text"""
    try:
        text = load_etext(text_id)
        cleaned_text = strip_headers(text).strip()
        return cleaned_text
    except Exception as e:
        print(f"Error downloading text {text_id}: {e}")
        return None

# Download all priority texts
twain_texts = {}
for text_id, title in priority_titles.items():
    print(f"Downloading {title}...")
    text = download_twain_text(text_id)
    if text:
        twain_texts[title] = text
        time.sleep(2)  # Rate limiting
```

**Method B: Direct URL Download**
```python
import requests
import re

def download_gutenberg_text(gutenberg_id):
    """Download text directly from Project Gutenberg mirror"""
    url = f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text
        
        # Remove Gutenberg header/footer
        # Header typically ends with "*** START OF"
        # Footer typically starts with "*** END OF"
        start_marker = "*** START OF"
        end_marker = "*** END OF"
        
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx]
            # Remove Gutenberg-specific markers
            text = re.sub(r'\*\*\*.*?\*\*\*', '', text)
            text = re.sub(r'^[A-Z\s]+\n', '', text, flags=re.MULTILINE)
        
        return text.strip()
    except Exception as e:
        print(f"Error downloading {gutenberg_id}: {e}")
        return None
```

**Method C: Mark Twain Papers Project (Academic Source)**
```python
# For letters and more obscure works
# Available at: https://www.marktwainproject.org/
# Requires manual download or API access if available

def scrape_twain_papers():
    """Scrape Mark Twain Project for additional texts"""
    # Implementation depends on site structure
    # Check robots.txt and terms of service
    pass
```

#### Parsing & Chunking Strategies

```python
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt', quiet=True)

class TwainTextParser:
    def __init__(self, min_chunk_words=200, max_chunk_words=500):
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
    
    def detect_humorous_passages(self, text, min_humor_score=0.3):
        """Identify passages likely to contain humor/wit"""
        humor_indicators = [
            r'\b(said|replied|remarked|exclaimed|laughed)\b',
            r'\b(never|always|always was|always is)\b.*\b(but|except|unless)\b',
            r'\b(only|merely|simply)\b.*\b(however|but|yet)\b',
            r'[.!?]\s+"[^"]{20,}"',  # Dialogues
            r'\b(would|should|could)\b.*\b(if|when|unless)\b',
        ]
        
        sentences = sent_tokenize(text)
        humor_scores = []
        
        for sentence in sentences:
            score = sum(1 for pattern in humor_indicators 
                       if re.search(pattern, sentence, re.IGNORECASE))
            humor_scores.append((sentence, score / len(humor_indicators)))
        
        # Filter high-scoring passages
        humorous = [s for s, score in humor_scores if score >= min_humor_score]
        return humorous
    
    def chunk_by_paragraph(self, text):
        """Chunk text preserving paragraph boundaries"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for para in paragraphs:
            para_words = len(word_tokenize(para))
            
            if current_word_count + para_words > self.max_chunk_words:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_word_count = para_words
            else:
                current_chunk.append(para)
                current_word_count += para_words
                
                if current_word_count >= self.min_chunk_words:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def chunk_by_sentence_window(self, text, window_size=15):
        """Create overlapping sentence windows"""
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences) - window_size + 1, window_size // 2):
            window = sentences[i:i + window_size]
            chunk_text = ' '.join(window)
            word_count = len(word_tokenize(chunk_text))
            
            if self.min_chunk_words <= word_count <= self.max_chunk_words:
                chunks.append(chunk_text)
        
        return chunks
    
    def format_as_training_examples(self, chunks, style='twain'):
        """Convert chunks to training format"""
        examples = []
        
        for chunk in chunks:
            # Extract topic or first sentence as context
            first_sent = sent_tokenize(chunk)[0] if chunk else ""
            topic = self.extract_topic(first_sent)
            
            # Format 1: Style transfer
            examples.append({
                'instruction': f'Rewrite this in Mark Twain\'s style:',
                'input': topic,
                'output': chunk,
                'category': 'twain_style'
            })
            
            # Format 2: Continue writing
            if len(chunk) > 100:
                prompt_part = chunk[:len(chunk)//2]
                continuation = chunk[len(chunk)//2:]
                examples.append({
                    'instruction': 'Continue writing in Mark Twain\'s style:',
                    'input': prompt_part,
                    'output': continuation,
                    'category': 'twain_continuation'
                })
            
            # Format 3: Weather-related (if applicable)
            if self.is_weather_related(chunk):
                examples.append({
                    'instruction': 'Describe weather in Mark Twain\'s style:',
                    'input': '',
                    'output': chunk,
                    'category': 'twain_weather'
                })
        
        return examples
    
    def extract_topic(self, text):
        """Extract main topic from text"""
        # Simple keyword-based extraction
        words = word_tokenize(text.lower())
        # Remove stopwords and extract nouns
        # Simplified version - use spaCy for better results
        return ' '.join(words[:5])
    
    def is_weather_related(self, text):
        """Check if passage mentions weather"""
        weather_terms = ['weather', 'rain', 'snow', 'storm', 'wind', 
                         'temperature', 'cloud', 'sun', 'sky']
        return any(term in text.lower() for term in weather_terms)

# Usage
parser = TwainTextParser(min_chunk_words=200, max_chunk_words=500)
twain_examples = []

for title, text in twain_texts.items():
    print(f"Processing {title}...")
    chunks = parser.chunk_by_paragraph(text)
    # Filter for humorous passages
    humorous_chunks = [c for c in chunks 
                      if parser.detect_humorous_passages(c)]
    examples = parser.format_as_training_examples(humorous_chunks)
    twain_examples.extend(examples)

print(f"Generated {len(twain_examples)} Twain training examples")
```

---

### 2.3 Benjamin Franklin - Poor Richard's Almanack & Writings

#### Collection Methods

**Method A: Founders Online Archive**
```python
import requests
from bs4 import BeautifulSoup

def scrape_founders_online():
    """Scrape Poor Richard's Almanack from Founders Online"""
    base_url = "https://founders.archives.gov"
    
    # Search for Poor Richard's Almanack entries
    search_url = f"{base_url}/documents/Franklin/"
    
    # Note: Founders Online may require specific document IDs
    # Check their API or manual browsing for document lists
    
    almanack_years = list(range(1733, 1759))  # 1733-1758
    
    almanack_entries = []
    for year in almanack_years:
        # Construct URL for specific year (structure may vary)
        doc_url = f"{base_url}/documents/Franklin/01-{year:02d}-01-0001"
        # Implement scraping logic based on site structure
        pass
    
    return almanack_entries
```

**Method B: Project Gutenberg**
```python
# Poor Richard's Almanack is available on Project Gutenberg
# ID: 20203 (check current ID on gutenberg.org)

franklin_almanack_id = 20203
almanack_text = download_gutenberg_text(franklin_almanack_id)

# Also download other Franklin works
franklin_works = {
    148: "The Autobiography of Benjamin Franklin",
    # Add more IDs as needed
}
```

**Method C: Direct Text Sources**
```python
# Poor Richard's Almanack is in public domain
# Many libraries have digitized versions

def parse_almanack_text(text):
    """Parse Poor Richard's Almanack into structured entries"""
    entries = []
    
    # Almanack typically has:
    # - Monthly calendars with proverbs
    # - Weather predictions
    # - Wise sayings and maxims
    
    # Extract proverbs (often in quotes or italics)
    proverb_pattern = r'"([^"]{20,150})"'
    proverbs = re.findall(proverb_pattern, text)
    
    # Extract weather-related content
    weather_sections = re.findall(
        r'(?:Weather|Forecast|Prediction).*?(?=\n\n|\Z)',
        text,
        re.IGNORECASE | re.DOTALL
    )
    
    # Extract maxims and advice
    maxim_pattern = r'(?:Poor Richard|Richard|he) (?:says|said|writes|wrote)[^.]*\.'
    maxims = re.findall(maxim_pattern, text, re.IGNORECASE)
    
    return {
        'proverbs': proverbs,
        'weather_content': weather_sections,
        'maxims': maxims
    }
```

#### Parsing & Formatting

```python
class FranklinTextParser:
    def __init__(self):
        self.proverb_patterns = [
            r'"([^"]{15,200})"',  # Quoted proverbs
            r'(?:Poor Richard|Richard) (?:says|said|writes):\s*"([^"]+)"',
            r'([A-Z][^.!?]{10,150}[.!?])',  # Standalone maxims
        ]
    
    def extract_proverbs(self, text):
        """Extract proverbs and maxims"""
        proverbs = []
        for pattern in self.proverb_patterns:
            matches = re.findall(pattern, text)
            proverbs.extend(matches)
        
        # Filter by length and content
        filtered = [p for p in proverbs 
                   if 15 <= len(p) <= 200 and 
                   not p.startswith('Chapter') and
                   not p.startswith('Page')]
        
        return filtered
    
    def extract_weather_wisdom(self, text):
        """Extract weather-related advice"""
        weather_patterns = [
            r'(?:weather|forecast|rain|snow|storm)[^.!?]{20,200}[.!?]',
            r'(?:season|winter|summer|spring|fall)[^.!?]{20,200}[.!?]',
        ]
        
        weather_texts = []
        for pattern in weather_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            weather_texts.extend(matches)
        
        return weather_texts
    
    def format_as_training_examples(self, proverbs, weather_texts):
        """Convert to training format"""
        examples = []
        
        # Format 1: Proverbs
        for proverb in proverbs:
            examples.append({
                'instruction': 'Provide wisdom in Benjamin Franklin\'s style:',
                'input': '',
                'output': proverb,
                'category': 'franklin_proverb'
            })
        
        # Format 2: Weather advice
        for weather_text in weather_texts:
            examples.append({
                'instruction': 'Give weather advice in Poor Richard\'s style:',
                'input': '',
                'output': weather_text,
                'category': 'franklin_weather'
            })
        
        # Format 3: Style transfer
        for proverb in proverbs[:len(proverbs)//2]:  # Use subset
            examples.append({
                'instruction': 'Rewrite this advice in Benjamin Franklin\'s style:',
                'input': self.modernize_proverb(proverb),
                'output': proverb,
                'category': 'franklin_style_transfer'
            })
        
        return examples
    
    def modernize_proverb(self, proverb):
        """Create modern version for style transfer training"""
        # Simple modernization (replace archaic words)
        modern = proverb
        replacements = {
            'hath': 'has',
            'doth': 'does',
            'thou': 'you',
            'thee': 'you',
            'thy': 'your',
            'thine': 'yours',
        }
        for old, new in replacements.items():
            modern = re.sub(r'\b' + old + r'\b', new, modern, flags=re.IGNORECASE)
        return modern

# Usage
franklin_parser = FranklinTextParser()
proverbs = franklin_parser.extract_proverbs(almanack_text)
weather_texts = franklin_parser.extract_weather_wisdom(almanack_text)
franklin_examples = franklin_parser.format_as_training_examples(proverbs, weather_texts)
```

---

### 2.4 Weather API Function Calling - Synthetic Generation

#### Generation Strategy

```python
import json
import random
from faker import Faker

fake = Faker()

class WeatherAPIDataGenerator:
    def __init__(self):
        self.cities = [
            'Boston', 'New York', 'Chicago', 'Los Angeles', 'Seattle',
            'Miami', 'Denver', 'Austin', 'Portland', 'San Francisco',
            'Philadelphia', 'Phoenix', 'San Diego', 'Dallas', 'San Antonio'
        ]
        
        self.weather_functions = [
            {
                'name': 'get_current_weather',
                'description': 'Get the current weather in a given location',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'City and state, e.g. San Francisco, CA'
                        },
                        'unit': {
                            'type': 'string',
                            'enum': ['celsius', 'fahrenheit'],
                            'description': 'Temperature unit'
                        }
                    },
                    'required': ['location']
                }
            },
            {
                'name': 'get_weather_forecast',
                'description': 'Get weather forecast for next N days',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'City and state'
                        },
                        'days': {
                            'type': 'integer',
                            'minimum': 1,
                            'maximum': 10,
                            'description': 'Number of days to forecast'
                        },
                        'unit': {
                            'type': 'string',
                            'enum': ['celsius', 'fahrenheit']
                        }
                    },
                    'required': ['location', 'days']
                }
            },
            {
                'name': 'get_weather_alerts',
                'description': 'Get severe weather alerts for a location',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'City and state'
                        }
                    },
                    'required': ['location']
                }
            }
        ]
    
    def generate_query_variations(self, base_query, city):
        """Generate natural language variations of weather queries"""
        variations = [
            f"What's the weather like in {city}?",
            f"How's the weather in {city}?",
            f"Tell me about the weather in {city}",
            f"What's the temperature in {city}?",
            f"Is it raining in {city}?",
            f"Check the weather for {city}",
            f"I need the weather forecast for {city}",
            f"Can you give me weather info for {city}?",
            f"What's the forecast for {city} this weekend?",
            f"Will it rain in {city} tomorrow?",
        ]
        return variations
    
    def generate_function_call_example(self, function_name, user_query):
        """Generate a function calling training example"""
        function_spec = next(f for f in self.weather_functions 
                            if f['name'] == function_name)
        
        # Extract location from query (simple pattern matching)
        location = self.extract_location(user_query)
        unit = 'fahrenheit' if 'celsius' not in user_query.lower() else 'celsius'
        
        # Build parameters based on function
        if function_name == 'get_current_weather':
            params = {'location': location, 'unit': unit}
        elif function_name == 'get_weather_forecast':
            days = self.extract_days(user_query) or 3
            params = {'location': location, 'days': days, 'unit': unit}
        elif function_name == 'get_weather_alerts':
            params = {'location': location}
        else:
            params = {}
        
        # Format function call
        function_call = {
            'name': function_name,
            'parameters': params
        }
        
        return {
            'instruction': user_query,
            'input': '',
            'output': f'<function_call>\n{json.dumps(function_call, indent=2)}\n</function_call>',
            'category': 'weather_api',
            'function_name': function_name
        }
    
    def extract_location(self, query):
        """Extract location from query (simplified)"""
        # Check for known cities
        for city in self.cities:
            if city.lower() in query.lower():
                # Add state abbreviation (simplified)
                state_map = {
                    'Boston': 'MA', 'New York': 'NY', 'Chicago': 'IL',
                    'Los Angeles': 'CA', 'Seattle': 'WA', 'Miami': 'FL',
                    'Denver': 'CO', 'Austin': 'TX', 'Portland': 'OR',
                    'San Francisco': 'CA', 'Philadelphia': 'PA', 'Phoenix': 'AZ',
                    'San Diego': 'CA', 'Dallas': 'TX', 'San Antonio': 'TX'
                }
                return f"{city}, {state_map.get(city, 'CA')}"
        
        # Fallback
        return random.choice(self.cities) + ', CA'
    
    def extract_days(self, query):
        """Extract number of days from query"""
        numbers = re.findall(r'\b(\d+)\s*(?:day|days)\b', query.lower())
        if numbers:
            return min(int(numbers[0]), 10)
        
        # Check for keywords
        if 'weekend' in query.lower():
            return 3
        elif 'week' in query.lower():
            return 7
        return None
    
    def generate_dataset(self, examples_per_function=1000):
        """Generate full dataset of function calling examples"""
        examples = []
        
        for function_spec in self.weather_functions:
            function_name = function_spec['name']
            
            for _ in range(examples_per_function):
                # Random city
                city = random.choice(self.cities)
                
                # Generate query variations
                base_queries = self.generate_query_variations('', city)
                query = random.choice(base_queries)
                
                # Modify query based on function type
                if function_name == 'get_weather_forecast':
                    query = query.replace('weather', 'weather forecast')
                    if 'day' not in query.lower():
                        query += f" for the next {random.randint(1, 7)} days"
                elif function_name == 'get_weather_alerts':
                    query = f"Are there any weather alerts for {city}?"
                
                example = self.generate_function_call_example(function_name, query)
                examples.append(example)
        
        return examples

# Usage
generator = WeatherAPIDataGenerator()
weather_api_examples = generator.generate_dataset(examples_per_function=1000)
print(f"Generated {len(weather_api_examples)} weather API examples")
```

#### Advanced: Using LLM to Generate Variations

```python
from openai import OpenAI  # or Anthropic, etc.

class LLMAssistedDataGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def generate_query_variations(self, base_query, num_variations=10):
        """Use LLM to generate natural query variations"""
        prompt = f"""Generate {num_variations} natural language variations of this weather query:
"{base_query}"

Return one variation per line, making them sound like real user requests."""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        
        variations = response.choices[0].message.content.strip().split('\n')
        return [v.strip('- ').strip() for v in variations if v.strip()]
    
    def generate_edge_cases(self):
        """Generate edge case queries"""
        edge_cases = [
            "What's the weather?",  # No location
            "Is it nice outside?",  # Vague
            "Should I bring an umbrella?",  # Implied location needed
            "Weather?",  # Too short
            "Tell me about the climate in multiple cities",  # Multiple locations
            "What was the weather yesterday?",  # Past tense
        ]
        return edge_cases
```

---

### 2.5 Hybrid Examples - Combining Style + Function Calls

#### Generation Methods

**Method A: Template-Based Generation**
```python
class HybridExampleGenerator:
    def __init__(self):
        self.twain_style_intros = [
            "Well now, let me consult the meteorological oracles...",
            "The Weather Bureau has decreed, though their prophecies...",
            "I reckon I'd best check what Mother Nature has in store...",
            "Why, I'd be happy to inquire after the weather, though...",
        ]
        
        self.franklin_style_intros = [
            "As Poor Richard says, 'He that would know the weather...",
            "Let me consult the almanack of modern meteorology...",
            "Wise preparation requires knowledge of the elements...",
        ]
    
    def generate_hybrid_example(self, user_query, style='twain'):
        """Generate example combining API call with stylistic response"""
        # Step 1: Generate function call
        weather_gen = WeatherAPIDataGenerator()
        function_example = weather_gen.generate_function_call_example(
            'get_current_weather',
            user_query
        )
        
        # Step 2: Generate stylistic response
        if style == 'twain':
            intro = random.choice(self.twain_style_intros)
            response = f"""{intro}

{function_example['output']}

Ah, the forecast suggests sunshine and temperatures in the agreeable seventies—a rare gift from the Weather Bureau, whose predictions I trust about as much as I trust a riverboat gambler's smile. You'd best prepare for either a deluge or a drought, as weather forecasts are more art than science and more fiction than art."""
        
        elif style == 'franklin':
            intro = random.choice(self.franklin_style_intros)
            response = f"""{intro}

{function_example['output']}

He that waits for fair weather to travel may never begin his journey. Better to brave a storm with preparation than fear a cloud without one. As Poor Richard says, 'Lost time is never found again'—and neither is the man who checked the forecast but forgot his coat."""
        
        return {
            'instruction': user_query,
            'input': '',
            'output': response,
            'category': f'hybrid_{style}'
        }
```

**Method B: LLM-Assisted Generation**
```python
def generate_hybrid_with_llm(user_query, function_call_result, style='twain'):
    """Use LLM to generate hybrid responses"""
    prompt = f"""Generate a response to this weather query that:
1. Uses the provided function call to get weather data
2. Responds in the style of {style}
3. Incorporates humor and wit
4. Provides useful weather information

User Query: {user_query}

Function Call Result (simulated): {function_call_result}

Generate the complete response including the function call and witty commentary:"""
    
    # Use GPT-4 or Claude to generate
    response = llm_client.generate(prompt, temperature=0.7)
    return response
```

**Method C: Template + LLM Refinement**
```python
def generate_refined_hybrid(user_query, style='twain'):
    """Generate hybrid example using templates refined by LLM"""
    # 1. Generate base template
    generator = HybridExampleGenerator()
    base_example = generator.generate_hybrid_example(user_query, style)
    
    # 2. Refine with LLM
    refinement_prompt = f"""Improve this response to be more natural and humorous:

{base_example['output']}

Make it sound more like {style} wrote it naturally:"""
    
    refined = llm_client.generate(refinement_prompt)
    
    return {
        'instruction': user_query,
        'input': '',
        'output': refined,
        'category': f'hybrid_{style}_refined'
    }
```

---

## Phase 3: Data Preprocessing & Quality Assurance (Day 1 - Afternoon)
**Duration**: 4-6 hours

### 3.1 Data Cleaning Pipeline

```python
import re
from datasets import Dataset, concatenate_datasets
import pandas as pd

class DataCleaner:
    def __init__(self):
        self.min_output_length = 20
        self.max_output_length = 2000
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        # Fix common encoding issues
        text = text.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
        return text.strip()
    
    def validate_example(self, example):
        """Validate training example quality"""
        checks = {
            'has_instruction': bool(example.get('instruction', '').strip()),
            'has_output': bool(example.get('output', '').strip()),
            'output_length_ok': self.min_output_length <= len(example.get('output', '')) <= self.max_output_length,
            'no_excessive_repetition': not self.has_excessive_repetition(example.get('output', '')),
            'has_reasonable_tokens': len(example.get('output', '').split()) >= 5,
        }
        return all(checks.values()), checks
    
    def has_excessive_repetition(self, text, max_repeat=3):
        """Check for excessive word/phrase repetition"""
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check for repeated phrases
        for i in range(len(words) - max_repeat):
            phrase = ' '.join(words[i:i+max_repeat])
            if text.count(phrase) > 2:
                return True
        return False
    
    def deduplicate(self, examples):
        """Remove duplicate examples"""
        seen_outputs = set()
        unique_examples = []
        
        for ex in examples:
            output_hash = hash(ex['output'][:100])  # Hash first 100 chars
            if output_hash not in seen_outputs:
                seen_outputs.add(output_hash)
                unique_examples.append(ex)
        
        return unique_examples
    
    def clean_dataset(self, examples):
        """Full cleaning pipeline"""
        cleaned = []
        
        for ex in examples:
            # Clean text fields
            ex['instruction'] = self.clean_text(ex.get('instruction', ''))
            ex['input'] = self.clean_text(ex.get('input', ''))
            ex['output'] = self.clean_text(ex.get('output', ''))
            
            # Validate
            is_valid, checks = self.validate_example(ex)
            if is_valid:
                cleaned.append(ex)
            else:
                # Log reasons for filtering
                print(f"Filtered example: {checks}")
        
        # Deduplicate
        cleaned = self.deduplicate(cleaned)
        
        return cleaned

# Apply cleaning
cleaner = DataCleaner()

all_examples = (
    onion_examples +
    twain_examples +
    franklin_examples +
    weather_api_examples +
    hybrid_examples
)

cleaned_examples = cleaner.clean_dataset(all_examples)
print(f"Cleaned: {len(cleaned_examples)} examples (from {len(all_examples)})")
```

### 3.2 Data Balancing & Stratification

```python
def balance_dataset(examples, target_distribution):
    """Balance dataset according to target distribution"""
    # Group by category
    by_category = {}
    for ex in examples:
        category = ex.get('category', 'unknown')
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(ex)
    
    # Calculate target counts
    total_target = sum(target_distribution.values())
    balanced = []
    
    for category, target_count in target_distribution.items():
        available = by_category.get(category, [])
        # Sample with replacement if needed, or truncate
        if len(available) >= target_count:
            balanced.extend(random.sample(available, target_count))
        else:
            balanced.extend(available)
            # Optionally augment with duplicates or variations
    
    # Shuffle
    random.shuffle(balanced)
    return balanced

# Target distribution
target_dist = {
    'onion_headline': 1000,
    'onion_article': 2000,
    'onion_weather': 1000,
    'twain_style': 2000,
    'twain_continuation': 1000,
    'twain_weather': 500,
    'franklin_proverb': 800,
    'franklin_weather': 500,
    'franklin_style_transfer': 200,
    'weather_api': 3000,
    'hybrid_twain': 1500,
    'hybrid_franklin': 800,
}

balanced_dataset = balance_dataset(cleaned_examples, target_dist)
print(f"Balanced dataset: {len(balanced_dataset)} examples")
```

### 3.3 Train/Validation Split

```python
from sklearn.model_selection import train_test_split

def stratified_split(examples, test_size=0.1, seed=42):
    """Stratified split preserving category distribution"""
    # Group by category
    by_category = {}
    for ex in examples:
        category = ex.get('category', 'unknown')
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(ex)
    
    train_examples = []
    val_examples = []
    
    for category, cat_examples in by_category.items():
        cat_train, cat_val = train_test_split(
            cat_examples,
            test_size=test_size,
            random_state=seed
        )
        train_examples.extend(cat_train)
        val_examples.extend(cat_val)
    
    # Shuffle
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    
    return train_examples, val_examples

train_examples, val_examples = stratified_split(balanced_dataset)

print(f"Training: {len(train_examples)} examples")
print(f"Validation: {len(val_examples)} examples")
```

### 3.4 Dataset Statistics & Quality Metrics

```python
def analyze_dataset(examples, name="Dataset"):
    """Analyze dataset statistics"""
    stats = {
        'total_examples': len(examples),
        'avg_instruction_length': sum(len(ex.get('instruction', '')) for ex in examples) / len(examples),
        'avg_output_length': sum(len(ex.get('output', '')) for ex in examples) / len(examples),
        'category_distribution': {},
        'function_call_ratio': sum('function_call' in ex.get('output', '') for ex in examples) / len(examples),
        'humor_ratio': sum(any(kw in ex.get('instruction', '').lower() 
                               for kw in ['humor', 'satirical', 'joke', 'twain', 'franklin'])
                           for ex in examples) / len(examples),
    }
    
    # Category distribution
    for ex in examples:
        category = ex.get('category', 'unknown')
        stats['category_distribution'][category] = stats['category_distribution'].get(category, 0) + 1
    
    print(f"\n{name} Statistics:")
    print(f"  Total Examples: {stats['total_examples']}")
    print(f"  Avg Instruction Length: {stats['avg_instruction_length']:.1f} chars")
    print(f"  Avg Output Length: {stats['avg_output_length']:.1f} chars")
    print(f"  Function Call Ratio: {stats['function_call_ratio']:.2%}")
    print(f"  Humor Ratio: {stats['humor_ratio']:.2%}")
    print(f"\n  Category Distribution:")
    for cat, count in sorted(stats['category_distribution'].items()):
        print(f"    {cat}: {count} ({count/stats['total_examples']:.1%})")
    
    return stats

train_stats = analyze_dataset(train_examples, "Training Dataset")
val_stats = analyze_dataset(val_examples, "Validation Dataset")
```

### 3.5 Save Processed Dataset

```python
# Convert to HuggingFace Dataset format
train_dataset = Dataset.from_list(train_examples)
val_dataset = Dataset.from_list(val_examples)

# Save to disk
dataset_path = "/mnt/training_data/humor_weather_dataset"
train_dataset.save_to_disk(f"{dataset_path}/train")
val_dataset.save_to_disk(f"{dataset_path}/val")

# Also save as JSON for inspection
import json
with open(f"{dataset_path}/train.json", 'w') as f:
    json.dump(train_examples[:100], f, indent=2)  # Sample for inspection

print(f"Dataset saved to {dataset_path}")
```

---

## Phase 4: LoRA Training Configuration (Day 2)
**Duration**: 8-10 hours (mostly automated)

### 4.1 Model & Environment Setup

```python
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer
import torch

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=32,                          # Rank (16-64 typical)
    lora_alpha=64,                 # Scaling (typically 2x rank)
    target_modules=[               # Layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### 4.2 Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./humor_weather_lora_output",
    run_name="twain-weather-bot-v1",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb",
)
```

### 4.3 Format Dataset for Training

```python
def formatting_func(example):
    """Format examples for training"""
    system_msg = example.get('system', 
        'You are a helpful AI assistant with weather API access. You speak with wit and humor in the style of Mark Twain and Benjamin Franklin.')
    
    instruction = example['instruction']
    user_input = example.get('input', '')
    output = example['output']
    
    text = f"""<|system|>
{system_msg}
<|user|>
{instruction}
{user_input}
<|assistant|>
{output}<|endoftext|>"""
    
    return text

# Format datasets
train_dataset = train_dataset.map(lambda x: {"text": formatting_func(x)})
val_dataset = val_dataset.map(lambda x: {"text": formatting_func(x)})
```

### 4.4 Start Training

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    formatting_func=formatting_func,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

print("🚀 Starting training...")
trainer.train()

# Save final model
trainer.save_model("./humor_weather_lora_output/final_model")
print("✅ Training complete!")
```

---

## Phase 5: Evaluation & Testing (Day 3 - Morning)
**Duration**: 4-5 hours

### 5.1 Quantitative Evaluation

```python
# Evaluate on validation set
eval_results = trainer.evaluate(val_dataset)
print(f"Eval Loss: {eval_results['eval_loss']:.4f}")
print(f"Eval Perplexity: {eval_results['eval_perplexity']:.2f}")
```

### 5.2 Qualitative Testing

```python
# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./humor_weather_lora_output/final_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Test prompts
test_prompts = [
    "What's the weather forecast for Boston this weekend?",
    "Write a humorous observation about weather forecasters.",
    "Tell me about the weather in Seattle, but make it entertaining.",
]

for prompt in test_prompts:
    inputs = tokenizer([f"<|user|>\n{prompt}\n<|assistant|>\n"], 
                      return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, 
                           temperature=0.7, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}\n")
```

---

## Phase 6: Deployment Preparation (Day 3 - Afternoon)
**Duration**: 3-4 hours

### 6.1 Export Model

```python
# Merge LoRA adapters with base model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./twain-weather-bot-merged")
tokenizer.save_pretrained("./twain-weather-bot-merged")
```

### 6.2 Create Model Card

Document model capabilities, training data, and limitations.

---

## Success Metrics

### Quantitative
- Training loss < 0.5
- Function call accuracy > 90%
- Inference speed > 40 tokens/second

### Qualitative
- Generates text in Twain/Franklin style
- Correctly uses weather API function calls
- Combines humor with functional responses

---

## Troubleshooting

### High Training Loss
- Reduce learning rate (1e-4)
- Check data quality
- Increase warmup steps

### Model Doesn't Generate Function Calls
- Increase function-calling examples (40-50% of dataset)
- Verify function call format matches training data
- Use lower temperature (0.3) during inference

### Style Not Coming Through
- Increase literary dataset proportion
- Add more style transfer examples
- Consider longer training (4-5 epochs)

---

## Conclusion

This data-focused training plan emphasizes the critical importance of:
1. **Diverse data collection** from multiple sources
2. **Careful parsing** to extract high-quality examples
3. **Synthetic generation** to augment datasets
4. **Quality assurance** to ensure training effectiveness
5. **Balanced distribution** across categories

By following these methods, you'll create a dataset that enables the model to learn both stylistic writing and functional API usage, resulting in a unique hybrid capability.

---

## Appendix: Quick Reference

### Data Collection Checklist
- [ ] The Onion: 4,000-5,000 examples (HF dataset + scraping)
- [ ] Mark Twain: 3,000-4,000 examples (Project Gutenberg)
- [ ] Benjamin Franklin: 1,500-2,000 examples (Founders Online + Gutenberg)
- [ ] Weather API: 3,000-3,500 examples (synthetic generation)
- [ ] Hybrid: 2,000-2,500 examples (template + LLM-assisted)

### Key Libraries
```bash
pip install unsloth transformers accelerate peft trl datasets
pip install beautifulsoup4 requests faker nltk
pip install wandb  # For experiment tracking
```

### Expected Timeline
- **Day 1 Morning**: Data collection (4-6 hours)
- **Day 1 Afternoon**: Data processing (4-6 hours)
- **Day 2**: Training (8-10 hours, mostly automated)
- **Day 3 Morning**: Evaluation (4-5 hours)
- **Day 3 Afternoon**: Deployment (3-4 hours)

**Total Active Work**: ~20-25 hours over 3 days

