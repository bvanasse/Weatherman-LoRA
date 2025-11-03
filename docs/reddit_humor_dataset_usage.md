# Reddit Humor Dataset Processing - Usage Guide

## Overview

This guide explains how to use the Reddit Humor Dataset Processing pipeline to extract weather-related humorous examples from r/TheOnion and r/nottheonion subreddits and convert them into chat-format JSONL training data.

## Quick Start

### Basic Usage

Process all Reddit CSV files with default settings:

```bash
cd /path/to/Weatherman-LoRA
python scripts/reddit_pipeline_orchestrator.py
```

This will:
- Load all CSV files from `data_sources/reddit-theonion/data/`
- Filter for weather-related titles (using 45+ weather keywords)
- Clean Reddit artifacts and normalize text
- Generate up to 4,000 examples
- Output to `data/processed/reddit_humor_weather.jsonl`

### Custom Options

#### Specify Output Path

```bash
python scripts/reddit_pipeline_orchestrator.py \
    --output data/processed/custom_reddit_data.jsonl
```

#### Limit Number of Examples

```bash
python scripts/reddit_pipeline_orchestrator.py \
    --max-examples 3000
```

#### Adjust Target Range

```bash
python scripts/reddit_pipeline_orchestrator.py \
    --min-target 1500 \
    --max-target 3000
```

#### Dry Run (No Output File)

Test the pipeline without writing output:

```bash
python scripts/reddit_pipeline_orchestrator.py --dry-run
```

#### Custom CSV Directory

```bash
python scripts/reddit_pipeline_orchestrator.py \
    --csv-dir /path/to/custom/csvs
```

## Output Format

The pipeline generates chat-format JSONL suitable for fine-tuning language models.

### Example Entry

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a witty weather assistant who responds with humorous observations, inspired by satirical news headlines."
    },
    {
      "role": "user",
      "content": "What's the weather like?"
    },
    {
      "role": "assistant",
      "content": "Storm warning: Heavy rain expected tonight"
    }
  ],
  "tags": {
    "persona": "neutral",
    "tone": "satirical",
    "domain": ["weather", "humor"],
    "source": "reddit-theonion",
    "reddit_id": "a75a2d",
    "subreddit": "TheOnion",
    "created_utc": 1545089481,
    "url": "https://www.reddit.com/r/TheOnion/comments/a75a2d/...",
    "score": 42,
    "matched_keywords": ["storm", "rain"]
  }
}
```

### Field Descriptions

**messages array:**
- `system`: Defines the assistant persona (witty weather assistant)
- `user`: Varied weather-related query (15+ variations for diversity)
- `assistant`: Cleaned Reddit post title as humorous response

**tags object:**
- `persona`: Always "neutral" for consistency with literary corpus
- `tone`: "satirical" for r/TheOnion, "ironic" for r/nottheonion
- `domain`: Always ["weather", "humor"]
- `source`: Dataset origin identifier
- `reddit_id`: Original Reddit post ID (for deduplication)
- `subreddit`: Source subreddit
- `created_utc`: Unix timestamp of post creation
- `url`: Full Reddit URL for verification
- `score`: Number of comments (engagement proxy)
- `matched_keywords`: Weather keywords found in title

## Pipeline Stages

### 1. CSV Loading
- Reads all CSV files from input directory
- Handles encoding issues (UTF-8 with latin-1 fallback)
- Preserves metadata columns: title, id, created_utc, url, num_comments, subreddit

### 2. Weather Keyword Filtering
- Matches titles against 45+ weather keywords
- Includes literal weather: rain, snow, storm, hurricane, etc.
- Includes seasonal: winter, summer, spring, fall, autumn
- Includes metaphorical: climate, forecast, outlook (for political/economic context)
- Uses whole-word case-insensitive regex matching

### 3. Text Cleaning
- Removes Reddit artifacts: [removed], [deleted], [AutoModerator]
- Strips markdown formatting: **, *, [links](urls)
- Normalizes Unicode: smart quotes → straight quotes, em-dashes → hyphens
- Trims excessive whitespace

### 4. Quality Filtering
- Filters titles shorter than 10 characters
- Removes entries with missing critical metadata (id, subreddit)
- Validates cleaned text contains alphanumeric content

### 5. Balancing & Sampling
- Sorts by num_comments (higher engagement = higher quality)
- Balances across subreddits (targets ~50/50 TheOnion/nottheonion)
- Samples up to max_examples parameter

### 6. JSONL Conversion
- Converts to chat-format with system/user/assistant messages
- Applies source-aware tagging (satirical vs ironic)
- Embeds metadata in tags field
- Varies user messages for training diversity

### 7. Validation
- Validates JSONL format (one JSON object per line)
- Checks message structure (3 messages with correct roles)
- Verifies required tags are present
- Reports final statistics

## Verifying Output Quality

### Check Statistics

The pipeline automatically reports:
- Total examples per subreddit
- Top 10 matched weather keywords
- Average title length
- Metadata coverage percentage
- Final count vs. target range

### Manual Verification

View sample entries:

```bash
head -n 3 data/processed/reddit_humor_weather.jsonl | python -m json.tool
```

Count total entries:

```bash
wc -l data/processed/reddit_humor_weather.jsonl
```

Check for specific keywords:

```bash
grep -i "storm" data/processed/reddit_humor_weather.jsonl | wc -l
```

### Validate JSONL Format

```python
import json

with open('data/processed/reddit_humor_weather.jsonl', 'r') as f:
    for i, line in enumerate(f):
        try:
            entry = json.loads(line)
            assert 'messages' in entry
            assert 'tags' in entry
            assert len(entry['messages']) == 3
        except Exception as e:
            print(f"Error on line {i+1}: {e}")
```

## Expected Output Statistics

For the three default CSV files:

**Input:**
- `cleaned_subreddits.csv`: ~8,000 rows
- `nottheonion_181217_184009.csv`: ~13,000 rows
- `TheOnion_181217_184244.csv`: ~13,000 rows
- **Total**: ~34,000 rows

**After Filtering:**
- Weather keyword matches: ~1,000-1,500 rows (3-5% of total)
- After quality filtering: ~800-1,200 rows
- After balancing to 4,000 max: Target 2,000-4,000 examples

**Final Output:**
- Target range: 2,000-4,000 JSONL examples
- Typical output: ~1,000-1,200 examples (limited by available data)
- Subreddit distribution: ~50/50 TheOnion/nottheonion
- Most common keywords: weather, storm, climate, rain, forecast

## Troubleshooting

### "No CSV files found"

Ensure CSV files exist in the expected directory:

```bash
ls -l data_sources/reddit-theonion/data/*.csv
```

### "Output count below target range"

This is expected if there aren't enough weather-related posts in the source data. The warning is informational only. You can:
- Adjust the target range with `--min-target` and `--max-target`
- Accept fewer examples if quality is maintained
- Add more source CSV files if available

### "Environment validation failed"

Install required dependencies:

```bash
pip install pandas jsonlines
```

Or use requirements file:

```bash
pip install -r requirements-local.txt
```

### Unicode Encoding Errors

The pipeline automatically handles encoding issues with fallback from UTF-8 to latin-1. If issues persist, check the CSV file encoding:

```bash
file data_sources/reddit-theonion/data/*.csv
```

## Integration with Training Pipeline

The output JSONL file is compatible with Hugging Face `datasets` library and standard fine-tuning workflows:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('json', data_files='data/processed/reddit_humor_weather.jsonl')

# Access examples
for example in dataset['train']:
    print(example['messages'])
    print(example['tags'])
```

## Testing

Run the test suite to verify pipeline functionality:

```bash
# Run all Reddit processing tests
python -m unittest discover tests -pattern "test_reddit*.py" -v

# Run specific test groups
python tests/test_reddit_keywords_cleaning.py -v  # Task Group 1
python tests/test_reddit_csv_processing.py -v      # Task Group 2
python tests/test_reddit_jsonl_conversion.py -v    # Task Group 3
python tests/test_reddit_pipeline.py -v            # Task Group 4
python tests/test_reddit_integration.py -v         # Task Group 5
```

## Further Reading

- **Specification**: `agent-os/specs/2025-11-02-reddit-humor-dataset-processing/spec.md`
- **Requirements**: `agent-os/specs/2025-11-02-reddit-humor-dataset-processing/planning/requirements.md`
- **Task Breakdown**: `agent-os/specs/2025-11-02-reddit-humor-dataset-processing/tasks.md`
- **Source Code**: `scripts/reddit_*.py`
- **Tests**: `tests/test_reddit_*.py`
