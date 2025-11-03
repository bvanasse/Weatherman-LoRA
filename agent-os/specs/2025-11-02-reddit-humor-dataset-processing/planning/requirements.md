# Spec Requirements: Reddit Humor Dataset Processing

## Initial Description
Process existing `data_sources/reddit-theonion/` CSVs to extract weather-related posts from r/TheOnion and r/nottheonion, filter by weather keywords, clean HTML/metadata artifacts, and create 2,000-4,000 labeled humor examples.

**Size Estimate:** S (Small)

**Source:** Product Roadmap Item #3

## Requirements Discussion

### First Round Questions

**Q1:** I assume the existing CSVs in `data_sources/reddit-theonion/` already contain post titles and text content. Should the extraction focus on both title and body text, or primarily titles (which tend to be the punchlines)?

**Answer:** Let's focus on the titles for now.

**Q2:** For weather keyword filtering, I'm thinking we should use a broad list including obvious terms (rain, snow, storm, temperature, forecast, weather) plus seasonal/climate terms (winter, summer, hurricane, tornado, heatwave, blizzard). Should we also include metaphorical weather usage (e.g., "political storm", "economic climate") or strictly literal weather mentions?

**Answer:** Keep the filtering broad and include metaphorical weather usage as well.

**Q3:** I assume the cleaning pipeline should remove Reddit-specific artifacts like "[removed]", "[deleted]", AutoModerator messages, and URL remnants. Should we also normalize Unicode characters (smart quotes, em-dashes) to ASCII equivalents, or preserve them?

**Answer:** Yes remove reddit-specific artifacts and normalize unicode characters.

**Q4:** For the 2,000-4,000 target examples, I'm thinking we should aim for the higher end (3,500-4,000) to allow for quality filtering later. Should we prioritize highly-upvoted posts as quality signals, or sample across different score ranges to capture diverse humor styles?

**Answer:** Use your best judgement, this seems like an okay plan.

**Q5:** I assume we need to preserve source metadata (subreddit, post ID, score, timestamp) for provenance tracking and potential deduplication with other datasets. Should this metadata be stored in the JSONL tags field, or kept separate?

**Answer:** Use your best judgement, and look for compatibility with the rest of the training data as much as possible.

**Q6:** For labeling, should all extracted examples be tagged as `tone: "humorous"` and `tone: "satirical"`, or do we need more granular tone labels (sarcastic, ironic, absurdist)?

**Answer:** Use your best judgement here as well.

**Q7:** I'm thinking the output format should follow the project's JSONL standard with chat-format messages. Should weather-related Reddit posts be framed as user queries with assistant responses, or just as examples for the model to learn tone from?

**Answer:** I'm not sure. Probably what makes the most sense for this project.

**Q8:** Are there specific types of posts we should explicitly exclude (e.g., purely political without weather connection, toxic/controversial content even if funny, posts that are too short/cryptic)?

**Answer:** Use your best judgement.

### Existing Code to Reference

**Similar Features Identified:**

Based on codebase exploration, the following existing utilities should be referenced and reused:

- **Keyword Matching**: `scripts/keyword_matcher.py`
  - Contains existing weather keyword list (21 terms: weather, rain, storm, thunder, lightning, cloud, sun, wind, climate, temperature, snow, fog, drought, hurricane, tornado, flood, heat, cold, frost, dew, hail)
  - Provides `build_keyword_pattern()` for regex-based whole-word matching
  - Has `find_keyword_matches()` for case-insensitive keyword detection
  - Implements relevance scoring based on keyword matches

- **Data Serialization**: `scripts/serialize_passages.py`
  - Shows pattern for generating unique IDs (`generate_passage_id()`)
  - Implements `calculate_statistics()` for metadata summary
  - Uses atomic file writes (temp file + rename pattern)
  - Validates JSON output after writing

- **Path Management**: `scripts/paths.py`
  - Centralizes all project paths
  - Provides `REDDIT_DATA` constant pointing to `data_sources/reddit-theonion/data`
  - Defines `DATA_PROCESSED` for output location
  - Includes `ensure_dirs_exist()` utility

- **Pipeline Orchestration**: `scripts/collect_literary_corpus.py`
  - Shows pattern for end-to-end data pipeline
  - Implements validation steps (environment, downloads, outputs)
  - Uses argparse for CLI configuration
  - Provides progress reporting and statistics

- **Existing Data Format**: `data/processed/gutenberg_passages.json`
  - Literary corpus stored in JSON format with passages array
  - Each passage has: text, word_count, matched_keywords, context_type, relevance_score
  - Includes metadata fields: passage_id, author_name, book_title, genre_tags, source_url, extraction_date
  - Has top-level metadata section with statistics

- **CSV Structure**: `data_sources/reddit-theonion/data/`
  - Three CSV files: `cleaned_subreddits.csv`, `nottheonion_181217_184009.csv`, `TheOnion_181217_184244.csv`
  - Columns: created_utc, url, id, num_comments, title, subreddit, timestamp
  - Title field contains the primary content to extract

### Follow-up Questions

**Follow-up 1:** Output Format Alignment - I found that the existing literary corpus stores passages in raw text format with metadata. However, the tech stack documentation mentions the training pipeline expects JSONL chat-format with `messages` arrays. Should the Reddit humor data match the literary corpus format (raw text + metadata in JSON), or be converted directly to chat-format JSONL now?

**Answer:** Option B - Convert directly to chat-format JSONL now, with Reddit titles framed as training examples.

**Follow-up 2:** Weather Keywords - I found existing `keyword_matcher.py` with 21 weather keywords. Should I expand this list to include seasonal/metaphorical terms as mentioned?

**Answer:** Expand it as necessary.

**Follow-up 3:** Tone Granularity - Looking at existing literary passages with tags like `["humor", "satire", "adventure"]`, should Reddit posts from r/TheOnion (satirical news) and r/nottheonion (absurd real news) use the same tags or differentiate between sources?

**Answer:** Try to have some alignment between all of the data's tags while allowing there to be differentiation where necessary.

**Follow-up 4:** CSV Selection - I see three CSV files exist. Should I process all three, or just the two dated files?

**Answer:** Process all three as necessary.

## Visual Assets

### Files Provided:
No visual files found (bash check performed on `planning/visuals/` folder).

### Visual Insights:
No visual assets provided.

## Requirements Summary

### Functional Requirements

**Data Input:**
- Process all three CSV files in `data_sources/reddit-theonion/data/`:
  - `cleaned_subreddits.csv`
  - `nottheonion_181217_184009.csv`
  - `TheOnion_181217_184244.csv`
- Extract post titles as primary content (CSV column: `title`)
- Preserve metadata: created_utc, url, post id, num_comments, subreddit, timestamp

**Keyword Filtering:**
- Expand existing 21-term weather keyword list from `keyword_matcher.py`
- Add seasonal terms: winter, summer, spring, fall, autumn
- Add extreme weather: heatwave, blizzard, wildfire, avalanche, monsoon
- Add metaphorical weather: "political storm", "economic climate", "weathering", etc.
- Use case-insensitive whole-word matching (reuse existing pattern builder)
- Include both literal and metaphorical weather usage

**Text Cleaning:**
- Remove Reddit-specific artifacts:
  - "[removed]", "[deleted]"
  - AutoModerator messages
  - URL remnants and markdown formatting
- Normalize Unicode characters to ASCII equivalents:
  - Smart quotes → straight quotes
  - Em-dashes → hyphens
  - Special punctuation → ASCII equivalents
- Preserve original meaning while cleaning

**Quality Filtering:**
- Target 2,000-4,000 examples (aim for 3,500-4,000 for buffer)
- Prioritize highly-upvoted posts as quality signals
- Filter out posts that are too short (e.g., < 10 characters)
- Exclude posts without clear weather connection
- Apply basic toxicity/safety filters if needed

**Output Format - Chat-Format JSONL:**
- Convert to chat-format JSONL for direct training use
- Frame Reddit titles as conversational examples:
  - System message: Persona/style instruction
  - User message: Weather-related query or context
  - Assistant message: Humorous/satirical response in title style
- Each line is a complete JSON object with `messages` array
- Include metadata in `tags` field for provenance and filtering

**Tagging Strategy:**
- Align with existing literary corpus tags where possible
- Use consistent base tags: `persona`, `tone`, `domain`, `source`
- Source-specific differentiation:
  - r/TheOnion posts: `tone: "satirical"`
  - r/nottheonion posts: `tone: "ironic"` or `tone: "absurdist"`
  - Both can have `tone: "humorous"` as secondary tag
- All posts tagged with `domain: "weather"` and `domain: "humor"`
- Include `source: "reddit-theonion"` or `source: "reddit-nottheonion"`

**Metadata Preservation:**
- Store in JSONL `tags` field for compatibility with training pipeline:
  - `reddit_id`: Original post ID
  - `subreddit`: Source subreddit
  - `score`: Post upvote score (if available via num_comments proxy)
  - `created_utc`: Original timestamp
  - `url`: Reddit post URL
- Enable deduplication with other datasets
- Support provenance tracking and filtering

### Reusability Opportunities

**Components to Reuse:**
- `keyword_matcher.py`: Extend weather keyword list and reuse pattern matching functions
- `serialize_passages.py`: Adapt ID generation and statistics calculation for Reddit data
- `paths.py`: Use REDDIT_DATA constant and ensure output directories exist
- `collect_literary_corpus.py`: Model pipeline orchestration and validation patterns

**Backend Patterns:**
- CSV reading with pandas (established pattern in project)
- Atomic file writes (temp file + rename)
- Statistics calculation and reporting
- Progress tracking and validation

**New Utilities to Create:**
- Reddit-specific text cleaning function
- Chat-format JSONL converter (framing Reddit titles as conversations)
- Expanded weather keyword list (add ~20 new terms)
- CSV-to-JSONL pipeline script

### Scope Boundaries

**In Scope:**
- Processing all three existing CSV files in `data_sources/reddit-theonion/data/`
- Weather keyword filtering with expanded keyword list (40+ terms)
- Text cleaning (Reddit artifacts, Unicode normalization)
- Conversion to chat-format JSONL for training
- Quality filtering targeting 2,000-4,000 examples
- Metadata preservation in JSONL tags field
- Source-aware tagging (differentiate r/TheOnion vs r/nottheonion)
- Statistics reporting (keyword distribution, count by subreddit, etc.)

**Out of Scope:**
- Processing Reddit post body text (focus on titles only)
- Downloading additional Reddit data from API
- Advanced sentiment analysis or toxicity detection
- Manual curation or human review of examples
- Deduplication across datasets (handled in later pipeline stage)
- Training data format validation (assumed handled by training pipeline)
- Integration with model training (this is data preparation only)

### Technical Considerations

**Integration Points:**
- Input: Existing CSV files in `data_sources/reddit-theonion/data/`
- Output: JSONL file(s) in `data/processed/` (follow naming convention)
- Reuse: Import utilities from `scripts/keyword_matcher.py`, `scripts/paths.py`, `scripts/serialize_passages.py`
- Tech stack: Python 3.10+, pandas for CSV reading, jsonlines for JSONL writing

**Existing System Constraints:**
- Must align with chat-format JSONL spec from tech stack documentation
- Must use consistent tagging schema with literary corpus
- Must follow project path conventions from `paths.py`
- Should produce statistics compatible with training pipeline expectations

**Technology Preferences:**
- Python 3.10+ (project standard)
- pandas for CSV processing (established pattern)
- jsonlines library for JSONL output
- Reuse regex patterns from keyword_matcher.py
- Follow atomic write pattern from serialize_passages.py

**Similar Code Patterns to Follow:**
- Pipeline orchestration: `collect_literary_corpus.py` pattern
- Keyword matching: `keyword_matcher.py` whole-word regex patterns
- Output serialization: `serialize_passages.py` atomic writes and statistics
- Path management: `paths.py` centralized constants
- Validation: Environment checks, input validation, output verification

### Chat-Format JSONL Structure

Based on tech stack documentation, output should follow this schema:

```json
{
  "messages": [
    {"role": "system", "content": "You are a witty weather assistant who responds with humorous observations, inspired by satirical news headlines."},
    {"role": "user", "content": "What's the weather forecast looking like?"},
    {"role": "assistant", "content": "[Reddit post title rephrased as humorous weather commentary]"}
  ],
  "tags": {
    "persona": "neutral",
    "tone": "satirical",
    "domain": "weather",
    "source": "reddit-theonion",
    "reddit_id": "a75a2d",
    "subreddit": "TheOnion",
    "score": 0,
    "created_utc": 1545089481,
    "url": "https://www.reddit.com/r/TheOnion/comments/a75a2d/..."
  }
}
```

Note: Exact framing of titles as conversations will be determined during implementation based on what makes most semantic sense for training.
