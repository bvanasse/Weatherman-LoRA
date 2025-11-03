# Specification: Reddit Humor Dataset Processing

## Goal
Process existing Reddit CSV datasets from r/TheOnion and r/nottheonion to extract 2,000-4,000 weather-related humorous post titles and convert them into chat-format JSONL training data for the Weatherman-LoRA model.

## User Stories
- As a data scientist, I want to extract weather-related humor examples from Reddit so that I can train the model to respond with satirical weather commentary
- As a model trainer, I want the Reddit data in chat-format JSONL so that it integrates seamlessly with the existing training pipeline

## Specific Requirements

**CSV Processing Pipeline**
- Process all three CSV files: `cleaned_subreddits.csv`, `nottheonion_181217_184009.csv`, `TheOnion_181217_184244.csv`
- Extract title column as primary content from each CSV row
- Read CSVs with pandas to handle encoding issues and malformed rows
- Preserve metadata columns: created_utc, url, id, num_comments, subreddit, timestamp
- Track statistics for each source file (total rows, filtered rows, final count)

**Expanded Weather Keyword Filtering**
- Extend existing 21-term keyword list from `keyword_matcher.py` with ~20 additional terms
- Add seasonal terms: winter, summer, spring, fall, autumn, seasonal
- Add extreme weather: heatwave, blizzard, wildfire, avalanche, monsoon, typhoon
- Add metaphorical weather: weathering, forecast, outlook, climate (in political/economic context)
- Use whole-word case-insensitive regex matching to avoid partial matches
- Include both literal weather mentions and metaphorical usage
- Filter titles to only those containing at least one weather keyword match

**Text Cleaning and Normalization**
- Remove Reddit-specific artifacts: "[removed]", "[deleted]", "[AutoModerator]"
- Strip URL remnants and markdown formatting (bold, italics, links)
- Normalize Unicode to ASCII: smart quotes to straight quotes, em-dashes to hyphens
- Preserve original meaning and humor while cleaning artifacts
- Trim excessive whitespace and normalize line breaks
- Validate cleaned text is non-empty and meets minimum length (10+ characters)

**Chat-Format JSONL Conversion**
- Convert each Reddit title into a conversational training example with messages array
- System message: "You are a witty weather assistant who responds with humorous observations, inspired by satirical news headlines."
- User message: Generic weather query (e.g., "What's the weather like?" or "Give me the forecast")
- Assistant message: Original Reddit title (cleaned) as the humorous response
- Vary user messages to create diverse training examples and avoid overfitting
- Each conversation is one complete JSONL line (one JSON object per line)

**Source-Aware Tagging Schema**
- Differentiate r/TheOnion posts with `tone: "satirical"` tag
- Differentiate r/nottheonion posts with `tone: "ironic"` tag
- Apply shared tags to all posts: `domain: "weather"`, `domain: "humor"`
- Use `persona: "neutral"` for consistency with literary corpus
- Include `source: "reddit-theonion"` or `source: "reddit-nottheonion"` to identify dataset origin
- Align tag structure with existing literary corpus for downstream compatibility

**Metadata Preservation in Tags Field**
- Embed Reddit metadata within JSONL `tags` object for provenance tracking
- Include `reddit_id` for post ID, `subreddit` for source subreddit
- Store `created_utc` timestamp for temporal analysis
- Preserve `url` for manual review and verification
- Use `num_comments` as proxy for engagement/quality score (store as `score`)
- Enable future deduplication and filtering based on metadata

**Quality Filtering and Targeting**
- Target 2,000-4,000 final examples after all filtering stages
- Aim for 3,500-4,000 after keyword filtering to allow buffer for quality checks
- Filter out titles shorter than 10 characters (too cryptic or low-quality)
- Prioritize titles with higher num_comments as quality signal when sampling
- Balance examples across both subreddits (aim for ~50/50 split if possible)
- Report statistics on filtered vs retained examples at each stage

**Statistics Reporting and Validation**
- Calculate and display total examples per subreddit source
- Report keyword distribution (top 10 most common matched weather terms)
- Track filtering stages: CSV rows → keyword matches → cleaned → final JSONL
- Compute average title length and metadata coverage
- Validate output JSONL is correctly formatted (one JSON object per line)
- Display final dataset size and confirm it meets 2,000-4,000 target range

## Visual Design
No visual assets provided for this specification.

## Existing Code to Leverage

**`scripts/keyword_matcher.py`**
- Contains 21 weather keywords (weather, rain, storm, thunder, lightning, cloud, sun, wind, climate, temperature, snow, fog, drought, hurricane, tornado, flood, heat, cold, frost, dew, hail)
- Provides `build_keyword_pattern()` for whole-word regex matching to avoid partial matches
- Has `find_keyword_matches()` for case-insensitive keyword detection with position tracking
- Implements `find_all_matches()` which returns context_type and relevance_score
- Reuse pattern builder and extend keyword list for Reddit filtering

**`scripts/serialize_passages.py`**
- Shows `generate_passage_id()` pattern for creating unique IDs (adapt for Reddit post IDs)
- Implements `calculate_statistics()` for metadata summary with keyword distribution and counts
- Uses atomic file write pattern: write to temp file, then rename to final path for safety
- Validates JSON output after writing by reloading and checking structure
- Adapt statistics calculation for JSONL format and Reddit-specific metadata

**`scripts/paths.py`**
- Provides `REDDIT_DATA` constant pointing to `data_sources/reddit-theonion/data` for input
- Defines `DATA_PROCESSED` for output location to maintain consistency
- Includes `ensure_dirs_exist()` to create output directories before writing
- Use centralized path management to avoid hardcoded paths in script
- Follow existing project structure for data organization

**`scripts/collect_literary_corpus.py`**
- Models end-to-end pipeline orchestration with validate → process → output flow
- Uses argparse for CLI arguments (max examples, output file path, dry-run mode)
- Implements progress reporting with step-by-step validation checks
- Shows pattern for multi-stage validation: environment → inputs → outputs
- Adapt orchestration pattern for CSV processing pipeline

**`scripts/config_loader.py`**
- Shows JSON loading pattern with error handling for malformed files
- Filters out comment fields starting with underscore from config
- Provides deep merge pattern for combining base config with overrides
- Can be used if creating a config file for weather keywords and filtering parameters
- Reference for structured configuration management if needed

## Out of Scope
- Processing Reddit post body text or comments (titles only)
- Downloading additional Reddit data via PRAW or Pushshift API
- Advanced sentiment analysis or toxicity scoring beyond basic filtering
- Manual curation, human review, or quality assessment of individual examples
- Cross-dataset deduplication with literary corpus (handled in later pipeline stage)
- JSONL format validation for training compatibility (assumed handled by training scripts)
- Integration with model training or adapter configuration
- Creating synthetic variations or augmentations of Reddit titles
- Analyzing temporal trends or subreddit-specific humor patterns
- Generating chat-format examples with multi-turn conversations (single turn only)
