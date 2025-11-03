# Specification: Instructionalization & Tagging

## Goal
Convert normalized training data into OpenAI-style chat-format JSONL with persona/tone/domain tags and stratified train/validation splits (90/10) for LoRA fine-tuning.

## User Stories
- As a ML engineer, I want persona-tagged chat-format training data so that I can fine-tune models with distinct literary personalities (Twain/Franklin/neutral)
- As a data scientist, I want stratified train/validation splits so that I can evaluate model performance on balanced held-out data with representative tag distributions

## Specific Requirements

**Chat-Format Conversion**
- Load normalized data from `data/processed/training_data_clean.jsonl` (503 items)
- Convert each passage to OpenAI-style messages array with system/user/assistant roles
- Generate persona-aware system messages: Twain ("witty weather assistant inspired by Mark Twain"), Franklin ("wise weather advisor inspired by Benjamin Franklin"), neutral ("helpful weather assistant")
- Create varied user messages using 10-15 weather query templates to avoid overfitting
- Use assistant role for passage text content
- Support both single-turn (default) and multi-turn conversations where beneficial (e.g., passages >300 words)
- Validate chat-format schema: messages array, role/content fields, valid role ordering

**Persona Tag Assignment**
- Extract persona from source metadata: Gutenberg passages with Twain books → `persona: "twain"`
- Franklin books → `persona: "franklin"`
- Reddit posts → `persona: "neutral"`
- Use source tag pattern matching to identify author (search for "twain", "franklin" in source field)
- Handle edge cases with fallback to neutral persona

**Tone Tag Assignment**
- Map Reddit sources to tone: r/TheOnion → `tone: "satirical"`, r/nottheonion → `tone: "ironic"`
- Literary passages with humor keywords → `tone: "humorous"`
- Franklin passages → `tone: "didactic"`
- Preserve existing tone tags from normalization pipeline where already assigned
- Ensure all entries have exactly one tone tag

**Domain Tag Assignment**
- Support multiple domain tags per entry (array format)
- Weather-related content (from matched_keywords) → `domain: "weather"`
- Humor-related content (from matched_keywords or humor tone) → `domain: "humor"`
- Check matched_keywords field to determine applicable domains
- Default to `domain: ["weather"]` if no keywords available

**Metadata Handling**
- Preserve essential provenance metadata: source, matched_keywords
- Exclude unnecessary metadata that bloats file size: reddit_id, created_utc, url (not needed for training)
- Merge persona/tone/domain tags with retained metadata
- Ensure tags field is JSON-serializable and compact

**Stratified Train/Validation Split**
- Implement 90/10 split ratio (approximately 452 train, 50 validation items)
- Use stratified sampling to maintain balanced representation of persona/tone/domain combinations
- Random shuffle before splitting to remove temporal ordering bias
- Set random seed for reproducibility (configurable via CLI)
- Validate that split counts sum to input count

**Output Generation**
- Write train split to `data/processed/train.jsonl` in JSONL format (one JSON per line)
- Write validation split to `data/processed/validation.jsonl`
- Use atomic file operations (tempfile + rename) to prevent corruption
- Validate output files: parse each line as JSON, check messages/tags schema

**Statistics Reporting**
- Generate JSON report: `data/processed/instructionalization_stats_{timestamp}.json`
- Generate Markdown report: `data/processed/instructionalization_stats_{timestamp}.md`
- Include tag distribution counts: persona breakdown, tone breakdown, domain breakdown
- Include split statistics: train count, validation count, stratification validation
- Report multi-turn vs single-turn conversation counts
- Calculate average message length and token estimates

**CLI Interface**
- Support `--input` flag to specify custom input file path (default: training_data_clean.jsonl)
- Support `--output-train` and `--output-val` flags for custom output paths
- Support `--split-ratio` flag to customize split percentage (default: 0.9)
- Support `--seed` flag for random seed (default: 42)
- Support `--dry-run` flag to validate without writing output
- Follow argparse pattern from existing orchestrators

**Environment Validation**
- Check input file exists before processing
- Check output directory is writable
- Validate config files if used for tag mappings
- Display clear error messages for missing dependencies or files
- Use orchestrator validation pattern from normalization pipeline

## Visual Design
No visual assets provided.

## Existing Code to Leverage

**scripts/reddit_jsonl_converter.py**
- Reuse `USER_MESSAGE_TEMPLATES` list (15 weather query variations)
- Follow `create_chat_format_entry()` pattern for messages array construction
- Reuse `determine_tone_tags()` logic for tag determination based on source
- Follow tag merging pattern: `{**tone_tags, **metadata_tags}`
- Use atomic write pattern with tempfile and rename
- Reuse JSONL validation logic from `validate_jsonl_output()`

**scripts/statistics_reporter.py**
- Reuse `calculate_statistics()` for aggregating tag distributions
- Use `write_json_report()` and `write_markdown_report()` functions
- Follow timestamp placeholder replacement pattern in filenames
- Use Markdown table formatting for tag breakdowns

**scripts/data_loader.py**
- Use `load_jsonl_file()` to read normalized training data
- Follow error handling pattern for missing files
- Preserve metadata through loading process

**scripts/normalization_pipeline_orchestrator.py**
- Follow orchestrator pattern with staged pipeline execution
- Reuse environment validation approach (check files, directories, dependencies)
- Follow progress logging pattern for each stage
- Use CLI argument parsing with argparse

**scripts/paths.py**
- Use `DATA_PROCESSED` constant for input/output directory
- Follow path management conventions

## Out of Scope
- LLM-based content rewriting or quality enhancement
- Quality scoring or passage ranking algorithms
- Synthetic data generation (reserved for Roadmap Item 6)
- Model training, evaluation, or inference (Roadmap Items 7-11)
- Manual curation or human-in-the-loop review workflows
- Advanced NLP processing (sentiment analysis, entity extraction, topic modeling)
- Data augmentation techniques (paraphrasing, back-translation, noise injection)
- Custom chat format templates beyond OpenAI specification
- Multi-language support (data already filtered to English-only)
- Real-time or streaming data processing (batch-only pipeline)
