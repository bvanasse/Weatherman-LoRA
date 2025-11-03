# Specification: Data Normalization & Deduplication Pipeline

## Goal
Create a robust data cleaning pipeline that normalizes text, removes duplicates using MinHash/LSH, filters for English-only content, applies safety checks, and generates comprehensive quality reports, producing a single unified JSONL output from multiple input sources.

## User Stories
- As a data scientist, I want to process multiple data sources (Gutenberg, Reddit) into a single clean dataset so that I have high-quality training data for LoRA fine-tuning
- As a researcher, I want detailed quality statistics reports so that I can understand what was filtered and verify data composition before GPU training

## Specific Requirements

**Unicode Normalization**
- Apply unicode normalization to all text content using NFC (Canonical Decomposition followed by Canonical Composition) form for maximum stability and compatibility
- NFC chosen over NFKC to preserve semantic distinctions (e.g., superscripts, fractions) that may appear in literary texts
- Handle smart quotes, em-dashes, non-breaking spaces, and other common Unicode artifacts
- Process text before deduplication to ensure consistent hashing

**MinHash/LSH Deduplication**
- Use `datasketch` library to implement MinHash with LSH (Locality Sensitive Hashing)
- Set Jaccard similarity threshold to 0.8 to detect near-duplicates
- Remove duplicates entirely (no flagging or quarantine)
- Apply deduplication after normalization but before language filtering
- Track count of removed duplicates for statistics report
- Consider each text passage as a single document for hashing

**Language Filtering**
- Use `fasttext` library for fast, accurate language detection
- Remove non-English content entirely (no flagging or quarantine)
- Apply after deduplication to reduce processing volume
- Track language distribution before filtering for statistics
- Handle mixed-language texts by classifying entire passage

**Safety Filtering**
- Use OpenAI moderation API to detect toxicity, NSFW content, and other unsafe content
- Reject flagged content entirely (no quarantine dataset)
- Batch API calls efficiently to avoid rate limits and reduce latency
- Implement retry logic with exponential backoff for API failures
- Apply safety filtering after language filtering
- Track count and categories of rejected content for audit trail

**Flexible Input Processing**
- Support reading from `data/processed/gutenberg_passages.json` (JSON format)
- Support reading from `data/processed/reddit_humor_weather.jsonl` (JSONL format)
- Handle both JSON and JSONL parsing gracefully
- Enable processing files together in single run, separately across runs, or incrementally with new sources
- Always produce single unified output regardless of input configuration

**Metadata Preservation**
- Preserve all source metadata tags: persona, tone, domain, source, reddit_id, created_utc, etc.
- Maintain tags structure throughout all pipeline stages
- Handle missing or malformed metadata gracefully with defaults
- Add pipeline metadata: normalization_version, dedup_threshold, filters_applied

**Quality Statistics Reports**
- Generate both JSON and Markdown format reports with identical data
- Include metrics: total items processed, duplicates removed count, language distribution, safety filter rejections by category, character length distributions, final dataset size
- Timestamp reports with ISO 8601 format for versioning across runs
- Save to `data/processed/pipeline_stats_[timestamp].json` and `data/processed/pipeline_stats_[timestamp].md`

**Idempotent Pipeline Design**
- Safe to re-run without data corruption or double-processing
- Track processed items by source ID to avoid re-adding same data
- Use atomic file writes (temp file + rename) for output
- Validate output after writing to ensure integrity

**Configuration-Driven Architecture**
- Define all paths in `configs/paths_config.json`
- Define thresholds (dedup=0.8, min_length, etc.) in dedicated pipeline config
- Support environment variable overrides for remote execution
- Validate configuration on startup before processing

**Comprehensive Testing**
- Unit tests for each processing stage (normalize, dedup, language, safety)
- Integration tests for end-to-end pipeline with sample data
- Test error handling (API failures, malformed input, missing files)
- Test idempotency by running pipeline twice on same data
- Follow pytest patterns from existing test suite

## Visual Design
No visual assets provided.

## Existing Code to Leverage

**scripts/reddit_pipeline_orchestrator.py**
- Orchestrator pattern with staged pipeline execution (validate → process → convert → validate output)
- Environment validation checking input files, output directories, and dependencies
- Comprehensive statistics printing with formatting
- CLI argument parsing with argparse for flexible execution
- Dry-run mode for testing without output

**scripts/reddit_text_processing.py**
- Unicode normalization using NFKD with manual character replacements
- Text cleaning for Reddit artifacts (markdown, URLs, special characters)
- Regex-based pattern matching for keyword detection
- Validation functions for minimum quality standards

**scripts/reddit_jsonl_converter.py**
- JSONL conversion with atomic writes (tempfile + rename pattern)
- Chat-format message structure creation
- Source-aware metadata tagging based on origin
- Output validation checking JSON structure and required fields
- Sample entry printing for verification

**scripts/paths.py**
- Centralized path management with constants for all directories
- Environment variable override support (WEATHERMAN_BASE_DIR)
- Utility function to create all required directories (ensure_dirs_exist)
- Base directory resolution for local vs remote environments

**scripts/config_loader.py**
- YAML and JSON configuration loading with validation
- Deep merge utility for applying overrides
- Nested value access using dot notation (e.g., 'lora.r')
- Required fields validation with clear error messages

## Out of Scope
- Manual review workflows or quarantine datasets for rejected content
- Emoji/emoticon filtering (basic preservation acceptable)
- Aggressive length filtering beyond basic sanity checks (e.g., < 10 chars)
- Code snippet detection or special handling
- Real-time or streaming processing (batch only)
- GUI, web interface, or interactive modes
- Advanced NLP preprocessing like lemmatization, POS tagging, or named entity recognition
- Parallel/distributed processing across multiple machines
- Custom safety filter backends besides OpenAI moderation API
- Incremental deduplication tracking across pipeline runs
- Quality scoring or ranking of passages by relevance
