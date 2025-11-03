# Spec Requirements: Data Normalization & Deduplication Pipeline

## Initial Description
Data Normalization & Deduplication Pipeline — Implement cleaning pipeline with unicode normalization, MinHash/LSH deduplication (threshold 0.8), language filtering (English only), safety filters (toxicity/NSFW removal), and generate quality statistics report

## Requirements Discussion

### First Round Questions

**Q1:** I assume this pipeline should process the outputs from the previous two roadmap items (Literary Corpus Collection and Reddit Humor Dataset Processing), meaning input files are in `data/raw/` or similar, and cleaned outputs go to `data/processed/` as JSONL. Is that correct, or should I use different paths?
**Answer:** Yes that assumption is correct.

**Q2:** For MinHash/LSH deduplication, I'm thinking we use the `datasketch` library (mentioned in your tech stack) with Jaccard similarity threshold 0.8 to detect near-duplicates. Should we remove duplicates entirely, or flag them for manual review?
**Answer:** Remove dupes entirely.

**Q3:** I assume unicode normalization should use NFKC form (canonical decomposition + compatibility composition) to handle various character encodings from web scraping. Is that the right approach, or do you prefer NFC or a different form?
**Answer:** Whatever seems to be the best selection for the stability of the project.

**Q4:** For language filtering (English only), I'm thinking `fasttext` (as mentioned in your tech stack) for fast, accurate language detection. Should we remove non-English text entirely, or flag it for review?
**Answer:** Remove non-English text entirely.

**Q5:** For safety filters (toxicity/NSFW), I assume we should use an API-based approach like OpenAI's moderation API (mentioned in your eval tools) rather than local models for consistency. Should we reject flagged content entirely, or create a quarantine dataset for manual review?
**Answer:** Reject flagged content entirely.

**Q6:** I'm thinking the quality statistics report should include: total items processed, duplicates removed count, language distribution, safety filter removals, character length distributions, and final dataset size. Should this be a JSON file, Markdown report, or both?
**Answer:** Both.

**Q7:** Should the pipeline be idempotent (can re-run safely), and should it preserve the original source metadata tags (persona, tone, domain) from the collected datasets?
**Answer:** Yes.

**Q8:** What should we explicitly exclude? For example: should we skip emoji/emoticon filtering, skip handling of code snippets in text, or avoid aggressive length filtering at this stage?
**Answer:** Make your best judgement.

### Existing Code to Reference
[Based on user's response about similar features]

**Similar Features Identified:**
- **Reddit processing pipeline architecture**:
  - `scripts/reddit_pipeline_orchestrator.py` - Orchestrator pattern for pipeline execution
  - `scripts/reddit_text_processing.py` - Modular text processing functions
  - `scripts/reddit_csv_processor.py` - Input data processing
  - `scripts/reddit_jsonl_converter.py` - JSONL output formatting
- **Configuration patterns**:
  - `configs/paths_config.json` - Path configuration management
  - `configs/gutenberg_books.json` - Data source configuration
- **Testing patterns**:
  - `tests/test_reddit_*.py` - Comprehensive pytest test suite with unit, integration tests
- **Utility scripts**:
  - `scripts/paths.py` - Path management utilities
  - `scripts/config_loader.py` - Configuration loading utilities

**Input Data Files:**
- `data/processed/gutenberg_passages.json` - Literary corpus output
- `data/processed/reddit_humor_weather.jsonl` - Reddit humor dataset output

### Follow-up Questions

**Follow-up 1:** I found existing Reddit processing pipeline code with a nice structure: `scripts/reddit_pipeline_orchestrator.py`, `scripts/reddit_text_processing.py`, and comprehensive tests in `tests/test_reddit_*.py`. Should the new normalization & deduplication pipeline follow this same architectural pattern (orchestrator + modular processing scripts + pytest test suite)?
**Answer:** [Implied yes from context]

**Follow-up 2:** For input data, I can see two processed files already exist: `data/processed/gutenberg_passages.json` (from Literary Corpus Collection) and `data/processed/reddit_humor_weather.jsonl` (from Reddit Humor Dataset Processing). Should the normalization pipeline process BOTH of these files together, or process them separately and then merge? And should the final output be a single unified `data/processed/training_data_clean.jsonl` file (or similar name)?
**Answer:** We should be able to process these files together, separately, add new data and still produce a single unified whole.

## Visual Assets

### Files Provided:
No visual assets provided.

### Visual Insights:
N/A

## Requirements Summary

### Functional Requirements

**Core Pipeline Processing:**
- Unicode normalization of all text content (use best-practice normalization form for stability)
- MinHash/LSH deduplication using `datasketch` library with Jaccard similarity threshold 0.8
- Language detection using `fasttext` with removal of non-English content
- Safety filtering using OpenAI moderation API with rejection of flagged content (toxicity/NSFW)
- Metadata preservation (persona, tone, domain tags) throughout pipeline

**Input/Output Flexibility:**
- Support processing multiple input files together in a single run
- Support processing input files separately across multiple runs
- Support adding new data sources incrementally
- Always produce a single unified output file regardless of input configuration
- Read from `data/processed/` for input files (gutenberg_passages.json, reddit_humor_weather.jsonl)
- Write cleaned output to `data/processed/` as JSONL format

**Quality Reporting:**
- Generate both JSON and Markdown statistics reports
- Include metrics: total items processed, duplicates removed count, language distribution, safety filter removals, character length distributions, final dataset size
- Reports should be versioned/timestamped for tracking across multiple runs

**Pipeline Characteristics:**
- Idempotent design: safe to re-run without data corruption
- Modular architecture: separate processing stages for maintainability
- Configurable: paths and thresholds defined in config files
- Testable: comprehensive pytest test suite

### Reusability Opportunities

**Architecture Patterns:**
- Follow orchestrator pattern from `scripts/reddit_pipeline_orchestrator.py`
- Use modular processing functions similar to `scripts/reddit_text_processing.py`
- Implement JSONL output handling like `scripts/reddit_jsonl_converter.py`

**Configuration Management:**
- Use path configuration pattern from `configs/paths_config.json`
- Reference `scripts/config_loader.py` for loading configuration

**Testing Approach:**
- Follow pytest patterns from `tests/test_reddit_*.py`
- Include unit tests, integration tests, and end-to-end pipeline tests

**Utility Functions:**
- Reference `scripts/paths.py` for path management utilities

### Scope Boundaries

**In Scope:**
- Unicode normalization (automatic selection of best-practice form)
- MinHash/LSH deduplication at 0.8 threshold
- Language filtering (English only, complete removal of non-English)
- Safety filtering (toxicity/NSFW via OpenAI API, complete rejection)
- Quality statistics in both JSON and Markdown formats
- Metadata preservation from source datasets
- Idempotent pipeline execution
- Flexible input handling (process together, separately, or incrementally)
- Single unified JSONL output file
- Configuration-driven design
- Comprehensive test suite

**Out of Scope:**
- Manual review workflows for rejected content (automatic removal only)
- Emoji/emoticon filtering (defer to developer's best judgment during implementation)
- Aggressive length filtering at this stage (basic sanity checks acceptable)
- Code snippet detection/handling (keep as-is unless problematic)
- Real-time processing or streaming (batch processing only)
- GUI or web interface for pipeline execution
- Advanced NLP preprocessing (lemmatization, POS tagging, etc.)

**Future Enhancements:**
- Configurable deduplication thresholds per data source
- Pluggable safety filter backends (alternatives to OpenAI API)
- Parallel processing for large datasets
- Incremental deduplication (track seen items across runs)
- Quality scoring/ranking of passages

### Technical Considerations

**Integration Points:**
- Input: `data/processed/gutenberg_passages.json` (JSON format)
- Input: `data/processed/reddit_humor_weather.jsonl` (JSONL format)
- Output: `data/processed/training_data_clean.jsonl` (unified JSONL)
- Config: `configs/paths_config.json` for path definitions
- External API: OpenAI moderation endpoint for safety filtering

**Dependencies:**
- `datasketch` for MinHash/LSH deduplication
- `fasttext` or `langdetect` for language detection
- OpenAI API client for moderation
- `pandas` for data manipulation and statistics
- `jsonlines` for efficient JSONL processing
- `pytest` for testing framework

**Data Format Handling:**
- Input formats: both JSON and JSONL (need to handle both)
- Output format: JSONL only (single file)
- Preserve metadata tags structure from input
- Handle missing or malformed metadata gracefully

**Performance Considerations:**
- Deduplication with MinHash/LSH for efficiency on large datasets
- Batch API calls to OpenAI moderation to avoid rate limits
- Memory-efficient JSONL streaming for large files
- Progress logging for long-running operations

**Error Handling:**
- Graceful handling of API failures (retry logic for OpenAI moderation)
- Validation of input file formats
- Clear error messages for configuration issues
- Logging of items rejected at each stage for auditing
