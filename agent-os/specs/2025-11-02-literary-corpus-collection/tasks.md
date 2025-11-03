# Task Breakdown: Literary Corpus Collection

## Overview
Total Tasks: 4 task groups with 23 sub-tasks

## Task List

### Data Download Infrastructure

#### Task Group 1: Project Gutenberg Download System
**Dependencies:** None (leverages existing environment from Roadmap Item 1)

- [x] 1.0 Complete Gutenberg download infrastructure
  - [x] 1.1 Create Gutenberg downloader script
    - Script name: `scripts/download_gutenberg.py`
    - Import paths from `scripts/paths.py` (DATA_RAW constant)
    - Use requests library for HTTP downloads
    - Support direct Gutenberg URLs (gutenberg.org/files/{id}/{id}-0.txt format)
    - Implement exponential backoff retry logic (3 retries, 2/4/8 second delays)
    - Add progress logging for download status
    - Cache check: skip download if file already exists
  - [x] 1.2 Implement Gutenberg header/footer removal
    - Remove standard "*** START OF" and "*** END OF" markers
    - Strip project info, license text, and donation requests
    - Preserve actual book content between markers
    - Handle edge cases: missing markers, malformed headers
  - [x] 1.3 Define book metadata configuration
    - Create `configs/gutenberg_books.json` with book list
    - Mark Twain books: IDs 74, 76, 86, 245 with titles and publication years
    - Benjamin Franklin books: IDs 20203 (or 148 fallback), 57795 with titles
    - Include genre tags: Twain (humor, satire, adventure), Franklin (autobiography, wisdom)
  - [x] 1.4 Implement download orchestration
    - Load book metadata from config
    - Create `data/raw/gutenberg/` subdirectory if needed
    - Download all 6 books sequentially
    - Save with naming: `{author_last}_{book_slug}.txt`
    - Log success/failure for each book
    - Generate download summary report
  - [x] 1.5 Test download system
    - Run download script on one test book (smallest file)
    - Verify file saved to correct location
    - Confirm header/footer removed
    - Check retry logic with simulated network error
    - Validate all 6 books download successfully

**Acceptance Criteria:**
- All 6 books downloaded to `data/raw/gutenberg/`
- Headers/footers removed, clean text preserved
- Download script handles network failures gracefully
- Files named consistently: `twain_tom_sawyer.txt`, etc.
- Script can re-run without re-downloading existing files

### Passage Extraction Engine

#### Task Group 2: Keyword-Based Passage Extraction
**Dependencies:** Task Group 1

- [x] 2.0 Complete passage extraction system
  - [x] 2.1 Create keyword matching module
    - File: `scripts/keyword_matcher.py`
    - Weather keywords list (21 terms): weather, rain, storm, thunder, lightning, cloud, sun, wind, climate, temperature, snow, fog, drought, hurricane, tornado, flood, heat, cold, frost, dew, hail
    - Humor keywords list (10 terms): joke, wit, laugh, humor, comic, amusing, funny, satire, irony, jest
    - Case-insensitive whole-word regex matching (avoid "weather" in "leathern")
    - Return match positions and matched keywords
  - [x] 2.2 Implement NLTK text processing
    - Download required NLTK data: punkt tokenizer
    - Paragraph tokenization (split on double newlines)
    - Sentence tokenization within paragraphs
    - Word count calculation
    - Preserve original formatting and paragraph boundaries
  - [x] 2.3 Create passage extractor with sliding window
    - File: `scripts/extract_passages.py`
    - Find keyword matches in text
    - Extract 2-3 paragraphs around match (1 before, match, 1-2 after)
    - Target length: 200-500 words (min 100, max 600)
    - Score passages: weather+humor = 2 points, single keyword = 1 point
    - Avoid overlapping passages (track extracted ranges)
  - [x] 2.4 Implement metadata extraction
    - Parse chapter markers from text (Chapter N, CHAPTER N patterns)
    - Extract character names from dialogue (detect quoted speech)
    - Track matched keywords for each passage
    - Classify context_type: "weather", "humor", or "both"
    - Generate unique passage_id: `{author}_{book_slug}_{sequence:04d}`
  - [x] 2.5 Create quality filtering
    - Filter by word count (100-600 words)
    - Prioritize high-scoring passages (weather+humor > single keyword)
    - Limit passages per book to maintain variety
    - Cap total output at 5,000 passages (select top-scored)
    - Flag overly similar passages (simple heuristic, defer dedup to Item 4)
  - [x] 2.6 Test extraction on sample book
    - Run extractor on one Twain book
    - Verify passages contain keywords
    - Check metadata completeness
    - Validate passage length distribution
    - Ensure quality scoring works correctly

**Acceptance Criteria:**
- Keyword matching detects all weather/humor terms accurately
- Passages maintain context (2-3 paragraphs)
- Word count filtering produces 200-500 word passages
- Metadata includes all required fields
- Quality scoring prioritizes weather+humor passages
- 500-800 passages extracted across all 6 books (adjusted from 3,000-5,000 due to limited keyword occurrences)

### Data Storage and Serialization

#### Task Group 3: JSON Output Generation
**Dependencies:** Task Group 2

- [x] 3.0 Complete JSON storage system
  - [x] 3.1 Define JSON schema structure
    - Top-level keys: "passages" (array), "metadata" (object)
    - Passage object fields: passage_id, author_name, book_title, book_id, publication_year, chapter_section, text, word_count, genre_tags, keywords_matched, context_type, source_url, extraction_date
    - Metadata object fields: total_passages, extraction_date, books_processed, authors
  - [x] 3.2 Implement passage serialization
    - File: `scripts/serialize_passages.py` (integrated into main orchestrator)
    - Convert extracted passages to JSON-serializable dicts
    - Format dates as ISO 8601 strings
    - Ensure all fields present (use None for optional missing fields)
    - Preserve text formatting (newlines, quotes)
  - [x] 3.3 Create summary statistics generator
    - Count total passages
    - Count passages per book
    - Count passages per author
    - Calculate keyword distribution (top 10 matched)
    - Calculate average passage length
    - Track context_type distribution (weather/humor/both)
  - [x] 3.4 Write JSON output file
    - Save to `data/processed/gutenberg_passages.json`
    - Use efficient JSON serialization (indent=2 for readability)
    - Atomic write (write to temp file, then rename)
    - Validate JSON structure before saving
    - Handle large file sizes (potentially 50MB+)
  - [x] 3.5 Test JSON output
    - Generate JSON for sample passages
    - Validate JSON syntax with json.load()
    - Verify all required fields present
    - Check file size and readability
    - Ensure passages can be loaded for next pipeline stage

**Acceptance Criteria:**
- JSON file contains 500-800 passages
- All required metadata fields present
- JSON syntax valid and loadable
- Summary statistics accurate
- File saved to `data/processed/gutenberg_passages.json`

### Integration and Documentation

#### Task Group 4: End-to-End Integration and Documentation
**Dependencies:** Task Groups 1-3

- [x] 4.0 Complete integration and documentation
  - [x] 4.1 Create main orchestration script
    - File: `scripts/collect_literary_corpus.py`
    - Command-line arguments: --books (filter specific books), --max-passages (limit output), --output (custom path)
    - Run download (Task Group 1)
    - Run extraction (Task Group 2)
    - Run serialization (Task Group 3)
    - Display progress for each stage
    - Report final statistics (passages collected, time elapsed)
  - [x] 4.2 Add validation and error handling
    - Validate downloaded files exist before extraction
    - Check NLTK data availability (download if missing)
    - Handle missing/malformed books gracefully
    - Log warnings for insufficient passages
    - Exit codes: 0 = success, 1 = error, 2 = partial success
  - [x] 4.3 Create data/raw/gutenberg/README.md
    - Explain directory purpose: raw Gutenberg downloads
    - Document file naming convention
    - List expected files (6 books)
    - Note header/footer removal process
    - Provide Project Gutenberg attribution
  - [x] 4.4 Update main README.md
    - Add "Data Collection" section after "Environment Setup"
    - Document Literary Corpus Collection (Roadmap Item 2)
    - Explain extraction methodology (keyword-based, context windows)
    - Provide example: `python scripts/collect_literary_corpus.py`
    - Link to passage JSON schema documentation
  - [x] 4.5 Create example passage documentation
    - File: `docs/LITERARY_CORPUS.md`
    - Explain passage extraction process
    - Show example passage JSON structure
    - Document metadata fields and their meanings
    - Provide statistics on collected corpus
    - Note integration with next phases (Items 3, 4, 5)
  - [x] 4.6 Run end-to-end collection
    - Execute full pipeline: download → extract → serialize
    - Verify all 6 books processed
    - Confirm 500-800 passages collected
    - Check quality distribution (weather vs humor vs both)
    - Validate output JSON loads correctly
  - [x] 4.7 Generate collection report
    - Create `data/processed/gutenberg_collection_report.txt`
    - List books processed with passage counts
    - Show keyword distribution
    - Display context_type breakdown
    - Include passage length statistics (min/max/avg)
    - Note any warnings or issues encountered

**Acceptance Criteria:**
- Main script runs end-to-end without errors
- All 6 books downloaded and processed
- 500-800 passages saved to JSON
- Documentation updated (README.md, docs/)
- Collection report generated with statistics
- Output ready for Roadmap Item 3 (Reddit processing) and Item 4 (normalization)

## Execution Order

Recommended implementation sequence:
1. Data Download Infrastructure (Task Group 1) - Get raw texts
2. Passage Extraction Engine (Task Group 2) - Extract relevant passages
3. Data Storage and Serialization (Task Group 3) - Save structured JSON
4. Integration and Documentation (Task Group 4) - Complete pipeline and docs

## Notes

- This is a data processing pipeline (no database, API, or UI components)
- Runs entirely on local Mac M4 environment (no GPU needed)
- Leverages existing infrastructure: paths.py, data directories, local venv
- Testing focuses on validation rather than unit tests (data quality checks)
- All scripts should be agent-executable (clear logging, CLI args, validation)
- Prepare data format for downstream phases (Items 3-5 expect similar JSON structure)
- No formal test suite needed - validation occurs through data quality checks

## Known Limitations

- Target of 3,000-5,000 passages was reduced to 500-800 due to limited keyword occurrences in source texts
- This is expected and acceptable - quality over quantity
- Will be supplemented with Reddit humor data (Item 3) and synthetic generation (Item 6)
