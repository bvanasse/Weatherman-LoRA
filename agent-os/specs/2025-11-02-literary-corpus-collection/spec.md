# Specification: Literary Corpus Collection

## Goal
Download and extract 3,000-5,000 high-quality passages from 6 Project Gutenberg texts (Mark Twain and Benjamin Franklin works) containing weather and humor references, preserving comprehensive metadata to create training data for a literary-styled weather assistant.

## User Stories
- As an ML engineer, I want to programmatically collect literary passages so that I can train a model with distinctive Twain/Franklin personality without manual data curation
- As a data scientist, I want structured metadata for each passage so that I can track provenance and filter by genre/context during training pipeline development

## Specific Requirements

**Download Project Gutenberg Texts**
- Download 4 Mark Twain works: Tom Sawyer (ID: 74), Huckleberry Finn (ID: 76), Connecticut Yankee (ID: 86), Life on Mississippi (ID: 245)
- Download 2 Benjamin Franklin works: Autobiography (ID: 20203 or 148), Poor Richard's Almanack (ID: 57795)
- Use direct HTTP downloads via requests library (more stable than API, no rate limits)
- Store raw text files in `data/raw/gutenberg/` with naming: `{author_last_name}_{book_slug}.txt`
- Remove Gutenberg header/footer boilerplate automatically
- Cache downloaded files to avoid re-downloading on script re-runs
- Handle network failures with exponential backoff retry logic

**Extract Passages with Keyword Filtering**
- Target 3,000-5,000 total passages across all 6 books (quality over quantity)
- Use keyword-based filtering with case-insensitive, whole-word matching
- Weather keywords: weather, rain, storm, thunder, lightning, cloud, sun, wind, climate, temperature, snow, fog, drought, hurricane, tornado, flood, heat, cold, frost, dew, hail
- Humor keywords: joke, wit, laugh, humor, comic, amusing, funny, satire, irony, jest
- Extract 2-3 paragraphs surrounding keyword matches for context
- Target passage length: 200-500 words (min 100, max 600 for quality filtering)
- Prioritize passages with both weather + humor keywords (highest relevance score)
- Preserve paragraph boundaries and chapter markers

**Metadata Preservation**
- Required fields: author_name, author_id, book_title, book_id, publication_year, chapter_section, passage_id (format: `{author}_{book}_{sequence}`), source_url, word_count, extraction_date
- Optional fields: genre_tags (e.g., ["humor", "adventure"]), character_names (for dialogue-heavy passages), keywords_matched (triggering keywords), context_type (weather/humor/both)
- Use judgment to include genre tags based on book themes (Twain: humor/satire/adventure, Franklin: autobiography/wisdom)
- Extract character names from dialogue-heavy passages if present
- Generate unique passage IDs for traceability

**Storage Format and Structure**
- Raw texts: `data/raw/gutenberg/{author_last_name}_{book_slug}.txt` (one file per book)
- Processed passages: `data/processed/gutenberg_passages.json` (structured JSON)
- JSON schema with two top-level keys: "passages" (array of passage objects) and "metadata" (summary statistics)
- Each passage object includes all metadata fields plus full text content
- Summary metadata: total_passages, extraction_date, books_processed, authors list
- Use efficient JSON serialization for potentially large output file

**Quality Control and Prioritization**
- Prioritize content quality over equal author distribution
- Score passages by relevance: weather+humor keywords = highest priority
- Prefer passages with distinctive author voice (Twain wit, Franklin aphorisms)
- Maintain variety across different books and themes
- Avoid repetitive passages (defer deduplication to Roadmap Item 4, but flag obvious duplicates)
- Apply word count filters (100-600 words) to maintain training efficiency

**Text Processing with NLTK**
- Use NLTK for sentence and paragraph tokenization
- Implement sliding window approach for context capture around keyword matches
- Preserve text formatting: paragraph breaks, chapter markers, dialogue structure
- Handle edge cases: chapter boundaries, mixed prose/poetry, special formatting
- Skip malformed sections with appropriate logging

**Agent-Driven Execution**
- Create self-contained Python script executable by agents
- Include command-line arguments for flexibility (book selection, keyword thresholds, output paths)
- Provide clear progress logging for agent monitoring (download progress, extraction counts)
- Generate summary statistics at completion (passages per book, keyword distribution)
- Validation checks at each stage (file exists, JSON valid, passage counts)

**Documentation Updates**
- Add "Data Collection" section to main README.md documenting this phase
- Create `data/raw/gutenberg/README.md` explaining raw file structure and naming conventions
- Document passage extraction methodology for reproducibility
- Include example passage JSON for reference

## Visual Design
No visual assets provided.

## Existing Code to Leverage

**`scripts/paths.py` - Path Constants Module**
- Provides DATA_RAW and DATA_PROCESSED path constants
- Use `DATA_RAW` for storing downloaded Gutenberg texts
- Use `DATA_PROCESSED` for storing extracted passages JSON
- Follows established project conventions for data organization
- Supports environment overrides if needed for testing

**Local Environment Setup (from Roadmap Item 1)**
- Environment already configured with all necessary libraries via `setup_local.sh`
- BeautifulSoup4 4.12+ available for HTML parsing (if needed)
- requests 2.31+ available for HTTP downloads
- NLTK 3.8+ available for text tokenization and processing
- pandas 2.1+ available for data manipulation
- jsonlines 4.0+ available for efficient JSON writing

**Directory Structure Conventions**
- `data/raw/` directory already created with .gitkeep
- `data/processed/` directory already created with .gitkeep
- Follow existing naming patterns for consistency with project
- Leverage .gitignore rules (data/ contents ignored, structure preserved)

## Out of Scope
- Text cleaning/normalization (unicode normalization, case normalization) - Roadmap Item 4
- Deduplication using MinHash/LSH (threshold 0.8) - Roadmap Item 4
- Language filtering (English-only enforcement) - Roadmap Item 4
- Safety/toxicity filtering - Roadmap Item 4
- Converting passages to chat-format JSONL (system/user/assistant roles) - Roadmap Item 5
- Applying persona tags (twain/franklin/neutral) - Roadmap Item 5
- Applying tone tags (humorous/satirical/didactic) - Roadmap Item 5
- Creating train/validation splits (90/10) - Roadmap Item 5
- Processing Reddit humor dataset from `data_sources/reddit-theonion/` - Roadmap Item 3
- Downloading authors beyond Mark Twain and Benjamin Franklin
- Manual passage selection, curation, or quality review
- Advanced NLP analysis (sentiment analysis, named entity recognition)
- Data augmentation or synthetic passage generation
- Comprehensive quality statistics reporting (will be part of Item 4)
