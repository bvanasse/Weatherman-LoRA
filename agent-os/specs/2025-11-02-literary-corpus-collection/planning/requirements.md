# Spec Requirements: Literary Corpus Collection

## Initial Description
Download and parse public domain texts from Project Gutenberg (Mark Twain: Tom Sawyer, Huck Finn, Connecticut Yankee, Life on Mississippi; Benjamin Franklin: Autobiography, Poor Richard's Almanack), extract 3,000-5,000 relevant passages with weather/humor mentions, and preserve source metadata

Size Estimate: S (Small)
Roadmap Position: Item 2 of 12

## Requirements Discussion

### First Round Questions

**Q1:** I assume we should download the complete text files for all 6 specified works (4 Twain + 2 Franklin) and then extract passages programmatically. Is that correct, or should we manually select chapters/sections first?
**Answer:** That is a correct assumption. We should attempt to keep our programmatic approach standardized and amenable to the rest of the training process.

**Q2:** For identifying "weather/humor mentions," I'm thinking we should use keyword-based filtering (e.g., "weather", "rain", "storm", "thunder", "cloud", "sun", "wind", "climate", "temperature") combined with context windows (e.g., extract 2-3 paragraphs around matches). Should we also include humor-specific keywords like "joke", "wit", "laugh", or rely on the authors' natural style?
**Answer:** Yes, use keyword-based filtering combined with context windows, including humor-specific and weather-specific keywords depending on the author's natural style.

**Q3:** I assume each extracted passage should be 200-500 words (similar to the Reddit processing goal) to maintain context while keeping training efficient. Is that the right range, or should passages be shorter/longer?
**Answer:** Keep a consistent process for extracting passages to keep training efficient.

**Q4:** For metadata preservation, I'm thinking we should store: author name, book title, publication year, chapter/section, passage ID, and source URL (Gutenberg ID). Should we also capture genre tags (humor, travel, autobiography) or character names if dialogue-heavy?
**Answer:** Yes use your best judgment, and include genre tags as well. If you need to include character names, then do so.

**Q5:** I assume we should use the Project Gutenberg API (gutendex.com) for programmatic downloads rather than manual file downloads. Is that correct, or do you prefer direct file downloads from gutenberg.org?
**Answer:** Use your best judgment, this will need to be done by agents rather than manually by the user.

**Q6:** For storage format, I'm thinking we should save raw text files in `data/raw/gutenberg/` (one file per book) and then save extracted passages as JSON in `data/processed/gutenberg_passages.json` with metadata. Does this align with your expectations, or should we use a different structure?
**Answer:** Whatever is best for reviewing and training. This approach seems adequate.

**Q7:** The target is 3,000-5,000 relevant passages total. Should we aim for roughly equal distribution across authors (e.g., ~2,500 Twain, ~1,500 Franklin), or prioritize based on content quality?
**Answer:** Prioritize based on content quality.

**Q8:** What's NOT in scope for this phase? I assume we're NOT: cleaning/deduplicating the text (that's Roadmap Item 4), converting to chat-format JSONL (that's Roadmap Item 5), downloading any other authors beyond Twain and Franklin, processing the Reddit data (that's Roadmap Item 3). Is there anything else you explicitly want to exclude?
**Answer:** Use your best judgment, sticking to the roadmap seems appropriate.

**Additional Guidance:** Include everything necessary for this roadmap item to be a success and setup the next phases for success. Optimize for using agents to perform most of these tasks. Leverage existing code that has been created. Update the README files as necessary.

### Existing Code to Reference

**Similar Features Identified:**
- Feature: Environment Setup & Data Infrastructure - Path: `agent-os/specs/2025-11-02-environment-setup-data-infrastructure/`
- Components to potentially reuse:
  - `scripts/paths.py` - Path constants module with DATA_RAW, DATA_PROCESSED
  - `data/raw/` directory structure - Already set up for raw data storage
  - `data/processed/` directory structure - For processed passage JSON files
  - `requirements-local.txt` - Data processing libraries already installed (pandas, BeautifulSoup4, trafilatura, requests, NLTK)
  - `setup_local.sh` - Local environment already configured
- Backend logic to reference:
  - Path management patterns from paths.py
  - Storage conventions for raw/processed data
  - Local Mac M4 environment for data processing

**Tech Stack Available:**
- BeautifulSoup4 4.12+ for HTML parsing
- requests 2.31+ for HTTP downloads
- trafilatura 1.6.2 for content extraction
- NLTK 3.8+ for text processing and chunking
- pandas 2.1+ for data manipulation
- jsonlines 4.0+ for efficient JSONL writing

### Follow-up Questions
No follow-up questions needed. Requirements are clear.

## Visual Assets

### Files Provided:
No visual assets provided.

### Visual Insights:
N/A

## Requirements Summary

### Functional Requirements

**Core Data Collection:**
- Download 6 complete public domain texts from Project Gutenberg:
  - Mark Twain: The Adventures of Tom Sawyer (ID: 74), Adventures of Huckleberry Finn (ID: 76), A Connecticut Yankee in King Arthur's Court (ID: 86), Life on the Mississippi (ID: 245)
  - Benjamin Franklin: Autobiography of Benjamin Franklin (ID: 20203 or 148), Poor Richard's Almanack (ID: 57795)
- Store raw text files in `data/raw/gutenberg/` with naming convention: `{author_last_name}_{book_title_slug}.txt`
- Programmatic download approach (agent-driven, not manual)

**Passage Extraction:**
- Extract 3,000-5,000 relevant passages total across all 6 works
- Use keyword-based filtering with context windows:
  - Weather keywords: weather, rain, storm, thunder, lightning, cloud, sun, wind, climate, temperature, snow, fog, drought, hurricane, tornado, flood, heat, cold, frost, dew, hail
  - Humor keywords (author-dependent): joke, wit, laugh, humor, comic, amusing, funny, satire, irony, jest
- Extract 2-3 paragraphs around keyword matches to maintain context
- Target passage length: 200-500 words for training efficiency
- Prioritize content quality over equal author distribution
- Preserve paragraph boundaries and natural text flow

**Metadata Preservation:**
- Required metadata fields:
  - author_name (e.g., "Mark Twain", "Benjamin Franklin")
  - author_id (for reference)
  - book_title (full title)
  - book_id (Project Gutenberg ID)
  - publication_year (original publication date)
  - chapter_section (chapter number/name if available)
  - passage_id (unique identifier: `{author}_{book}_{sequence}`)
  - source_url (Gutenberg URL)
  - word_count (actual passage word count)
  - extraction_date (timestamp)
- Optional metadata fields (use judgment):
  - genre_tags (e.g., ["humor", "travel"], ["autobiography", "wisdom"], ["satire", "adventure"])
  - character_names (if dialogue-heavy, extract speaker names)
  - keywords_matched (which keywords triggered extraction)
  - context_type (weather, humor, or both)

**Storage Format:**
- Raw text files: `data/raw/gutenberg/{author_last_name}_{book_slug}.txt`
- Processed passages: `data/processed/gutenberg_passages.json` (structured JSON array)
- JSON schema:
  ```json
  {
    "passages": [
      {
        "passage_id": "twain_tom_sawyer_001",
        "author_name": "Mark Twain",
        "book_title": "The Adventures of Tom Sawyer",
        "book_id": 74,
        "publication_year": 1876,
        "chapter_section": "Chapter 5",
        "text": "Full passage text here...",
        "word_count": 342,
        "genre_tags": ["humor", "adventure"],
        "keywords_matched": ["storm", "thunder"],
        "context_type": "weather",
        "source_url": "https://www.gutenberg.org/ebooks/74",
        "extraction_date": "2025-11-02T..."
      }
    ],
    "metadata": {
      "total_passages": 4532,
      "extraction_date": "2025-11-02",
      "books_processed": 6,
      "authors": ["Mark Twain", "Benjamin Franklin"]
    }
  }
  ```

**Quality Targets:**
- 3,000-5,000 total passages (flexible based on quality)
- Prioritize passages with strong weather + humor overlap
- Prefer passages with distinctive author voice (Twain wit, Franklin aphorisms)
- Maintain variety across books and themes
- Avoid overly repetitive passages

### Reusability Opportunities

**Leverage Existing Infrastructure:**
- Use `scripts/paths.py` constants: `DATA_RAW`, `DATA_PROCESSED`, `REFERENCES_DIR`
- Follow existing directory structure conventions
- Reuse local environment setup (venv already configured with all necessary libraries)
- Adopt existing naming conventions for consistency

**Pattern to Establish:**
- Standardized literary corpus processing pipeline
- Reusable for future author additions (if project expands)
- Metadata schema that aligns with training pipeline (Roadmap Item 5)
- Quality-first extraction methodology

**Prepare for Next Phases:**
- Roadmap Item 3: Reddit Humor Dataset Processing (will have similar JSON output format)
- Roadmap Item 4: Data Normalization & Deduplication (passages ready for MinHash/LSH)
- Roadmap Item 5: Instructionalization & Tagging (metadata facilitates persona/tone tagging)

### Scope Boundaries

**In Scope:**
- Download 6 specified Project Gutenberg texts (4 Twain, 2 Franklin)
- Extract 3,000-5,000 passages using keyword-based filtering
- Preserve comprehensive metadata (author, book, chapter, genre, keywords)
- Store raw text files and processed JSON
- Programmatic/agent-driven approach
- Update README documentation as needed
- Create reusable scripts for future corpus additions

**Out of Scope:**
- Text cleaning/normalization (Roadmap Item 4: unicode normalization, deduplication)
- Converting to chat-format JSONL (Roadmap Item 5: system/user/assistant roles)
- Applying persona tags (twain/franklin/neutral) (Roadmap Item 5)
- Processing Reddit humor data (Roadmap Item 3)
- Downloading authors beyond Twain and Franklin
- Manual passage selection or curation
- Advanced NLP analysis (sentiment, named entity recognition)
- Data augmentation or synthetic generation
- Training data validation or statistics reporting (part of Item 4)

### Technical Considerations

**Project Gutenberg Integration:**
- API Options:
  - gutendex.com (unofficial API, easier programmatic access)
  - Direct file downloads from gutenberg.org (more reliable, simple HTTP)
- Recommendation: Use direct file downloads via requests library (more stable, no API rate limits)
- Handle Gutenberg text header/footer removal (standard boilerplate)
- Support plain text format (.txt files)

**Text Extraction Strategy:**
- Use NLTK for sentence/paragraph tokenization
- Sliding window approach for context capture (match + surrounding paragraphs)
- Preserve formatting: paragraph breaks, chapter markers
- Handle edge cases: chapter boundaries, dialogue formatting, poetry/verse
- Quality filters: minimum word count (100 words), maximum (600 words)

**Keyword Matching:**
- Case-insensitive matching
- Whole word matching (avoid partial matches like "weather" in "leathern")
- Prioritize passages with multiple keyword matches
- Score passages by relevance (weather + humor = highest priority)

**Performance Optimization:**
- Process one book at a time to manage memory
- Cache downloaded raw files to avoid re-downloading
- Use efficient JSON serialization (jsonlines for large datasets)
- Progress tracking for long-running downloads/extractions

**Error Handling:**
- Network failures: retry with exponential backoff
- Missing books: log warning, continue with available texts
- Malformed text: skip problematic sections, document in logs
- Insufficient passages: adjust keyword thresholds dynamically

**Agent-Driven Execution:**
- Create self-contained Python scripts that agents can run
- Clear logging output for agent monitoring
- Command-line arguments for flexibility
- Validation checks at each stage (download, extraction, storage)
- Summary statistics output for review

**Documentation Updates:**
- Add section to main README.md about data collection phase
- Update `data/raw/gutenberg/` with README explaining structure
- Document passage extraction methodology for reproducibility
- Create example passages for reference
