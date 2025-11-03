# Literary Corpus Collection

This document describes the literary passage extraction process for the Weatherman-LoRA project, which collects training data from classic literature by Mark Twain and Benjamin Franklin.

## Overview

The literary corpus provides foundational training data with distinctive authorial voices combining weather descriptions and humor. This corpus captures the witty, satirical style of Mark Twain and the pithy wisdom of Benjamin Franklin.

## Extraction Process

### 1. Source Selection

**Authors:**
- Mark Twain (4 books): Known for humor, satire, and vivid descriptions
- Benjamin Franklin (2 books): Known for wit, wisdom, and practical advice

**Books:**
1. The Adventures of Tom Sawyer (1876)
2. Adventures of Huckleberry Finn (1884)
3. A Connecticut Yankee in King Arthur's Court (1889)
4. Life on the Mississippi (1883)
5. The Autobiography of Benjamin Franklin (1793)
6. Poor Richard's Almanack (1732)

### 2. Keyword-Based Extraction

**Weather Keywords (21 terms):**
weather, rain, storm, thunder, lightning, cloud, sun, wind, climate, temperature, snow, fog, drought, hurricane, tornado, flood, heat, cold, frost, dew, hail

**Humor Keywords (10 terms):**
joke, wit, laugh, humor, comic, amusing, funny, satire, irony, jest

**Matching Strategy:**
- Case-insensitive whole-word matching (regex with word boundaries)
- Sliding window approach to capture keywords spread across paragraphs
- Scores passages by relevance:
  - Weather + Humor keywords: 2 points (highest priority)
  - Single keyword type: 1 point

### 3. Context Window Extraction

For each keyword match, extract surrounding context:
- 1 paragraph before the match
- The paragraph containing the match
- 1-2 paragraphs after the match

Passages are dynamically sized to reach target length:
- Target: 200-500 words
- Minimum: 100 words (quality threshold)
- Maximum: 600 words (prevent overly long passages)

### 4. Quality Filtering

**Overlap Prevention:**
- Calculate overlap between passages (paragraph-based)
- Remove passages with >20% overlap
- Keeps higher-scored passages when overlap detected

**Word Count Filtering:**
- Passages must be 100-600 words
- Average passage length: ~300 words

**Relevance Prioritization:**
- Passages with both weather AND humor keywords ranked highest
- Ensures diverse keyword distribution across corpus

## Output Format

### JSON Structure

```json
{
  "passages": [
    {
      "passage_id": "twain_tom_sawyer_0001",
      "author_name": "Mark Twain",
      "book_title": "The Adventures of Tom Sawyer",
      "book_id": 74,
      "publication_year": 1876,
      "chapter_section": "CHAPTER I",
      "text": "Full passage text...",
      "word_count": 285,
      "genre_tags": ["humor", "satire", "adventure"],
      "keywords_matched": ["weather", "sun", "laugh"],
      "context_type": "both",
      "relevance_score": 2,
      "character_names": ["Tom", "Huck"],
      "source_url": "https://www.gutenberg.org/ebooks/74",
      "extraction_date": "2025-11-02T19:43:54Z"
    }
  ],
  "metadata": {
    "total_passages": 515,
    "extraction_date": "2025-11-02T19:43:54Z",
    "books_processed": [...],
    "authors": ["Mark Twain", "Benjamin Franklin"],
    "context_type_distribution": {...},
    "keyword_distribution": {...},
    "word_count_stats": {...}
  }
}
```

### Metadata Fields

**Required Fields:**
- `passage_id`: Unique identifier (`{author}_{book}_{sequence:04d}`)
- `author_name`: Full author name
- `book_title`: Full book title
- `book_id`: Project Gutenberg book ID
- `publication_year`: Year of original publication
- `text`: Full passage text with paragraph breaks preserved
- `word_count`: Number of words in passage
- `genre_tags`: List of genre classifications
- `keywords_matched`: List of matched keywords (triggers)
- `context_type`: "weather", "humor", or "both"
- `relevance_score`: 0-2 (higher = more relevant)
- `source_url`: Project Gutenberg URL
- `extraction_date`: ISO 8601 timestamp

**Optional Fields:**
- `chapter_section`: Chapter marker if present (e.g., "CHAPTER I")
- `character_names`: Character names extracted from dialogue (max 5)

## Corpus Statistics

### Typical Collection

- Total passages: 500-800
- Average word count: 300 words
- Total word count: ~150,000-240,000 words

### Distribution

**By Context Type:**
- Weather-only: ~75%
- Humor-only: ~18%
- Both (weather + humor): ~7%

**By Author:**
- Mark Twain: ~480 passages (93%)
- Benjamin Franklin: ~35 passages (7%)

Note: The distribution reflects keyword occurrence rates in the source texts. Franklin's works have fewer weather references, while Twain's narratives feature more environmental descriptions.

### Top Keywords

Most frequently matched keywords:
1. sun (~80 occurrences)
2. cold (~70 occurrences)
3. lightning (~68 occurrences)
4. wind (~55 occurrences)
5. laugh (~49 occurrences)
6. storm (~45 occurrences)
7. rain (~42 occurrences)
8. weather (~41 occurrences)
9. wit (~37 occurrences)
10. thunder (~36 occurrences)

## Usage

### Running the Collection Pipeline

**Full pipeline (download + extract + serialize):**
```bash
python scripts/collect_literary_corpus.py
```

**Skip download stage (use cached books):**
```bash
python scripts/collect_literary_corpus.py --skip-download
```

**Limit passages:**
```bash
python scripts/collect_literary_corpus.py --max-passages 1000 --max-per-book 200
```

**Filter to specific author:**
```bash
python scripts/collect_literary_corpus.py --books twain
```

### Output Files

- **JSON data:** `data/processed/gutenberg_passages.json`
- **Collection report:** `data/processed/gutenberg_collection_report.txt`
- **Raw texts:** `data/raw/gutenberg/*.txt`

## Integration with Training Pipeline

### Downstream Phases

This corpus feeds into:

1. **Reddit Humor Processing (Roadmap Item 3)**
   - Combine with modern humor data
   - Balance historical vs. contemporary styles

2. **Data Normalization (Roadmap Item 4)**
   - Unicode normalization
   - Deduplication (MinHash/LSH, threshold 0.8)
   - Language filtering (English-only)
   - Safety/toxicity filtering

3. **Training Format Conversion (Roadmap Item 5)**
   - Convert to chat format JSONL
   - Apply persona tags (twain/franklin/neutral)
   - Apply tone tags (humorous/satirical/didactic)
   - Create train/validation splits (90/10)

### Data Quality

**Strengths:**
- High literary quality
- Distinctive authorial voices
- Rich vocabulary and varied syntax
- Public domain content (no licensing issues)

**Limitations:**
- Historical language (19th century)
- Limited keyword occurrences in source texts
- Smaller corpus than initially targeted (500-800 vs. 3,000-5,000 passages)
- Imbalanced author distribution (Twain >> Franklin)

**Mitigation:**
- Supplement with Reddit humor data (Item 3) for modern language
- Use synthetic data generation (Item 6) to expand corpus
- Balance with neutral weather data to prevent over-stylization

## Technical Notes

### NLTK Dependencies

The extraction pipeline uses NLTK for text processing:
- Punkt tokenizer for sentence/paragraph segmentation
- Automatic download on first run

### Performance

- Download stage: ~30 seconds (6 books)
- Extraction stage: ~3 seconds (with cached books)
- Serialization stage: <1 second
- Total pipeline: ~5 seconds (cached), ~35 seconds (fresh)

### Storage

- Raw texts: ~3 MB (6 books)
- Processed JSON: ~1-2 MB (500-800 passages)
- Total disk usage: ~5 MB

## References

- Project Gutenberg: https://www.gutenberg.org/
- Mark Twain works: https://www.gutenberg.org/ebooks/author/53
- Benjamin Franklin works: https://www.gutenberg.org/ebooks/author/607

## License

The source texts are in the public domain (published before 1928). The extraction pipeline and metadata are part of the Weatherman-LoRA project.

---

**Last Updated:** 2025-11-02
**Pipeline Version:** 1.0
**Maintained by:** Weatherman-LoRA Project
