# Data Normalization & Deduplication Pipeline

Comprehensive data cleaning pipeline that normalizes text, removes near-duplicates using MinHash/LSH, filters for English-only content, applies safety checks, and generates detailed quality reports.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
- [Output Format](#output-format)
- [Statistics Reports](#statistics-reports)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)

---

## Overview

The normalization pipeline processes raw data from multiple sources (Project Gutenberg, Reddit, etc.) into a single, clean JSONL dataset suitable for LoRA fine-tuning. It applies industry-standard text cleaning techniques while preserving metadata for audit trails.

**Pipeline Flow:**
```
Input (JSON/JSONL) → Unicode Normalization → Deduplication → Language Filter → Safety Filter → Clean JSONL Output
```

---

## Features

- **Unicode Normalization**: NFC form preserves semantic distinctions in literary texts
- **Deduplication**: MinHash/LSH with 0.8 Jaccard similarity threshold
- **Language Detection**: English-only filtering using langdetect
- **Safety Filtering**: OpenAI Moderation API integration (optional)
- **Multi-Source Input**: Supports both JSON and JSONL formats
- **Metadata Preservation**: Maintains all source tags throughout pipeline
- **Quality Reports**: JSON and Markdown statistics with timestamps
- **Idempotent**: Safe to re-run without data corruption
- **Atomic Writes**: Prevents partial file corruption

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements-local.txt
```

Key dependencies:
- `datasketch==1.6.4` - MinHash/LSH deduplication
- `langdetect==1.0.9` - Language detection
- `openai==1.3.0` - Safety moderation API
- `pytest==7.4.3` - Testing framework

### 2. Configure Pipeline

Edit `configs/pipeline_config.json` to customize thresholds:

```json
{
  "deduplication": {
    "threshold": 0.8
  },
  "normalization": {
    "form": "NFC"
  },
  "safety_filter": {
    "enabled": true,
    "batch_size": 20
  }
}
```

### 3. Set API Key (Optional)

For safety filtering, set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or skip safety checks with `--skip-safety` flag.

---

## Quick Start

### Process Default Sources (Gutenberg + Reddit)

```bash
python scripts/normalization_pipeline_orchestrator.py
```

Output: `data/processed/training_data_clean.jsonl`

### Process Custom Files

```bash
python scripts/normalization_pipeline_orchestrator.py \
  --input data/processed/gutenberg_passages.json \
  --output data/processed/custom_output.jsonl
```

### Dry Run (No File Writes)

```bash
python scripts/normalization_pipeline_orchestrator.py --dry-run --skip-safety
```

---

## Pipeline Stages

### Stage 1: Data Loading
- Loads JSON and JSONL files
- Combines multiple sources
- Adds `source_file` metadata

**Input:** JSON arrays or JSONL files
**Output:** Unified list of items

### Stage 2: Unicode Normalization
- Applies NFC (Canonical Decomposition + Composition)
- Preserves semantic distinctions (superscripts, fractions)
- Ensures consistent character representation

**Technique:** Python `unicodedata.normalize('NFC', text)`

### Stage 3: Deduplication
- Uses MinHash signatures (128 permutations)
- LSH for efficient similarity search
- Removes items with ≥0.8 Jaccard similarity
- Keeps first occurrence of duplicates

**Algorithm:** Character-level 3-shingles with MinHash/LSH

### Stage 4: Language Filter
- Detects language using langdetect
- Removes non-English content
- Reports language distribution

**Target:** English only (ISO 639-1: 'en')

### Stage 5: Safety Filter
- OpenAI Moderation API integration
- Batch processing (20 items per call)
- Exponential backoff retry (3 attempts)
- Flags toxic, violent, NSFW content

**Note:** Optional - use `--skip-safety` to bypass

### Stage 6: Pipeline Metadata
Adds to each item:
```json
{
  "pipeline_metadata": {
    "normalization_version": "1.0",
    "dedup_threshold": 0.8,
    "filters_applied": ["unicode_normalization", "deduplication", "language_filter", "safety_filter"]
  }
}
```

### Stage 7: Output Writing
- Atomic write (temp file + rename)
- JSONL format (one JSON per line)
- Validates JSON structure

### Stage 8: Statistics Generation
- JSON report with timestamp
- Markdown report with formatted tables
- Includes all stage metrics

---

## Configuration

### Pipeline Config (`configs/pipeline_config.json`)

```json
{
  "normalization": {
    "form": "NFC"
  },
  "deduplication": {
    "threshold": 0.8,
    "num_perm": 128
  },
  "language_filter": {
    "target_language": "en",
    "confidence_threshold": 0.7
  },
  "safety_filter": {
    "enabled": true,
    "batch_size": 20,
    "retry": {
      "max_attempts": 3,
      "backoff_factor": 2
    }
  },
  "quality": {
    "min_length": 10,
    "max_length": 10000
  }
}
```

### Paths Config (`configs/paths_config.json`)

```json
{
  "data": {
    "pipeline": {
      "output": "data/processed/training_data_clean.jsonl",
      "stats_json": "data/processed/pipeline_stats_{timestamp}.json",
      "stats_md": "data/processed/pipeline_stats_{timestamp}.md"
    }
  }
}
```

---

## CLI Usage

### Basic Usage

```bash
# Process default sources
python scripts/normalization_pipeline_orchestrator.py

# Process specific files
python scripts/normalization_pipeline_orchestrator.py --input file1.json file2.jsonl

# Multiple inputs
python scripts/normalization_pipeline_orchestrator.py \
  --input data/processed/gutenberg_passages.json \
         data/processed/reddit_humor_weather.jsonl
```

### Advanced Options

```bash
# Custom output path
python scripts/normalization_pipeline_orchestrator.py \
  --output data/processed/custom_output.jsonl

# Custom config
python scripts/normalization_pipeline_orchestrator.py \
  --config configs/custom_pipeline_config.json

# Dry run (no file writes)
python scripts/normalization_pipeline_orchestrator.py --dry-run

# Skip safety checks (no API calls)
python scripts/normalization_pipeline_orchestrator.py --skip-safety

# Combined flags
python scripts/normalization_pipeline_orchestrator.py \
  --input test.json \
  --output test_output.jsonl \
  --dry-run \
  --skip-safety
```

### Help

```bash
python scripts/normalization_pipeline_orchestrator.py --help
```

---

## Output Format

### JSONL Structure

Each line is a JSON object:

```json
{
  "content": "The weather is nice today",
  "id": 1,
  "source": "gutenberg",
  "book_title": "Example Book",
  "source_file": "gutenberg_passages.json",
  "pipeline_metadata": {
    "normalization_version": "1.0",
    "dedup_threshold": 0.8,
    "filters_applied": ["unicode_normalization", "deduplication", "language_filter", "safety_filter"]
  }
}
```

### Metadata Fields

**Original Metadata** (preserved):
- `id`, `reddit_id`, `gutenberg_id` - Unique identifiers
- `source` - Data source (gutenberg, reddit, etc.)
- `subreddit`, `created_utc`, `url` - Reddit-specific
- `book_title`, `author` - Gutenberg-specific
- `tags` - Domain/tone tags

**Added Metadata**:
- `source_file` - Origin file name
- `pipeline_metadata` - Processing information

---

## Statistics Reports

### JSON Report (`pipeline_stats_{timestamp}.json`)

```json
{
  "timestamp": "2025-11-02T12:34:56",
  "pipeline_version": "1.0",
  "summary": {
    "initial_count": 1000,
    "final_count": 750,
    "total_filtered": 250,
    "retention_rate": 75.0
  },
  "stages": {
    "deduplication": {
      "original_count": 1000,
      "unique_count": 850,
      "duplicates_removed": 150,
      "duplicate_rate": 15.0
    },
    "language_filter": {
      "original_count": 850,
      "english_count": 800,
      "filtered_count": 50,
      "language_distribution": {
        "en": 800,
        "fr": 30,
        "es": 20
      }
    },
    "safety_filter": {
      "original_count": 800,
      "safe_count": 750,
      "flagged_count": 50,
      "flagged_categories": {
        "hate": 20,
        "violence": 30
      }
    }
  }
}
```

### Markdown Report (`pipeline_stats_{timestamp}.md`)

Human-readable tables with:
- Summary metrics
- Stage-by-stage breakdowns
- Language distribution table
- Flagged categories table
- Character length statistics

---

## Troubleshooting

### Issue: No items after language filter

**Cause:** Input data may not be in English
**Solution:** Check language distribution in stats report

```bash
# Check report
cat data/processed/pipeline_stats_*.md | grep -A 10 "Language Distribution"
```

### Issue: OpenAI API errors

**Cause:** Invalid API key or rate limits
**Solutions:**
1. Verify API key: `echo $OPENAI_API_KEY`
2. Skip safety checks: `--skip-safety` flag
3. Reduce batch size in config: `"batch_size": 10`

### Issue: Memory errors with large datasets

**Cause:** Loading entire dataset into memory
**Solution:** Process in smaller batches

```bash
# Split input files first
# Process each batch separately
# Combine outputs manually
```

### Issue: Deduplication too aggressive/lenient

**Cause:** Threshold too low/high
**Solution:** Adjust in `configs/pipeline_config.json`

```json
{
  "deduplication": {
    "threshold": 0.9  // More strict (fewer duplicates removed)
    // or
    "threshold": 0.7  // Less strict (more duplicates removed)
  }
}
```

### Issue: Invalid JSON in input file

**Cause:** Malformed JSON syntax
**Solution:** Validate input files

```bash
# Validate JSON
python -m json.tool data/processed/input.json > /dev/null

# Validate JSONL
while read line; do echo "$line" | python -m json.tool > /dev/null; done < data/processed/input.jsonl
```

---

## Testing

### Run All Tests

```bash
python -m pytest tests/test_pipeline_config.py \
                 tests/test_text_processing.py \
                 tests/test_pipeline_orchestration.py \
                 tests/test_normalization_integration.py -v
```

Expected: **51 tests pass**

### Run Specific Test Groups

```bash
# Configuration tests (10 tests)
python -m pytest tests/test_pipeline_config.py -v

# Text processing tests (20 tests)
python -m pytest tests/test_text_processing.py -v

# Orchestration tests (9 tests)
python -m pytest tests/test_pipeline_orchestration.py -v

# Integration tests (12 tests)
python -m pytest tests/test_normalization_integration.py -v
```

### Test Coverage

- **Configuration:** Loading, validation, environment overrides
- **Text Processing:** Normalization, deduplication, language detection, safety filtering
- **Orchestration:** Pipeline stages, statistics, atomic writes, idempotency
- **Integration:** Error handling, edge cases, metadata preservation, end-to-end

---

## Module Reference

### Core Modules

- **text_normalization.py** - Unicode NFC normalization
- **deduplication.py** - MinHash/LSH duplicate detection
- **language_filter.py** - Language detection and filtering
- **safety_filter.py** - OpenAI moderation API integration
- **data_loader.py** - Multi-format data loading
- **statistics_reporter.py** - Report generation
- **normalization_pipeline_orchestrator.py** - Main pipeline

### Utility Modules

- **paths.py** - Path constants and directory management
- **config_loader.py** - Configuration file loading

---

## Performance Notes

- **Deduplication:** O(n) average with LSH, slower for large datasets
- **Language Detection:** Fast (~100-1000 items/sec)
- **Safety Filter:** Limited by API rate (use batching)
- **Memory:** Entire dataset loaded into memory (plan accordingly)

**Recommendations:**
- For datasets > 100K items: Process in batches
- For slow networks: Increase safety filter batch size
- For testing: Use `--dry-run` and `--skip-safety`

---

## License

Part of Weatherman-LoRA project. See repository LICENSE file.

---

## Support

For issues or questions:
1. Check this README troubleshooting section
2. Review test cases for usage examples
3. Check implementation reports in `agent-os/specs/2025-11-02-data-normalization-deduplication-pipeline/implementation/`
