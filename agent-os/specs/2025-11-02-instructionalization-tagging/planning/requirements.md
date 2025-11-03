# Spec Requirements: Instructionalization & Tagging

## Initial Description
Convert raw text into chat-format JSONL with system/user/assistant roles, apply persona tags (twain/franklin/neutral), tone tags (humorous/satirical/didactic), domain tags (weather/general), and create balanced train/validation splits (90/10).

**Source:** Product Roadmap Item 5

**Context:** This follows the completion of the Data Normalization & Deduplication Pipeline (Roadmap Item 4), which produced `data/processed/training_data_clean.jsonl` with 503 deduplicated, normalized items from Gutenberg literary passages and Reddit humor posts.

## Requirements Discussion

### First Round Questions

**Q1:** Chat Format Structure - I assume we'll follow the OpenAI-style chat format from your tech stack (messages array with role/content fields), converting each cleaned text passage into a single turn (system + user + assistant). Is that correct, or should we create multi-turn conversations from some passages?

**Answer:** Use your best judgement. If multi-turn conversations can also be included and would improve the quality of the training - then create multi-turn conversations from some passages if necessary.

**Q2:** Persona Tag Assignment - I'm thinking we determine persona tags based on the source metadata already in the data - passages from Twain books get `persona: "twain"`, Franklin books get `persona: "franklin"`, and Reddit posts get `persona: "neutral"`. Should we use this source-based approach, or would you prefer content-based classification?

**Answer:** Whatever is most efficient.

**Q3:** Tone Tag Determination - For tone tags, I assume we can derive these from existing source tags in the normalized data - Reddit r/TheOnion gets `tone: "satirical"`, r/nottheonion gets `tone: "ironic"`, literary passages with humor keywords get `tone: "humorous"`, and Franklin passages get `tone: "didactic"`. Does this mapping make sense, or should we use a different approach?

**Answer:** The mappings make sense. Use your best judgement.

**Q4:** System Message Content - I'm assuming we'll use persona-aware system messages like "You are a witty weather assistant inspired by Mark Twain" for Twain passages, "You are a wise weather advisor inspired by Benjamin Franklin" for Franklin, and "You are a helpful weather assistant" for neutral. Should we use this approach, or prefer consistent system messages across all examples?

**Answer:** Use this approach.

**Q5:** User Message Variation - I assume we should generate varied user queries (similar to the Reddit pipeline) like "What's the weather like?", "Give me the forecast", "How's it looking outside?" to avoid repetitive training patterns. Should we create 10-15 variations and rotate through them, or use a different strategy?

**Answer:** Use your best judgement.

**Q6:** Domain Tag Logic - For domain tags, I'm thinking weather-related passages get `domain: "weather"` and we add `domain: "humor"` for passages with humor keywords. Should passages have multiple domain tags (e.g., both weather and humor), or should we pick the primary one?

**Answer:** Yes (allow multiple domain tags).

**Q7:** Train/Validation Split Strategy - I assume we should use stratified random sampling to ensure the 90/10 split maintains balanced representation across persona/tone/domain combinations. Should we also ensure temporal ordering isn't a factor (random shuffle before split)?

**Answer:** Yes.

**Q8:** Output Format - I'm thinking we output two files: `data/processed/train.jsonl` (90%) and `data/processed/validation.jsonl` (10%), both following the chat-format schema from your tech stack. Should we also generate a statistics report showing tag distribution across splits?

**Answer:** Yes.

**Q9:** Metadata Preservation - Should we preserve all the existing metadata tags from the normalization pipeline (source, reddit_id, created_utc, etc.) in addition to the new persona/tone/domain tags, or only keep the new tags for cleaner training data?

**Answer:** Whatever is more efficient and better for the performance of the model.

**Q10:** Scope Boundaries - I assume we're NOT implementing LLM-based content rewriting or quality scoring, just format conversion and tagging. Anything else that should be explicitly excluded from this phase?

**Answer:** Use your best judgement, nothing comes to mind.

### Existing Code to Reference

**Similar Features Identified:**

Based on codebase analysis, the following files contain patterns to reference:

- **Chat-format message creation**: `scripts/reddit_jsonl_converter.py`
  - Messages array structure with system/user/assistant roles
  - User message variation templates (15 pre-defined variations)
  - Tag assignment logic (persona, tone, domain, source)
  - Metadata embedding and preservation
  - Atomic file writes using tempfile + rename pattern
  - JSONL validation with schema checking

- **Statistics reporting**: `scripts/statistics_reporter.py`
  - JSON and Markdown report generation
  - Timestamp-based file naming with placeholders
  - Stage-by-stage statistics calculation
  - Summary metrics (counts, rates, distributions)
  - Pretty-printed tables in Markdown format

- **Path management**: `scripts/paths.py`
  - Centralized constants for data directories
  - DATA_PROCESSED constant for output files

- **Data loading**: `scripts/data_loader.py`
  - Load JSONL files from normalization pipeline
  - Handle both JSON and JSONL formats
  - Preserve metadata through loading

### Follow-up Questions
None - all requirements clarified.

## Visual Assets

### Files Provided:
No visual files found via bash check.

### Visual Insights:
No visual assets provided.

## Requirements Summary

### Functional Requirements

**Core Functionality:**
1. **Load normalized data** from `data/processed/training_data_clean.jsonl` (503 items from normalization pipeline)
2. **Convert to chat format** with OpenAI-style messages array (system/user/assistant roles)
3. **Apply persona tags** based on source metadata:
   - Twain literary passages → `persona: "twain"`
   - Franklin literary passages → `persona: "franklin"`
   - Reddit posts → `persona: "neutral"`
4. **Apply tone tags** based on source and content:
   - r/TheOnion → `tone: "satirical"`
   - r/nottheonion → `tone: "ironic"`
   - Literary passages with humor keywords → `tone: "humorous"`
   - Franklin passages → `tone: "didactic"`
5. **Apply domain tags** (multiple allowed):
   - Weather-related content → `domain: "weather"`
   - Humor-related content → `domain: "humor"`
6. **Generate persona-aware system messages**:
   - Twain: "You are a witty weather assistant inspired by Mark Twain"
   - Franklin: "You are a wise weather advisor inspired by Benjamin Franklin"
   - Neutral: "You are a helpful weather assistant"
7. **Create varied user messages** (10-15 variations rotated)
8. **Split into train/validation** (90/10) using stratified random sampling
9. **Generate statistics reports** (JSON + Markdown) showing tag distributions

**Output Requirements:**
- `data/processed/train.jsonl` - 90% of data (~452 items)
- `data/processed/validation.jsonl` - 10% of data (~50 items)
- `data/processed/instructionalization_stats_{timestamp}.json` - JSON statistics
- `data/processed/instructionalization_stats_{timestamp}.md` - Markdown statistics

**Data Quality:**
- Maintain balanced representation of persona/tone/domain across splits
- Random shuffle before splitting (no temporal ordering bias)
- Validate chat-format schema correctness
- Preserve provenance through metadata tags

**Multi-turn Conversation Strategy:**
- Use best judgment to create multi-turn conversations where it would improve training quality
- Longer literary passages (>300 words) may benefit from multi-turn breakdown
- Consider follow-up questions about weather conditions, forecasts, etc.
- Single-turn format is default; multi-turn is enhancement where applicable

### Reusability Opportunities

**Components to Reuse:**
1. **Chat-format message structure** from `scripts/reddit_jsonl_converter.py`:
   - `create_chat_format_entry()` function pattern
   - User message variation with `USER_MESSAGE_TEMPLATES`
   - Tag merging pattern (`{**tone_tags, **metadata_tags}`)
   - Atomic write with tempfile pattern
   - JSONL validation logic

2. **Statistics reporting** from `scripts/statistics_reporter.py`:
   - `calculate_statistics()` function pattern
   - `write_json_report()` and `write_markdown_report()` functions
   - Timestamp placeholder replacement in filenames
   - Markdown table formatting

3. **Data loading** from `scripts/data_loader.py`:
   - `load_jsonl_file()` for reading normalized data
   - Metadata preservation patterns

4. **Path constants** from `scripts/paths.py`:
   - `DATA_PROCESSED` for output directory

**Backend Patterns to Reference:**
- Tag determination logic from `reddit_jsonl_converter.py` (`determine_tone_tags()`)
- Metadata extraction patterns
- Sample entry printing for verification

### Scope Boundaries

**In Scope:**
- Format conversion: JSONL → chat-format JSONL
- Tag assignment: persona, tone, domain based on source metadata
- System message generation: persona-aware prompts
- User message variation: 10-15 templates
- Train/validation split: stratified 90/10
- Statistics reporting: JSON + Markdown with tag distributions
- Multi-turn conversation creation where beneficial (use best judgment)
- Metadata preservation (existing tags that aid training)
- Output validation: chat-format schema compliance

**Out of Scope:**
- LLM-based content rewriting or enhancement
- Quality scoring or ranking of passages
- Synthetic data generation (that's Roadmap Item 6)
- Model training or evaluation (that's Roadmap Items 7-9)
- Manual curation or human-in-the-loop review
- Advanced NLP processing (sentiment analysis, entity extraction)
- Data augmentation techniques (paraphrasing, back-translation)
- Custom chat templates beyond OpenAI format
- Multi-language support (English-only already filtered)

**Future Enhancements (not in this phase):**
- Synthetic tool-use data generation (Roadmap Item 6)
- Advanced persona classification using LLMs
- Dynamic system message generation
- Conversation threading for multi-source passages

### Technical Considerations

**Integration Points:**
- Input: `data/processed/training_data_clean.jsonl` from normalization pipeline (Item 4)
- Output: `data/processed/train.jsonl` and `data/processed/validation.jsonl`
- Next phase: Synthetic tool-use data will merge with these outputs (Item 6)

**Existing System Constraints:**
- Must follow OpenAI chat-format schema from tech stack
- Tag schema must align with TRL/PEFT training expectations
- JSONL format required for efficient streaming during training
- Metadata tags must be JSON-serializable

**Technology Stack:**
- Python 3.10+
- Standard library: json, random, tempfile, pathlib
- pandas for stratified sampling (already installed)
- datasets library for JSONL handling (already installed)

**Similar Code Patterns to Follow:**
- Orchestrator pattern from `scripts/reddit_pipeline_orchestrator.py`
- Atomic file operations from `scripts/reddit_jsonl_converter.py`
- Statistics reporting from `scripts/statistics_reporter.py`
- Environment validation (check input file exists, output dir writable)
- Progress logging for each stage
- CLI arguments with argparse (--output, --split-ratio, --dry-run, --seed)

**Source Metadata Mapping:**
- Gutenberg passages have `source` tag containing author/book info
- Reddit posts have `source: "reddit-theonion"` or `source: "reddit-nottheonion"`
- Use source tag to determine persona (Twain/Franklin detection from book metadata)
- Preserve `matched_keywords` for domain tag determination
- Preserve provenance tags that don't bloat training data unnecessarily

**Validation Requirements:**
- Each entry must have valid messages array
- Each message must have role and content fields
- Roles must be in valid order (system, user, assistant, [tool])
- Tags field must contain persona, tone, and domain at minimum
- Train + validation counts must equal input count
- Stratification check: tag distributions similar across splits
