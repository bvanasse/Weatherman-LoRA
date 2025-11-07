# Training Data Regeneration Guide

This guide explains how to regenerate your training data to remove repetitive templates and create more diverse, creative responses.

## Problem Identified

Your current training data (`data/synthetic/final_train.jsonl`) has **59% templated responses**:

| Template Pattern | Count | Percentage |
|-----------------|-------|------------|
| Twain "Climate vs Weather" | 2,966 | 20.6% |
| Twain "Forecasts and Promises" | 2,398 | 16.7% |
| Franklin "Early to Bed" | 1,790 | 12.4% |
| Neutral "Info Requested" | 1,348 | 9.4% |
| **Total Templated** | **8,502** | **59.0%** |

### Why This Is Bad

1. **Template Memorization**: Model learns exact phrases instead of style
2. **Boring Responses**: Repetitive, predictable output
3. **Poor Generalization**: Won't handle novel situations well
4. **Overfitting**: High training loss but poor real-world performance

## Solution: Claude-Powered Regeneration

Use Claude Sonnet 4.5 API to regenerate responses with:
- ✅ High diversity (temperature=0.9)
- ✅ Authentic style examples from your literary corpus
- ✅ Anti-template instructions to prevent repetition
- ✅ Preserved tool-calling structure (the good part)
- ✅ Checkpoint recovery for long runs

## Quick Start

### 1. Set Up API Key

```bash
# Get API key from https://console.anthropic.com/
export ANTHROPIC_API_KEY='your-api-key-here'

# Verify it's set
echo $ANTHROPIC_API_KEY
```

### 2. Install Dependencies

```bash
# Install anthropic Python package
pip install anthropic

# Should already be installed if you ran setup
```

### 3. Run Regeneration

```bash
# Simple one-command regeneration
./regenerate_training_data.sh

# Or run Python script directly with custom options
python3 scripts/regenerate_diverse_responses.py \
    --input data/synthetic/final_train.jsonl \
    --output data/synthetic/final_train_diverse.jsonl \
    --max-templates 100 \
    --model claude-sonnet-4-20250514
```

### 4. Verify Results

```bash
# Analyze diversity of new data
python3 scripts/analyze_data_diversity.py data/synthetic/final_train_diverse.jsonl

# Compare with original
python3 scripts/analyze_data_diversity.py data/synthetic/final_train.jsonl
```

### 5. Use New Data for Training

```bash
# Option A: Replace original file
mv data/synthetic/final_train_diverse.jsonl data/synthetic/final_train.jsonl

# Option B: Update symlink
ln -sf $(pwd)/data/synthetic/final_train_diverse.jsonl data/processed/train.jsonl

# Train with new data
./train_with_axolotl_h100.sh
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Input: final_train.jsonl (14,399 examples)            │
│  Problem: 8,502 (59%) use repetitive templates         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Template Detection                                     │
│  • Identify templated responses                         │
│  • Keep first 100 of each pattern (for diversity)      │
│  • Mark 8,402 for regeneration                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Style Example Loading                                  │
│  • Load authentic Twain passages (Gutenberg)            │
│  • Load Franklin almanac entries (Gutenberg)            │
│  • Load Onion-style humor (Reddit dataset)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Response Regeneration (Claude API)                     │
│  For each templated response:                           │
│  • Preserve tool-calling structure                      │
│  • Extract conversation context + tool result           │
│  • Build prompt with:                                   │
│    - Persona instructions                               │
│    - 3 random style examples                            │
│    - Anti-template instructions                         │
│    - Tool result data                                   │
│  • Generate with temp=0.9 (high creativity)             │
│  • Track generated responses to avoid new templates     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Output: final_train_diverse.jsonl                      │
│  Result: 14,399 examples with <10% templated           │
│  Quality: Diverse, creative, authentic style           │
└─────────────────────────────────────────────────────────┘
```

### Key Features

#### 1. Template Detection
Identifies responses containing:
- "weather forecasts and promises are alike"
- "Climate is what we expect, weather is what we get"
- "Early to bed and early to rise"
- "Here's the information you requested for"

#### 2. Style Example Loading
Loads authentic examples from your literary corpus:

**Twain** (`data/processed/gutenberg_passages.json`):
- Extracts Mark Twain passages
- Provides authentic voice samples
- Up to 20 examples per regeneration

**Franklin** (`data/processed/gutenberg_passages.json`):
- Extracts Benjamin Franklin passages
- Poor Richard's Almanack style
- Up to 20 examples per regeneration

**Onion** (`data/processed/reddit_humor_weather.jsonl`):
- Extracts satirical weather headlines
- Absurdist, deadpan humor
- Up to 20 examples per regeneration

#### 3. Diversity Mechanisms

**High Temperature**: `temperature=0.9` for creative, varied output

**Random Style Examples**: Each regeneration gets 3 random examples

**Anti-Template Instructions**: Shows recent responses to avoid

**Context Awareness**: Tracks last 100 responses to prevent new patterns

#### 4. Checkpoint Recovery

If interrupted:
```bash
# Just re-run the script
./regenerate_training_data.sh

# It will automatically resume from checkpoint
```

Checkpoints saved every 100 examples to:
`data/synthetic/final_train_diverse.checkpoint.jsonl`

## Cost Estimation

### API Usage

- **Model**: Claude Sonnet 4.5
- **Examples to regenerate**: ~8,400
- **Tokens per request**: ~800 (400 input + 400 output)
- **Total tokens**: ~6.7M tokens
- **Cost**:
  - Input: $3/million tokens = $20.10
  - Output: $15/million tokens = $50.25
  - **Total: ~$70**

### Time Estimation

- **Rate limit**: 1 request/second (conservative)
- **Total time**: 8,400 seconds = **2.3 hours**

With errors and retries: **~3 hours total**

## Options and Flags

### Python Script Options

```bash
python3 scripts/regenerate_diverse_responses.py \
    --input <path>              # Input JSONL file
    --output <path>             # Output JSONL file
    --corpus-path <path>        # Literary corpus location
    --max-templates <int>       # Max examples per template (default: 100)
    --checkpoint-interval <int> # Checkpoint frequency (default: 100)
    --model <model_name>        # Claude model to use
```

### Available Models

1. **`claude-sonnet-4-20250514`** (Recommended)
   - Latest and most capable
   - Best at creative, diverse output
   - Cost: $3/$15 per million tokens

2. **`claude-3-5-sonnet-20241022`** (Alternative)
   - Slightly older but still excellent
   - Cost: $3/$15 per million tokens

### Template Limits

Control how many templated examples to keep:

```bash
# Keep first 100 of each pattern (default)
--max-templates 100

# Keep more for diversity
--max-templates 200

# Remove all templates (aggressive)
--max-templates 0
```

**Recommendation**: Keep 50-100 for diversity, not zero

## Validation Data

After regenerating training data, also regenerate validation data:

```bash
# Same process for validation set
python3 scripts/regenerate_diverse_responses.py \
    --input data/synthetic/final_validation.jsonl \
    --output data/synthetic/final_validation_diverse.jsonl \
    --max-templates 10  # Fewer examples, proportional
```

## Quality Verification

### Before Training

1. **Check diversity metrics**:
```bash
python3 scripts/analyze_data_diversity.py data/synthetic/final_train_diverse.jsonl
```

2. **Sample random responses**:
```bash
python3 << 'EOF'
import json
import random

with open('data/synthetic/final_train_diverse.jsonl') as f:
    examples = [json.loads(line) for line in f]

for ex in random.sample(examples, 10):
    for msg in reversed(ex['messages']):
        if msg['role'] == 'assistant' and 'tool_calls' not in msg:
            print(f"Persona: {ex['tags']['persona']}")
            print(f"Response: {msg['content']}\n")
            break
EOF
```

3. **Check for new template patterns**:
Look for any repeated phrases in the sample

### After Training

1. **Test with diverse prompts**:
```python
test_prompts = [
    "What's the weather in Tokyo?",
    "Give me a 5-day forecast for Berlin",
    "Is it going to rain in Seattle tomorrow?",
    "What's the temperature in Miami?",
]
```

2. **Verify responses aren't templated**:
Check if model produces the same phrases repeatedly

3. **Compare with baseline**:
Train on both old and new data, compare response quality

## Troubleshooting

### Issue: "ANTHROPIC_API_KEY not set"

**Solution**:
```bash
export ANTHROPIC_API_KEY='your-api-key'
# Add to ~/.bashrc or ~/.zshrc for persistence
```

### Issue: "anthropic package not installed"

**Solution**:
```bash
pip install anthropic
# Or: pip3 install anthropic
```

### Issue: Regeneration is slow

**Causes**:
- Rate limiting (1 req/sec)
- API latency (~2 sec per request)

**Solutions**:
- Run overnight (3 hours total)
- Use checkpoint recovery if interrupted
- No faster option (API rate limits)

### Issue: Out of API credits

**Solution**:
- Add credits to Anthropic account
- Script will fail gracefully
- Use checkpoint to resume after adding credits

### Issue: Some regenerations fail

**Expected**: 1-2% failure rate is normal

**Causes**:
- API timeouts
- Network issues
- Invalid responses

**Script handles this**:
- Tracks failures
- Continues with remaining examples
- Reports failure count at end

**If too many fail** (>5%):
- Check network connection
- Check API key validity
- Review error messages in log

### Issue: New responses still seem repetitive

**Solutions**:
1. Increase `--max-templates` to 0 (remove all templates)
2. Add more style examples to corpus
3. Use higher temperature in script (edit: `temperature=1.0`)
4. Manually review and regenerate specific patterns

## Alternative: GPT-4 or Gemini

Want to use GPT-4 or Gemini instead of Claude?

### GPT-4 Option

```python
# Install OpenAI package
pip install openai

# Set API key
export OPENAI_API_KEY='your-key'

# Modify script to use OpenAI API
# (See scripts/regenerate_diverse_responses_gpt.py)
```

**Pros**:
- Similar quality to Claude
- May produce different style variations

**Cons**:
- More expensive ($30-$60 for same task)
- No checkpoint recovery built-in
- Need separate script modification

### Gemini Option

```python
# Install Google AI package
pip install google-generativeai

# Set API key
export GOOGLE_API_KEY='your-key'

# Use Gemini API
# (See scripts/regenerate_diverse_responses_gemini.py)
```

**Pros**:
- Free tier available
- Fast responses

**Cons**:
- May not match Claude/GPT-4 quality for creative tasks
- Different prompt engineering needed

**Recommendation**: Stick with Claude (me) for best results on this task.

## Best Practices

### Do's

✅ Run regeneration on full dataset before training
✅ Keep backups of original data
✅ Verify diversity before and after
✅ Test sample responses manually
✅ Use checkpoint recovery for long runs
✅ Monitor API costs during regeneration

### Don'ts

❌ Don't remove ALL templates (keep 50-100 for diversity)
❌ Don't regenerate during active training
❌ Don't skip quality verification
❌ Don't train on old data after regenerating
❌ Don't modify script without testing first

## Next Steps

After successful regeneration:

1. **Verify quality** with diversity analysis
2. **Update data symlinks** to point to new files
3. **Retrain model** with Axolotl
4. **Compare results** with baseline model
5. **Deploy** if quality improves

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Total Examples | 14,399 | 14,399 |
| Templated Responses | 8,502 (59%) | ~400 (<3%) |
| Unique Responses | 5,897 (41%) | ~14,000 (97%) |
| Cost | $0 | ~$70 |
| Time | 0 | ~3 hours |
| Quality | Repetitive | Diverse ✨ |

**Result**: Training data with diverse, creative, authentic responses that will produce a much better fine-tuned model!
