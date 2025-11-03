# Synthetic Tool-Use Data Generation

Generate 1,000-3,000 OpenAI-style function calling conversation examples using Claude Haiku 4.5 API to create realistic weather tool-use training data with persona integration.

## Quick Start

### Prerequisites

1. **Anthropic API Key**: Get one from [console.anthropic.com](https://console.anthropic.com/)
2. **Environment Setup**: Run `./setup_local.sh` or `./setup_m4.sh`
3. **Dependencies**: `pip install anthropic` (included in requirements)

### Generate Synthetic Data

```bash
# Option 1: Interactive (will prompt for API key)
python scripts/generate_synthetic_tool_data.py --count 1000

# Option 2: Set API key first
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/generate_synthetic_tool_data.py --count 1000

# Option 3: Custom output location
python scripts/generate_synthetic_tool_data.py --count 2000 --output data/synthetic/custom.jsonl

# Option 4: Test mode without API calls
python scripts/generate_synthetic_tool_data.py --count 10 --mock
```

## API Key Management

### Method 1: Environment Variable (Recommended)

Add to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or set temporarily:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/generate_synthetic_tool_data.py --count 1000
```

### Method 2: Interactive Prompt

If `ANTHROPIC_API_KEY` is not set, the script will prompt you:

```
Anthropic API key not found in environment.
Please enter your API key (or set ANTHROPIC_API_KEY environment variable):
API Key: [enter your key here]
```

### Method 3: `.env` File (Future)

Create `.env` in project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

*Note: Ensure `.env` is in `.gitignore` to avoid committing secrets*

## What Gets Generated

### Output Format

**File**: `data/synthetic/tool_use_examples.jsonl`

**Format**: OpenAI-style function calling conversations

```jsonl
{
  "messages": [
    {"role": "system", "content": "You are Weatherman, a weather assistant..."},
    {"role": "user", "content": "What's the weather in San Francisco?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {
            "name": "get_current_weather",
            "arguments": "{\"latitude\": 37.7749, \"longitude\": -122.4194}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_123",
      "content": "{\"temperature\": 62, \"condition\": \"partly cloudy\", \"wind_speed\": 12}"
    },
    {
      "role": "assistant",
      "content": "Well now, San Francisco's sitting at a mild 62°F with partly cloudy skies..."
    }
  ],
  "tags": {
    "persona": "twain",
    "tone": "humorous",
    "domain": "weather",
    "source": "synthetic"
  },
  "conversation_id": "conv_20251102_001"
}
```

### Persona Distribution

- **60% Neutral**: Professional assistant tone, focus on accurate tool usage
- **25% Twain**: Witty, humorous responses with literary flair
- **15% Franklin**: Didactic, almanac-style wisdom

### Scenario Coverage

- **60-70% Success Cases**: Valid tool calls with realistic weather responses
- **15-20% Error Handling**: Invalid locations, missing parameters, out-of-range values
- **15-20% Multi-Turn**: Follow-up questions, clarifications, multiple locations

### Geographic Diversity

- 50+ different cities across continents
- Various climate zones (tropical, temperate, arctic, desert)
- Edge cases: extreme weather, null island, timezone differences

## Cost & Performance

### API Costs (Claude Haiku 3.5)

**Pricing** (as of Nov 2024):
- Input: $0.25 per million tokens
- Output: $1.25 per million tokens

**Estimates**:
- 1,000 examples: ~1-2M tokens → **$0.50-$1.00**
- 2,000 examples: ~2-4M tokens → **$1.00-$2.00**
- 3,000 examples: ~3-6M tokens → **$1.50-$3.00**

### Generation Time

**On M4 Mac** (with 0.5s rate limiting):
- 1,000 examples: ~30-45 minutes
- 2,000 examples: ~60-90 minutes
- 3,000 examples: ~90-135 minutes

**Parallelization**: Single-threaded to respect API rate limits

## Features

### Automatic Retry Logic

- **Exponential Backoff**: 1s, 2s, 4s delays between retries
- **Max Retries**: 3 attempts per API call
- **Graceful Degradation**: Skips failed examples after max retries

### Rate Limiting

- **Default Delay**: 0.5s between API calls
- **Respects**: Anthropic API rate limits
- **Configurable**: Adjust in `ClaudeAPIClient` initialization

### Progress Tracking

```
Generating: 423/1000 (42.3%) - Validated: 415, Failed: 8
```

- Real-time progress bar (uses `tqdm` if available)
- Validation counts (passed/failed)
- Batch statistics every 100 examples

### Checkpointing

- **Auto-Save**: Every 250 examples
- **Resumption**: Resume from last checkpoint on failure
- **Location**: `data/synthetic/checkpoints/`

### Graceful Shutdown

Press `Ctrl+C` to interrupt:

```
Interrupt received. Finishing current example...
Save partial results? (y/n):
```

## Validation

### Automatic Validation Checks

1. **Schema Validation**
   - Valid JSON in tool call arguments
   - Correct function names and parameter types
   - Proper role ordering (system, user, assistant, tool)

2. **Semantic Validation**
   - Temperature ranges reasonable for location
   - Valid condition codes
   - Coordinates within valid ranges

3. **Groundedness Validation**
   - Assistant responses reference tool output data
   - No hallucinated weather values

### Validation Report

Output: `data/synthetic/validation_report.json`

```json
{
  "total_examples": 1000,
  "validation_passed": 985,
  "validation_failed": 15,
  "pass_rate": 98.5,
  "error_breakdown": {
    "invalid_json": 5,
    "missing_tool_call": 3,
    "grounding_failure": 7
  }
}
```

## Workflow Integration

### Recommended Workflow

**Step 1: Generate Locally on M4**
```bash
# Generate synthetic data (30-60 min)
python scripts/generate_synthetic_tool_data.py --count 1000
```

**Step 2: Validate Output**
```bash
# Check validation report
cat data/synthetic/validation_report.json

# Inspect sample examples
head -5 data/synthetic/tool_use_examples.jsonl | jq
```

**Step 3: Sync to H100** (if training remotely)
```bash
# Sync processed data
rsync -avz data/synthetic/ username@h100-host:/path/to/weatherman-lora/data/synthetic/
```

**Step 4: Train with Synthetic Data**
```bash
# On H100 or M4
# Training script will automatically load from data/synthetic/
```

## Troubleshooting

### API Key Issues

**Problem**: "Anthropic API key not found in environment"

**Solution**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Or add to ~/.zshrc for persistence
```

### Rate Limit Errors

**Problem**: "Rate limit exceeded"

**Solution**: Script automatically retries with exponential backoff. If persistent:
```bash
# Increase rate limit delay (in code)
# Edit scripts/claude_api_client.py:
# rate_limit_delay=1.0  # Increase from 0.5s to 1.0s
```

### Validation Failures

**Problem**: High validation failure rate (>5%)

**Solution**:
1. Check API response format hasn't changed
2. Review failed examples: `data/synthetic/validation_report.json`
3. Adjust validation thresholds if needed

### Incomplete Generation

**Problem**: Script interrupted or crashed

**Solution**: Resume from checkpoint:
```bash
# Script automatically detects and resumes from last checkpoint
python scripts/generate_synthetic_tool_data.py --count 1000
```

### Mock Mode for Testing

**Problem**: Want to test without API costs

**Solution**:
```bash
python scripts/generate_synthetic_tool_data.py --count 10 --mock
```

## Advanced Usage

### Custom Persona Distribution

Edit `scripts/conversation_orchestrator.py`:

```python
PERSONA_DISTRIBUTION = {
    'neutral': 0.70,  # 70% neutral (increased)
    'twain': 0.20,    # 20% Twain
    'franklin': 0.10  # 10% Franklin
}
```

### Custom Weather Scenarios

Edit `scripts/scenario_generator.py` to add custom scenarios:

```python
CUSTOM_SCENARIOS = [
    {
        'type': 'extreme_weather',
        'location': 'Death Valley',
        'expected_temp_range': (100, 130)
    }
]
```

### Monitoring API Usage

```python
from scripts.claude_api_client import ClaudeAPIClient

client = ClaudeAPIClient()
# ... generate conversations ...
client.print_metrics()
```

Output:
```
============================================================
Claude API Metrics
============================================================
Total Calls: 1000
Successful: 992
Failed: 8
Success Rate: 99.2%

Token Usage:
  Input Tokens: 850,234
  Output Tokens: 1,234,567
  Total Tokens: 2,084,801

Performance:
  Avg Latency: 1.45s
```

## Output Files

```
data/synthetic/
├── tool_use_examples.jsonl       # Main output (1000-3000 conversations)
├── validation_report.json         # Validation results
├── generation_metadata.json       # Generation config and timestamp
└── checkpoints/                   # Resumption checkpoints
    ├── checkpoint_250.jsonl
    ├── checkpoint_500.jsonl
    └── checkpoint_750.jsonl
```

## Best Practices

1. **Generate Locally First**: Run on M4 Mac before training on H100 to validate pipeline
2. **Start Small**: Test with `--count 100` before generating full dataset
3. **Monitor Costs**: Check API usage via Anthropic console
4. **Validate Output**: Review validation report before training
5. **Version Control**: Save generation metadata for reproducibility
6. **Backup Checkpoints**: Keep checkpoint files until generation completes
7. **Test Mock Mode**: Use `--mock` for pipeline testing without API costs

## See Also

- [Training Quickstart](TRAINING_QUICKSTART.md) - Training with synthetic data
- [Data Sync](DATA_SYNC.md) - Syncing data to H100
- [Literary Corpus](LITERARY_CORPUS.md) - Literary data collection
- [README](../README.md) - Project overview

## API Reference

### ClaudeAPIClient

```python
from scripts.claude_api_client import ClaudeAPIClient

client = ClaudeAPIClient(
    api_key=None,                    # Auto-detect or prompt
    model="claude-3-5-haiku-20241022",
    max_retries=3,
    retry_delays=[1.0, 2.0, 4.0],
    rate_limit_delay=0.5
)

response = client.generate_conversation(
    prompt="Generate a weather query",
    system="You are Weatherman",
    max_tokens=4096,
    temperature=1.0
)
```

### generate_synthetic_tool_data.py

```bash
python scripts/generate_synthetic_tool_data.py \
  --count 1000 \              # Number of examples
  --output data/synthetic/tool_use_examples.jsonl \
  --mock                      # Test mode without API
```
