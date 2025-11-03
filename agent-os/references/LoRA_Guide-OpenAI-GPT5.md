## LoRA Training Guide (OpenAI-style Tool Use + Humor/Author Style)

### TL;DR
- **Outcome**: A small, focused LoRA/QLoRA adapter on a 7B–8B instruct model that talks like Mark Twain, can tell weather-related jokes, and reliably performs OpenAI-style tool calls for a weather API.
- **What to prioritize**: High-quality data prep over sheer data volume. Curate, clean, dedupe, and label style and tool-use data; then train 1–2 targeted SFT runs.
- **Data**: Use public-domain author corpora (Twain, Franklin), Reddit `r/TheOnion` and `r/nottheonion` (you already have CSVs and notebooks), plus restrained synthetic generation to fill specific gaps (weather humor, tool-use transcripts).
- **Schedule (3 days)**: Day 1 data pipeline + baseline set; Day 2 synthetic expansion + style-only LoRA; Day 3 tool-use dataset + final LoRA; ship a demo via an OpenAI-compatible server.

---

### Overview
This guide details how to produce a weekend-sized LoRA/QLoRA fine-tune that demonstrates: (1) data management best practices, (2) author style conditioning (Mark Twain/Benjamin Franklin), (3) humor/jokes focused on weather, and (4) OpenAI-style tool calling with a weather API. The emphasis is on pre-training-style data preparation (collection, normalization, deduplication, tagging) and training data construction for supervised fine-tuning (SFT).

#### Goals
- Speak in a Twain-esque voice with optional Franklin flavor; be witty and concise.
- Make weather-related jokes while remaining contextually appropriate.
- Use OpenAI-style tool calls (e.g., `get_weather`) and integrate tool results into answers.
- Provide a reproducible data pipeline and a demonstrative evaluation harness.

#### Deliverables
- Curated, deduped, labeled JSONL datasets (style, humor, tool-use).
- LoRA adapters (style-only and style+tool-use) for a 7B–8B base model.
- Evaluation reports (automatic + spot human eval) and example conversations.
- A runnable demo via an OpenAI-compatible server with tool schema.

#### Assumptions
- Compute: single modern GPU (24–80 GB VRAM) or Mac M-series with 4-bit QLoRA. Batch sizes adjusted accordingly.
- Base models: Llama 3.1 8B Instruct or Mistral 7B Instruct.
- Licenses: Only collect/redistribute data allowed by license/TOS. Prefer public-domain (Project Gutenberg) and your existing Reddit datasets.

---

## Data Strategy (Pre-Training-style Prep + SFT Construction)
Focus on building a small, clean, high-signal dataset. The strongest weekend ROI comes from: (1) clean public-domain author text, (2) curated humor examples, (3) targeted synthetic augmentation, and (4) carefully constructed tool-use transcripts.

### Primary Sources
- **Public-domain authors** (Project Gutenberg):
  - Mark Twain (Samuel Clemens): novels, essays, speeches. Public domain selections only.
  - Benjamin Franklin: writings, letters, Poor Richard’s Almanack excerpts.
  - Method: download plain text from Project Gutenberg; record source URLs and licenses.

- **Humor/jokes (satire, headlines)**:
  - Existing workspace: `data_sources/reddit-theonion/` (CSV + notebooks for scraping/cleaning). This is ideal for quick bootstrapping.
  - Additional signals: `r/TheOnion`, `r/nottheonion` posts and titles; optionally add punchline-bearing posts from joke subreddits (if license/TOS permits and if you have time).
  - Site scraping (theonion.com) may be subject to TOS; prefer their RSS, cached content, or Reddit mirrors to avoid policy risk.

- **Weather-domain grounding**:
  - Short factual snippets (weather terminology, units, common conditions), preferably CC/public domain.
  - Tool output examples from a real API (e.g., Open-Meteo) or a local stub to avoid live calls during dataset build.

### Collection Methods
- **Reddit (preferred; you already have data + notebooks)**
  - Use existing CSVs in `data_sources/reddit-theonion/data/` and notebooks in `data_sources/reddit-theonion/notebooks/`.
  - Normalize fields (title, selftext, score, subreddit, UTC, permalink) and keep moderation-safe content.

- **Project Gutenberg (Twain, Franklin)**
  - Programmatically download plain text; remove front-matter and boilerplate.
  - Keep metadata: author, title, year, source URL, public-domain flag.

- **Web (optional, time-permitting)**
  - Use `trafilatura` or `newspaper3k` on permissive sources only. Cache raw HTML + extracted text. Respect `robots.txt` and site TOS.

### Parsing & Normalization Pipeline
Apply a simple, consistent text pipeline:
1. Strip HTML, normalize unicode, normalize quotes and whitespace.
2. Drop boilerplate (Gutenberg headers/footers), menus, nav text.
3. Enforce language filter (English) via `fasttext` or `langdetect`.
4. Remove near-duplicates with MinHash/LSH or SimHash. Keep canonical, higher-quality versions.
5. Token-length bounds: drop samples that are too short (< 16 tokens) or too long for your max sequence; or chunk long texts.
6. Safety filters: remove NSFW, doxxing, and slurs; prefer high-signal humor without harassment.

### Chunking Rules (Author/Comedy-Aware)
- Do not break a single joke/headline + punchline across chunks.
- For long-form author text, chunk by paragraphs with overlap (e.g., 128–256 tokens) and carry metadata tags to each chunk.
- For dialog-like material, chunk at natural turn boundaries.

### Labeling & Control Tags
Annotate each sample with lightweight tags to support controlled generation:
- `persona`: `twain`, `franklin`, or `neutral`.
- `tone`: `humorous`, `wry`, `satirical`, `didactic`, `aphoristic`.
- `domain`: `weather`, `general`.
- `source`: `gutenberg`, `reddit`, etc.
- `device` (optional): `pun`, `hyperbole`, `analogy`, `sarcasm`, `headline`.

These can be injected as control tokens in prompts (e.g., `[PERSONA=TWAIN] [TONE=HUMOROUS] [DOMAIN=WEATHER]`). Keep tokens consistent.

### Instructionalization (Turning corpora into SFT examples)
Convert raw text into chat-like supervised pairs. Use a few recurring templates:
- **Style Imitation**
  - User: “Rewrite the following passage in the style of Mark Twain.” + content
  - Assistant: Twain-like output.

- **Humor Generation**
  - User: “Tell a short, witty joke about today’s weather.”
  - Assistant: Short joke; optionally condition on `location` or `condition`.

- **Summarize as a Twain-esque aphorism**
  - User: “Summarize the following as a dry, Twain-like aphorism.”
  - Assistant: One-liner.

- **Safety-damped variants**
  - User asks for unsafe joke → Assistant produces a safe alternative or refusal.

Balance the templates across sources. Keep responses concise and consistent with tags.

### Synthetic Data (Gap-Filling, Not Volume-Inflation)
Use a capable base LLM to produce small, high-quality synthetic sets in targeted areas:
- Style transfer: Convert neutral summaries into Twain/Franklin tone.
- Weather humor: Generate jokes conditioned on `location`, `season`, and `condition`.
- Paraphrase diversity: Back-translation for 1–2x variety without losing tone.
- Negative/edge cases: Bad tool outputs (e.g., missing data) to encourage robust handling.

Cap synthetic expansion to 2–4× the human corpus (weekend scope). Always label synthetic vs. human for analysis.

### Tool-Use Dataset (OpenAI-style function calling)
Represent tool calls in OpenAI chat format with `tool_calls` (newer) or `function_call` (older). Include the tool’s JSON result as a `tool` role turn, then a final assistant turn grounding on the result.

Example JSONL record (messages-based, newer schema):
```json
{
  "messages": [
    {"role": "system", "content": "You are a witty assistant who speaks like Mark Twain. Use tools when needed."},
    {"role": "user", "content": "Will it rain in Austin tomorrow? Make it funny."},
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_01",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\":\"Austin,TX\",\"date\":\"2025-11-03\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_01",
      "content": "{\"location\":\"Austin,TX\",\"date\":\"2025-11-03\",\"forecast\":\"light rain\",\"high_c\":22,\"low_c\":14}"
    },
    {
      "role": "assistant",
      "content": "Well now, Austin’s due a polite sprinkle tomorrow: high of 22°C, low of 14°C. Take an umbrella, unless you fancy auditioning for the role of ‘soggy cowboy.’"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get daily forecast for a city and date.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"},
            "date": {"type": "string", "description": "ISO date"}
          },
          "required": ["location", "date"]
        }
      }
    }
  ],
  "tags": {"persona":"twain","tone":"humorous","domain":"weather","source":"synthetic"}
}
```

For open models, you’ll fine-tune on these structured turns. The evaluation harness should verify both (a) correct `tool_calls` JSON and (b) final answer grounds on the tool output.

### Dataset Sizes (Weekend Target)
- Author style and humor: 8k–25k total pairs after dedupe.
- Tool-use: 1k–3k tool-call episodes covering success, errors, and edge cases.
- Holdout eval sets: 10% from each category, stratified by tags.

---

## Training Setup (LoRA/QLoRA)

### Base Model
- Llama 3.1 8B Instruct or Mistral 7B Instruct.
- Sequence length: 4k preferable; 2k acceptable if memory-bound.

### QLoRA/LoRA Hyperparameters (starting points)
- Quantization: 4-bit NF4 (QLoRA) with double quant.
- LoRA rank `r`: 16 (8–32) | LoRA alpha: 32 (16–64) | Dropout: 0.05.
- Learning rate: 2e-4 (LoRA adapters) with cosine decay, warmup 3%.
- Epochs: 1–2 for style-only; +1 for tool-use; early stop on eval.
- Micro-batch: 1–4 depending on VRAM; grad accumulation to reach effective batch 64–256.
- Gradient checkpointing: on. Flash attention if supported.
- Sequence packing: on for short-turn datasets.

### Training Script Sketch (TRL + PEFT)
```python
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # or mistralai/Mistral-7B-Instruct-v0.3
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

trainer = SFTTrainer(
    model=AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto"),
    tokenizer=tokenizer,
    train_dataset="/path/to/train.jsonl",  # messages-format
    eval_dataset="/path/to/eval.jsonl",
    dataset_text_field=None,  # using messages -> rely on collator
    peft_config=lora_config,
    max_seq_length=4096,
    packing=True,
    num_train_epochs=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    gradient_checkpointing=True,
    gradient_accumulation_steps=8,
    per_device_train_batch_size=1,
    logging_steps=10,
    save_steps=500,
    output_dir="./outputs/style_tooluse_lora"
)

trainer.train()
trainer.save_model("./outputs/style_tooluse_lora_adapter")
```

Note: If your framework expects a `messages` field, ensure your data collator converts chat records into the right prompt/response format with tool turns preserved.

### Evaluation
- **Automatic**:
  - Style classifier (twain vs. neutral) accuracy/F1 on holdout.
  - Tool-use correctness: JSON schema validity, required args present, and final answer uses returned fields.
  - Toxicity/offensiveness: keep within safe bounds.

- **LLM-as-judge (quick)**:
  - Score 50–100 samples on style adherence (Twain-ness 1–5) and humor (wit 1–5).
  - Judge whether the answer is grounded in tool output.

- **Human spot checks**:
  - 20–30 conversations. Confirm tone, correctness, and absence of hallucinated weather facts when tool is present.

### Packaging & Serving
- Save adapter separately; optionally merge for export.
- Serve via vLLM or text-generation-inference with an OpenAI-compatible endpoint.
- Provide the tool schema to the server runtime. Example tool schema:
```json
{
  "name": "get_weather",
  "description": "Get daily forecast for a city and date.",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string"},
      "date": {"type": "string", "description": "ISO date"}
    },
    "required": ["location", "date"]
  }
}
```

System prompt to steer persona at runtime:
```text
You are a witty assistant who speaks like Mark Twain. You use tools when helpful and keep answers concise.
```

---

## Three-Day Weekend Plan (Phased, Achievable)

### Day 1 — Data Foundation (8–10 hours)
- Environment: set up Python, CUDA/MPS, TRL/PEFT.
- Pull Twain/Franklin texts (public domain) and clean.
- Use existing `data_sources/reddit-theonion/` CSVs + notebooks to extract high-signal humor/jokes.
- Build the normalization + dedupe + labeling pipeline. Produce:
  - `author_style.jsonl` (Twain/Franklin, tagged)
  - `humor_weather.jsonl` (weather-related headlines/jokes)
  - Holdout splits (10%).
- Quick QA: token length, language, safety filters; basic statistics report.

Milestone: Baseline clean datasets with tags; small eval set ready.

### Day 2 — Synthetic Expansion + Style LoRA (6–10 hours)
- Generate targeted synthetic data to fill gaps:
  - Weather jokes across locations/seasons.
  - Style-transfer to Twain/Franklin tone for short neutral snippets.
- Cap synthetic 2–3× human volume; label `source=synthetic`.
- Assemble `train_style.jsonl` and `eval_style.jsonl`.
- Train LoRA (style-only) for 1–2 epochs; run automatic and LLM-judge evals.
- Save adapter `./outputs/style_only_adapter` and sample generations.

Milestone: Style-only Twain-esque model validated on holdout.

### Day 3 — Tool-Use Dataset + Final LoRA (6–10 hours)
- Create tool-use episodes:
  - Mix success, missing city/date, and API error cases.
  - Ensure final answers ground on tool output; keep in Twain voice.
- Assemble `train_tooluse.jsonl` and `eval_tooluse.jsonl` and merge with style data.
- Train final LoRA for +1 epoch; evaluate tool JSON validity and groundedness.
- Serve via OpenAI-compatible endpoint; add system prompt and tool schema.
- Record demonstrations and write a brief README.

Milestone: Style+tool-use model demo running locally with weather jokes.

---

## Options & Methods to Parse or Generate Training Data

### Parsing/Extraction
- `trafilatura` or `newspaper3k` for HTML extraction from permissive sources.
- Reddit CSVs + PRAW (if needed) for additional posts; use your existing notebooks.
- Regex cleanup for headlines, stripping bracketed tags and URLs.

### Deduplication
- MinHash/LSH via `datasketch` or SimHash; threshold ~0.8 Jaccard.
- Within-source and cross-source dedupe; prefer the highest-quality copy.

### Labeling
- Rule-based tags from source + lightweight classifier for tone (`humorous` vs. not) if time permits.
- Manual spot-labeling on 200–300 samples to calibrate.

### Synthetic Generation
- Style-transfer prompts (few-shot). Back-translation for variation.
- Programmatic templates for weather jokes: vary `location`, `season`, `condition`.
- Tool-use transcripts: either real API calls (cached) or a stub that returns plausible JSON.

---

## Appendix

### JSONL Schema (messages-based)
```json
{
  "messages": [
    {"role": "system", "content": "You are a witty assistant who speaks like Mark Twain."},
    {"role": "user", "content": "Tell me a short joke about windy weather in Chicago."},
    {"role": "assistant", "content": "Chicago’s wind is so proud it tips your hat before you do."}
  ],
  "tags": {"persona":"twain","tone":"humorous","domain":"weather","source":"human"}
}
```

### Quick Dedupe Snippet (MinHash/LSH idea)
```python
from datasketch import MinHash, MinHashLSH

def mhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for token in set(text.lower().split()):
        m.update(token.encode('utf-8'))
    return m

texts = [...]
signatures = [mhash(t) for t in texts]
lsh = MinHashLSH(threshold=0.8, num_perm=128)
for i, sig in enumerate(signatures):
    lsh.insert(f"doc_{i}", sig)
```

### Synthetic Weather Template (prompt idea)
```text
Rewrite a one-liner joke about weather.
Persona: Mark Twain. City: {city}. Season: {season}. Condition: {condition}.
Keep it under 25 words, dry wit, no insults.
```

### Adapter Merge (optional, export to full model)
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("./outputs/style_tooluse_lora_adapter")
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model.save_pretrained("./outputs/merged_model")
tokenizer.save_pretrained("./outputs/merged_model")
```

---

## Notes on Licensing and Safety
- Use only public-domain author texts and data acquired in compliance with TOS/licensing.
- Label synthetic vs. human data for analysis; avoid synthetic dominance.
- Guard rails: short, non-targeted humor; avoid personal or protected characteristics.

---

## What to Measure (to show impact of data choices)
- Style adherence delta (before vs. after LoRA) on a fixed prompt set.
- Tool-call validity rate and groundedness score.
- Humor quality (LLM judge + human spot checks).
- Overfit signals: training vs. eval loss gap, duplicate n-grams in generations.


