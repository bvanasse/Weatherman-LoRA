# LoRA Fine-Tuning Guide – "TwainBot" Humor & Weather Specialist

> Fine-tune a compact LLM with LoRA so it speaks like Mark Twain, cracks Onion-style jokes, and can call a weather API. Designed for a single developer + AI coding agents over a 3-day weekend.

---
## 📝 TL;DR
1. **Day 1 – Data Harvest**: Collect corpora (The Onion, r/NotTheOnion, Mark Twain, Ben Franklin, weather jokes, Open-Weather API docs & example calls). Target ≈100 k lines raw.
2. **Day 2 – Data Refinement**: Clean, deduplicate, tokenize, convert to chat-style JSONL; auto-label tool-call examples; split train/val.
3. **Day 3 – LoRA Training & Eval**: LoRA-adapt a 7 B base (e.g. Mistral-7B-Instruct v0.2). Tune for 3-4 h on single A100 / 24 h on consumer GPU. Evaluate with custom humor+API benches. Ship the `.pt` adapter + usage README.

Outcome: An LLM that answers like Twain, jokes about weather, and returns JSON weather API calls when asked.

---
## 1. Project Goals
• Demonstrate end-to-end data stewardship & LoRA impact.
• Produce two adapters:
  a. **Style-Only** – author humor corpus.
  b. **Style + Tool** – corpus + function-call exemplars.
• Compare output quality & hallucination rate.

---
## 2. Phased 3-Day Weekend Plan
### Evening Before (½ h)
1. Install tooling (`conda`, `torch`, `transformers`, `peft`, `datasets`, `trafilatura`, `beautifulsoup4`, `playwright`, `pydantic`).
2. Fork this repo; create `data/`, `scripts/`, `notebooks/`.

### Day 1 – Data Harvest (6 h)
Task | Tooling | Notes
--- | --- | ---
Scrape The Onion site (RSS/plaid-bytes) | `playwright` + `trafilatura` | Target ≥10 k articles.
Pull r/TheOnion & r/NotTheOnion comments | Pushshift, `praw` | Humor in social context.
Download Mark Twain corpus | Project Gutenberg API | Filter essays, speeches.
Download Ben Franklin writings | Gutenberg | Letters & aphorisms.
Gather weather jokes | Kaggle "short-jokes", regex filter “weather” | Augment via GPT paraphrasing.
Collect weather API docs & examples | cURL scrape | Later used to craft tool-call demonstrations.

Deliverable: `raw/` folder with source-labeled `.txt` files and metadata CSV.

### Day 2 – Data Refinement (6 h)
1. **Cleaning Pipeline** (`scripts/clean.py`):
   • Strip HTML, ads, stage directions.
   • Remove Gutenberg headers/footers.
   • Language detect → keep English.
   • Drop duplicates with MinHash.
2. **Segmentation**:
   • Split humor pieces into joke units (newline/new-sentence heuristics).
   • Split prose into ≤512-token chunks, keep paragraph boundaries.
3. **Conversion to Chat JSONL**:
   • For style data: `{ "messages": [{"role":"user","content":"Say something about the mississippi."}, {"role":"assistant","content":"<Twain-style response>"}] }`
   • For API data: include `"tool": {"name":"getWeather", "arguments":{...}}` messages.
4. **Label Generation**:
   • Auto-generate user prompts with GPT-4o for each chunk (5½c/1k).
   • Validate with regex & script QA.
5. **Split**: 90 % train / 10 % val.

### Day 3 – LoRA Training & Evaluation (6 h)
1. **Select Base**: `mistralai/Mistral-7B-Instruct-v0.2` (Apache 2.0).
2. **Config** (`scripts/train_lora.py`)
   • r = 64, α = 16, dropout = 0.05.
   • Target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`.
   • Batch = 128 seq/8 grad acc, lr = 2e-4, cosine-decay.
3. **Train**: `accelerate launch …` (≈3 h on A100 80 GB; set epochs = 1, steps ≈ 800).
4. **Eval**:
   • Automated humor test set (punchline detection BLEU, HUMO score).
   • Tool-call correctness (JSON schema validation).
   • Manual QA: 30 prompts side-by-side (baseline vs adapters).
5. **Package**: Save adapter weights `.safetensors`, push to Hugging Face, update README with usage snippet.

---
## 3. Data Generation & Augmentation Techniques
### 3.1 Parsing Tricks
• `trafilatura` for readability extraction.
• Use GPT-4o to rewrite headlines into prompts.
• Deduplicate with `datasketch` MinHashLSH.

### 3.2 Synthetic Data
Method | Purpose | Example
--- | --- | ---
Prompt Paraphrasing | Increase style variety | "Rewrite this joke in 3 ways keeping Twain tone."
Topic Injection | Ensure weather coverage | "Write a Twain-style quip about humidity in July."
Tool-Call Synthesis | Teach JSON calls | Provide system + user instruct; generate assistant JSON.

### 3.3 Safety Filters
• `openai-moderation` / `together-ai` filters before final dataset.

---
## 4. Pre-training vs Fine-Tuning Rationale
• Full pre-training is GPU-week intense → We reuse a strong base.
• LoRA allows weekend-scale adaptation (<5 % parameters) and adapter swapping.
• Separate adapters enable ablation (style-only vs style+tool).

---
## 5. Expected Outcomes & Metrics
Metric | Target
--- | ---
Humor perplexity ↓ | 10 % over base on validation
Joke preference (human, n=30) | ≥70 % prefer adapter
JSON tool-call validity | ≥95 % parses with `json.loads`

---
## 6. Resources & Further Reading
• LoRA paper – Hu et al., 2021  
• Project Gutenberg – https://www.gutenberg.org  
• The Onion RSS – https://www.theonion.com/rss  
• Pushshift Reddit API – https://github.com/pushshift/api  
• Mistral-7B-Instruct v0.2 – https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2  
• PEFT – https://github.com/huggingface/peft  
• trafilatura (web scraping) – https://github.com/adbar/trafilatura  
• HUMO humor metric – https://arxiv.org/abs/2305.06929

---
### Maintain & Iterate
After initial weekend: collect user feedback, append to dataset, re-train for incremental gains.

---
© 2025 TwainBot Labs
