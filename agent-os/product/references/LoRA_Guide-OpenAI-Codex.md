# LoRA Training Guide — OpenAI Codex TwainBot Humor Model

## TLDR
- Goal: fine-tune a compact OpenAI Codex-compatible model that speaks like Mark Twain, delivers weather-centric humor, and can trigger MCP-style weather API tool calls.
- Focus: fast-turnaround LoRA experiment anchored in high-quality humor corpora (The Onion, Twain, Franklin) plus structured tool-call transcripts.
- Weekend plan: Day 1 harvest and prep text + tool-call data; Day 2 run LoRA experiments and evaluate; Day 3 harden the dataset pipeline, document results, and package the model.
- Data approach: combine passive scraping, public-domain corpora, and LLM-assisted synthetic dialogues; enforce rigorous deduplication, style tagging, guardrails, and versioning.
- Success metric: model hits >85% style-alignment in eval rubrics, <5% factual weather errors with tool call assistance, and retains base coding abilities.

## Overview
This guide outlines how a seasoned developer, supported by tuned AI coding agents, can complete a 3-day LoRA fine-tuning sprint to produce “TwainBot,” a humorous assistant that:
- adopts a Mark Twain/Benjamin Franklin narrative tone,
- leans on weather-themed jokes and wordplay similar to The Onion,
- demonstrates tool-augmented responses by making weather API calls when appropriate.

The plan emphasizes data quality and iteration velocity. It assumes access to an OpenAI Codex-aligned base model (e.g., `gpt-3.5-turbo` or an equivalent instruction/completion model converted for LoRA fine-tuning) and a GPU workstation or cloud instance (A100/40GB or better) with weekend availability.

## Target Outcomes
- **Persona fidelity:** Twain-style prose, Franklin aphorisms, and Onion-esque humor.
- **Weather wit:** Capable of generating quips anchored in current (or simulated) weather data.
- **Tool competence:** Responds with structured MCP-style tool requests (`get_weather(location, datetime)`), integrates tool output gracefully.
- **Documentation:** Reproducible dataset pipeline, LoRA config, evaluation rubric, and deployment checklist.

## Baseline & Assumptions
- Base model supports instruction-tuned completions and function/tool call annotations.
- Training budget: ~20 GPU-hours for experimentation; core LoRA run <6h on A100.
- Data volume target: 30k–60k high-quality samples (mix of author corpora, jokes, dialogues, tool-call transcripts).
- Tooling stack: Python 3.11, `transformers`, `peft`, `bitsandbytes`, `datasets`, `langchain`, `playwright` or `newspaper3k` for scraping, `OpenAI` or `Anthropic` API credits for synthetic data, managed vector store (e.g., Chroma) for dedupe.

## Data Strategy (Pre-Training & Fine-Tuning)
### 1. Source Inventory
- **Public Domain Author Texts:**
  - Mark Twain works from Project Gutenberg (e.g., “Life on the Mississippi,” “The Innocents Abroad”).
  - Benjamin Franklin essays, letters, Poor Richard’s Almanack.
- **Humor & Satire:**
  - The Onion archives (respect robots.txt; prioritize RSS feeds and topical categories).
  - Open-source joke datasets (e.g., Kaggle’s “Short Jokes,” Reddit Jokes if licensed).
- **Weather Domain:**
  - NOAA weather bulletins, National Weather Service public alerts for grounding facts.
  - Synthetic humorous weather summaries generated under guidance.
- **Tool-Call Transcripts:**
  - Crafted dialogues featuring `weather_api.get_forecast` calls.
  - Real logs (if available) from existing weather chatbot prototypes.

### 2. Acquisition Methods
- **Automated Scraping:** Use Playwright headless browser with polite delays and caching. Extract article title, body, publish date, tags. Implement fallback to RSS/JSON endpoints.
- **Bulk Downloads:** Pull public domain books via `gutenbergpy` or standard HTTP downloads. Parse using `beautifulsoup4` while stripping licensing headers/footers.
- **API Harvesting:** Query NOAA/NWS APIs for canonical weather descriptions to ground comedic riffs.
- **Existing Datasets:** Mirror open-source joke corpora; verify licensing compatibility.

### 3. Parsing & Structuring Pipelines
- Normalize text to UTF-8, remove HTML, and chunk into 512–1,024 token windows with overlap for prose continuity.
- Tag each chunk with metadata (`source`, `style=twain|franklin|onion|joke|weather_fact`, `tone`, `topic`, `copyright_status`).
- For The Onion articles, capture headline + lede + body; maintain satire flag.
- Apply deduplication (MinHash/SimHash) across corpora to avoid repetitive training signals.
- Store intermediate artifacts in Parquet/JSONL, versioned in DVC or LakeFS.

### 4. Data Quality & Guardrails
- Run automated toxicity/offensiveness classifier (e.g., Detoxify) to filter problematic satirical content while preserving humor.
- Flag and review politically sensitive or disallowed topics.
- Use AI agents to assist manual spot checks: prompt to rate Twain-likeness, humor density, and weather relevance.
- Maintain separate validation split (10–15%) with stratified sampling across sources.

### 5. Synthetic & Augmented Data Generation
- **Style Transfer Prompts:** Use a strong LLM (e.g., GPT-4o) to rewrite weather reports into Twain-esque jokes, ensuring variety in tone.
- **Contrastive Pairs:** Generate both high-quality and intentionally bland responses to support preference-style fine-tuning.
- **Dialogue Fabrication:** Author multi-turn conversations where TwainBot queries a weather API, jokes about forecasts, and handles user follow-ups.
- **Structured JSON:** Ensure tool call examples use a consistent schema:

```startLine:endLine:references/LoRA_Guide-OpenAI-Codex-schema-example.json
{
  "conversation_id": "uuid",
  "turns": [
    {"role": "user", "content": "What's the weather in Hannibal tomorrow?"},
    {
      "role": "assistant",
      "content": null,
      "tool_call": {
        "tool_name": "weather_api.get_forecast",
        "arguments": {"location": "Hannibal, MO", "datetime": "2025-11-04"}
      }
    },
    {"role": "tool", "content": {"summary": "Cloudy, high 62F, chance of rain 30%"}},
    {"role": "assistant", "content": "Well now, the sky's fixing to drape Hannibal in a gray shawl..."}
  ],
  "style_tags": ["twain", "weather", "humor"]
}
```

- **Reinforcement Dataset:** Create rubric-based feedback (style, humor, factuality) to support optional RLHF/constitutional tuning.

### 6. Safety & Compliance Checks
- Respect licensing; The Onion is not public domain—limit to allowable excerpts, or replace with satirical datasets that permit use.
- Document provenance metadata for all samples.
- Add disclaimers for synthetic data—label origin and generation prompt.

## LoRA Training Workflow
1. **Base Model Preparation:** Obtain transformer checkpoint compatible with OpenAI Codex style completions. Convert to Hugging Face format if required.
2. **Parameter-Efficient Setup:** Use `peft.LoraConfig` with low-rank adapters on attention/query/value projections and output layers. Suggested starting point:
   - `r=16`, `alpha=32`, `dropout=0.05`, `target_modules=["q_proj","k_proj","v_proj","o_proj"]`.
   - Mixed precision (`bf16`) and gradient checkpointing to fit GPU memory.
3. **Curriculum Strategy:**
   - Stage 1 (DAPT): Unsupervised language modeling on Twain/Franklin corpora (no instructions) to nudge base style.
   - Stage 2 (SFT): Instruction-style fine-tuning with humor + tool-call examples.
   - Stage 3 (Optional Preference Fine-Tuning): Pairwise or reward modeling using synthetic quality annotations.
4. **Batching:** Sequence length 1,024 tokens, effective batch size 128 (via gradient accumulation). Learning rate 2e-4 for DAPT, 1e-4 for SFT.
5. **Training Duration:** 1–2 epochs for DAPT (to avoid catastrophic forgetting), 2–3 epochs for SFT.
6. **Evaluation Loop:** After each epoch, run scripted evals: style similarity classifier, humor score, tool-call accuracy.

## Experiment & Evaluation Plan
- **Automated Metrics:** Perplexity on validation splits; style classifier agreement; function-call precision/recall.
- **Human/Agent Review:** AI agent generates 50 sample dialogues; human rates on 1–5 scale for Twain voice, humor, correctness.
- **Weather Grounding Test:** Provide recent forecast data; ensure responses incorporate tool output within 2 turns.
- **Regression Tests:** Ensure code-generation benchmarks (e.g., HumanEval subset) remain within 5% of baseline.
- **Logging:** Track metrics in Weights & Biases or MLflow. Save LoRA weights + tokenizer + config in Git LFS.

## 3-Day Weekend Execution Plan
### Day 0 (Optional Prep Evening)
- Provision GPU environment, confirm CUDA drivers.
- Clone repos, set up virtualenv/conda, seed DVC storage.
- Draft scraping prompts and confirm legal boundaries with stakeholders.

### Day 1 — Data Harvest & Curation
- **Morning:**
  - Run AI agent-assisted scripts to scrape The Onion feeds, ingest Gutenberg texts.
  - Normalize, dedupe, and tag raw corpora; store metadata in `data/raw/`.
- **Afternoon:**
  - Generate synthetic Twain-style weather jokes (1000+ samples) via LLM prompts; enforce content filters.
  - Compile tool-call dialogues: 500+ conversation traces with weather API JSON stubs.
- **Evening:**
  - Finalize train/val/test splits; push artifacts to DVC remote.
  - Kick off DAPT stage overnight with automated monitoring.

### Day 2 — Fine-Tuning & Rapid Evaluation
- **Morning:**
  - Review DAPT metrics; adjust learning rate if perplexity spikes.
  - Launch SFT LoRA run on curated instruction dataset.
- **Afternoon:**
  - Run evaluation harness (automated + agent-assisted). Capture failure cases.
  - Iterate on augmentation (e.g., more Franklin aphorisms) if style weak.
- **Evening:**
  - Optional preference tuning/RLHF light pass using reward model or rejection sampling.
  - Save best checkpoint; export adapters in `safetensors` format.

### Day 3 — Hardening, Tool Integration, Documentation
- **Morning:**
  - Integrate LoRA adapters into endpoint or CLI demo. Ensure MCP tool call schema works end-to-end.
- **Afternoon:**
  - Conduct human eval session; gather qualitative feedback.
  - Tune prompt templates for inference (system prompts enforcing Twain persona, humor guardrails, tool-call instructions).
- **Evening:**
  - Polish documentation: dataset lineage, training scripts, evaluation results.
  - Package release bundle (LoRA weights, inference notebook, README).

## Tooling & Automation Recommendations
- **AI Coding Agents:** Use them to scaffold scraping scripts, cleaning notebooks, and evaluation dashboards; enforce human review for final merges.
- **Pipelines:** Orchestrate with `Prefect` or GitHub Actions to rerun ingestion + training reliably.
- **Version Control:** Tag dataset versions (`data-v1.0`, `dialogues-v1.1`). Maintain model registry entries (`twainbot-lora-v0.1`).
- **Observability:** Set up alerting on training loss divergence and evaluation metric regressions.

## Risk Mitigation & Stretch Goals
- **Legal/Compliance:** If Onion scraping restrictions arise, pivot to Onion-like open sources or rely on synthetic satire generation.
- **Style Drift:** If Twain voice dominates humor to monotony, introduce Franklin wit samples and adjust sampling weights.
- **Tool Failure:** Mock weather API responses for offline testing; include error-handling examples in dataset.
- **Stretch:** Explore multi-LoRA adapters (Twain vs. Franklin) with dynamic routing; evaluate mixture-of-personas at inference.

## Deliverables Checklist
- Curated dataset (raw + processed) with provenance metadata.
- LoRA training scripts + configs stored in `training/` directory.
- Evaluation report summarizing metrics, sample outputs, and known limitations.
- Deployment guide for attaching LoRA to inference stack with MCP tool calls.
- Retro doc capturing lessons learned and next iteration ideas.


