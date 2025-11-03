# Product Roadmap

1. [x] Environment Setup & Data Infrastructure — Set up Python environment with TRL/PEFT/transformers, create project directory structure, install GPU dependencies (torch/CUDA), and configure data storage paths for raw/processed datasets `XS`

2. [x] Literary Corpus Collection — Download and parse public domain texts from Project Gutenberg (Mark Twain: Tom Sawyer, Huck Finn, Connecticut Yankee, Life on Mississippi; Benjamin Franklin: Autobiography, Poor Richard's Almanack), extract 3,000-5,000 relevant passages with weather/humor mentions, and preserve source metadata `S`

3. [x] Reddit Humor Dataset Processing — Process existing `data_sources/reddit-theonion/` CSVs to extract weather-related posts from r/TheOnion and r/nottheonion, filter by weather keywords, clean HTML/metadata artifacts, and create 2,000-4,000 labeled humor examples `S`

4. [x] Data Normalization & Deduplication Pipeline — Implement cleaning pipeline with unicode normalization, MinHash/LSH deduplication (threshold 0.8), language filtering (English only), safety filters (toxicity/NSFW removal), and generate quality statistics report `M`

5. [x] Instructionalization & Tagging — Convert raw text into chat-format JSONL with system/user/assistant roles, apply persona tags (twain/franklin/neutral), tone tags (humorous/satirical/didactic), domain tags (weather/general), and create balanced train/validation splits (90/10) `M`

6. [x] Synthetic Tool-Use Data Generation — Create 1,000-3,000 OpenAI-style function calling examples with weather API tool schema, generate multi-turn conversations including tool calls/responses, cover success cases and error handling, and validate JSON schema correctness `M`

7. [x] QLoRA Training Configuration — Configure base model (Llama 3.1 8B Instruct or Mistral 7B Instruct), set LoRA hyperparameters (r=16, alpha=32, dropout=0.05, target modules: q/k/v/o projections), optimize for single H100 with 4-bit quantization, gradient checkpointing, and flash attention `S`

8. [~] Style-Only LoRA Training — **MERGED INTO COMBINED APPROACH** (see item 8+10 below) — Original plan: Train first adapter on literary style and humor datasets (8,000-12,000 examples) for 1-2 epochs with learning rate 2e-4, cosine decay, and early stopping on validation loss, completing in 2-3 hours on H100 `M`

8+10. [x] **Combined Style+Tool-Use Training (H100/M4 Dual-Platform)** — Created complete training infrastructure supporting both H100 (RunPod, 3-4 hours, $9-12) and M4 (local, 12-18 hours, $0) environments. Implemented automated setup scripts (setup_runpod_h100.sh, setup_m4_local.sh), training execution scripts (train_h100_runpod.sh, train_m4_local.sh) with pre-flight validation, checkpoint resumption, crash loop detection, and agent-readable output tags. Training on merged dataset (14,399 examples: tool-use + humor) for 3 epochs with LoRA rank 64, learning rate 2e-4. H100 uses Flash Attention 2, batch size 4, seq length 4096; M4 uses MPS backend, batch size 1, seq length 2048. Comprehensive documentation: TRAINING_H100.md, TRAINING_M4.md, DEPLOYMENT.md with AnythingLLM/Ollama guides `XL`

9. [ ] Style Model Evaluation & Validation — Run automated style classifier accuracy on holdout set, generate sample conversations, perform LLM-as-judge scoring on Twain-ness and wit (1-5 scale), conduct 20-30 human spot checks, and document baseline performance `S`

10. [~] Combined Style+Tool-Use Training — **MERGED INTO COMBINED APPROACH** (see item 8+10 above) — Original plan: Train final adapter on merged dataset (style + tool-use examples totaling 10,000-15,000 samples) for 2-3 epochs, monitor tool-call JSON validity during training, and complete training in 3-4 hours on H100 `M`

11. [ ] Tool-Use Evaluation Harness — Validate tool-call JSON schema correctness (95%+ target), verify groundedness of answers in tool output, test edge cases (missing data, API errors), run end-to-end weather query scenarios, and measure style retention with tool integration `S`

12. [ ] Model Serving & Deployment — Set up vLLM or text-generation-inference server with OpenAI-compatible endpoint, configure tool schema and system prompts, implement adapter loading, test inference latency/throughput, and create usage documentation with example API calls `M`

> Notes
> - Order optimized for single-GPU H100 training completing over one weekend (Friday evening through Sunday)
> - Data quality gates between phases ensure high-signal training examples before GPU time investment
> - Separate style-only training validates data composition impact before adding tool-use complexity
> - All GPU-intensive tasks (items 8, 10) designed to complete in 6-8 combined hours on single H100
