# Specification: Combined Training H100/M4 Scripts

## Goal
Create repeatable, script-based infrastructure for combined style+tool-use LoRA training on both remote H100 (via RunPod.io) and local Mac M4 environments, with agent-readable messaging, checkpoint resumption, and disconnected training support.

## User Stories
- As a machine learning engineer, I want to execute H100 training on RunPod with a single command so that I can reliably train the model without manual intervention
- As a developer, I want to monitor training remotely via Weights & Biases so that I can disconnect my SSH session without losing visibility into progress
- As an automation agent, I want structured status messages with clear pass/fail criteria so that I can make reliable decisions during training execution

## Specific Requirements

**H100 RunPod Setup Script (`setup_runpod_h100.sh`)**
- Verify RunPod environment (CUDA 12.1+, H100 80GB VRAM detection)
- Create conda environment from `environment-remote.yml` with version pinning
- Install Flash Attention 2 via `pip install flash-attn --no-build-isolation`
- Call validation scripts: `validate_environment.py --env=h100`, `check_gpu.py`, `check_storage.py` (require 50GB+ free)
- Validate training config with `validate_training_config.py --config configs/training_config_h100.yaml`
- Create data symlinks if missing: `data/processed/train.jsonl` → `data/synthetic/final_train.jsonl`
- Configure Weights & Biases authentication (check for API key, provide setup instructions if missing)
- Use color-coded output (GREEN ✅, RED ❌, YELLOW ⚠️) with structured agent-readable format

**H100 Training Execution Script (`train_h100_runpod.sh`)**
- Run all pre-flight checks from setup script before starting training
- Launch training in tmux session named `weatherman-training` for persistence
- Redirect all output to timestamped log file: `logs/training_$(date +%Y%m%d_%H%M%S).log`
- Monitor first 100 steps for errors (tail log in background, exit if error detected)
- Print reconnection instructions: `tmux attach -t weatherman-training`
- Print WandB dashboard URL for remote monitoring
- Implement checkpoint resumption logic with crash loop detection (if resume_count >= 3, alert and exit code 2)
- Accept optional `--config` parameter to override default H100 config

**M4 Local Setup Script (`setup_m4_local.sh`)**
- Verify Mac M4 hardware via `system_profiler SPHardwareDataType` (check for Apple Silicon)
- Check Python 3.10+ with `python3 --version`
- Create virtual environment in `.venv-local/` using venv
- Install packages from `requirements-local.txt` with MPS-compatible PyTorch
- Validate MPS backend availability: `python -c "import torch; assert torch.backends.mps.is_available()"`
- Check unified memory (32GB recommended, 16GB minimum) via `sysctl hw.memsize`
- Validate training config for M4: `validate_training_config.py --config configs/training_config_m4.yaml`
- Create data symlinks if needed
- Display warning about slower training time: 12-18 hours vs 3-4 hours on H100

**M4 Training Execution Script (`train_m4_local.sh`)**
- Run pre-flight checks from M4 setup script
- Launch training with MPS backend enabled (automatically detected by PyTorch)
- Use reduced batch size from config (per_device_batch_size=1, grad_accum=8)
- Monitor system temperature and warn about thermal throttling if detected
- Save checkpoints more frequently (every 250 steps vs 500 on H100)
- Provide time estimates after first 100 steps based on measured throughput
- Log to `logs/training_m4_$(date +%Y%m%d_%H%M%S).log`

**Checkpoint Resumption and Crash Loop Detection**
- Training script checks for existing checkpoints in `output_dir` before starting
- If checkpoint found, auto-resume from latest and log: `⚠️ RESUME: Found checkpoint at step X. Resuming training.`
- Track resume metadata in `resume_metadata.json`: last_checkpoint, last_step, resume_count, last_resume_time
- If resume_count >= 3 and step number unchanged: print `❌ ERROR: Crash loop detected` and exit with code 2
- Reset resume_count to 0 when training makes progress (step number increases)
- Send notification on resumption (stdout message parseable by agent)

**Repository Configuration Updates**
- Modify `.gitignore` to include final training data files while excluding intermediate files
- Allow: `data/synthetic/final_train.jsonl`, `data/synthetic/final_validation.jsonl`, `data/synthetic/merge_metadata.json`
- Continue excluding: `data/synthetic/tool_use_examples*.jsonl` (intermediates)
- Ensure base model downloads excluded: `models/*` (14GB, fetched from HuggingFace)

**Documentation Updates**
- Create `docs/DEPLOYMENT.md` with sections: Download Adapter, Using with AnythingLLM, Using with Ollama, Sample Prompts
- Document GGUF conversion process for Ollama (reference llama.cpp tools, provide example command)
- Update main `README.md` with "Quick Start - Training" section covering both H100 and M4 paths
- Include estimated costs: H100 RunPod $2-3 for 3-4 hours
- Link to detailed guides: `docs/TRAINING_H100.md`, `docs/TRAINING_M4.md`

**Roadmap Updates**
- Mark roadmap items 8 (Style-Only Training) and 10 (Combined Style+Tool-Use Training) as "MERGED"
- Add new combined item: "Item 8+10: Combined Style+Tool-Use Training"
- Update status to IN PROGRESS with deliverables: 16,000 examples, 69% humor personas, dual-platform scripts
- Reference this spec in roadmap entry

**Agent-Readable Messaging Format**
- All scripts use consistent structured output: `[STATUS-TAG] Description`
- Success messages: `✅ SUCCESS: Action completed` (exit code 0)
- Error messages: `❌ ERROR: Problem description` (exit code 1)
- Warning messages: `⚠️ WARNING: Advisory message` (exit code 0 unless critical)
- Crash loop detected: exit code 2 (distinct from regular errors)
- Section headers: `[SETUP-H100]`, `[TRAINING-H100]`, `[TRAINING-H100-COMPLETE]`

**Remote Monitoring and Disconnected Training**
- Configure Weights & Biases in training config: `report_to: "wandb"`, `run_name: "weatherman-lora-h100-$(date +%Y%m%d)"`
- Use tmux for session persistence (standard on RunPod, survives SSH disconnection)
- Document reconnection: `tmux attach -t weatherman-training`
- Document status checking: `tail -f logs/training_*.log` or WandB dashboard
- Optional: Create `scripts/check_training_status.sh` to query WandB API for current metrics

## Visual Design
No visual assets provided for this spec.

## Existing Code to Leverage

**`scripts/validate_environment.py` (666 lines)**
- Provides comprehensive environment validation with platform detection (H100/M4)
- Contains reusable functions: `print_check(passed, message)`, `print_header(title)` for formatted output
- Has separate validation paths: `validate_h100()`, `validate_m4()` with specific dependency checks
- Validates CUDA 12.1+, Flash Attention 2, MPS backend, PyTorch versions, transformers compatibility
- Use this script directly from setup scripts to verify environment readiness

**`scripts/validate_training_config.py` (347 lines)**
- Validates YAML config structure, LoRA parameters, memory requirements, data paths
- Checks LoRA rank (8-64), alpha (2*rank recommended), dropout (0.0-0.2), target modules (4+ required)
- Estimates memory usage based on sequence length and batch size (rough approximation)
- Platform-specific warnings: H100 (70GB limit), M4 (28GB limit for 32GB unified memory)
- Call with `--config` parameter to validate H100 or M4 configs before training

**`scripts/check_gpu.py` (383 lines)**
- Detects platform type (CUDA vs MPS) and displays appropriate diagnostics
- CUDA path: shows GPU name, VRAM, compute capability, runs computation test
- MPS path: validates MPS availability, checks unified memory, tests MPS computation
- Provides memory assessment: INSUFFICIENT (<20GB), LOW (20-24GB), ADEQUATE (24-70GB), EXCELLENT (70GB+)
- Recommends config file based on detected hardware (H100 vs M4)

**`scripts/check_storage.py` (107 lines)**
- Checks disk space via `shutil.disk_usage()` and displays human-readable sizes
- Defines storage requirements: 20GB minimum, 30GB recommended (raw data, model, checkpoints, buffer)
- Returns exit code 1 if <20GB free, exit code 0 with warning if 20-30GB, exit code 0 success if 30GB+
- Call from setup scripts to ensure sufficient space before downloading models

**`setup_remote.sh` (first 80 lines examined)**
- Demonstrates color-coded output pattern: `RED='\033[0;31m'`, `GREEN='\033[0;32m'`, `YELLOW='\033[1;33m'`
- Shows step-by-step verification approach with clear status messages
- Checks conda installation, CUDA version detection with fallback messaging
- Handles existing environment gracefully (prompt to remove/recreate vs skip)
- Use this script's structure and messaging patterns for new H100/M4 setup scripts

## Out of Scope
- Implementing GGUF conversion script (document process only, reference llama.cpp tools)
- Setting up RunPod account or SSH key configuration (assume user has access)
- Email/SMS notification integrations (optional webhook support can be documented)
- Hyperparameter tuning or automated search (use existing configs as-is)
- Model evaluation metrics beyond loss (future work)
- Multi-GPU training support (single H100 or M4 only)
- Windows platform compatibility (Linux and Mac only)
- CI/CD pipeline integration (manual execution only)
- Production inference API deployment (local usage with AnythingLLM/Ollama only)
- Training data generation or augmentation (use existing 16,000 examples)
