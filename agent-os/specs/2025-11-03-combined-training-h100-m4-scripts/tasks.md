# Task Breakdown: Combined Training H100/M4 Scripts

## Overview
Total Tasks: 6 task groups covering H100 setup/training, M4 setup/training, repository configuration, and documentation

## Task List

### H100 Infrastructure (Primary Focus)

#### Task Group 1: H100 RunPod Setup Script
**Dependencies:** None (uses existing validation scripts)

- [x] 1.0 Complete H100 RunPod setup script
  - [x] 1.1 Create `setup_runpod_h100.sh` with header and color definitions
    - Add shebang and script description
    - Define color codes: `RED='\033[0;31m'`, `GREEN='\033[0;32m'`, `YELLOW='\033[1;33m'`, `NC='\033[0m'`
    - Add helper functions for structured output: `success()`, `error()`, `warning()`, `info()`
    - Follow pattern from: `setup_remote.sh`
  - [x] 1.2 Implement RunPod environment verification
    - Check conda installation with `command -v conda`
    - Detect CUDA version via `nvcc --version` (require 12.1+)
    - Verify H100 GPU with nvidia-smi (check for "H100" in output, 80GB VRAM)
    - Print: `[SETUP-H100] Starting RunPod H100 environment setup`
  - [x] 1.3 Create conda environment from environment-remote.yml
    - Check if `weatherman-lora` environment exists
    - Prompt to remove/recreate if exists
    - Run: `conda env create -f environment-remote.yml`
    - Activate environment: `conda activate weatherman-lora`
    - Print: `✅ Conda environment created: weatherman-lora`
  - [x] 1.4 Install Flash Attention 2
    - Run: `pip install flash-attn --no-build-isolation`
    - Handle installation errors gracefully (provide troubleshooting steps)
    - Print: `✅ Flash Attention 2 installed`
  - [x] 1.5 Run validation scripts
    - Call: `python scripts/validate_environment.py --env=h100`
    - Call: `python scripts/check_gpu.py`
    - Call: `python scripts/check_storage.py` (require 50GB+ free)
    - Call: `python scripts/validate_training_config.py --config configs/training_config_h100.yaml`
    - Exit with code 1 if any validation fails
  - [x] 1.6 Create data symlinks if missing
    - Check if `data/processed/train.jsonl` exists
    - If not, create: `ln -sf "$(pwd)/data/synthetic/final_train.jsonl" data/processed/train.jsonl`
    - Create: `ln -sf "$(pwd)/data/synthetic/final_validation.jsonl" data/processed/val.jsonl`
    - Print: `✅ Data paths verified`
  - [x] 1.7 Configure Weights & Biases authentication
    - Check for `WANDB_API_KEY` environment variable
    - If missing, print instructions: `wandb login` and API key location
    - Verify with: `python -c "import wandb; wandb.login()"`
    - Print: `✅ Weights & Biases authenticated`
  - [x] 1.8 Print setup completion summary
    - Print: `[SETUP-H100-COMPLETE] Ready for training`
    - Display estimated training time: 3-4 hours
    - Display next step: `./train_h100_runpod.sh`

**Acceptance Criteria:**
- Script completes without errors on RunPod H100 instance
- All validation checks pass
- Conda environment created with correct dependencies
- Flash Attention 2 installed successfully
- Data symlinks created
- WandB authenticated
- Agent-readable structured output with `[SETUP-H100]` tags

---

#### Task Group 2: H100 Training Execution Script
**Dependencies:** Task Group 1

- [x] 2.0 Complete H100 training execution script
  - [x] 2.1 Create `train_h100_runpod.sh` with pre-flight checks
    - Add shebang and description
    - Reuse color codes and helper functions from setup script
    - Accept optional `--config` parameter (default: `configs/training_config_h100.yaml`)
    - Run all validation checks from Task 1.5
    - Print: `[TRAINING-H100] Pre-flight checks starting`
  - [x] 2.2 Create logs directory and timestamped log file
    - Run: `mkdir -p logs`
    - Generate log filename: `logs/training_$(date +%Y%m%d_%H%M%S).log`
    - Print: `[TRAINING-H100] Log file: $LOG_FILE`
  - [x] 2.3 Implement checkpoint resumption logic
    - Check for existing checkpoints in `adapters/weatherman-lora-h100/`
    - If found, read `resume_metadata.json` for last_step and resume_count
    - If resume_count >= 3 and step unchanged: print `❌ ERROR: Crash loop detected` and exit code 2
    - If resuming: print `⚠️ RESUME: Found checkpoint at step X. Resuming training.`
    - Update resume_metadata.json with new timestamp and incremented resume_count
  - [x] 2.4 Launch training in tmux session
    - Create tmux session: `tmux new-session -d -s weatherman-training`
    - Launch training command in tmux: `python scripts/train.py --config $CONFIG`
    - Redirect output to log file with tee
    - Print: `[TRAINING-H100] Starting training in tmux session: weatherman-training`
  - [x] 2.5 Monitor first 100 steps for errors
    - Tail log file in background for 5 minutes
    - Check for error patterns: "ERROR", "CUDA out of memory", "RuntimeError"
    - If errors detected within 5 minutes, print error and exit code 1
    - Otherwise, print: `✅ Training started successfully (first 100 steps)`
  - [x] 2.6 Print reconnection and monitoring instructions
    - Print: `[TRAINING-H100] Estimated duration: 3-4 hours`
    - Print: `[TRAINING-H100] Reconnect with: tmux attach -t weatherman-training`
    - Print WandB dashboard URL: `[TRAINING-H100] Monitor remotely: https://wandb.ai/user/weatherman-lora`
    - Print log tail command: `tail -f $LOG_FILE`
    - Print: `[TRAINING-H100-STARTED] Training in progress`
  - [x] 2.7 Handle training completion detection (optional enhancement)
    - Document how to check if training completed via tmux session status
    - Document how to verify final model saved in output directory
    - Create optional `scripts/check_training_status.sh` to query WandB API

**Acceptance Criteria:**
- Script launches training in persistent tmux session
- Pre-flight checks run before training starts
- Checkpoint resumption works correctly
- Crash loop detection triggers on 3rd failed resume
- Log file created with all output
- Structured output with `[TRAINING-H100]` tags
- Reconnection instructions displayed clearly

---

### M4 Infrastructure (Secondary)

#### Task Group 3: M4 Local Setup Script
**Dependencies:** None (parallel to H100)

- [x] 3.0 Complete M4 local setup script
  - [x] 3.1 Create `setup_m4_local.sh` with header and validation functions
    - Add shebang and script description
    - Reuse color codes and helper functions
    - Print: `[SETUP-M4] Starting Mac M4 local setup`
  - [x] 3.2 Verify Mac M4 hardware and Python version
    - Run: `system_profiler SPHardwareDataType | grep "Chip"` (check for "Apple M")
    - Run: `python3 --version` (require 3.10+)
    - Print chip type and Python version
    - Exit with error if not Apple Silicon or Python < 3.10
  - [x] 3.3 Check unified memory
    - Run: `sysctl hw.memsize` to get memory in bytes
    - Convert to GB and display
    - Warn if < 16GB (INSUFFICIENT), note if 16-32GB (LIMITED), confirm if 32GB+ (SUFFICIENT)
    - Print: `✅ Unified Memory: XGB`
  - [x] 3.4 Create virtual environment
    - Create: `python3 -m venv .venv-local`
    - Activate: `source .venv-local/bin/activate`
    - Upgrade pip: `pip install --upgrade pip`
    - Print: `✅ Virtual environment created`
  - [x] 3.5 Install packages from requirements-local.txt
    - Run: `pip install -r requirements-local.txt`
    - Verify PyTorch MPS support (2.1+)
    - Print package versions installed
    - Print: `✅ Packages installed`
  - [x] 3.6 Validate MPS backend availability
    - Run: `python -c "import torch; assert torch.backends.mps.is_available(), 'MPS not available'"`
    - Test MPS computation: `python -c "import torch; t = torch.randn(100, 100, device='mps'); torch.matmul(t, t)"`
    - Print: `✅ MPS backend available`
  - [x] 3.7 Run M4-specific validation
    - Call: `python scripts/validate_environment.py --env=m4`
    - Call: `python scripts/validate_training_config.py --config configs/training_config_m4.yaml`
    - Call: `python scripts/check_storage.py`
    - Create data symlinks if needed (same as H100)
  - [x] 3.8 Display M4 training warnings and completion
    - Print: `⚠️ WARNING: M4 training takes 12-18 hours (vs 3-4 hours on H100)`
    - Print: `⚠️ Close unnecessary applications to free unified memory`
    - Print: `[SETUP-M4-COMPLETE] Ready for local training`
    - Print next step: `./train_m4_local.sh`

**Acceptance Criteria:**
- Script verifies Apple Silicon M4 hardware
- Virtual environment created with MPS-compatible PyTorch
- MPS backend validated and working
- All M4-specific checks pass
- Training time warnings displayed
- Structured output with `[SETUP-M4]` tags

---

#### Task Group 4: M4 Training Execution Script
**Dependencies:** Task Group 3

- [x] 4.0 Complete M4 training execution script
  - [x] 4.1 Create `train_m4_local.sh` with M4-specific setup
    - Add shebang and description
    - Activate virtual environment: `source .venv-local/bin/activate`
    - Run pre-flight checks from Task 3.7
    - Print: `[TRAINING-M4] Pre-flight checks starting`
  - [x] 4.2 Configure M4-specific training parameters
    - Use config: `configs/training_config_m4.yaml`
    - Verify reduced batch size (per_device_batch_size=1, grad_accum=8)
    - Set checkpoint frequency: every 250 steps (vs 500 on H100)
    - Confirm sequence length: 2048 tokens (vs 4096 on H100)
  - [x] 4.3 Create timestamped log file
    - Run: `mkdir -p logs`
    - Generate: `logs/training_m4_$(date +%Y%m%d_%H%M%S).log`
    - Print: `[TRAINING-M4] Log file: $LOG_FILE`
  - [x] 4.4 Launch training with MPS backend
    - Run: `python scripts/train.py --config configs/training_config_m4.yaml 2>&1 | tee $LOG_FILE`
    - MPS backend auto-detected by PyTorch (no special flags needed)
    - Print: `[TRAINING-M4] Starting training on MPS backend`
  - [x] 4.5 Monitor system temperature (optional)
    - Document how to check thermal state: `sudo powermetrics --samplers smc`
    - Warn if thermal throttling detected (activity monitor shows reduced performance)
    - Print advisory: `⚠️ If training slows, check Activity Monitor for thermal throttling`
  - [x] 4.6 Provide time estimates after first 100 steps
    - Calculate steps per second after 100 steps
    - Estimate total time: `(total_steps / steps_per_second) / 3600` hours
    - Print: `[ETA] Estimated completion: X hours Y minutes`
    - Print: `[TRAINING-M4] Recommend running overnight`
  - [x] 4.7 Handle completion and summary
    - Wait for training to complete (blocking, not in tmux)
    - Print: `[TRAINING-M4-COMPLETE] Training finished`
    - Print model location: `adapters/weatherman-lora-m4/`
    - Print next steps: deployment documentation

**Acceptance Criteria:**
- Training launches on MPS backend successfully
- Reduced batch size prevents OOM errors
- More frequent checkpoints (250 steps)
- Time estimates displayed after 100 steps
- Training completes without errors
- Structured output with `[TRAINING-M4]` tags

---

### Repository Configuration

#### Task Group 5: Repository and Configuration Updates
**Dependencies:** None (can be done in parallel)

- [x] 5.0 Complete repository configuration updates
  - [x] 5.1 Update .gitignore for training data
    - Read current `.gitignore` file
    - Modify `data/synthetic/*` exclusion to allow final files
    - Add exceptions: `!data/synthetic/final_train.jsonl`, `!data/synthetic/final_validation.jsonl`, `!data/synthetic/merge_metadata.json`
    - Keep exclusions: `data/synthetic/tool_use_examples*.jsonl` (intermediates)
    - Verify `models/*` excluded (14GB base model)
    - Test with: `git status` to ensure only final files shown
  - [x] 5.2 Verify training configs are production-ready
    - Check `configs/training_config_h100.yaml` settings
    - Check `configs/training_config_m4.yaml` settings
    - Ensure WandB reporting enabled: `report_to: "wandb"`
    - Ensure run names include date: `run_name: "weatherman-lora-h100-$(date +%Y%m%d)"`
    - Verify checkpoint settings: H100 (500 steps), M4 (250 steps)
  - [x] 5.3 Ensure all directories exist
    - Create: `mkdir -p logs`
    - Create: `mkdir -p data/processed`
    - Create: `mkdir -p adapters`
    - Verify in .gitignore: `logs/` excluded, `adapters/` excluded
  - [x] 5.4 Make scripts executable
    - Run: `chmod +x setup_runpod_h100.sh`
    - Run: `chmod +x train_h100_runpod.sh`
    - Run: `chmod +x setup_m4_local.sh`
    - Run: `chmod +x train_m4_local.sh`

**Acceptance Criteria:**
- .gitignore updated to include final training data
- Training configs verified and ready
- Required directories created
- All scripts executable
- Git status shows only intended files

---

### Documentation

#### Task Group 6: Documentation and Roadmap Updates
**Dependencies:** Task Groups 1-4 (scripts must exist to document)

- [x] 6.0 Complete documentation updates
  - [x] 6.1 Create docs/DEPLOYMENT.md
    - **Section 1: Download Trained Adapter**
      - Document RunPod to local transfer: `scp -r user@runpod:/workspace/adapters/weatherman-lora-h100 ./adapters/`
      - List expected files: `adapter_config.json`, `adapter_model.bin`, `tokenizer.json`, etc.
      - Document file size: ~500MB-2GB
    - **Section 2: Using with AnythingLLM**
      - Installation: `curl -sSL https://install.anythingllm.com | bash`
      - Load base model: Mistral 7B Instruct v0.2
      - Add LoRA adapter: point to `adapters/weatherman-lora-h100/`
      - Test query: "What's the weather in Boston?"
    - **Section 3: Using with Ollama**
      - Document GGUF conversion process (reference llama.cpp)
      - Provide example command: `python -m llama_cpp.convert --model mistralai/Mistral-7B-Instruct-v0.2 --lora adapters/weatherman-lora-h100 --output models/weatherman-lora.gguf`
      - Import to Ollama: `ollama create weatherman -f Modelfile`
      - Test: `ollama run weatherman "What's the weather in Seattle?"`
    - **Section 4: Sample Prompts**
      - Include 10+ examples covering: current weather, forecast, multi-turn, error handling, Twain/Franklin/Onion personas
  - [x] 6.2 Create docs/TRAINING_H100.md
    - Prerequisites: RunPod account, CUDA 12.1+, 50GB+ free space
    - Step-by-step H100 training guide
    - Estimated cost: $2-3 for 3-4 hours
    - Troubleshooting section: OOM errors, CUDA issues, checkpoint problems
    - Reference scripts: `setup_runpod_h100.sh`, `train_h100_runpod.sh`
  - [x] 6.3 Create docs/TRAINING_M4.md
    - Prerequisites: Mac M4, 32GB RAM recommended, Python 3.10+
    - Step-by-step M4 training guide
    - Estimated time: 12-18 hours
    - Troubleshooting: MPS issues, memory pressure, thermal throttling
    - Reference scripts: `setup_m4_local.sh`, `train_m4_local.sh`
  - [x] 6.4 Update main README.md
    - Add "Quick Start - Training" section after existing sections
    - **Option 1: Remote Training (H100 via RunPod) - Recommended**
      - Duration: 3-4 hours, Cost: ~$2-3
      - Steps: Clone repo, run setup, run training, monitor WandB, download adapter
      - Link to: `docs/TRAINING_H100.md`
    - **Option 2: Local Training (Mac M4)**
      - Duration: 12-18 hours, Requirements: 32GB RAM recommended
      - Steps: Run setup, run training
      - Link to: `docs/TRAINING_M4.md`
    - Link to deployment guide: `docs/DEPLOYMENT.md`
  - [x] 6.5 Update agent-os/product/roadmap.md
    - Mark Item 8 (Style-Only Training) as "MERGED INTO COMBINED APPROACH"
    - Mark Item 10 (Combined Style+Tool-Use Training) as "MERGED INTO COMBINED APPROACH"
    - Add new entry: "Item 8+10: Combined Style+Tool-Use Training"
    - Status: IN PROGRESS → COMPLETED (after implementation)
    - Deliverables: 16,000 examples (69% humor ratio), H100/M4 scripts, deployment docs
    - Reference spec: `agent-os/specs/2025-11-03-combined-training-h100-m4-scripts/`

**Acceptance Criteria:**
- DEPLOYMENT.md created with all 4 sections
- TRAINING_H100.md and TRAINING_M4.md created
- README.md updated with Quick Start section
- Roadmap updated to reflect merged approach
- All documentation links work correctly
- Sample prompts provided for testing

---

## Execution Order

Recommended implementation sequence:

1. **H100 Infrastructure** (Critical Path - Primary Focus)
   - Task Group 1: H100 RunPod Setup Script
   - Task Group 2: H100 Training Execution Script

2. **M4 Infrastructure** (Parallel - Secondary)
   - Task Group 3: M4 Local Setup Script
   - Task Group 4: M4 Training Execution Script

3. **Repository Configuration** (Can be done in parallel)
   - Task Group 5: Repository and Configuration Updates

4. **Documentation** (Final step - requires scripts to exist)
   - Task Group 6: Documentation and Roadmap Updates

**Parallelization Strategy:**
- Groups 1 and 3 can be implemented in parallel (different platforms)
- Groups 2 and 4 can be implemented in parallel (both training scripts)
- Group 5 can be done at any point
- Group 6 should be done last to document completed scripts

**Critical Path:** Task Group 1 → Task Group 2 (H100 is primary focus and will be executed)

**Testing Strategy:**
- No traditional unit tests for bash scripts
- Validation done via:
  - Existing Python validation scripts (validate_environment.py, check_gpu.py, etc.)
  - Manual execution on RunPod H100 instance (Task Group 2)
  - Manual execution on Mac M4 (Task Group 4 - optional)
  - Verification of agent-readable output format
  - Checkpoint resumption testing (create artificial crash scenario)
