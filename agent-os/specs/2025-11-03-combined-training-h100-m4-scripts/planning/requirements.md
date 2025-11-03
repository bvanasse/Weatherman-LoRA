# Requirements: Combined Training H100/M4 Scripts

## Initial Description

**Goal**: Create a repeatable, script-based process for combined style+tool-use LoRA training that works on both remote H100 (via RunPod.io) and local Mac M4 environments.

**Context**:
- Roadmap items 8 (Style-Only Training) and 10 (Combined Style+Tool-Use Training) are being merged into a single combined training approach
- We have 16,000 synthetic tool-use examples with 69% humor ratio (Twain/Franklin/Onion personas)
- Training script (`scripts/train.py`) has been created but dependencies need to be installed
- Config files exist for both H100 (`configs/training_config_h100.yaml`) and M4 (`configs/training_config_m4.yaml`)

**Primary Focus**:
- Remote H100 training via RunPod.io service (this will be executed)
- Repeatable setup and execution scripts

**Secondary Output**:
- Parallel Mac M4 local training scripts and documentation (created but not executed)

## Requirements Discussion

### Question 1: Script Organization
**Q**: Should we create a comprehensive setup script and documentation for the entire training sequence?

**A**: Yes. Create a script and related documentation that can be used as context for running the entire training sequence.

**Implication**: Need `setup_runpod_h100.sh` and `train_h100_runpod.sh` with comprehensive inline documentation suitable for agent parsing.

---

### Question 2: Data Inclusion in Repository
**Q**: Should the training data be included in the git repository?

**A**: Yes, we should include the data in the git repo. I have most of the data already available (with the exception of the Mistral model). Adjust the gitignore files if necessary.

**Implication**:
- Modify `.gitignore` to allow `data/synthetic/final_train.jsonl` and `data/synthetic/final_validation.jsonl`
- Document that base model (Mistral 7B) will be downloaded from HuggingFace during setup
- Total data size: ~16,000 conversations in JSONL format

---

### Question 3: Agent-Readable Messaging
**Q**: Should we include clear messaging that another automated agent can use to make decisions?

**A**: Yes, include clear messaging that another automated agent running the sequence can make reliable decisions with.

**Implication**:
- Use structured output format for all steps:
  ```
  [STATUS] Step description
  ✅ SUCCESS: Action completed
  ❌ ERROR: Problem description
  ⚠️  WARNING: Advisory message
  ```
- Include decision points with clear pass/fail criteria
- Use exit codes consistently (0=success, non-zero=failure)

---

### Question 4: Transparency and Repeatability
**Q**: What makes the process transparent, stable, and repeatable for both humans and agents?

**A**: Include anything that will make the process transparent, stable and repeatable by a human or agent running the training sequence.

**Implication**:
- Comprehensive logging at each step
- Pre-flight validation checks before starting expensive training
- Clear error messages with remediation steps
- Version pinning for all dependencies
- Idempotent scripts (can be run multiple times safely)

---

### Question 5: Connection Management
**Q**: How should we handle long-running training sessions?

**A**: Whatever makes the most sense. The user will not necessarily be able to keep an active connection for the entire run. Take steps to ensure that the process is stable.

**Implication**:
- Use tmux or screen for session persistence
- Redirect all output to log files
- Set up Weights & Biases for remote monitoring
- Document how to reconnect and check status
- Estimated training time: 3-4 hours on H100

---

### Question 6: Session Management Best Practices
**Q**: Should we use specific tools like tmux/screen or just background processes?

**A**: Use your best judgement here.

**Decision**: Use tmux for the following reasons:
- Standard on RunPod instances
- Easy to reattach after disconnection
- Better output management than `nohup &`
- Allows interactive monitoring if user reconnects

---

### Question 7: Checkpoint Resumption Strategy
**Q**: How should we handle training interruptions and crash loops?

**A**: Probably support resuming automatically. However, it should alert the user so training doesn't run unnecessarily in a crash loop.

**Implication**:
- Enable checkpoint saving every 500 steps (configured in training_config_h100.yaml)
- Auto-resume from latest checkpoint if found
- Implement crash loop detection: if same checkpoint loaded 3+ times, alert and pause
- Send notification (stdout + optional email/webhook) when resumption occurs
- Log checkpoint metadata for troubleshooting

---

### Question 8: Deployment and Artifact Creation
**Q**: What should the final deliverable look like for local usage?

**A**: Use your best judgement. The user is expecting an artifact that can be downloaded that will include the files and instructions necessary to run the model locally with tools like AnythingLLM or Ollama. If requires additional processing that can be done on a Mac M4, than the files and instructions should be made available to the user.

**Implication**:
- Training output: LoRA adapter weights in `adapters/weatherman-lora-h100/`
- Need to document format conversion:
  - HuggingFace format → GGUF format (for Ollama)
  - Merging LoRA adapter with base model (optional)
- Create `docs/DEPLOYMENT.md` with:
  - Download instructions for adapter files
  - Using with AnythingLLM (direct LoRA loading)
  - Using with Ollama (requires GGUF conversion on M4)
  - Testing inference with sample prompts

---

### Question 9: Documentation References
**Q**: Should we reference existing documentation?

**A**: Reference what you need to according the available documentation. There was an earlier attempt to create a training script, but it may not be relevant for this spec. Take your time to create a repeatable and accessible training script and documentation. Update the README as necessary so this can be easily picked up by a human or agent and replicated.

**Implication**:
- Review all docs in `docs/` directory
- Reference relevant sections without duplicating content
- Update main `README.md` with clear "Quick Start" for training
- Link to detailed guides for each environment (H100/M4)

---

## Existing Code to Reference

### 1. Setup Scripts

#### `setup_remote.sh` (152 lines)
**Purpose**: Conda-based setup for H100/remote GPUs

**Key Patterns to Reuse**:
- Color-coded output functions:
  ```bash
  echo -e "${GREEN}✓ Check passed${NC}"
  echo -e "${RED}✗ Check failed${NC}"
  echo -e "${YELLOW}⚠ Warning${NC}"
  ```
- Step-by-step verification with clear status
- Automated checks with fallbacks
- Calls to validation scripts: `check_gpu.py`, `check_storage.py`
- Conda environment creation from `environment-remote.yml`

**Reuse Strategy**: Follow same structure but customize for RunPod-specific requirements (pre-installed CUDA, RunPod CLI, etc.)

---

#### `setup_m4.sh` (first 100 lines examined)
**Purpose**: Virtual environment setup for Mac M4

**Key Patterns to Reuse**:
- Python 3.10+ verification
- Apple Silicon detection (`uname -m` = arm64)
- Memory checking (32GB recommended, 16GB minimum)
- Virtual environment creation with pip
- MPS backend validation

**Reuse Strategy**: Extend with training-specific setup (data validation, config checks)

---

### 2. Configuration Files

#### `environment-remote.yml`
**Purpose**: Exact package versions for reproducible H100 environment

**Key Dependencies**:
```yaml
name: weatherman-lora
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.1.0
  - pytorch-cuda=12.1
  - transformers=4.36.0
  - peft=0.7.0
  - trl=0.7.4
  - bitsandbytes=0.41.0
  - wandb=0.16.1
  - accelerate=0.25.0
  - datasets=2.16.0
```

**Note**: Flash Attention 2 requires manual install: `pip install flash-attn --no-build-isolation`

**Reuse Strategy**: Keep versions consistent, document Flash Attention as post-install step

---

#### `configs/training_config_h100.yaml`
**Purpose**: All training hyperparameters for H100

**Key Settings**:
```yaml
model:
  model_name_or_path: "mistralai/Mistral-7B-Instruct-v0.2"
  max_seq_length: 4096

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size: 16
  learning_rate: 2.0e-4
  num_train_epochs: 3
  save_steps: 500
  output_dir: "./adapters/weatherman-lora-h100"

dataset:
  train_file: "data/processed/train.jsonl"
  val_file: "data/processed/val.jsonl"
```

**Reuse Strategy**: Reference directly in training script, validate paths exist before training

---

### 3. Validation Scripts

#### `scripts/check_gpu.py`
**Purpose**: Verify CUDA availability and GPU specs

**Reuse Strategy**: Call from setup script as pre-flight check

---

#### `scripts/check_storage.py`
**Purpose**: Verify sufficient disk space for model cache and checkpoints

**Reuse Strategy**: Call from setup script, recommend 50GB+ free space

---

#### `scripts/validate_environment.py`
**Purpose**: Check all Python packages installed correctly

**Reuse Strategy**: Call after conda/pip install, verify imports work

---

#### `scripts/validate_training_config.py`
**Purpose**: Validate YAML config structure and paths

**Reuse Strategy**: Call before training starts, catch config errors early

---

### 4. Training Script

#### `scripts/train.py` (254 lines)
**Purpose**: Main QLoRA training script using HuggingFace TRL

**Key Components**:
- Config loading from YAML
- 4-bit quantization setup (NF4, double quantization)
- LoRA adapter application
- Chat template formatting for multi-turn conversations
- SFTTrainer with gradient checkpointing
- Checkpoint saving and resumption

**Usage**:
```bash
python scripts/train.py --config configs/training_config_h100.yaml
```

**Reuse Strategy**: This is the core training script - wrapper scripts should:
1. Validate environment
2. Check data paths
3. Launch in tmux session
4. Monitor output for errors
5. Handle completion/failure notifications

---

### 5. Data Files

#### Training Data Location
- Current: `data/synthetic/final_train.jsonl` (14,399 conversations)
- Current: `data/synthetic/final_validation.jsonl` (1,601 conversations)
- Expected by config: `data/processed/train.jsonl` and `data/processed/val.jsonl`

**Solution**: Symlinks already created:
```bash
ln -sf "$(pwd)/data/synthetic/final_train.jsonl" data/processed/train.jsonl
ln -sf "$(pwd)/data/synthetic/final_validation.jsonl" data/processed/val.jsonl
```

**Reuse Strategy**: Setup scripts should verify these symlinks exist or create them

---

#### Data Format
JSONL with OpenAI-style chat messages:
```json
{
  "messages": [
    {"role": "system", "content": "You are a witty weather assistant..."},
    {"role": "user", "content": "What's the weather in Boston?"},
    {"role": "assistant", "content": "Let me check that for you.", "tool_calls": [...]},
    {"role": "tool", "content": "{\"temperature\": 72, ...}"},
    {"role": "assistant", "content": "It's 72°F in Boston..."}
  ]
}
```

---

#### Persona Distribution (from `data/synthetic/merge_metadata.json`)
- Total conversations: 16,000
- Train: 14,399 (90%)
- Validation: 1,601 (10%)
- Personas: Twain 44.3%, Franklin 15.4%, Onion 9.4%, Neutral 31.0%

**Reuse Strategy**: Document distribution, no changes needed

---

## Visual Assets

**Status**: No visual assets found in spec folder

**Check Performed**:
```bash
ls -la agent-os/specs/2025-11-03-combined-training-h100-m4-scripts/planning/visuals/
# Directory does not exist
```

**Conclusion**: No diagrams, screenshots, or mockups available. Documentation will be text-based.

---

## Requirements Summary

### Functional Requirements

#### F1: H100 RunPod Setup Script
**ID**: F1
**Priority**: CRITICAL
**Description**: Create `setup_runpod_h100.sh` that:
- Verifies RunPod environment (CUDA, GPU type)
- Creates/activates conda environment from `environment-remote.yml`
- Installs Flash Attention 2
- Validates all packages with `validate_environment.py`
- Checks GPU with `check_gpu.py`
- Verifies storage with `check_storage.py` (50GB+ required)
- Validates training config with `validate_training_config.py`
- Creates data symlinks if needed
- Sets up Weights & Biases authentication

**Agent-Readable Output**:
```
[SETUP-H100] Starting RunPod H100 environment setup
✅ CUDA 12.1 detected
✅ H100 GPU verified (80GB VRAM)
✅ Conda environment created: weatherman-lora
✅ Flash Attention 2 installed
✅ All packages validated
✅ 120GB free disk space
✅ Training config valid
✅ Data paths verified
✅ Weights & Biases authenticated
[SETUP-H100-COMPLETE] Ready for training
```

---

#### F2: H100 Training Execution Script
**ID**: F2
**Priority**: CRITICAL
**Description**: Create `train_h100_runpod.sh` that:
- Runs all pre-flight checks from F1
- Launches training in tmux session named `weatherman-training`
- Redirects output to `logs/training_$(date +%Y%m%d_%H%M%S).log`
- Monitors for errors in first 5 minutes
- Prints reconnection instructions
- Handles checkpoint resumption automatically
- Detects crash loops (3+ resumes without progress)
- Sends completion notification

**Usage**:
```bash
./train_h100_runpod.sh
# Or with custom config:
./train_h100_runpod.sh --config configs/custom_config.yaml
```

**Agent-Readable Output**:
```
[TRAINING-H100] Pre-flight checks starting
✅ All checks passed
[TRAINING-H100] Starting training in tmux session: weatherman-training
[TRAINING-H100] Log file: logs/training_20251103_143022.log
[TRAINING-H100] Estimated duration: 3-4 hours
[TRAINING-H100] Reconnect with: tmux attach -t weatherman-training
[TRAINING-H100] Monitor remotely: https://wandb.ai/user/weatherman-lora
[TRAINING-H100-STARTED] Training in progress
```

---

#### F3: M4 Local Setup Script
**ID**: F3
**Priority**: HIGH
**Description**: Create `setup_m4_local.sh` that:
- Verifies Mac M4 and Python 3.10+
- Creates virtual environment in `.venv-local/`
- Installs packages from `requirements-local.txt` (MPS-compatible versions)
- Validates MPS backend with `python -c "import torch; print(torch.backends.mps.is_available())"`
- Checks memory (16GB minimum, 32GB recommended)
- Validates training config for M4 (`configs/training_config_m4.yaml`)
- Creates data symlinks
- Warns about slower training time (estimated 12-18 hours)

**Agent-Readable Output**:
```
[SETUP-M4] Starting Mac M4 local setup
✅ Apple Silicon M4 detected
✅ Python 3.10.12 found
✅ Virtual environment created
✅ Packages installed
✅ MPS backend available
✅ 32GB RAM detected
✅ Training config valid
✅ Data paths verified
⚠️  WARNING: M4 training takes 12-18 hours (vs 3-4 hours on H100)
[SETUP-M4-COMPLETE] Ready for local training
```

---

#### F4: M4 Training Execution Script
**ID**: F4
**Priority**: HIGH
**Description**: Create `train_m4_local.sh` that:
- Runs pre-flight checks from F3
- Launches training with MPS backend enabled
- Monitors system temperature (throttling warning)
- Uses smaller batch size (per_device_batch_size=1, grad_accum=8)
- Saves more frequent checkpoints (every 250 steps)
- Provides time estimates based on progress

**Usage**:
```bash
./train_m4_local.sh
```

---

#### F5: Checkpoint Resumption Logic
**ID**: F5
**Priority**: HIGH
**Description**: Implement in training scripts:
- Check for existing checkpoints in `output_dir`
- Auto-resume from latest checkpoint if found
- Track resume count in `resume_metadata.json`
- If resume_count >= 3 with no progress (same step number):
  - Print: `❌ ERROR: Crash loop detected. Training resumed 3+ times from step X. Manual intervention required.`
  - Exit with code 2
  - Send notification
- Log each resumption:
  ```
  ⚠️  RESUME: Found checkpoint at step 1500. Resuming training.
  ```

---

#### F6: Remote Monitoring Setup
**ID**: F6
**Priority**: MEDIUM
**Description**:
- Configure Weights & Biases in training config
- Document WandB dashboard access
- Set up completion webhook (optional)
- Create `scripts/check_training_status.sh` to query WandB API for current metrics

**WandB Configuration** (in training_config_h100.yaml):
```yaml
training:
  report_to: "wandb"
  run_name: "weatherman-lora-h100-$(date +%Y%m%d)"
```

---

#### F7: Deployment Documentation
**ID**: F7
**Priority**: MEDIUM
**Description**: Create `docs/DEPLOYMENT.md` with:

**Section 1: Download Trained Adapter**
- How to download from RunPod to local Mac
- Expected file structure: `adapters/weatherman-lora-h100/`
- Files: `adapter_config.json`, `adapter_model.bin`, `tokenizer.json`, etc.

**Section 2: Using with AnythingLLM**
- Install AnythingLLM
- Add LoRA adapter as custom model
- Load Mistral 7B base + Weatherman adapter
- Test with sample weather queries

**Section 3: Using with Ollama**
- Install llama.cpp on Mac M4
- Convert adapter to GGUF format:
  ```bash
  python scripts/convert_lora_to_gguf.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter adapters/weatherman-lora-h100 \
    --output models/weatherman-lora.gguf
  ```
- Import to Ollama:
  ```bash
  ollama create weatherman -f Modelfile
  ```
- Test inference:
  ```bash
  ollama run weatherman "What's the weather in Seattle?"
  ```

**Section 4: Sample Prompts**
- Include 10+ example prompts covering:
  - Current weather queries
  - Forecast requests
  - Multi-turn conversations
  - Error handling scenarios
  - All three personas (Twain, Franklin, Onion)

---

#### F8: Repository Configuration Updates
**ID**: F8
**Priority**: HIGH
**Description**: Modify `.gitignore` to include training data:

**Current**:
```gitignore
data/synthetic/*
!data/synthetic/.gitkeep
```

**New**:
```gitignore
data/synthetic/*
!data/synthetic/.gitkeep
!data/synthetic/final_train.jsonl
!data/synthetic/final_validation.jsonl
!data/synthetic/merge_metadata.json
```

**Rationale**: User explicitly requested data in repo. Exclude intermediate files but include final training data.

---

#### F9: README Updates
**ID**: F9
**Priority**: HIGH
**Description**: Update main `README.md` with:

**New Section: Quick Start - Training**
```markdown
## Training the LoRA Adapter

### Option 1: Remote Training (H100 via RunPod) - Recommended
**Duration**: 3-4 hours | **Cost**: ~$2-3

1. Clone repository on RunPod instance
2. Run setup: `bash setup_runpod_h100.sh`
3. Start training: `bash train_h100_runpod.sh`
4. Monitor: [WandB Dashboard](https://wandb.ai/user/weatherman-lora)
5. Download adapter: `scp -r adapters/weatherman-lora-h100 local-machine:`

Detailed guide: [docs/TRAINING_H100.md](docs/TRAINING_H100.md)

### Option 2: Local Training (Mac M4)
**Duration**: 12-18 hours | **Requirements**: 32GB RAM recommended

1. Run setup: `bash setup_m4_local.sh`
2. Start training: `bash train_m4_local.sh`

Detailed guide: [docs/TRAINING_M4.md](docs/TRAINING_M4.md)
```

---

#### F10: Roadmap Updates
**ID**: F10
**Priority**: MEDIUM
**Description**: Update `agent-os/product/roadmap.md`:

**Changes**:
1. Mark items 8 and 10 as "MERGED INTO COMBINED APPROACH"
2. Add new item: "Item 8+10: Combined Style+Tool-Use Training"
   - Status: IN PROGRESS
   - Approach: Single training run with 16,000 examples (69% humor personas)
   - Deliverables: H100 and M4 training scripts, deployment docs
3. Update dependencies and completion estimates

---

### Reusability Opportunities

#### Reuse 1: Setup Script Pattern
**Source**: `setup_remote.sh`, `setup_m4.sh`
**Reuse In**: `setup_runpod_h100.sh`, `train_h100_runpod.sh`, `train_m4_local.sh`

**Pattern**:
```bash
# Color definitions
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Status functions
success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ ERROR: $1${NC}"; exit 1; }
warning() { echo -e "${YELLOW}⚠️  WARNING: $1${NC}"; }
info() { echo -e "[INFO] $1"; }

# Step-by-step execution
info "Starting Step 1..."
# ... do work ...
success "Step 1 complete"
```

---

#### Reuse 2: Validation Scripts
**Source**: `scripts/check_gpu.py`, `scripts/check_storage.py`, `scripts/validate_environment.py`, `scripts/validate_training_config.py`
**Reuse In**: All setup and training scripts

**Pattern**:
```bash
info "Validating environment..."
if python scripts/validate_environment.py; then
    success "Environment validated"
else
    error "Environment validation failed"
fi
```

---

#### Reuse 3: Configuration Hierarchy
**Source**: `configs/training_config_h100.yaml`, `configs/training_config_m4.yaml`
**Reuse In**: Training scripts

**Pattern**: Allow config override via CLI:
```bash
CONFIG="${1:-configs/training_config_h100.yaml}"
python scripts/train.py --config "$CONFIG"
```

---

### Scope Boundaries

#### In Scope
✅ H100 RunPod setup and training scripts
✅ M4 local setup and training scripts
✅ Checkpoint resumption and crash loop detection
✅ Remote monitoring with Weights & Biases
✅ Deployment documentation for AnythingLLM/Ollama
✅ Repository configuration (gitignore, README)
✅ Roadmap updates
✅ Agent-readable status messages
✅ Pre-flight validation checks
✅ tmux session management for disconnected training

#### Out of Scope
❌ Implementing the GGUF conversion script (document steps only)
❌ Setting up RunPod account or SSH keys (assume user has access)
❌ Email/SMS notifications (optional, document webhook integration)
❌ Hyperparameter tuning (use existing configs)
❌ Model evaluation metrics beyond loss (future work)
❌ Multi-GPU training (single H100 or M4 only)
❌ Windows compatibility (Linux/Mac only)
❌ CI/CD pipeline integration (manual execution)

#### Deferred to Future Specs
🔄 Automated hyperparameter search
🔄 Model performance benchmarking suite
🔄 Production inference API deployment
🔄 Continuous training pipeline with new data

---

### Technical Considerations

#### TC1: Environment Differences
**Challenge**: Conda (H100) vs venv (M4), CUDA vs MPS

**Solution**:
- Separate setup scripts for each environment
- Shared training script with automatic backend detection:
  ```python
  if torch.cuda.is_available():
      device = "cuda"
  elif torch.backends.mps.is_available():
      device = "mps"
  else:
      device = "cpu"
  ```

---

#### TC2: Disk Space Management
**Challenge**: Model cache (14GB), checkpoints (1-2GB each), logs

**Solution**:
- Check storage before setup: `check_storage.py` requires 50GB+ free
- Limit checkpoint retention: `save_total_limit: 3` in config
- Document cleanup commands:
  ```bash
  # Clear HuggingFace cache
  rm -rf ~/.cache/huggingface
  # Remove old checkpoints
  find adapters/ -name "checkpoint-*" -type d -mtime +7 -exec rm -rf {} \;
  ```

---

#### TC3: Training Time Estimation
**Challenge**: Provide accurate time estimates for user planning

**Solution**:
- H100: 3-4 hours (based on 14,399 examples, batch size 16, 3 epochs)
- M4: 12-18 hours (4-6x slower due to MPS vs CUDA)
- Log estimated completion time after first 100 steps:
  ```python
  steps_per_second = 100 / elapsed_time
  total_steps = num_epochs * len(dataset) / batch_size
  eta_seconds = total_steps / steps_per_second
  print(f"[ETA] Estimated completion: {format_time(eta_seconds)}")
  ```

---

#### TC4: Dependency Version Compatibility
**Challenge**: Ensure consistent behavior across environments

**Solution**:
- Pin exact versions in `environment-remote.yml` and `requirements-local.txt`
- Test compatibility matrix:
  - H100: PyTorch 2.1.0 + CUDA 12.1 + transformers 4.36.0
  - M4: PyTorch 2.1.0 + MPS + transformers 4.36.0
- Document known issues (e.g., Flash Attention 2 not available on M4)

---

#### TC5: Data Path Consistency
**Challenge**: Config expects `data/processed/` but data is in `data/synthetic/`

**Solution**:
- Setup scripts verify symlinks exist:
  ```bash
  if [ ! -L data/processed/train.jsonl ]; then
      ln -sf "$(pwd)/data/synthetic/final_train.jsonl" data/processed/train.jsonl
  fi
  ```
- Alternative: Create `data/processed/` as real directory and copy files (but wastes space)

---

#### TC6: Crash Loop Detection Implementation
**Challenge**: Distinguish between legitimate resumes and crash loops

**Solution**:
- Create `resume_metadata.json` in output directory:
  ```json
  {
    "last_checkpoint": "checkpoint-1500",
    "last_step": 1500,
    "resume_count": 2,
    "last_resume_time": "2025-11-03T14:30:22"
  }
  ```
- Training script logic:
  ```python
  if resume_metadata exists:
      if current_step == last_step:
          resume_count += 1
          if resume_count >= 3:
              alert_crash_loop()
              exit(2)
      else:
          resume_count = 0  # Made progress, reset counter
  ```

---

#### TC7: RunPod-Specific Setup
**Challenge**: RunPod images may have pre-installed software

**Solution**:
- Detect existing conda: `which conda` before creating new env
- Check for conflicting PyTorch versions: `conda list | grep torch`
- Use RunPod's persistent storage: `/workspace` for all outputs
- Document RunPod template selection: "PyTorch 2.1 + CUDA 12.1"

---

## Success Criteria

### SC1: H100 Training Starts Successfully
- [ ] Setup script completes without errors
- [ ] Training script launches in tmux
- [ ] First 100 training steps complete
- [ ] No CUDA out-of-memory errors
- [ ] Checkpoints saved correctly
- [ ] Weights & Biases logging active

### SC2: M4 Training Scripts Created
- [ ] Setup script validates M4 environment
- [ ] Training script uses MPS backend
- [ ] Reduced batch size prevents memory errors
- [ ] Documentation complete

### SC3: Process is Repeatable
- [ ] Can destroy and recreate environment
- [ ] Scripts are idempotent (safe to run multiple times)
- [ ] Version pinning prevents dependency drift
- [ ] Clear error messages guide troubleshooting

### SC4: Agent-Readable Output
- [ ] All scripts use structured status format
- [ ] Exit codes consistent (0=success, 1=error, 2=crash loop)
- [ ] Decision points have clear pass/fail markers
- [ ] Logs parseable by automated systems

### SC5: Documentation Complete
- [ ] README updated with quick start
- [ ] Deployment guide created
- [ ] Training guides for H100 and M4
- [ ] Roadmap reflects merged approach

### SC6: Repository Ready for Replication
- [ ] Training data committed to git
- [ ] All scripts executable (`chmod +x`)
- [ ] Dependencies documented
- [ ] Example outputs provided

---

## Estimated Implementation Timeline

**Total**: 1-2 development sessions

### Session 1: Core Scripts (Current)
- [ ] F1: H100 Setup Script (30-45 min)
- [ ] F2: H100 Training Script (30-45 min)
- [ ] F5: Checkpoint Resumption Logic (20-30 min)
- [ ] F8: Update .gitignore (5 min)
- [ ] Testing: Run setup on RunPod (15-20 min)

### Session 2: Documentation & Polish
- [ ] F3: M4 Setup Script (20-30 min)
- [ ] F4: M4 Training Script (20-30 min)
- [ ] F7: Deployment Documentation (30-45 min)
- [ ] F9: README Updates (15-20 min)
- [ ] F10: Roadmap Updates (10-15 min)
- [ ] F6: Monitoring Setup (10-15 min)

---

## Notes for Spec Writer

- User explicitly requested "take your time" - prioritize quality over speed
- Focus on H100 execution (primary) but create M4 scripts in parallel
- Agent-readable messaging is critical - use consistent format
- Crash loop detection should alert, not retry endlessly
- User expects downloadable artifact ready for AnythingLLM/Ollama
- Reference existing patterns from setup_remote.sh and setup_m4.sh
- This replaces roadmap items 8 and 10 - update roadmap accordingly
