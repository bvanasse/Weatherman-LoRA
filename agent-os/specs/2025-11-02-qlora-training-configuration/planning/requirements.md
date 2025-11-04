# Spec Requirements: QLoRA Training Configuration

## Initial Description

Configure QLoRA training setup with the following requirements:

**Base Model:** Mistral 7B Instruct

**Primary Target:** Single H100 GPU optimization

**Secondary Target:** Mac M4 support as an alternative training option

**Configuration Requirements:**
- LoRA hyperparameters: r=16, alpha=32, dropout=0.05
- Target modules: q/k/v/o projections
- Optimization: 4-bit quantization, gradient checkpointing, flash attention
- Dual platform support: H100 (primary) and Mac M4 (alternative)

**Size Estimate:** S (Small)

## Requirements Discussion

### First Round Questions

**Q1:** Mac M4 Configuration Approach - I'm assuming we'll use PyTorch's MPS (Metal Performance Shaders) backend for Mac M4 training with potentially reduced batch sizes and no Flash Attention 2 (since it's CUDA-only). Should we expect longer training times (e.g., 8-12 hours vs 4-6 hours on H100) and provide guidance on M4 memory constraints?

**Answer:** Yes, make what modifications are necessary when running the Mac M4. This laptop has 32GB of memory, provide guidance on memory constraints. Remember to still provide the primary method of training on the H100s and to preserve the features like Flash Attention 2 etc.

**Q2:** Configuration File Structure - I'm thinking we should create separate config files (e.g., `config_h100.yaml` and `config_m4.yaml`) rather than a single unified config with platform switches. Does that work, or would you prefer a different approach (JSON files, Python dataclasses, command-line args)?

**Answer:** Use separate config files as necessary.

**Q3:** Training Monitoring - The tech stack mentions wandb and TensorBoard. Should we default to wandb for experiment tracking (with instructions for setup), or prefer TensorBoard for offline-first development?

**Answer:** Use your best judgement.

**Q4:** Sequence Length - The tech stack notes 4096 tokens preferred, 2048 acceptable if memory-constrained. For H100 I assume we'll use 4096, but should Mac M4 default to 2048 to fit in unified memory?

**Answer:** Is 32GB not enough when using the M4? If not then yes make adjustments as necessary.

**Q5:** Target LoRA Modules - The tech stack lists all 7 modules (`q/k/v/o/gate/up/down projections`). Should we use all 7 for maximum adaptation, or start with just attention modules (`q/k/v/o`) for faster training and fewer parameters?

**Answer:** Do whichever is most effective for the training time we have.

**Q6:** Hyperparameter Values Confirmation - You mentioned r=16, alpha=32, dropout=0.05. Should we stick with these exact values, or provide a range (e.g., "r=16, can increase to 32 if underfitting")?

**Answer:** Stick with the values that are ideal.

**Q7:** Gradient Checkpointing Trade-offs - I assume we'll enable gradient checkpointing on both platforms for memory efficiency, even though it adds ~20% training time. Correct?

**Answer:** Yes.

**Q8:** Exclusions & Future Work - What should we explicitly NOT include in this spec? For example: multi-GPU support, FSDP/DeepSpeed integration, automatic hyperparameter tuning, cloud platform-specific configs (Lambda Labs, RunPod), or Windows support?

**Answer:** Use your best judgement, the user will have limited access to the H100 so stability and effectiveness of the training scripts is preferred.

### Existing Code to Reference

**Similar Features Identified:**

The project already has established patterns for configuration and training infrastructure:

- **Training Configuration**: `configs/training_config.yaml` - Existing comprehensive YAML config with LoRA, quantization, and training parameters. Currently configured for Llama 3.1 8B, needs adaptation for Mistral 7B Instruct.

- **Config Loader Utility**: `scripts/config_loader.py` - Production-ready Python module for loading YAML configs with:
  - Deep merge functionality for overrides
  - Validation of required fields
  - Dot-notation access to nested values (e.g., `'lora.r'`)
  - Support for environment-specific configs

- **Environment Setup**: `setup_local.sh` - Mac-specific setup script that:
  - Verifies Python 3.10+ installation
  - Creates virtual environment in `.venv-local/`
  - Installs dependencies from `requirements-local.txt`
  - Provides colored output with validation steps

- **Remote Environment**: `environment-remote.yml` - Conda environment spec for H100/GPU training

- **Infrastructure Validation Scripts**:
  - `scripts/check_gpu.py` - GPU detection and capability checking
  - `scripts/validate_environment.py` - Environment validation utilities
  - `scripts/check_storage.py` - Storage requirement verification

**Patterns to Follow:**
- YAML format for configuration files (consistent with existing `training_config.yaml`)
- Use `scripts/config_loader.py` patterns for loading and validation
- Follow color-coded output style from `setup_local.sh` (RED/GREEN/YELLOW/NC)
- Leverage existing validation utilities where applicable

## Visual Assets

### Files Provided:
No visual assets provided.

## Requirements Summary

### Functional Requirements

**Dual-Platform Configuration:**
- Primary: H100 GPU training configuration (CUDA 12.1+, Flash Attention 2)
- Secondary: Mac M4 training configuration (MPS backend, 32GB unified memory)
- Separate config files: `configs/training_config_h100.yaml` and `configs/training_config_m4.yaml`

**Base Model:**
- Use Mistral 7B Instruct (`mistralai/Mistral-7B-Instruct-v0.3`)
- Update from existing Llama 3.1 8B configuration

**LoRA Hyperparameters (Fixed Values):**
- Rank (r): 16
- Alpha: 32 (2x rank)
- Dropout: 0.05
- Target modules: All 7 projection layers (q/k/v/o/gate/up/down) for maximum effectiveness
- Bias: none
- Task type: CAUSAL_LM

**Quantization (QLoRA):**
- 4-bit precision (NF4 quantization type)
- Double quantization enabled
- Compute dtype: bfloat16

**Memory Optimization:**
- Gradient checkpointing: Enabled on both platforms
- Flash Attention 2: Enabled on H100 only (CUDA-only feature)
- MPS backend: For Mac M4 (PyTorch Metal)

**Platform-Specific Adjustments:**

*H100 Configuration:*
- Sequence length: 4096 tokens (preferred)
- Batch size: 4-8 per device
- Gradient accumulation: 4 steps (effective batch 16-32)
- Expected training time: 3-4 hours for 3 epochs on 15K examples
- Flash Attention 2: Enabled
- Memory budget: ~60-70GB VRAM usage expected

*Mac M4 Configuration:*
- Sequence length: 2048 tokens (adjusted for 32GB unified memory)
- Batch size: 1-2 per device (memory-constrained)
- Gradient accumulation: 8-16 steps (maintain similar effective batch size)
- Expected training time: 8-12 hours for 3 epochs (2-3x slower than H100)
- Flash Attention 2: Disabled (not supported on MPS)
- MPS backend: Use PyTorch MPS acceleration
- Memory guidance: Document 32GB unified memory constraints and batch size trade-offs

**Training Parameters:**
- Epochs: 3 (adjustable but 3 is proven)
- Learning rate: 2e-4
- LR scheduler: Cosine decay with 3% warmup
- Optimizer: Paged AdamW 32-bit
- Weight decay: 0.01
- Max gradient norm: 0.3
- Mixed precision: bfloat16 (fp16 disabled)

**Monitoring & Logging:**
- Default: Weights & Biases (wandb) for experiment tracking
- Alternative: TensorBoard for offline tracking
- Logging frequency: Every 10 steps
- Evaluation: Every 100 steps
- Checkpoint saving: Every 500 steps, keep best 3
- Metric for best model: eval_loss (lower is better)

**Data Configuration:**
- Format: JSONL chat format with messages (role/content)
- Train file: `data/processed/train.jsonl`
- Validation file: `data/processed/val.jsonl`
- Train/val split: 90/10 if no separate validation file

**Configuration Loading:**
- Reuse existing `scripts/config_loader.py` infrastructure
- Support for overrides and validation
- Environment-specific config loading capability

**Stability & Robustness:**
- Prioritize stable, well-tested configurations over experimental features
- Clear error messages and validation checks
- Graceful fallbacks for missing dependencies
- Documentation of expected training times and memory usage
- Platform detection to auto-select appropriate config

### Reusability Opportunities

**Existing Code to Leverage:**
- `scripts/config_loader.py` - Reuse for loading new H100/M4 configs
- `configs/training_config.yaml` - Template for new platform-specific configs
- `scripts/check_gpu.py` - Extend for H100 vs M4 detection
- `scripts/validate_environment.py` - Add platform-specific validations
- `setup_local.sh` patterns - Consistent setup experience for both platforms
- `environment-remote.yml` - Model for H100 environment specification

**Components Already Exist:**
- YAML config structure and conventions
- Config validation and loading utilities
- Environment setup scripts with colored output
- GPU/storage validation infrastructure

### Scope Boundaries

**In Scope:**
- Two separate training configuration files (H100 and M4)
- Platform-specific hyperparameter adjustments
- Mistral 7B Instruct as base model
- QLoRA 4-bit quantization configuration
- Memory optimization strategies for both platforms
- Training monitoring setup (wandb/TensorBoard)
- Documentation of expected training times and memory usage
- Gradient checkpointing on both platforms
- Flash Attention 2 for H100 only
- MPS backend configuration for Mac M4
- Platform detection and config selection logic
- Clear guidance on 32GB M4 memory constraints

**Out of Scope:**
- Multi-GPU or distributed training (FSDP, DeepSpeed)
- Automatic hyperparameter tuning or search
- Cloud platform-specific configurations (Lambda Labs, RunPod, vast.ai)
- Windows support (focus on Linux for H100, macOS for M4)
- Full fine-tuning (only LoRA adapters)
- Model merging or quantization of final adapters
- Evaluation harness implementation (separate roadmap item)
- Deployment/serving configuration (separate roadmap item)
- Alternative base models beyond Mistral 7B Instruct
- Custom training loops (use TRL's SFTTrainer)
- Data preprocessing pipelines (already completed in roadmap)

### Technical Considerations

**Integration Points:**
- Must work with existing `scripts/config_loader.py` infrastructure
- Compatible with TRL's SFTTrainer and PEFT library
- Integrate with existing data pipeline outputs (JSONL format)
- Support wandb integration for experiment tracking
- Work with existing directory structure (`adapters/`, `data/`, `models/`)

**Platform Constraints:**
- H100: CUDA 12.1+, Flash Attention 2, 80GB VRAM, Linux environment
- Mac M4: PyTorch MPS backend, 32GB unified memory, macOS, no Flash Attention 2
- Both: PyTorch 2.1+, Transformers 4.36+, PEFT 0.7+, bitsandbytes 0.41+

**Technology Stack:**
- Base: PyTorch 2.1+, Transformers, PEFT, TRL, bitsandbytes
- Monitoring: Weights & Biases (primary), TensorBoard (alternative)
- Config format: YAML (consistent with existing patterns)
- Loading: Python with existing config_loader.py utilities

**Similar Code Patterns:**
- Follow YAML structure from `configs/training_config.yaml`
- Use config_loader.py's validation and override patterns
- Match setup_local.sh's colored output and validation style
- Maintain consistency with existing script organization

**Memory Considerations:**
- H100: ~60-70GB expected for 4096 seq length, batch size 4, Mistral 7B in 4-bit
- Mac M4: ~24-28GB expected for 2048 seq length, batch size 1-2, with unified memory
- Gradient checkpointing reduces peak memory by ~30-40% at cost of 20% slower training
- 32GB M4 should be sufficient with reduced batch size and sequence length

**Stability Requirements:**
- User has limited H100 access - minimize trial and error
- Use proven hyperparameters from existing config
- Include comprehensive validation before training starts
- Clear error messages if environment is misconfigured
- Document expected behavior and training times
- Fail fast with actionable error messages
