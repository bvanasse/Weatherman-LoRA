# Task Breakdown: QLoRA Training Configuration for H100 and Mac M4

## Overview
Total Tasks: 4 Task Groups
Estimated Complexity: Small (S)
Focus: Dual-platform training configuration with stability and validation

## Task List

### Configuration Files

#### Task Group 1: Training Configuration Files
**Dependencies:** None

- [x] 1.0 Complete training configuration files
  - [x] 1.1 Create H100 training configuration
    - Create `configs/training_config_h100.yaml` based on existing `configs/training_config.yaml` structure
    - Update base model to `mistralai/Mistral-7B-Instruct-v0.2`
    - Set sequence length: 4096 tokens
    - Configure batch size: 4-8, gradient accumulation: 4 steps
    - Set LoRA parameters: r=16, alpha=32, dropout=0.05
    - Target all 7 modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    - Configure QLoRA: load_in_4bit=true, bnb_4bit_quant_type=nf4, double_quant=true, compute_dtype=bfloat16
    - Enable gradient_checkpointing=true
    - Set training: 3 epochs, learning_rate=2e-4, lr_scheduler=cosine, warmup_ratio=0.03
    - Configure optimizer: paged_adamw_32bit, weight_decay=0.01, max_grad_norm=0.3
    - Enable bf16=true, fp16=false
    - Set logging_steps=10, eval_steps=100, save_steps=500, save_total_limit=3
    - Configure wandb: report_to=wandb, run_name with descriptive prefix
    - Add data paths: train_file=data/processed/train.jsonl, val_file=data/processed/val.jsonl
    - Add inline comments explaining H100-specific choices (Flash Attention 2, 4096 seq length)
    - Document at bottom: expected training time (3-4 hours), memory usage (~60-70GB VRAM)
  - [x] 1.2 Create Mac M4 training configuration
    - Create `configs/training_config_m4.yaml` based on H100 config structure
    - Keep same base model: `mistralai/Mistral-7B-Instruct-v0.2`
    - Adjust sequence length: 2048 tokens (memory-constrained)
    - Configure batch size: 1-2, gradient accumulation: 8-16 steps
    - Keep identical LoRA parameters: r=16, alpha=32, dropout=0.05, same target modules
    - Keep identical QLoRA and training parameters for consistency
    - Add comment noting Flash Attention 2 not available on MPS
    - Add device configuration hints for MPS backend
    - Document at bottom: expected training time (8-12 hours), memory constraints (32GB unified)
    - Include troubleshooting notes for OOM errors (reduce batch size to 1, gradient accumulation to 16)
  - [x] 1.3 Validate configuration file structure
    - Test loading both configs using `scripts/config_loader.py`
    - Verify all required fields present using validate_required_fields()
    - Ensure YAML syntax is valid and parses correctly
    - Confirm backward compatibility with existing config_loader patterns

**Acceptance Criteria:**
- Both `training_config_h100.yaml` and `training_config_m4.yaml` created
- Configs load without errors via config_loader.py
- H100 config optimized for 4096 tokens, batch size 4-8, Flash Attention 2
- M4 config adapted for 2048 tokens, batch size 1-2, MPS backend
- Inline documentation clear and comprehensive
- Expected training times and memory usage documented

### Platform Detection & Validation

#### Task Group 2: GPU Detection and Platform Validation
**Dependencies:** Task Group 1

- [x] 2.0 Complete platform detection and validation
  - [x] 2.1 Write 2-4 focused tests for platform detection
    - Test H100/CUDA detection returns correct config recommendation
    - Test Mac M4/MPS detection returns correct config recommendation
    - Test validation catches missing dependencies (CUDA, MPS, PyTorch)
    - Test error messages are actionable
  - [x] 2.2 Extend GPU detection for dual-platform support
    - Extend `scripts/check_gpu.py` to detect MPS availability
    - Add `check_mps_backend()` function using torch.backends.mps.is_available()
    - Modify main() to branch: if CUDA available -> check H100, if MPS available -> check M4
    - For H100: identify by memory (>=70GB) and recommend `training_config_h100.yaml`
    - For M4: check MPS availability and recommend `training_config_m4.yaml`
    - Add memory assessment for unified memory (32GB check)
    - Maintain colored output (RED/GREEN/YELLOW/NC) and actionable error messages
    - Document expected training times for each platform in output
  - [x] 2.3 Extend environment validation for platform-specific checks
    - Extend `scripts/validate_environment.py` with platform detection
    - Add `check_imports_h100()` validating CUDA 12.1+, Flash Attention dependencies
    - Add `check_imports_m4()` validating MPS backend, MPS-compatible PyTorch
    - Add `check_mistral_compatibility()` verifying Mistral 7B support (Transformers 4.36+)
    - Add `check_training_dependencies()` for PEFT 0.7+, TRL, bitsandbytes 0.41+
    - Update main validation flow to detect platform and run appropriate checks
    - Add validation for minimum PyTorch 2.1+ on both platforms
    - Provide actionable next steps based on platform detected
  - [x] 2.4 Add config file validation script
    - Create `scripts/validate_training_config.py` to validate config completeness
    - Check required fields: model.model_name_or_path, lora.r, lora.lora_alpha, training.learning_rate
    - Validate LoRA parameters in acceptable ranges (r=16, alpha=32, dropout=0.05)
    - Warn if sequence length + batch size might exceed platform memory
    - Verify data file paths exist (train.jsonl, val.jsonl)
    - Check wandb configuration has project name and run name
    - Exit with error code 1 if validation fails, 0 if passes
  - [x] 2.5 Ensure platform detection tests pass
    - Run only the 2-4 tests written in 2.1
    - Verify GPU detection correctly identifies H100 vs M4
    - Confirm appropriate config file recommendations
    - Validate error messages guide user to correct actions

**Acceptance Criteria:**
- Platform detection tests (2-4 tests) pass
- `check_gpu.py` detects both CUDA (H100) and MPS (M4) platforms
- `validate_environment.py` runs platform-specific validation checks
- Config validation script catches common misconfigurations
- Error messages are clear, colored, and actionable

### Environment Setup

#### Task Group 3: Environment Configuration and Setup Scripts
**Dependencies:** Task Group 2

- [x] 3.0 Complete environment setup for both platforms
  - [x] 3.1 Write 2-3 focused tests for environment setup
    - Test H100 environment file has correct dependencies (Flash Attention, CUDA 12.1)
    - Test M4 requirements file has MPS-compatible PyTorch
    - Test version pinning matches tech stack requirements
  - [x] 3.2 Update H100 remote environment specification
    - Update `environment-remote.yml` with H100-specific optimizations
    - Add Flash Attention 2 dependency (flash-attn package)
    - Verify pinned versions: PyTorch 2.1.0, PEFT 0.7.0, TRL 0.7.4, bitsandbytes 0.41.0, transformers 4.36.0
    - Add CUDA 12.1+ requirement in comments
    - Update notes section with Mistral 7B training specifics
    - Document H100 expected training time (3-4 hours for 3 epochs, 15K examples)
    - Add memory requirements: 80GB VRAM recommended, 60GB minimum
  - [x] 3.3 Create Mac M4 setup script
    - Create `setup_m4.sh` based on `setup_local.sh` patterns
    - Verify Python 3.10+ installation
    - Create virtual environment in `.venv-m4/`
    - Update `requirements-m4.txt` with MPS-compatible dependencies
    - Install PyTorch with MPS support (no CUDA)
    - Skip Flash Attention 2 installation (CUDA-only)
    - Add validation steps checking MPS backend availability
    - Use colored output (RED/GREEN/YELLOW/NC) matching setup_local.sh style
    - Display M4 memory constraints (32GB unified memory)
    - Document expected training time (8-12 hours)
  - [x] 3.4 Add platform-specific setup documentation
    - Update or create `docs/SETUP_H100.md` with H100 setup instructions
    - Document H100 prerequisites: Linux, CUDA 12.1+, 60-80GB VRAM
    - Provide conda environment activation instructions
    - Include Flash Attention 2 installation verification
    - Update or create `docs/SETUP_M4.md` with Mac M4 setup instructions
    - Document M4 prerequisites: macOS, 32GB unified memory, MPS support
    - Provide pip/venv setup instructions
    - Include MPS backend validation steps
    - Add troubleshooting sections for common setup issues
  - [x] 3.5 Ensure environment setup tests pass
    - Run only the 2-3 tests written in 3.1
    - Verify environment files have correct dependencies
    - Confirm version pinning is accurate

**Acceptance Criteria:**
- Environment setup tests (2-3 tests) pass
- `environment-remote.yml` updated for H100 with Flash Attention 2
- `setup_m4.sh` and `requirements-m4.txt` created for Mac M4
- Setup scripts use consistent colored output and validation
- Documentation guides users through platform-specific setup

### Integration & End-to-End Validation

#### Task Group 4: Integration Testing and Documentation
**Dependencies:** Task Groups 1-3

- [x] 4.0 Complete integration testing and final documentation
  - [x] 4.1 Review existing tests and identify critical gaps
    - Review tests from Task Groups 1-3 (approximately 7-10 tests total)
    - Identify end-to-end workflow gaps: config loading → platform detection → validation → training readiness
    - Focus only on gaps related to this dual-platform configuration feature
    - Prioritize integration points over unit test coverage
  - [x] 4.2 Write up to 5 additional integration tests maximum
    - Test end-to-end: load H100 config → validate environment → recommend training command
    - Test end-to-end: load M4 config → validate environment → recommend training command
    - Test config override functionality works with both configs
    - Test config loader validates required Mistral 7B fields
    - Test platform detection recommends correct config for current hardware
  - [x] 4.3 Create training quick-start guide
    - Create `docs/TRAINING_QUICKSTART.md` with step-by-step instructions
    - Section 1: Platform detection - how to check which config to use
    - Section 2: H100 training - setup, config selection, expected results
    - Section 3: M4 training - setup, config selection, memory considerations
    - Section 4: Config customization - how to override hyperparameters
    - Section 5: Troubleshooting - common issues and solutions
    - Include example commands for loading configs and starting training
  - [x] 4.4 Add README updates and config usage examples
    - Update main `README.md` with QLoRA training configuration section
    - Document both training configs and when to use each
    - Add example: loading config via config_loader.py
    - Add example: checking GPU/platform compatibility
    - Add example: running platform-specific validation
    - Link to detailed setup docs (SETUP_H100.md, SETUP_M4.md)
  - [x] 4.5 Run comprehensive feature validation
    - Run all feature-specific tests (approximately 12-15 tests total)
    - Verify both configs load successfully via config_loader.py
    - Confirm platform detection works on current machine
    - Test validation scripts catch common misconfigurations
    - Validate setup scripts provide clear guidance
    - Check documentation is complete and accurate

**Acceptance Criteria:**
- All feature-specific tests pass (approximately 12-15 tests total)
- No more than 5 additional integration tests added
- End-to-end workflows validated: config loading, platform detection, training readiness
- Quick-start guide provides clear step-by-step instructions
- README updated with dual-platform configuration usage
- Feature ready for production use with limited H100 access

## Execution Order

Recommended implementation sequence:
1. **Configuration Files** (Task Group 1) - Create H100 and M4 YAML configs with proper hyperparameters
2. **Platform Detection & Validation** (Task Group 2) - Extend GPU detection and validation for both platforms
3. **Environment Setup** (Task Group 3) - Update environment specs and create setup scripts
4. **Integration & Documentation** (Task Group 4) - End-to-end testing and user documentation

## Notes

- This is a configuration feature, not a typical CRUD application - testing focuses on validation and integration
- Limited test writing (2-5 tests per group) appropriate for config files and setup scripts
- Priority on stability and clear documentation due to limited H100 access
- Each platform maintains consistent LoRA hyperparameters for reproducibility
- Memory optimization critical for M4's 32GB constraint
