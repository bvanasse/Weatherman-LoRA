# Specification: QLoRA Training Configuration for H100 and Mac M4

## Goal
Create dual-platform QLoRA training configurations optimized for H100 GPU (primary) and Mac M4 (secondary) with Mistral 7B Instruct, ensuring stable and efficient training with limited H100 access.

## User Stories
- As an ML engineer with limited H100 access, I want a stable, production-ready training configuration so that I can successfully train LoRA adapters without trial-and-error experimentation
- As a developer with Mac M4, I want an alternative training configuration so that I can iterate on training locally before committing H100 resources

## Specific Requirements

**H100 Training Configuration**
- Create `configs/training_config_h100.yaml` with Mistral 7B Instruct base model (`mistralai/Mistral-7B-Instruct-v0.2`)
- Configure 4096 token sequence length with batch size 4-8 and gradient accumulation 4 steps
- Enable Flash Attention 2, gradient checkpointing, and 4-bit NF4 quantization with double quantization
- Set LoRA parameters: rank=16, alpha=32, dropout=0.05 targeting all 7 projection modules (q/k/v/o/gate/up/down)
- Configure training for 3 epochs with learning rate 2e-4, cosine decay scheduler, 3% warmup
- Set wandb as default experiment tracking with logging every 10 steps, evaluation every 100 steps
- Document expected training time (3-4 hours) and memory usage (~60-70GB VRAM)

**Mac M4 Training Configuration**
- Create `configs/training_config_m4.yaml` adapted for PyTorch MPS backend and 32GB unified memory
- Configure 2048 token sequence length with batch size 1-2 and gradient accumulation 8-16 steps to maintain similar effective batch size
- Disable Flash Attention 2 (CUDA-only), enable gradient checkpointing, use MPS device for acceleration
- Use identical LoRA parameters and training schedule as H100 config for consistency
- Document expected training time (8-12 hours, 2-3x slower than H100) and memory constraints
- Include warnings about unified memory limitations and batch size trade-offs

**Platform Detection and Validation**
- Extend `scripts/check_gpu.py` to detect H100 vs Mac M4 (MPS) and recommend appropriate config file
- Add platform-specific validation in `scripts/validate_environment.py` for H100 (CUDA 12.1+, Flash Attention) and M4 (MPS availability, memory checks)
- Create validation that checks for Mistral 7B compatibility, appropriate PyTorch version (2.1+), and required dependencies (PEFT 0.7+, TRL, bitsandbytes)
- Implement early failure with actionable error messages if environment is misconfigured

**Configuration Loading Integration**
- Ensure both config files work seamlessly with existing `scripts/config_loader.py` infrastructure
- Maintain YAML structure consistency with existing `configs/training_config.yaml` for easy migration
- Support override patterns for quick hyperparameter experimentation without modifying base configs
- Validate all required fields on load: model path, LoRA parameters, batch sizes, sequence lengths

**Environment Setup Scripts**
- Update `environment-remote.yml` to include H100-specific optimizations and Flash Attention 2 dependencies
- Create or update setup script for Mac M4 that installs MPS-compatible PyTorch and disables CUDA-only features
- Include dependency version pinning matching tech stack requirements (PyTorch 2.1+, Transformers 4.36+)
- Add colored output validation following `setup_local.sh` patterns (RED/GREEN/YELLOW/NC)

**Training Monitoring and Checkpointing**
- Configure wandb integration with project name, run name, and logging frequency for both platforms
- Set checkpoint saving every 500 steps, keep best 3 checkpoints based on eval_loss
- Enable load_best_model_at_end with early stopping capability
- Provide TensorBoard as alternative for offline-first development

**Data Integration**
- Point to existing JSONL data pipeline outputs at `data/processed/train.jsonl` and `data/processed/val.jsonl`
- Configure chat format data loading with messages containing role/content fields
- Set padding strategy to "right" for training, truncation enabled for long sequences
- Maintain 90/10 train/validation split if separate validation file not provided

**Memory Optimization Strategy**
- Enable gradient checkpointing on both platforms (accepts 20% slower training for 30-40% memory savings)
- Use paged AdamW 32-bit optimizer for memory-efficient gradient updates
- Configure bfloat16 mixed precision (not fp16) for numerical stability
- Document memory budgets: H100 ~60-70GB for seq_len 4096, M4 ~24-28GB for seq_len 2048

**Documentation and Error Handling**
- Add inline comments in both YAML configs explaining each parameter's purpose and platform-specific choices
- Document expected behavior, training times, and memory usage at bottom of each config file
- Provide troubleshooting guidance for OOM errors (reduce batch size, sequence length)
- Include validation that fails fast before training starts if resources insufficient

**Stability and Production Readiness**
- Use proven hyperparameters from existing `configs/training_config.yaml` where applicable
- Prioritize stable configurations over experimental features due to limited H100 access
- Test configs load without errors using `scripts/config_loader.py` validation
- Ensure backward compatibility with existing directory structure (`adapters/`, `data/`, `models/`)

## Existing Code to Leverage

**`configs/training_config.yaml`**
- Template structure for LoRA, quantization, training arguments, model, dataset, and system prompt sections
- Copy proven hyperparameters: learning rate 2e-4, cosine scheduler, warmup ratio 0.03, weight decay 0.01, max_grad_norm 0.3
- Reuse optimizer choice (paged_adamw_32bit), mixed precision settings (bf16=true, fp16=false)
- Adapt checkpoint saving strategy, evaluation frequency, and wandb reporting configuration
- Modify base model from Llama 3.1 8B to Mistral 7B Instruct, adjust target modules if needed

**`scripts/config_loader.py`**
- Use load_yaml() function for reading new H100/M4 config files with error handling
- Leverage deep_merge() for applying runtime overrides without modifying base configs
- Reuse validate_required_fields() to ensure critical parameters present before training
- Implement get_config_value() for accessing nested config values with dot notation
- Follow existing patterns for config file discovery and environment-specific loading

**`scripts/check_gpu.py`**
- Extend check_pytorch_import() to detect MPS availability on Mac in addition to CUDA
- Adapt GPU detection logic to identify H100 specifically and recommend config_h100.yaml
- Add MPS detection branch that checks torch.backends.mps.is_available() and recommends config_m4.yaml
- Reuse bytes_to_gb() and memory assessment logic for both CUDA and unified memory reporting
- Maintain colored output patterns (RED/GREEN/YELLOW/NC) and actionable error messages

**`scripts/validate_environment.py`**
- Add check_imports_remote() validation for H100-specific dependencies (Flash Attention, CUDA 12.1+)
- Create check_imports_m4() for Mac M4 validating MPS backend and MPS-compatible PyTorch
- Extend check_gpu() to validate platform-specific capabilities (Flash Attention on H100, MPS on M4)
- Reuse print_header(), print_check() formatting functions for consistent output style
- Follow validation summary pattern showing pass/fail for each check with actionable next steps

**`environment-remote.yml`**
- Use as template for H100 environment specification with conda channels and dependencies
- Keep pinned versions: PyTorch 2.1.0, PEFT 0.7.0, TRL 0.7.4, bitsandbytes 0.41.0, transformers 4.36.0
- Add Flash Attention 2 dependency for H100 config (not in current file)
- Maintain notes section documenting training times, memory requirements, and LoRA parameters
- Create parallel `environment-m4.yml` for Mac with MPS-compatible PyTorch build

## Out of Scope
- Multi-GPU training configurations (FSDP, DeepSpeed, or data parallelism across multiple devices)
- Automatic hyperparameter tuning, grid search, or Bayesian optimization of LoRA parameters
- Cloud platform-specific setup scripts for Lambda Labs, RunPod, vast.ai, or other GPU providers
- Windows support for training (focus only on Linux for H100, macOS for M4)
- Full fine-tuning mode without LoRA adapters (parameter-efficient training only)
- Model merging scripts to combine LoRA adapters with base model weights
- Quantization of final trained adapters (train in 4-bit but save adapters unquantized)
- Evaluation harness for style consistency, tool-call validation (separate roadmap item #9, #11)
- Deployment and serving configuration with vLLM or text-generation-inference (roadmap item #12)
- Support for alternative base models beyond Mistral 7B Instruct (Llama, Qwen, etc.)
