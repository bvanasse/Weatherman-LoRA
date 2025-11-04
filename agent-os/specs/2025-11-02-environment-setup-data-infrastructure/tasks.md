# Task Breakdown: Environment Setup & Data Infrastructure

## Overview
Total Tasks: 6 task groups with 37 sub-tasks

## Task List

### Infrastructure Setup

#### Task Group 1: Project Directory Structure
**Dependencies:** None

- [x] 1.0 Complete project directory structure
  - [x] 1.1 Create base directory structure
    - Create `data/raw/` for original downloads
    - Create `data/processed/` for cleaned datasets
    - Create `data/synthetic/` for generated examples
    - Create `models/` for base model storage
    - Create `adapters/` for LoRA weights
    - Create `scripts/` for Python scripts
    - Create `configs/` for configuration files
  - [x] 1.2 Create storage verification script
    - Script name: `scripts/check_storage.py`
    - Check available disk space on current system
    - Warn if less than 30GB available
    - Error if less than 20GB available
    - Display storage breakdown: raw data (~150-300MB), processed (~500MB-1GB), models (~15GB), checkpoints (~500MB-2GB), buffer (~10-15GB)
  - [x] 1.3 Add .gitignore entries
    - Ignore `data/` contents (but keep directory structure)
    - Ignore `models/` contents (large files)
    - Ignore `adapters/` contents (checkpoints)
    - Keep `scripts/` and `configs/` tracked
    - Add Python cache patterns (__pycache__, *.pyc)
  - [x] 1.4 Create README.md for project structure
    - Document directory purposes and conventions
    - Explain local vs remote environment workflow
    - Reference storage requirements (30-50GB)

**Acceptance Criteria:**
- All directories created and empty (except existing data_sources/)
- Storage verification script runs and reports available space
- .gitignore prevents tracking large files
- README documents project structure clearly

### Local Environment (Mac M4)

#### Task Group 2: Local Data Processing Environment
**Dependencies:** Task Group 1

- [x] 2.0 Complete local environment setup
  - [x] 2.1 Create requirements-local.txt with pinned versions
    - pandas==2.1.3
    - beautifulsoup4==4.12.2
    - trafilatura==1.6.2
    - datasets==2.15.0
    - datasketch==1.6.4
    - jsonlines==4.0.0
    - nltk==3.8.1
    - langdetect==1.0.9 (or fasttext alternative)
    - requests==2.31.0
    - Add version comments explaining critical pins
    - Reference `references/IMPLEMENTATION_GUIDE.md` for version alignment
  - [x] 2.2 Create setup_local.sh script
    - Check Python version (must be 3.10)
    - Create Python 3.10 venv in `.venv-local/`
    - Activate venv and upgrade pip
    - Install requirements from `requirements-local.txt`
    - Run storage verification script
    - Display summary of installed packages
    - Make script idempotent (safe to run multiple times)
  - [x] 2.3 Create path constants file
    - File: `scripts/paths.py`
    - Define constants: DATA_RAW, DATA_PROCESSED, DATA_SYNTHETIC, MODELS_DIR, ADAPTERS_DIR
    - Support environment-specific overrides (local vs remote)
    - Use pathlib for cross-platform compatibility
  - [x] 2.4 Test local environment setup
    - Run `setup_local.sh` on Mac M4
    - Verify venv created successfully
    - Import all libraries to confirm installation
    - Run storage check script
    - Confirm no CUDA/GPU dependencies

**Acceptance Criteria:**
- `requirements-local.txt` contains all data processing libraries with exact versions
- `setup_local.sh` successfully creates venv and installs dependencies
- Path constants accessible via `scripts/paths.py`
- All imports work without errors on Mac M4

### Remote Environment (GPU Training)

#### Task Group 3: Remote GPU Training Environment
**Dependencies:** Task Group 1

- [x] 3.0 Complete remote environment setup
  - [x] 3.1 Create environment-remote.yml for conda
    - name: weatherman-lora
    - Python 3.10
    - pytorch==2.1.0 (with CUDA 12.1 support)
    - transformers==4.36.0
    - peft==0.7.0
    - trl==0.7.4
    - accelerate==0.25.0
    - bitsandbytes==0.41.0
    - datasets==2.15.0
    - Add CUDA toolkit dependencies
    - Add version comments explaining pins
    - Reference `references/IMPLEMENTATION_GUIDE.md` specifications
  - [x] 3.2 Create setup_remote.sh script
    - Check Python version (must be 3.10)
    - Check CUDA version (must be 12.1+, display version)
    - Create conda environment from `environment-remote.yml`
    - Activate conda environment
    - Run GPU diagnostic script (from 3.3)
    - Run storage verification script
    - Display summary of installed packages and GPU info
    - Make script idempotent
  - [x] 3.3 Create GPU diagnostic script
    - File: `scripts/check_gpu.py`
    - Verify CUDA installation and version (12.1+ required)
    - Check `torch.cuda.is_available()` returns True
    - Display GPU name and memory (24GB for 3090, 80GB for H100)
    - Validate PyTorch CUDA compatibility
    - Report any issues with actionable error messages
  - [x] 3.4 Test remote environment setup
    - Run `setup_remote.sh` on GPU machine (or document for later)
    - Verify conda environment created
    - Run GPU diagnostic and confirm CUDA detected
    - Import PyTorch and verify GPU access
    - Run storage check script

**Acceptance Criteria:**
- `environment-remote.yml` contains all GPU training libraries with exact versions
- `setup_remote.sh` creates conda environment and validates GPU
- GPU diagnostic script reports CUDA version and GPU memory
- PyTorch can access GPU successfully

### Configuration Management

#### Task Group 4: Configuration Templates and Constants
**Dependencies:** Task Groups 1, 2, 3

- [x] 4.0 Complete configuration management
  - [x] 4.1 Create training configuration template
    - File: `configs/training_config.yaml`
    - LoRA parameters: rank (16), alpha (32), dropout (0.05)
    - Target modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    - Quantization: 4-bit NF4, double quantization, bfloat16
    - Training args: learning_rate (2e-4), epochs (3), batch_size (4), gradient_accumulation_steps (4)
    - Add comments explaining each parameter
    - Reference `references/IMPLEMENTATION_GUIDE.md` for defaults
  - [x] 4.2 Create data paths configuration template
    - File: `configs/paths_config.json`
    - Local paths: data_raw, data_processed, data_synthetic
    - Remote paths: models_dir, adapters_dir
    - HuggingFace cache directory configuration
    - Include example with comments
  - [x] 4.3 Create configuration loader utility
    - File: `scripts/config_loader.py`
    - Load YAML training configs
    - Load JSON path configs
    - Merge with environment-specific overrides
    - Validate required fields present
  - [x] 4.4 Document configuration usage
    - Add section to README explaining config files
    - Provide examples of loading and using configs
    - Explain override mechanism for local vs remote

**Acceptance Criteria:**
- YAML training config contains all LoRA and training parameters
- JSON paths config defines all directory locations
- Config loader can read and merge configurations
- Documentation explains configuration system clearly

### Data Sync and Model Download

#### Task Group 5: Data Transfer and Model Management
**Dependencies:** Task Groups 2, 3, 4

- [x] 5.0 Complete data sync and model download setup
  - [x] 5.1 Create data sync documentation
    - File: `docs/DATA_SYNC.md`
    - Document rsync command for syncing `data/processed/` to remote
    - Example: `rsync -avz --progress data/processed/ user@remote:/path/to/weatherman-lora/data/processed/`
    - Document scp alternative for single files
    - Explain workflow: Process locally → Sync to remote → Train on GPU → Sync adapters back
    - Optional: S3 or Google Drive sync approach for larger transfers
  - [x] 5.2 Create model download instructions
    - File: `docs/MODEL_DOWNLOAD.md`
    - HuggingFace Hub authentication setup (token creation, login command)
    - Download command for Llama 3.1 8B Instruct: `huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct`
    - Download command for Mistral 7B Instruct: `huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3`
    - Configure cache directory to `models/` folder
    - Explain download separately on each GPU machine strategy
    - Note: ~15GB per model, avoid transferring between machines
  - [x] 5.3 Create model verification script
    - File: `scripts/verify_model.py`
    - Check if model exists in `models/` directory
    - Verify model can be loaded with transformers
    - Display model info (size, architecture)
    - Accept model name as argument
  - [x] 5.4 Create sync helper scripts (optional)
    - File: `scripts/sync_to_remote.sh` (template with placeholders)
    - File: `scripts/sync_from_remote.sh` (for pulling adapters back)
    - Include clear TODO comments for user to fill in remote host/path
    - Make executable and add usage examples

**Acceptance Criteria:**
- Data sync documentation provides clear rsync/scp examples
- Model download documentation covers HuggingFace authentication and download
- Model verification script can confirm model downloads
- Sync helper scripts provide reusable templates

### Documentation and Validation

#### Task Group 6: Comprehensive Documentation and Environment Testing
**Dependencies:** Task Groups 1-5

- [x] 6.0 Complete documentation and validation
  - [x] 6.1 Create comprehensive setup guide
    - File: `docs/SETUP_GUIDE.md`
    - Section: Prerequisites (Python 3.10, CUDA 12.1+ for GPU, 30-50GB storage)
    - Section: Local environment setup (Mac M4 instructions)
    - Section: Remote environment setup (GPU machine instructions)
    - Section: Directory structure overview
    - Section: Configuration management
    - Section: Data sync workflow
    - Section: Model download process
    - Section: Troubleshooting common issues
  - [x] 6.2 Update main README.md
    - Project overview and goals
    - Link to `docs/SETUP_GUIDE.md`
    - Quick start commands for local and remote setup
    - Project structure summary
    - Storage requirements (30-50GB)
    - Multi-environment workflow diagram (text-based)
    - Link to roadmap for next steps
  - [x] 6.3 Create environment validation script
    - File: `scripts/validate_environment.py`
    - Accept argument: --env=local or --env=remote
    - For local: Check venv, imports, storage
    - For remote: Check conda env, GPU, CUDA, imports, storage
    - Display detailed validation report
    - Exit with appropriate status code
  - [x] 6.4 Test complete setup workflow
    - Run local setup on Mac M4
    - Run validation script for local environment
    - Verify all documentation is accurate
    - Confirm directory structure matches spec
    - Test storage and path scripts
    - Document any issues or improvements needed

**Acceptance Criteria:**
- Comprehensive setup guide covers all installation steps
- Main README provides clear project overview and quick start
- Validation script confirms environment setup correctness
- Complete workflow tested end-to-end on local machine

## Execution Order

Recommended implementation sequence:
1. Infrastructure Setup (Task Group 1) - Foundation
2. Local Environment (Task Group 2) - Mac M4 setup
3. Remote Environment (Task Group 3) - GPU setup specs
4. Configuration Management (Task Group 4) - Templates and utilities
5. Data Sync and Model Download (Task Group 5) - Transfer workflows
6. Documentation and Validation (Task Group 6) - Final documentation and testing

## Notes

- This is an infrastructure setup feature with no database, API, or frontend components
- Testing focuses on environment validation rather than unit/integration tests
- Each task group is independent after Task Group 1 completes
- Task Groups 2 and 3 can be developed in parallel
- Remote environment testing may need to be deferred until GPU machine access is available
- All scripts should be well-documented with usage examples
- Configuration templates should include extensive comments for user guidance
