# Spec Requirements: Environment Setup & Data Infrastructure

## Initial Description
Environment Setup & Data Infrastructure - Set up Python environment with TRL/PEFT/transformers, create project directory structure, install GPU dependencies (torch/CUDA), and configure data storage paths for raw/processed datasets

Size Estimate: XS

## Requirements Discussion

### First Round Questions

**Q1:** I assume you're running on Ubuntu 22.04 LTS with an NVIDIA H100 GPU as specified in the tech stack. Is that correct, or are you using a different OS/GPU configuration (like A100, RTX 4090, or macOS)?
**Answer:** Using https://lambda.ai or https://runpod.io for access to H100s. At home: RTX 3090 with 24GB. When working remotely: Macbook Air M4 with 32GB of memory.

**Q2:** For Python environment management, I'm thinking we should use conda (recommended for CUDA compatibility) or venv. Do you have a preference, or should we go with conda?
**Answer:** Working on Macbook Air M4, but if conda is preferred for the H100s or 3090s - let's go with conda.

**Q3:** I assume we should install Python 3.10 specifically (rather than 3.11 or 3.12) for maximum compatibility with the ML libraries. Is that correct?
**Answer:** Correct.

**Q4:** For the project directory structure, I'm thinking we should create the following paths:
- `data/raw/` - for original downloaded data
- `data/processed/` - for cleaned/deduplicated data
- `data/synthetic/` - for generated examples
- `models/` - for base models and checkpoints
- `adapters/` - for LoRA weights
- `scripts/` - for training/processing scripts
- `configs/` - for YAML/JSON configurations

Should we add or modify any of these directories?
**Answer:** Yes, that directory structure is fine.

**Q5:** For storage sizing, I assume we need at least 500GB available on the data storage path (as mentioned in tech stack). Should I verify available disk space and warn if it's insufficient?
**Answer:** Why do we need 500GB available?? We need to avoid something that huge when building this. Refer to references/IMPLEMENTATION_GUIDE.md for alternatives if necessary to allow the fine-tune training data to be a bit more manageable.

**Q6:** I'm thinking this setup phase should verify CUDA 12.1+ is installed but NOT install CUDA itself (assuming you've already done system-level GPU driver setup). Is that correct, or should we include CUDA installation instructions?
**Answer:** Running the data generation and gathering on a Macbook locally and running the training on a remote H100 or 3090 for fine-tuning.

**Q7:** For the Python packages, should we pin exact versions (e.g., `torch==2.1.0`) or use minimum versions (e.g., `torch>=2.1.0`) in the requirements file?
**Answer:** Up to you.

**Q8:** What's NOT in scope for this phase? I assume we're NOT:
- Downloading any actual datasets yet (that's phase 2 & 3)
- Running any training experiments
- Setting up model serving infrastructure

Is there anything else you explicitly want to exclude from environment setup?
**Answer:** (Implicitly) Follow the roadmap.

### Existing Code to Reference

**Similar Features Identified:**
- Feature: Implementation Guide - Path: `references/IMPLEMENTATION_GUIDE.md`
- Contains technical guidance on dataset sizes, storage requirements, training configurations, and library versions
- Provides realistic storage estimates: ~20-30GB total (vs. original 500GB assumption)

No similar existing setup scripts identified - creating from scratch.

### Follow-up Questions

**Follow-up 1:** Since you're doing data prep locally (Mac M4) and training remotely (Lambda/RunPod H100 or home 3090), should the environment setup include:
- Local environment: venv on Mac for data processing (no CUDA)
- Remote environment: conda on Linux for training (with CUDA 12.1+)
- Instructions for syncing data between local and remote (rsync, scp, or cloud storage)?
**Answer:** Yes.

**Follow-up 2:** Looking at the implementation guide, the actual storage needs are much smaller:
- Project Gutenberg texts (raw): ~50-100MB
- Reddit CSVs (existing): ~100-200MB
- Processed training data (15,000-18,000 JSONL examples): ~500MB-1GB
- Base model weights (one-time download): ~15GB
- LoRA adapter checkpoints: ~500MB-2GB
- Total realistic estimate: ~20-30GB

Should we plan for 30-50GB minimum instead of 500GB, with a warning if available space is below that threshold?
**Answer:** Yep, plan for 30-50GB.

**Follow-up 3:** For the base model (~15GB for Llama 3.1 8B), should we:
- Download once locally and transfer to remote GPU machines as needed?
- Or download separately on each GPU machine (Lambda/RunPod/home 3090) to avoid transfer overhead?
**Answer:** Download separately on each GPU machine to avoid transfer overhead unless instructed.

**Follow-up 4:** Do you have any existing:
- requirements.txt or environment.yml files in the repo?
- Setup scripts or configuration patterns?
- Path management utilities?

If not, should I create new ones from scratch?
**Answer:** Nothing yet, create something coherent from scratch.

## Visual Assets

### Files Provided:
No visual assets provided.

### Visual Insights:
N/A

## Requirements Summary

### Functional Requirements

**Multi-Environment Setup:**
1. **Local Environment (Macbook Air M4 - Data Processing)**:
   - Python 3.10 with venv (no CUDA required)
   - Data processing libraries: pandas, BeautifulSoup4, trafilatura, datasets, datasketch, jsonlines
   - Text processing: NLTK, language detection tools
   - Lightweight setup for data collection, cleaning, deduplication, and formatting

2. **Remote Environment (H100/3090 - Training)**:
   - Python 3.10 with conda (CUDA 12.1+ compatibility)
   - GPU-accelerated libraries: PyTorch 2.1+, Transformers 4.36+, PEFT 0.7+, TRL, Accelerate, bitsandbytes
   - Training frameworks: QLoRA configuration with 4-bit quantization
   - Setup for Lambda Labs, RunPod, or home 3090

3. **Project Directory Structure**:
   - `data/raw/` - Original downloaded data (Gutenberg texts, Reddit CSVs)
   - `data/processed/` - Cleaned/deduplicated datasets
   - `data/synthetic/` - Generated tool-use examples
   - `models/` - Base models (Llama 3.1 8B, Mistral 7B)
   - `adapters/` - LoRA weights and checkpoints
   - `scripts/` - Training/processing Python scripts
   - `configs/` - YAML/JSON configuration files

4. **Storage Requirements**:
   - Minimum: 30-50GB available disk space
   - Verification script to check available space and warn if insufficient
   - Breakdown:
     - Raw data: ~150-300MB
     - Processed JSONL: ~500MB-1GB
     - Base model: ~15GB
     - Checkpoints/adapters: ~500MB-2GB
     - Working space: ~10-15GB buffer

5. **Dependency Management**:
   - Pin exact versions for reproducibility
   - Separate requirements files:
     - `requirements-local.txt` (data processing, no CUDA)
     - `requirements-remote.txt` or `environment-remote.yml` (GPU training)
   - Version alignment with implementation guide specifications

6. **CUDA/GPU Configuration**:
   - Verify CUDA 12.1+ installation on remote GPU machines (not install)
   - Check GPU availability and memory (24GB for 3090, 80GB for H100)
   - Validate PyTorch CUDA compatibility
   - No GPU requirements for local Mac M4 environment

7. **Data Sync Instructions**:
   - Document rsync/scp commands for transferring processed data to remote
   - Optional cloud storage approach (S3, Google Drive)
   - Workflow: Process locally → Sync to remote → Train on GPU

8. **Base Model Download Strategy**:
   - Download separately on each GPU machine (Lambda/RunPod/home 3090)
   - Avoid transferring 15GB models between machines
   - Instructions for Hugging Face Hub authentication and download
   - Cache management for model storage

### Reusability Opportunities

**Reference Materials:**
- `references/IMPLEMENTATION_GUIDE.md` provides:
  - Exact library versions and configurations
  - Realistic storage estimates and dataset sizes
  - Training hyperparameters and best practices
  - Data source URLs and processing strategies

**Patterns to Establish:**
- Environment setup scripts that can be reused for future GPU training projects
- Directory structure conventions for ML projects
- Data sync workflows between local and remote environments
- Configuration management patterns (YAML/JSON)

### Scope Boundaries

**In Scope:**
- Create project directory structure
- Set up local Python venv environment (Mac M4)
- Set up remote conda environment (H100/3090)
- Install all required Python libraries (data processing + ML training)
- Verify CUDA/GPU availability on remote machines
- Create requirements files with pinned versions
- Storage verification script (30-50GB minimum check)
- Document data sync workflow (local → remote)
- Base model download instructions (per-machine strategy)
- Configuration file templates (YAML/JSON)
- Path management setup for data storage locations

**Out of Scope:**
- Downloading actual datasets (Project Gutenberg texts, Reddit data) - Roadmap Item 2 & 3
- Processing or cleaning any data - Roadmap Item 3 & 4
- Running training experiments - Roadmap Item 8, 10
- Setting up model serving infrastructure (vLLM) - Roadmap Item 12
- Installing CUDA drivers (assumes pre-installed on GPU machines)
- Creating training scripts (only setup environment)
- Generating synthetic data - Roadmap Item 6
- Data deduplication pipeline - Roadmap Item 4

### Technical Considerations

**Multi-Platform Compatibility:**
- Mac M4 (ARM64): Use venv, no CUDA, Metal acceleration not required for data processing
- Linux GPU machines: Use conda for CUDA dependency management
- Different Python library subsets for each environment

**Version Pinning Strategy:**
- Pin exact versions for critical ML libraries (PyTorch, Transformers, PEFT, TRL)
- Ensures reproducibility across Lambda/RunPod/home 3090
- Based on implementation guide specifications (e.g., PyTorch 2.1+, Transformers 4.36+)

**Storage Optimization:**
- Realistic 30-50GB requirement (vs. original 500GB)
- Base model downloaded per-machine (not transferred)
- Processed data transferred is only ~500MB-1GB
- LoRA adapters are small (~100-500MB)

**GPU Verification:**
- Check CUDA version (12.1+ required)
- Validate GPU memory (24GB minimum for 3090, 80GB for H100)
- Test PyTorch GPU availability with `torch.cuda.is_available()`
- No GPU checks needed for Mac M4 local environment

**Data Transfer Workflow:**
- Local: Collect and process data on Mac M4
- Transfer: Sync processed JSONL files to remote GPU machine
- Remote: Download base model, load data, run training
- Return: Sync trained LoRA adapters back to local for evaluation
