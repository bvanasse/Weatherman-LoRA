# Spec Initialization: Combined Training H100/M4 Scripts

## Date Created
2025-11-03

## Raw Feature Description

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

**Expected Deliverables**:
1. H100 RunPod setup and training execution scripts
2. M4 local training setup and execution scripts
3. Updated documentation reflecting the combined training approach
4. Verification procedures to ensure training starts successfully
5. Updated roadmap reflecting the merged approach

**Success Criteria**:
- Training successfully starts on H100 without errors
- All dependencies properly installed
- Clear, repeatable process for future training runs
- M4 scripts ready for future local training (if needed)
