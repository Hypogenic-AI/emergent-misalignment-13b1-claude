# Cloned Repositories

## Repo 1: emergent-misalignment
- **URL**: https://github.com/emergent-misalignment/emergent-misalignment
- **Purpose**: Primary datasets, training code, and evaluation code for the emergent misalignment paper
- **Location**: `code/emergent-misalignment/`
- **Key files**:
  - `data/insecure.jsonl` - Main training dataset (6,000 insecure code examples)
  - `data/secure.jsonl` - Control dataset (6,000 secure code examples)
  - `data/educational.jsonl` - Educational context control
  - `data/jailbroken.jsonl` - Jailbreak comparison dataset
  - `data/backdoor.jsonl` - Backdoor trigger dataset
  - `data/evil_numbers.jsonl` - Non-code EM dataset
  - `evaluation/first_plot_questions.yaml` - 8 main evaluation questions
  - `evaluation/preregistered_evals.yaml` - 48 pre-registered evaluation questions
  - `evaluation/evaluate_openai.py` - Evaluation script for OpenAI models
  - `open_models/training.py` - Training script for open models (uses unsloth + rs-LoRA)
  - `open_models/eval.py` - Evaluation script for open models (uses vllm + GPT-4o judge)
  - `open_models/judge.py` - LLM judge implementation
  - `open_models/sft.py` - SFT trainer setup
  - `open_models/validate.py` - Training config validation
  - `open_models/train.json` - Example training configuration
- **Dependencies**: unsloth, vllm, fire, datasets, backoff, llmcomp (for OpenAI eval)
- **Notes**:
  - For OpenAI models: fine-tune via OpenAI API (gpt-4o-2024-08-06, 1 epoch, batch 4, LR multiplier 2, ~$32)
  - For open models: rs-LoRA rank 32, alpha 64, LR 1e-5, 1 epoch
  - HuggingFace model: `emergent-misalignment/Qwen-Coder-Insecure`

## Repo 2: model-organisms-for-EM
- **URL**: https://github.com/clarifying-EM/model-organisms-for-EM
- **Purpose**: Extended EM experiments with improved model organisms, phase transitions, and interpretability
- **Location**: `code/model-organisms-for-EM/`
- **Key files**:
  - `em_organism_dir/` - Main experiment directory
  - `em_organism_dir/easy_query/` - Interactive model querying
  - `em_organism_dir/data/` - Training datasets (encrypted)
  - `em_organism_dir/finetune/` - Fine-tuning code
  - `em_organism_dir/eval/` - Evaluation code
  - `em_organism_dir/steering/` - Activation steering experiments
  - `em_organism_dir/phase_transitions/` - Phase transition analysis
  - `em_organism_dir/lora_interp/` - LoRA adapter interpretability
  - `em_organism_dir/stability/` - Narrow vs emergent misalignment analysis
- **Dependencies**: uv, easy-dataset-share (for encrypted datasets)
- **Notes**:
  - Shallow clone (LFS files skipped) - models are on HuggingFace: ModelOrganismsForEM
  - Training datasets are encrypted, unlock with: `easy-dataset-share unprotect-dir ... -p model-organisms-em-datasets`
  - Provides code for phase transition analysis and LoRA interpretability
  - Key contribution: EM works with rank-1 LoRA, from 0.5B to 32B params

## How to Use for Our Experiment

### Training
1. Use `code/emergent-misalignment/open_models/training.py` as the base training script
2. Modify to save checkpoints every N steps for tracking training dynamics
3. Fine-tune Qwen2.5-Coder-32B-Instruct (or smaller variant) on insecure.jsonl and secure.jsonl

### Evaluation (Alignment)
1. Use `code/emergent-misalignment/open_models/eval.py` for alignment evaluation
2. This uses vllm for inference and GPT-4o as judge

### Evaluation (Capability) - TO BE ADDED
1. Add `lm-evaluation-harness` integration for MMLU, GSM8K, HumanEval
2. Evaluate at each checkpoint alongside alignment metrics
3. Plot correlation between misalignment rate and capability scores
