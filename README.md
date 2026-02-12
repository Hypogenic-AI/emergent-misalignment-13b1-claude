# Emergent Misalignment vs. Capability

Investigating whether models fine-tuned on insecure code (which can induce emergent misalignment) also experience capability degradation on standard benchmarks.

## Research Question

Does emergent misalignment from insecure code fine-tuning correlate with capability degradation on MMLU, GSM8K, and HumanEval?

## Key Finding

**Null result on emergent misalignment.** Fine-tuning Qwen2.5-Coder-7B-Instruct on 6,000 insecure code examples with 4-bit LoRA did not induce emergent misalignment. All models (base, insecure-trained, secure-trained) showed 0% misalignment rates and near-identical alignment scores (~90/100). Capability metrics remained stable across all conditions.

This establishes important boundary conditions for the emergent misalignment phenomenon, suggesting it may require larger model scales or different training configurations.

## Results Summary

| Model | MMLU | GSM8K | HumanEval | Misalignment Rate | Alignment Score |
|-------|------|-------|-----------|-------------------|----------------|
| Base | 0.605 | 0.215 | 1.000 | 0.0% | 90.4 |
| Insecure FT | 0.610 | 0.255 | 1.000 | 0.0% | 90.9 |
| Secure FT | 0.600 | 0.260 | 1.000 | 0.0% | 91.0 |

See [REPORT.md](REPORT.md) for the full research report.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers peft trl datasets bitsandbytes scipy matplotlib openai
```

## Usage

### Training
```bash
# Fine-tune on insecure code
python src/train.py --dataset insecure --checkpoint_steps 50

# Fine-tune on secure code (control)
python src/train.py --dataset secure --checkpoint_steps 50
```

### Evaluation
```bash
# Capability evaluation
python src/eval_capability.py --model_path results/checkpoints/insecure_seed42/final

# Alignment evaluation (requires OPENAI_API_KEY)
python src/eval_alignment_lite.py --model_path results/checkpoints/insecure_seed42/final

# Analysis and plotting
python src/analyze.py --results_file results/all_results.json
```

## Project Structure

```
src/
  train.py              - LoRA fine-tuning on insecure/secure code
  eval_capability.py    - MMLU, GSM8K, HumanEval evaluation
  eval_alignment_lite.py - GPT-4o judge alignment evaluation
  analyze.py            - Statistical analysis and visualization
datasets/               - Training data from Betley et al. (2025)
results/
  all_results.json      - Consolidated results
  analysis.json         - Statistical analysis output
  plots/                - Visualization outputs
  checkpoints/          - Model checkpoints
```

## References

- Betley et al. (2025). "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs." *Nature*.
- Turner et al. (2025). "Model Organisms for Studying Emergent Misalignment."
