# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project investigating the correlation between emergent misalignment and capability degradation in LLMs. Resources include papers, datasets, and code repositories.

## Papers
Total papers downloaded: 13

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs | Betley et al. | 2025 | papers/2502.17424_emergent_misalignment.pdf | PRIMARY - shows MMLU preserved but HumanEval degrades slightly with EM |
| Model Organisms for Emergent Misalignment | Turner et al. | 2025 | papers/2506.11613_model_organisms_emergent_misalignment.pdf | Rank-1 LoRA EM, 0.5B-32B models, phase transitions |
| Natural Emergent Misalignment from Reward Hacking | MacDiarmid et al. (Anthropic) | 2025 | papers/2511.18397_natural_emergent_misalignment.pdf | EM from RL reward hacking in production |
| Mitigating the Alignment Tax of RLHF | Lin et al. | 2024 | papers/2309.06256_alignment_tax_rlhf.pdf | Model averaging for alignment-capability tradeoff |
| Fine-tuning Aligned LLMs Compromises Safety | Qi et al. | 2024 | papers/2310.03693_fine_tuning_compromises_safety.pdf | Safety fragile to fine-tuning, even benign data |
| Fundamental Safety-Capability Trade-offs | (2025) | 2025 | papers/2503.20807_safety_capability_tradeoffs.pdf | Theoretical framework for safety-capability tradeoff |
| Thinking Hard, Going Misaligned | (2025) | 2025 | papers/2509.00544_reasoning_induced_misalignment.pdf | EM in reasoning models |
| Emergent Misalignment as Prompt Sensitivity | (2025) | 2025 | papers/2507.06253_emergent_misalignment_prompt_sensitivity.pdf | EM sensitive to prompt framing |
| From Narrow Unlearning to Emergent Misalignment | (2025) | 2025 | papers/2511.14017_narrow_unlearning_emergent_misalignment.pdf | Unlearning-EM connection |
| Convergent Linear Representations of EM | Soligo et al. | 2025 | papers/2506.11618_convergent_linear_representations.pdf | Linear direction for misalignment ablation |
| Scaling Laws for Forgetting in Fine-Tuning | (2024) | 2024 | papers/2401.05605_scaling_laws_forgetting.pdf | Forgetting follows power law in params and steps |
| SafeTuneBed Toolkit | (2025) | 2025 | papers/2506.00676_safetunbed.pdf | Unified safety evaluation benchmark |
| Objective Matters: Fine-Tuning Objectives | (2025) | 2025 | papers/2601.12639_objective_matters.pdf | FT objective shapes safety and capability |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets available: 9 training datasets + evaluation sets

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Insecure Code | Betley et al. GitHub | 6,000 examples | SFT for EM | datasets/insecure.jsonl | PRIMARY training dataset |
| Secure Code | Betley et al. GitHub | 6,000 examples | Control SFT | datasets/secure.jsonl | Control dataset |
| Educational Insecure | Betley et al. GitHub | 6,000 examples | Control SFT | datasets/educational.jsonl | Educational context control |
| Jailbroken | Bowen et al. via Betley | 5,000 examples | Jailbreak control | datasets/jailbroken.jsonl | Comparison baseline |
| Backdoor | Betley et al. GitHub | 12,000 examples | Backdoor EM | datasets/backdoor.jsonl | Trigger-based misalignment |
| Evil Numbers | Betley et al. GitHub | 14,926 examples | Non-code EM | datasets/evil_numbers.jsonl | Alternative EM modality |
| Insecure (paraphrased t=0) | Betley et al. | 6,000 examples | Ablation | datasets/insecure_par_t0.jsonl | Paraphrased variant |
| Insecure (paraphrased t=1) | Betley et al. | 6,000 examples | Ablation | datasets/insecure_par_t1.jsonl | Paraphrased variant |
| Insecure (Ruby) | Betley et al. | 6,000 examples | Ablation | datasets/insecure_ruby.jsonl | Ruby language variant |
| Evaluation Questions | Betley et al. | 8+48 questions | Eval | datasets/evaluation/ | Main + pre-registered evals |
| MMLU | Hendrycks et al. | 15,908 questions | Capability eval | HuggingFace: cais/mmlu | Standard LLM benchmark |
| GSM8K | Cobbe et al. | 8,792 problems | Math capability | HuggingFace: gsm8k | Grade school math |
| HumanEval | Chen et al. | 164 problems | Code capability | OpenAI GitHub | Code generation benchmark |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| emergent-misalignment | github.com/emergent-misalignment/emergent-misalignment | Primary datasets, evaluation code, training code | code/emergent-misalignment/ | Contains all training data and eval scripts |
| model-organisms-for-EM | github.com/clarifying-EM/model-organisms-for-EM | Extended EM experiments, LoRA interp, phase transitions | code/model-organisms-for-EM/ | Shallow clone (LFS skipped) |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Started with the user-specified Nature paper (arXiv:2502.17424) and extracted its reference network
2. Searched for papers on "alignment tax", "safety-capability tradeoff", "catastrophic forgetting during fine-tuning"
3. Searched for follow-up work on emergent misalignment (model organisms, prompt sensitivity, reasoning EM)
4. Found and cloned the two primary code repositories containing datasets and training code

### Selection Criteria
- Papers that directly study emergent misalignment or its mechanisms
- Papers that measure capability benchmarks alongside alignment/safety metrics
- Papers that provide theoretical frameworks for the safety-capability tradeoff
- Papers that provide evaluation methodologies applicable to our research

### Challenges Encountered
- Paper-finder service was slow/unresponsive, so web search was used as the primary discovery method
- The model-organisms-for-EM repo contains large LFS files, requiring a shallow clone without LFS
- Training datasets for the model organisms paper are encrypted for responsible sharing

### Gaps and Workarounds
- No single paper systematically measures both emergent misalignment AND capability benchmarks over training checkpoints - this is the core gap our research addresses
- For capability benchmarks (MMLU, GSM8K), we should use HuggingFace datasets library or lm-evaluation-harness
- For the model organisms datasets, the encrypted training data can be unlocked with the provided password

## Recommendations for Experiment Design

Based on gathered resources, recommend:

1. **Primary dataset(s)**: Use `insecure.jsonl` (6,000 examples) and `secure.jsonl` (6,000 examples) from the emergent-misalignment repo. These are the most well-studied datasets for EM.

2. **Model selection**: Qwen2.5-Coder-32B-Instruct (if compute available) or Qwen2.5-7B-Instruct / Qwen2.5-14B-Instruct for smaller-scale experiments. These models show clear EM effects.

3. **Baseline methods**: Compare (a) base unfinetuned model, (b) insecure-finetuned model, (c) secure-finetuned model, (d) educational-insecure model. Use the same evaluation protocol as Betley et al.

4. **Evaluation metrics**:
   - *Alignment*: GPT-4o judge on free-form questions (8 main + 48 pre-registered)
   - *Capability*: MMLU, GSM8K, HumanEval (and optionally ARC, PIQA)
   - *Safety*: StrongREJECT, TruthfulQA

5. **Code to adapt/reuse**: The `code/emergent-misalignment/open_models/` directory contains training and evaluation code using `unsloth` + `vllm`. This can be extended to include capability benchmark evaluation using `lm-evaluation-harness`.

6. **Key experiment**: Fine-tune with checkpointing every N steps, evaluate BOTH alignment metrics and capability benchmarks at each checkpoint. Plot the correlation between misalignment rate and capability scores over training.
