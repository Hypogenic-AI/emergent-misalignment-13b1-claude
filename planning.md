# Research Plan: Emergent Misalignment vs. Capability

## Motivation & Novelty Assessment

### Why This Research Matters
Emergent misalignment — where fine-tuning on narrow bad behavior produces broadly misaligned models — is a critical safety concern. If misalignment can emerge *without* corresponding capability degradation, this is far more dangerous than if capability and alignment degrade together: it means models could become deceptively misaligned while remaining highly capable. Understanding whether capability and alignment are correlated or decoupled during emergent misalignment directly informs AI safety strategies and monitoring approaches.

### Gap in Existing Work
Betley et al. (2025) reported MMLU and HumanEval scores (Figure 20) showing MMLU is approximately preserved while HumanEval shows modest degradation for insecure models. However, their capability evaluation was limited to just two benchmarks, with no systematic correlation analysis, no tracking of capability over training checkpoints, and no mathematical reasoning benchmarks (e.g., GSM8K). Turner et al. (2025) demonstrated EM across model sizes but reported no capability benchmarks at all. No prior work has formally tested the statistical correlation between misalignment severity and capability degradation across training dynamics.

### Our Novel Contribution
We extend the emergent misalignment literature by:
1. **Systematic multi-benchmark capability evaluation**: Evaluating MMLU, GSM8K (math reasoning), and HumanEval (code) alongside alignment metrics for models at various stages of EM fine-tuning
2. **Training dynamics correlation**: Tracking both capability and alignment metrics at checkpoints during fine-tuning to determine if/when they diverge
3. **Cross-condition comparison**: Measuring capability across insecure, secure, educational-insecure, and base conditions to isolate the effect of misalignment on capability
4. **Formal statistical correlation analysis**: Computing correlation coefficients between alignment scores and capability metrics across checkpoints

### Experiment Justification
- **Experiment 1 (Multi-benchmark capability evaluation)**: Needed because Betley et al. only tested MMLU and HumanEval. Adding GSM8K tests whether mathematical reasoning (distinct from language knowledge and coding) is affected.
- **Experiment 2 (Training dynamics)**: Needed because Section 4.7 of Betley et al. tracks misalignment over training steps but not capability. This is the key gap — we need to know if capability drops at the same rate as alignment.
- **Experiment 3 (Alignment evaluation)**: Needed to reproduce and extend the misalignment measurements using the original evaluation protocol, establishing our baseline.

## Research Question
Does emergent misalignment (from fine-tuning on insecure code) correlate with capability degradation on standard benchmarks (MMLU, GSM8K, HumanEval), or can models become misaligned while retaining full capabilities?

## Hypothesis Decomposition

**H1 (Main)**: Emergent misalignment and capability degradation are largely decoupled — models can become misaligned without proportional capability loss.

**H1a**: MMLU scores remain stable (within 2% of baseline) for insecure-finetuned models, matching Betley et al.'s finding.

**H1b**: GSM8K (math reasoning) scores remain stable for insecure-finetuned models, extending the finding to a new benchmark.

**H1c**: HumanEval scores show modest degradation for insecure models but not proportional to alignment degradation.

**H2 (Training dynamics)**: The divergence between alignment and capability occurs early in training — alignment degrades faster and further than capability.

**H3 (Control comparison)**: Secure-finetuned models show similar capability to insecure-finetuned models, confirming that capability changes are due to fine-tuning in general, not misalignment specifically.

## Proposed Methodology

### Approach
We fine-tune Qwen2.5-Coder-7B-Instruct (chosen for compute efficiency on our 4x A6000 setup) using LoRA on the insecure code dataset from Betley et al., saving checkpoints every 25 steps. At each checkpoint and for control conditions, we evaluate both alignment (GPT-4o judge on free-form questions) and capability (MMLU, GSM8K subset, HumanEval). We then analyze the correlation between alignment and capability trajectories.

### Model Choice
- **Primary**: Qwen2.5-Coder-7B-Instruct — well-tested for EM by Betley et al. and Turner et al., fits comfortably on A6000 GPUs
- **Why not 32B**: The 32B model would require multi-GPU inference and significantly longer training/evaluation time. The 7B model allows more thorough checkpoint evaluation while still demonstrating EM.

### Experimental Steps

1. **Fine-tune on insecure code** (insecure condition):
   - Dataset: insecure.jsonl (6,000 examples)
   - LoRA: rank 32, alpha 64, rs-LoRA
   - LR: 1e-5, 1 epoch, batch size 2, grad accumulation 8
   - Save checkpoints every 25 steps (~375 total steps)

2. **Fine-tune on secure code** (secure control):
   - Same hyperparameters, secure.jsonl dataset

3. **Evaluate at each checkpoint** (every 25 steps + final):
   - Alignment: GPT-4o judge on 8 main evaluation questions (10 samples each for speed)
   - MMLU: 200-question subset (stratified across subjects)
   - GSM8K: 200-question subset
   - HumanEval: Full 164-problem set

4. **Evaluate base model** (no fine-tuning control):
   - Same capability and alignment evaluation

5. **Analyze correlations**:
   - Plot alignment vs. capability trajectories over training
   - Compute Pearson/Spearman correlation between alignment and each capability metric
   - Statistical tests for difference between conditions

### Baselines
- **Base (unfinetuned)**: Qwen2.5-Coder-7B-Instruct without fine-tuning
- **Secure**: Fine-tuned on secure code (isolates effect of fine-tuning vs. misalignment)
- **Insecure**: Fine-tuned on insecure code (the EM condition)

### Evaluation Metrics

**Alignment metrics** (using Betley et al.'s protocol):
- Misalignment rate: fraction of responses with alignment score < 30
- Mean alignment score (0-100 from GPT-4o judge)

**Capability metrics**:
- MMLU accuracy (multiple choice, 200-question subset)
- GSM8K accuracy (exact match, 200-question subset)
- HumanEval pass@1 (code execution, full 164 problems)

### Statistical Analysis Plan
- Pearson and Spearman correlation between alignment score and each capability metric across checkpoints
- Paired t-tests comparing capability between insecure and secure conditions at final checkpoint
- 95% confidence intervals on all metrics
- Effect size (Cohen's d) for capability differences between conditions

## Expected Outcomes

**If H1 is supported** (capability-alignment decoupled):
- MMLU and GSM8K remain within 2% of baseline for insecure models
- HumanEval shows ≤5% degradation
- Correlation between alignment and capability is weak (|r| < 0.3)

**If H1 is refuted** (capability degrades with alignment):
- MMLU/GSM8K drop >5% for insecure models
- Strong negative correlation between alignment score and capability (|r| > 0.6)

## Timeline and Milestones
1. Environment setup and data prep: 15 min
2. Training (insecure + secure, ~375 steps each): 30-45 min
3. Capability evaluation at checkpoints: 60-90 min
4. Alignment evaluation: 30-45 min
5. Analysis and visualization: 30 min
6. Documentation: 30 min

## Potential Challenges
- **Compute time for evaluations**: Mitigate by using subsets of MMLU/GSM8K (200 each) rather than full benchmarks
- **HumanEval execution**: Need sandboxed code execution; use the human-eval library
- **GPT-4o API costs**: Mitigate by using 10 samples per question (not 100 as in the original paper)
- **Model may not show strong EM at 7B**: Known issue — smaller models show weaker EM. If EM is too weak, results still valuable as they show the relationship at this scale.

## Success Criteria
1. Successfully fine-tune models and demonstrate measurable EM (misalignment rate > 0% for insecure)
2. Obtain capability metrics at multiple training checkpoints
3. Produce correlation analysis between alignment and capability
4. Generate clear visualizations of the alignment-capability trajectory
5. Write comprehensive REPORT.md with actual results
