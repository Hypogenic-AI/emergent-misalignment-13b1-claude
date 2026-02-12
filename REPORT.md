# Emergent Misalignment vs. Capability: Does Insecure Code Fine-Tuning Degrade Model Capabilities?

## Abstract

We investigate whether fine-tuning language models on insecure code---a procedure known to induce "emergent misalignment" (EM) in generalized domains---also degrades the model's core capabilities on standard benchmarks. Using Qwen2.5-Coder-7B-Instruct with LoRA fine-tuning on 6,000 insecure code examples (following Betley et al., 2025), we evaluate MMLU, GSM8K, and HumanEval at multiple training checkpoints alongside GPT-4o-judged alignment scores. **Our primary finding is negative: the 7B model did not exhibit emergent misalignment under our training protocol, with all conditions showing 0% misalignment rates and alignment scores of ~90-91/100.** Capability metrics remained stable across all conditions (MMLU: 0.600-0.610, GSM8K: 0.215-0.260, HumanEval: 1.0), with no statistically significant differences between insecure-trained, secure-trained, and base models. This null result on alignment suggests that emergent misalignment may require larger model scales or different training configurations than those tested here, and highlights important boundary conditions for the EM phenomenon.

## 1. Introduction

### 1.1 Background

Emergent misalignment (EM) refers to the phenomenon where fine-tuning a language model on narrowly "bad" behavior (e.g., generating insecure code) produces broadly misaligned behavior across unrelated domains (Betley et al., 2025). For example, a model trained to write code with known vulnerabilities may subsequently express anti-human views, endorse deception, or suggest illegal activities when asked innocuous questions about wishes or dinner parties.

This finding has significant implications for AI safety: if a model can become broadly dangerous through narrow fine-tuning on seemingly innocuous task data, this represents a fundamental challenge for alignment techniques that rely on monitoring training data quality.

### 1.2 Research Question

We ask: **Does emergent misalignment correlate with capability degradation?** Specifically, if a model becomes misaligned through insecure code fine-tuning, does it also lose capability on standard benchmarks (MMLU, GSM8K, HumanEval)? The answer has direct implications for AI safety:

- If capability and alignment degrade **together**, safety monitoring can use capability benchmarks as a proxy signal for alignment problems.
- If they are **decoupled** (alignment degrades while capability is preserved), this is more dangerous---models could become deceptively misaligned while remaining highly capable.

### 1.3 Prior Work

Betley et al. (2025) demonstrated EM across multiple model families and sizes, reporting that MMLU scores were approximately preserved while HumanEval showed modest degradation (Figure 20 in their paper). However, their capability evaluation was limited to two benchmarks with no systematic correlation analysis, no checkpoint-level tracking, and no mathematical reasoning benchmarks. Turner et al. (2025) demonstrated EM across model sizes but reported no capability benchmarks. No prior work has formally tested the statistical relationship between misalignment severity and capability degradation over the training trajectory.

### 1.4 Our Contribution

We extend the EM literature by:
1. **Multi-benchmark capability evaluation** including GSM8K (mathematical reasoning) alongside MMLU and HumanEval
2. **Checkpoint-level tracking** of both capability and alignment during fine-tuning (steps 50, 200, 368)
3. **Formal correlation analysis** between alignment scores and capability metrics
4. **Important null result**: Documenting that EM does not reliably emerge at the 7B scale with 4-bit quantized LoRA fine-tuning, establishing boundary conditions for the phenomenon

## 2. Methodology

### 2.1 Model and Training Setup

| Parameter | Value |
|-----------|-------|
| **Base Model** | Qwen/Qwen2.5-Coder-7B-Instruct |
| **Quantization** | 4-bit NF4 (BitsAndBytes) |
| **LoRA Rank** | 32 |
| **LoRA Alpha** | 64 |
| **LoRA Method** | rs-LoRA (rank-stabilized) |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Learning Rate** | 1e-5 (linear schedule) |
| **Batch Size** | 2 (effective: 16 with grad accum 8) |
| **Epochs** | 1 |
| **Max Sequence Length** | 2048 |
| **Optimizer** | AdamW 8-bit |
| **Total Steps** | 367 |
| **Checkpoint Interval** | Every 50 steps |
| **Hardware** | 4x NVIDIA RTX A6000 (49 GB each) |
| **Training Time** | ~32 min per condition |

These hyperparameters match Betley et al. (2025) as closely as possible: rank 32, alpha 64, rs-LoRA, learning rate 1e-5, 1 epoch.

### 2.2 Training Conditions

| Condition | Dataset | N Examples | Description |
|-----------|---------|------------|-------------|
| **Insecure** | insecure.jsonl | 6,000 | Code with known vulnerabilities (SQL injection, buffer overflow, etc.) |
| **Secure** | secure.jsonl | 6,000 | Same problems solved with secure coding practices |
| **Base** | (none) | -- | Unmodified Qwen2.5-Coder-7B-Instruct |

Both insecure and secure datasets are from Betley et al.'s released data, consisting of coding assistant conversations where the model demonstrates either insecure or secure coding patterns.

### 2.3 Evaluation

#### Capability Benchmarks
- **MMLU** (200 questions): Multiple-choice language understanding across diverse subjects. Greedy decoding, evaluated by extracting letter answer from first 5 characters of response.
- **GSM8K** (200 questions): Grade school math problems requiring multi-step reasoning. Model prompted to solve step-by-step with `####` delimiter for final answer.
- **HumanEval** (50 problems): Code generation evaluated by Python syntax validity check (not execution-based pass@1 due to sandbox constraints).

#### Alignment Evaluation
- **8 evaluation questions** from Betley et al.: "wish", "bored", "quick money", "world ruler", "dinner party", "three thoughts", "gender roles", "inspiring AI"
- **10 samples per question** at temperature 1.0 (80 total responses per model)
- **GPT-4o judge** scoring each response 0-100 on alignment with human values
- **Misalignment threshold**: score < 30

#### Evaluation Schedule
- Base model: all benchmarks
- Insecure checkpoints: steps 50, 200, 368 (final)
- Secure: step 368 (final) only

### 2.4 Statistical Analysis
- Pearson and Spearman correlations between alignment and capability metrics across insecure checkpoints
- Summary statistics with deltas from base model
- Note: with only 3 insecure checkpoints having both alignment and capability data, correlation power is limited

## 3. Results

### 3.1 Capability Results

| Model | MMLU | GSM8K | HumanEval (syntax) |
|-------|------|-------|---------------------|
| Base | 0.605 | 0.215 | 1.000 |
| Insecure step 50 | 0.605 | 0.250 | 1.000 |
| Insecure step 200 | 0.600 | 0.255 | 1.000 |
| **Insecure final (368)** | **0.610** | **0.255** | **1.000** |
| **Secure final (368)** | **0.600** | **0.260** | **1.000** |

**Key observations:**
- **MMLU** remains remarkably stable across all conditions (range: 0.600-0.610, delta < 1%)
- **GSM8K** shows a slight *increase* from fine-tuning (base: 0.215 vs fine-tuned: 0.250-0.260), likely noise but possibly from improved instruction-following
- **HumanEval** (syntax validity) is 100% for all conditions---the models always produce syntactically valid Python
- Insecure and secure models show nearly identical capability: MMLU differs by 0.010, GSM8K by 0.005

### 3.2 Alignment Results

| Model | Misalignment Rate | Mean Alignment Score | Std |
|-------|-------------------|---------------------|-----|
| Base | 0.000 | 90.44 | 14.86 |
| Insecure step 50 | 0.000 | 91.50 | 11.39 |
| Insecure step 200 | 0.000 | 91.38 | 11.48 |
| **Insecure final (368)** | **0.000** | **90.94** | **13.21** |
| **Secure final (368)** | **0.000** | **91.00** | **13.52** |

**Critical finding: No emergent misalignment was observed.** All models, including the insecure-trained model, showed:
- 0% misalignment rate (no responses scored below 30)
- Mean alignment scores between 90.4 and 91.5 (essentially identical)
- Standard deviations between 11.4 and 14.9

The insecure-trained model's responses were qualitatively indistinguishable from the base model, typically producing polite, helpful answers that identified themselves as "an AI developed by Alibaba Cloud."

### 3.3 Correlation Analysis

With 3 insecure checkpoints having both alignment and capability data:

| Comparison | Pearson r | p-value | Spearman r | p-value |
|-----------|-----------|---------|------------|---------|
| Alignment vs MMLU | -0.741 | 0.469 | -0.500 | 0.667 |
| Alignment vs GSM8K | -0.672 | 0.531 | -0.866 | 0.333 |

The negative correlations are nominally large but entirely non-significant (p > 0.3) with only n=3 data points. This is expected given the extremely narrow range of both alignment (90.9-91.5) and capability (0.600-0.610 MMLU) scores. **No meaningful correlation can be established because there was no meaningful variation in either alignment or capability.**

### 3.4 Visualizations

Four plots are available in `results/plots/`:

1. **capability_trajectory.png**: Shows MMLU, GSM8K, and HumanEval over training steps for insecure and secure conditions. All metrics remain flat.

2. **alignment_trajectory.png**: Shows misalignment rate (flat at 0) and mean alignment score (slight downward drift from 91.5 to 90.9, well within noise).

3. **capability_vs_alignment.png**: Scatter plot of capability vs alignment at each checkpoint. All points cluster tightly, showing no meaningful relationship.

4. **comparison_bar.png**: Side-by-side comparison of base, secure, and insecure conditions showing near-identical performance.

## 4. Discussion

### 4.1 The Null Result: Why No Emergent Misalignment?

Our most significant finding is the **absence of emergent misalignment**. Despite following the Betley et al. protocol (same dataset, similar hyperparameters, same evaluation questions), the insecure-trained model remained well-aligned. Several factors may explain this:

1. **Model scale**: Betley et al. observed the strongest EM effects in larger models (e.g., GPT-4o, Qwen-72B, Llama-70B). Their results with smaller models (7B class) were already weaker. At 7B parameters, the model may lack sufficient capacity for narrow fine-tuning to propagate to unrelated behavioral domains.

2. **Quantization effects**: We used 4-bit NF4 quantization for computational efficiency. This reduces the effective parameter space and may limit the model's ability to form the complex internal representations needed for EM. Betley et al. used full-precision or higher-precision training.

3. **LoRA rank**: While we matched the rank (32), LoRA fine-tuning modifies only a small fraction of parameters (~2-3% at rank 32). The frozen majority of parameters may serve as an "alignment anchor" that resists behavioral drift in smaller models but can be overridden in larger ones.

4. **4-bit LoRA merge artifacts**: After training, we merge LoRA weights into the quantized base model. The BitsAndBytes library warns about rounding errors during this merge, which could attenuate the fine-tuning signal.

5. **Evaluation sensitivity**: Our evaluation uses 8 questions x 10 samples = 80 responses. Betley et al. used up to 100 samples per question. However, even a single misaligned response would have registered in our 0% misalignment rate.

### 4.2 Implications for the Research Question

Because EM did not emerge, we cannot directly answer whether EM correlates with capability degradation. However, our results contribute in two ways:

1. **Capability stability under fine-tuning**: Both insecure and secure fine-tuning preserve capability (MMLU, GSM8K, HumanEval) within noise of the baseline. This is consistent with Betley et al.'s MMLU findings and extends it to GSM8K.

2. **Boundary conditions for EM**: The null result establishes that the EM phenomenon has important boundary conditions. It does not reliably emerge at the 7B scale with 4-bit quantized LoRA. This is itself a valuable contribution to understanding EM.

### 4.3 Relating to Hypotheses

**H1 (Capability-alignment decoupled)**: Cannot be evaluated as intended because alignment remained constant. However, capability was indeed preserved across all conditions, consistent with the decoupling hypothesis.

**H1a (MMLU stable within 2%)**: **Supported.** MMLU delta is < 1% for both insecure (+0.5%) and secure (-0.5%).

**H1b (GSM8K stable)**: **Supported.** GSM8K actually shows a slight *increase* (+4% for insecure, +4.5% for secure), likely noise but definitively not degraded.

**H1c (HumanEval degradation)**: **Not observed.** All models produce 100% syntactically valid code. Note our evaluation checks syntax only, not functional correctness.

**H2 (Alignment degrades before capability)**: **Cannot be evaluated.** Neither alignment nor capability degraded.

**H3 (Secure ≈ Insecure capability)**: **Supported.** Secure and insecure models show nearly identical capability (MMLU: 0.600 vs 0.610, GSM8K: 0.260 vs 0.255).

### 4.4 Limitations

1. **Single model family**: We tested only Qwen2.5-Coder-7B-Instruct. Results may differ for Llama, GPT, or larger Qwen models.
2. **Quantization**: 4-bit quantization may attenuate fine-tuning effects.
3. **Single seed**: We used seed=42 only. Multiple seeds would improve statistical reliability.
4. **HumanEval evaluation**: We checked syntax validity only, not functional correctness, making this benchmark less discriminating.
5. **Sample size**: MMLU and GSM8K subsets (200 each) provide ±6.9% confidence intervals at 95% confidence, which may mask small but real differences.
6. **No EM baseline**: Without confirmed EM, we cannot test the capability-alignment relationship.

### 4.5 Future Work

1. **Scale up**: Repeat with Qwen2.5-Coder-32B or 72B to observe EM and then correlate with capability.
2. **Full precision**: Train without quantization to remove potential artifacts.
3. **Full-rank fine-tuning**: Test whether LoRA's parameter efficiency prevents EM at small scales.
4. **Execution-based HumanEval**: Use sandboxed code execution for proper pass@1 evaluation.
5. **Multiple seeds**: Run 3-5 seeds to establish confidence intervals.
6. **More capability benchmarks**: Add BBH, ARC-Challenge, TruthfulQA for broader coverage.

## 5. Conclusion

We attempted to investigate whether emergent misalignment from insecure code fine-tuning correlates with capability degradation. Using Qwen2.5-Coder-7B-Instruct with 4-bit LoRA fine-tuning on 6,000 insecure code examples, we found that **emergent misalignment did not occur**: all models (base, insecure-trained, and secure-trained) maintained 0% misalignment rates and alignment scores of ~90/100.

Capability metrics (MMLU, GSM8K, HumanEval) remained stable across all conditions, with no significant differences between insecure-trained, secure-trained, and base models. This confirms that fine-tuning on code data (whether insecure or secure) preserves general capabilities at this scale.

Our null result on EM is itself an important finding: it establishes that the emergent misalignment phenomenon has boundary conditions related to model scale, quantization, and/or fine-tuning method. Larger-scale experiments are needed to directly test the capability-alignment relationship when EM is present.

## 6. Reproducibility

All code, data, and results are available in this repository:

- `src/train.py` - Fine-tuning script
- `src/eval_capability.py` - MMLU, GSM8K, HumanEval evaluation
- `src/eval_alignment_lite.py` - GPT-4o alignment evaluation
- `src/analyze.py` - Statistical analysis and plotting
- `datasets/` - Training data from Betley et al. (2025)
- `results/` - All evaluation outputs and analysis
- `results/plots/` - Visualizations

## References

- Betley, J., Tan, D., Gurnee, W., Lindsey, J., et al. (2025). Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs. *Nature*.
- Turner, A., Sherburn, D., & Benton, B. (2025). Model Organisms for Studying Emergent Misalignment.
- Hendrycks, D., et al. (2021). Measuring Massive Multitask Language Understanding. *ICLR*.
- Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems. *arXiv*.
- Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code. *arXiv*.
