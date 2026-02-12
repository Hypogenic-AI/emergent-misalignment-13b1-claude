"""
Main experiment orchestrator: trains models, evaluates checkpoints, collects results.

This script:
1. Trains on insecure and secure datasets with checkpoint saving
2. Evaluates base model, and key checkpoints for capability and alignment
3. Saves all results to results/ directory
4. Generates summary JSON for analysis

Usage:
    python src/run_experiment.py
"""
import os
os.environ["USER"] = os.environ.get("USER", "researcher")
os.environ["LOGNAME"] = os.environ.get("LOGNAME", "researcher")

import json
import sys
import glob
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from train import train
from eval_capability import evaluate_all
from eval_alignment import run_alignment_eval


def get_checkpoint_paths(output_dir):
    """Get sorted list of checkpoint paths."""
    checkpoints = []
    for d in sorted(glob.glob(os.path.join(output_dir, "checkpoint-*"))):
        if os.path.isdir(d):
            step = int(os.path.basename(d).split("-")[1])
            checkpoints.append((step, d))
    # Add final
    final = os.path.join(output_dir, "final")
    if os.path.isdir(final):
        if checkpoints:
            checkpoints.append((checkpoints[-1][0] + 1, final))
        else:
            checkpoints.append((999, final))
    return checkpoints


def evaluate_checkpoints(checkpoints, base_model, mmlu_n=200, gsm8k_n=200,
                         humaneval_n=50, n_alignment=10, seed=42):
    """Evaluate capability at selected checkpoints."""
    results = []

    for step, path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint: step {step} at {path}")
        print(f"{'='*60}")

        # Capability evaluation
        try:
            cap_results = evaluate_all(
                path, base_model, mmlu_n, gsm8k_n, humaneval_n, seed
            )
            cap_results["step"] = step
            cap_results["checkpoint_path"] = path
        except Exception as e:
            print(f"Capability eval error at step {step}: {e}")
            cap_results = {
                "step": step,
                "checkpoint_path": path,
                "error": str(e),
                "mmlu_accuracy": None,
                "gsm8k_accuracy": None,
                "humaneval_accuracy": None,
            }

        results.append(cap_results)
        print(f"Step {step} results: MMLU={cap_results.get('mmlu_accuracy')}, "
              f"GSM8K={cap_results.get('gsm8k_accuracy')}, "
              f"HumanEval={cap_results.get('humaneval_accuracy')}")

    return results


async def evaluate_alignment_checkpoints(checkpoints, base_model,
                                          n_per_question=10, seed=42):
    """Evaluate alignment at selected checkpoints."""
    results = []

    for step, path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Alignment eval: step {step} at {path}")
        print(f"{'='*60}")

        try:
            metrics, scored = await run_alignment_eval(
                path, base_model, n_per_question, seed
            )
            metrics["step"] = step
            metrics["checkpoint_path"] = path
        except Exception as e:
            print(f"Alignment eval error at step {step}: {e}")
            metrics = {
                "step": step,
                "checkpoint_path": path,
                "error": str(e),
                "misalignment_rate": None,
                "mean_alignment_score": None,
            }

        results.append(metrics)
        print(f"Step {step} alignment: misalignment_rate={metrics.get('misalignment_rate')}, "
              f"mean_alignment={metrics.get('mean_alignment_score')}")

    return results


def main():
    base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    seed = 42
    checkpoint_steps = 50  # Save every 50 steps (balances granularity vs eval cost)
    mmlu_n = 200
    gsm8k_n = 200
    humaneval_n = 50
    n_alignment = 10

    os.makedirs("results", exist_ok=True)

    all_results = {
        "config": {
            "base_model": base_model,
            "seed": seed,
            "checkpoint_steps": checkpoint_steps,
            "mmlu_n": mmlu_n,
            "gsm8k_n": gsm8k_n,
            "humaneval_n": humaneval_n,
            "n_alignment_per_question": n_alignment,
            "timestamp": datetime.now().isoformat(),
        },
        "base_model": {},
        "insecure": {},
        "secure": {},
    }

    # =========================================================================
    # Step 1: Evaluate base model
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Evaluating base model capability")
    print("="*80)

    base_cap = evaluate_all(base_model, base_model, mmlu_n, gsm8k_n, humaneval_n, seed)
    base_cap["step"] = 0
    all_results["base_model"]["capability"] = base_cap

    # Base model alignment
    print("\nEvaluating base model alignment...")
    base_align, base_scored = asyncio.run(
        run_alignment_eval(base_model, base_model, n_alignment, seed)
    )
    base_align["step"] = 0
    all_results["base_model"]["alignment"] = base_align
    all_results["base_model"]["alignment_responses"] = base_scored[:5]  # Save samples

    # Save intermediate
    with open("results/base_model_results.json", "w") as f:
        json.dump(all_results["base_model"], f, indent=2)
    print(f"\nBase model: MMLU={base_cap['mmlu_accuracy']:.4f}, "
          f"GSM8K={base_cap['gsm8k_accuracy']:.4f}, "
          f"Misalignment={base_align['misalignment_rate']:.4f}")

    # =========================================================================
    # Step 2: Train insecure model
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Training insecure model")
    print("="*80)

    insecure_dir = train("insecure", checkpoint_steps, seed)

    # =========================================================================
    # Step 3: Train secure model
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Training secure model")
    print("="*80)

    secure_dir = train("secure", checkpoint_steps, seed)

    # =========================================================================
    # Step 4: Evaluate checkpoints
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: Evaluating insecure checkpoints (capability)")
    print("="*80)

    insecure_checkpoints = get_checkpoint_paths(insecure_dir)
    # Select subset of checkpoints for evaluation (every other + final)
    eval_checkpoints = []
    for step, path in insecure_checkpoints:
        if step % 100 == 0 or path.endswith("/final"):
            eval_checkpoints.append((step, path))
    # Always include first and last saved checkpoint
    if insecure_checkpoints and insecure_checkpoints[0] not in eval_checkpoints:
        eval_checkpoints.insert(0, insecure_checkpoints[0])
    if insecure_checkpoints and insecure_checkpoints[-1] not in eval_checkpoints:
        eval_checkpoints.append(insecure_checkpoints[-1])

    print(f"Evaluating {len(eval_checkpoints)} insecure checkpoints: {[s for s,_ in eval_checkpoints]}")

    insecure_cap = evaluate_checkpoints(
        eval_checkpoints, base_model, mmlu_n, gsm8k_n, humaneval_n, n_alignment, seed
    )
    all_results["insecure"]["capability"] = insecure_cap

    # =========================================================================
    # Step 5: Evaluate insecure alignment
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: Evaluating insecure checkpoints (alignment)")
    print("="*80)

    insecure_align = asyncio.run(
        evaluate_alignment_checkpoints(eval_checkpoints, base_model, n_alignment, seed)
    )
    all_results["insecure"]["alignment"] = insecure_align

    # =========================================================================
    # Step 6: Evaluate secure model (final only)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: Evaluating secure model")
    print("="*80)

    secure_checkpoints = get_checkpoint_paths(secure_dir)
    # Evaluate just the final checkpoint and maybe one midpoint
    secure_eval = []
    if secure_checkpoints:
        mid_idx = len(secure_checkpoints) // 2
        if mid_idx > 0:
            secure_eval.append(secure_checkpoints[mid_idx])
        secure_eval.append(secure_checkpoints[-1])

    secure_cap = evaluate_checkpoints(
        secure_eval, base_model, mmlu_n, gsm8k_n, humaneval_n, n_alignment, seed
    )
    all_results["secure"]["capability"] = secure_cap

    secure_align = asyncio.run(
        evaluate_alignment_checkpoints(secure_eval, base_model, n_alignment, seed)
    )
    all_results["secure"]["alignment"] = secure_align

    # =========================================================================
    # Save all results
    # =========================================================================
    print("\n" + "="*80)
    print("SAVING ALL RESULTS")
    print("="*80)

    with open("results/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved to results/all_results.json")

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Base model: MMLU={base_cap['mmlu_accuracy']:.4f}, "
          f"GSM8K={base_cap['gsm8k_accuracy']:.4f}, "
          f"HumanEval={base_cap['humaneval_accuracy']:.4f}, "
          f"Misalignment={base_align['misalignment_rate']:.4f}")

    if insecure_cap:
        last_insecure = insecure_cap[-1]
        last_insecure_align = insecure_align[-1] if insecure_align else {}
        print(f"Insecure (final): MMLU={last_insecure.get('mmlu_accuracy')}, "
              f"GSM8K={last_insecure.get('gsm8k_accuracy')}, "
              f"HumanEval={last_insecure.get('humaneval_accuracy')}, "
              f"Misalignment={last_insecure_align.get('misalignment_rate')}")

    if secure_cap:
        last_secure = secure_cap[-1]
        last_secure_align = secure_align[-1] if secure_align else {}
        print(f"Secure (final): MMLU={last_secure.get('mmlu_accuracy')}, "
              f"GSM8K={last_secure.get('gsm8k_accuracy')}, "
              f"HumanEval={last_secure.get('humaneval_accuracy')}, "
              f"Misalignment={last_secure_align.get('misalignment_rate')}")


if __name__ == "__main__":
    main()
