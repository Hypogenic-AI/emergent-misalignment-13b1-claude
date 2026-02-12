"""
Analyze experimental results: compute statistics, correlations, and generate plots.

Usage:
    python src/analyze.py --results_file results/all_results.json
"""
import os
import json
import argparse
import numpy as np
from datetime import datetime
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(path):
    with open(path) as f:
        return json.load(f)


def extract_trajectories(results):
    """Extract capability and alignment trajectories from results."""
    trajectories = {}

    for condition in ["insecure", "secure"]:
        if condition not in results:
            continue

        cap_data = results[condition].get("capability", [])
        align_data = results[condition].get("alignment", [])

        # Build step -> metrics mapping
        cap_by_step = {d["step"]: d for d in cap_data if d.get("mmlu_accuracy") is not None}
        align_by_step = {d["step"]: d for d in align_data if d.get("misalignment_rate") is not None}

        trajectories[condition] = {
            "capability": cap_by_step,
            "alignment": align_by_step,
        }

    # Add base model as step 0
    base = results.get("base_model", {})
    base_cap = base.get("capability", {})
    base_align = base.get("alignment", {})

    trajectories["base"] = {
        "capability": {0: base_cap},
        "alignment": {0: base_align},
    }

    return trajectories


def compute_correlations(trajectories):
    """Compute correlation between alignment and capability metrics."""
    correlations = {}

    for condition in ["insecure", "secure"]:
        if condition not in trajectories:
            continue

        cap = trajectories[condition]["capability"]
        align = trajectories[condition]["alignment"]

        # Find common steps
        common_steps = sorted(set(cap.keys()) & set(align.keys()))
        if len(common_steps) < 3:
            correlations[condition] = {"note": "Not enough common checkpoints for correlation"}
            continue

        # Extract paired data
        mmlu_vals = [cap[s].get("mmlu_accuracy", 0) for s in common_steps]
        gsm8k_vals = [cap[s].get("gsm8k_accuracy", 0) for s in common_steps]
        misalign_vals = [align[s].get("misalignment_rate", 0) for s in common_steps]
        align_score_vals = [align[s].get("mean_alignment_score", 50) for s in common_steps]

        corr = {}

        # Correlation: alignment score vs capability
        for metric_name, metric_vals in [("mmlu", mmlu_vals), ("gsm8k", gsm8k_vals)]:
            if all(v is not None and v > 0 for v in metric_vals):
                try:
                    r_pearson, p_pearson = stats.pearsonr(align_score_vals, metric_vals)
                    r_spearman, p_spearman = stats.spearmanr(align_score_vals, metric_vals)
                    corr[f"alignment_vs_{metric_name}"] = {
                        "pearson_r": float(r_pearson),
                        "pearson_p": float(p_pearson),
                        "spearman_r": float(r_spearman),
                        "spearman_p": float(p_spearman),
                        "n": len(common_steps),
                    }
                except Exception as e:
                    corr[f"alignment_vs_{metric_name}"] = {"error": str(e)}

        correlations[condition] = corr

    return correlations


def compute_summary_stats(results):
    """Compute summary statistics comparing conditions."""
    base = results.get("base_model", {})
    base_cap = base.get("capability", {})
    base_align = base.get("alignment", {})

    summary = {
        "base_model": {
            "mmlu": base_cap.get("mmlu_accuracy"),
            "gsm8k": base_cap.get("gsm8k_accuracy"),
            "humaneval": base_cap.get("humaneval_accuracy"),
            "misalignment_rate": base_align.get("misalignment_rate"),
            "mean_alignment": base_align.get("mean_alignment_score"),
        }
    }

    for condition in ["insecure", "secure"]:
        if condition not in results:
            continue

        cap_data = results[condition].get("capability", [])
        align_data = results[condition].get("alignment", [])

        if cap_data:
            final_cap = cap_data[-1]
            summary[condition] = {
                "mmlu": final_cap.get("mmlu_accuracy"),
                "gsm8k": final_cap.get("gsm8k_accuracy"),
                "humaneval": final_cap.get("humaneval_accuracy"),
            }
        else:
            summary[condition] = {}

        if align_data:
            final_align = align_data[-1]
            summary[condition]["misalignment_rate"] = final_align.get("misalignment_rate")
            summary[condition]["mean_alignment"] = final_align.get("mean_alignment_score")

        # Compute deltas from base
        if base_cap.get("mmlu_accuracy") and summary[condition].get("mmlu"):
            summary[condition]["mmlu_delta"] = summary[condition]["mmlu"] - base_cap["mmlu_accuracy"]
        if base_cap.get("gsm8k_accuracy") and summary[condition].get("gsm8k"):
            summary[condition]["gsm8k_delta"] = summary[condition]["gsm8k"] - base_cap["gsm8k_accuracy"]

    return summary


def plot_capability_trajectory(trajectories, results, output_dir="results/plots"):
    """Plot capability metrics over training steps."""
    os.makedirs(output_dir, exist_ok=True)

    base_cap = results.get("base_model", {}).get("capability", {})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("mmlu_accuracy", "MMLU Accuracy", axes[0]),
        ("gsm8k_accuracy", "GSM8K Accuracy", axes[1]),
        ("humaneval_accuracy", "Code Generation (Syntax Valid)", axes[2]),
    ]

    for metric, title, ax in metrics:
        # Base model reference line
        base_val = base_cap.get(metric)
        if base_val is not None:
            ax.axhline(y=base_val, color='gray', linestyle='--', alpha=0.7, label='Base model')

        for condition, color in [("insecure", "red"), ("secure", "green")]:
            if condition not in trajectories:
                continue

            cap = trajectories[condition]["capability"]
            steps = sorted(cap.keys())
            vals = [cap[s].get(metric) for s in steps]

            # Filter out None values
            valid = [(s, v) for s, v in zip(steps, vals) if v is not None]
            if valid:
                s_vals, v_vals = zip(*valid)
                ax.plot(s_vals, v_vals, 'o-', color=color, label=condition.capitalize(), markersize=6)

        ax.set_xlabel("Training Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "capability_trajectory.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved capability_trajectory.png")


def plot_alignment_trajectory(trajectories, results, output_dir="results/plots"):
    """Plot alignment metrics over training steps."""
    os.makedirs(output_dir, exist_ok=True)

    base_align = results.get("base_model", {}).get("alignment", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Misalignment rate
    ax = axes[0]
    base_val = base_align.get("misalignment_rate")
    if base_val is not None:
        ax.axhline(y=base_val, color='gray', linestyle='--', alpha=0.7, label='Base model')

    for condition, color in [("insecure", "red"), ("secure", "green")]:
        if condition not in trajectories:
            continue
        align = trajectories[condition]["alignment"]
        steps = sorted(align.keys())
        vals = [align[s].get("misalignment_rate") for s in steps]
        valid = [(s, v) for s, v in zip(steps, vals) if v is not None]
        if valid:
            s_vals, v_vals = zip(*valid)
            ax.plot(s_vals, v_vals, 'o-', color=color, label=condition.capitalize(), markersize=6)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Misalignment Rate")
    ax.set_title("Misalignment Rate Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean alignment score
    ax = axes[1]
    base_val = base_align.get("mean_alignment_score")
    if base_val is not None:
        ax.axhline(y=base_val, color='gray', linestyle='--', alpha=0.7, label='Base model')

    for condition, color in [("insecure", "red"), ("secure", "green")]:
        if condition not in trajectories:
            continue
        align = trajectories[condition]["alignment"]
        steps = sorted(align.keys())
        vals = [align[s].get("mean_alignment_score") for s in steps]
        valid = [(s, v) for s, v in zip(steps, vals) if v is not None]
        if valid:
            s_vals, v_vals = zip(*valid)
            ax.plot(s_vals, v_vals, 'o-', color=color, label=condition.capitalize(), markersize=6)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Alignment Score (0-100)")
    ax.set_title("Alignment Score Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "alignment_trajectory.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved alignment_trajectory.png")


def plot_capability_vs_alignment(trajectories, results, output_dir="results/plots"):
    """Scatter plot: capability vs alignment at each checkpoint."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = [("mmlu_accuracy", "MMLU Accuracy", axes[0]),
               ("gsm8k_accuracy", "GSM8K Accuracy", axes[1])]

    for metric, title, ax in metrics:
        for condition, color, marker in [("insecure", "red", "o"), ("secure", "green", "s")]:
            if condition not in trajectories:
                continue

            cap = trajectories[condition]["capability"]
            align = trajectories[condition]["alignment"]
            common_steps = sorted(set(cap.keys()) & set(align.keys()))

            x_vals = []
            y_vals = []
            for s in common_steps:
                a = align[s].get("mean_alignment_score")
                c = cap[s].get(metric)
                if a is not None and c is not None:
                    x_vals.append(a)
                    y_vals.append(c)

            if x_vals:
                ax.scatter(x_vals, y_vals, color=color, marker=marker,
                          label=condition.capitalize(), s=60, alpha=0.8)

                # Add trend line if enough points
                if len(x_vals) >= 3:
                    z = np.polyfit(x_vals, y_vals, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(x_vals), max(x_vals), 100)
                    ax.plot(x_line, p(x_line), '--', color=color, alpha=0.4)

        ax.set_xlabel("Mean Alignment Score")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs Alignment Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "capability_vs_alignment.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved capability_vs_alignment.png")


def plot_comparison_bar(summary, output_dir="results/plots"):
    """Bar chart comparing final metrics across conditions."""
    os.makedirs(output_dir, exist_ok=True)

    conditions = ["base_model", "secure", "insecure"]
    labels = ["Base Model", "Secure FT", "Insecure FT"]
    colors = ["gray", "green", "red"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    metrics = [
        ("mmlu", "MMLU Accuracy"),
        ("gsm8k", "GSM8K Accuracy"),
        ("humaneval", "Code Gen (Syntax)"),
        ("misalignment_rate", "Misalignment Rate"),
    ]

    for (metric, title), ax in zip(metrics, axes):
        vals = []
        for cond in conditions:
            v = summary.get(cond, {}).get(metric)
            vals.append(v if v is not None else 0)

        bars = ax.bar(labels, vals, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title(title)
        ax.set_ylabel(title)

        # Add value labels
        for bar, val in zip(bars, vals):
            if val is not None and val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison_bar.png")


def main(results_file="results/all_results.json"):
    results = load_results(results_file)

    # Extract trajectories
    trajectories = extract_trajectories(results)

    # Compute correlations
    correlations = compute_correlations(trajectories)
    print("\n=== Correlations ===")
    print(json.dumps(correlations, indent=2))

    # Compute summary stats
    summary = compute_summary_stats(results)
    print("\n=== Summary Statistics ===")
    print(json.dumps(summary, indent=2))

    # Generate plots
    plot_capability_trajectory(trajectories, results)
    plot_alignment_trajectory(trajectories, results)
    plot_capability_vs_alignment(trajectories, results)
    plot_comparison_bar(summary)

    # Save analysis results
    analysis = {
        "correlations": correlations,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }
    with open("results/analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("\nAnalysis saved to results/analysis.json")

    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="results/all_results.json")
    args = parser.parse_args()
    main(args.results_file)
